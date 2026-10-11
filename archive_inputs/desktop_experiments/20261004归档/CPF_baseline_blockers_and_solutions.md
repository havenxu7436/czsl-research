# CPF 基线复现：阻塞问题、解决方案与验收门禁

更新日期：2026-09-28

适用输入：`CRMask_copy.tar.gz`，目标为论文中使用 ViT-B/16 + GloVe 的 CPF，而不是 CLIP 版 CPF。

## 0. 结论

当前压缩包不能不加修改地复现论文 CPF。它可以通过 Python 语法编译，但训练入口缺少评测器，测试入口缺少两个模块；更关键的是，`train.py`/`test.py` 实际导入的 `models/ddm_v2.py` 在普通训练和验证路径里没有启用论文的 TEO（text-enhanced object learning）和 OGA（object-guided attribute learning）。直接把缺失 import 补齐后开跑，只会得到一个可以运行但算法不等于完整 CPF 的结果。

应把任务分成两层：

1. **可运行的重构版 CPF**：依据主论文 Eq. 1-8 和补充材料 Algorithm 1 重构模型，恢复评测和 checkpoint；可以完成可信的工程复现。
2. **逐数值严格复现**：除上述工作外，还需要作者确认或提供官方完整源码、准确 seed、学习率衰减时点、归一化方式、组合投影实现和官方 evaluator 版本。目前这些信息不完整。

在第 6 节正式训练前，必须依次通过本文的 P0、P1、P2 门禁。

## 1. 论文实现必须满足的不可变条件

| 项目 | 论文要求 |
|---|---|
| Backbone | fine-tuned ViT-B/16 |
| 深层特征 | 第 12 block 的 class token 和 patch tokens |
| 浅层特征 | 第 3、6、9 block 的 patch tokens |
| 文本特征 | 300 维 GloVe，冻结 |
| TEO | Eq. 1-2：深层 class token 选择 object 文本描述，再用该描述关注深层 patch，得到增强 object feature |
| OGA | Eq. 4：增强 object feature 关注浅层 patch，得到 attribute feature |
| 损失 | `Lcom + 0.6 * Latt + 0.4 * Lobj` |
| 温度 | 三项交叉熵共同使用 `tau=0.05` |
| 推理 | composition、attribute、object 三路得分相加 |
| 优化器 | Adam |
| 训练长度 | 三个数据集均为 10 epoch |
| 主学习率 | `1e-4`，衰减系数 `0.1` |
| 评测 | CW 与 OW；AUC、best HM、Seen、Unseen |

## 2. P0：程序无法形成完整训练闭环的问题

### P0-1：训练入口缺少 `evaluator_ge.py`

**现象**

```text
ModuleNotFoundError: No module named 'evaluator_ge'
```

**原因**

`train.py` 第 14 行导入 `evaluator_ge`，但压缩包中没有该文件。压缩包结构明显继承自 CoT/OADis 风格代码，其 evaluator 接口是 `Evaluator.score_model(...)` 与 `Evaluator.evaluate_predictions(...)`。

**解决方案**

首选向 CPF 作者取得实验实际使用的 `evaluator_ge.py` 并记录 SHA256。为了先打通工程链路，可暂用 CoT 官方 evaluator 作为兼容候选，但必须在实验记录中标为第三方依赖，不能声称它已经等同于 CPF 官方 evaluator：

```bash
cd "$CPF_REPRO"
curl -fL \
  'https://raw.githubusercontent.com/HanjaeKim98/CoT/master/evaluator_ge.py' \
  -o evaluator_ge.py
sha256sum evaluator_ge.py | tee evaluator_ge.sha256
```

**验收**

```bash
test -s evaluator_ge.py
python -c 'import evaluator_ge; print("EVALUATOR IMPORT PASSED")'
```

### P0-2：`test.py` 导入不存在的可视化数据集和评测器

**现象**

```text
ModuleNotFoundError: No module named 'dataset_visualization'
```

随后还会缺少 `evaluator_ge_v.py`。

**原因**

`test.py` 是可视化实验残留版本，但论文数值评测不需要这两个可视化模块。

**解决方案**

数值评测入口应改为：

```diff
-from dataset_visualization import CompositionDataset
-import evaluator_ge_v
+from dataset import CompositionDataset
+import evaluator_ge
```

并把：

```diff
-evaluator_test_ge = evaluator_ge_v.Evaluator(testset, model)
+evaluator_test_ge = evaluator_ge.Evaluator(testset, model)
```

删除未使用的 `COT`、`sys`、`torch.nn` import。论文表格按 top-1 评测；调试时可打印 top-2/top-3，但正式记录必须单独保留 top-1。

**验收**

```bash
python -m compileall -q .
python test.py --help
```

### P0-3：训练过程不返回 AUC/HM，也不保存模型

**现象**

- `validate_ge()` 中组装预测、计算 AUC/HM、返回结果的主体被整段注释。
- `save_checkpoint(...)` 调用被注释。
- 训练结束时 `best_records` 永远为 0。

**解决方案**

恢复以下完整闭环：

1. `validate_ge()` 汇总所有 batch 的 pair score。
2. 调用 evaluator 得到 `AUC`、`best_hm`、`best_seen`、`best_unseen`。
3. 每个 epoch 只在 validation set 上选择最佳模型。
4. 当 validation AUC 提升时保存 `best_auc.pth`。
5. 训练完成后加载 `best_auc.pth`，只在 test set 上做最终报告。

不要用 test AUC 选择 checkpoint，否则会产生测试集泄漏。建议保存内容包括模型、优化器、epoch、config 和随机种子，而不是只有裸 `state_dict`。

**验收**

```bash
grep -n "return stats\['AUC'\], stats\['best_hm'\]" train.py
grep -n "best_auc.pth" train.py
```

单 epoch 后必须同时存在日志和非空 checkpoint：

```bash
test -s "$RUN_ROOT/cpf/smoke/ckpt/ut-zappos-czsl_utz_seed0/best_auc.pth"
```

## 3. P1：代码能跑但不等于论文 CPF 的问题

### P1-1：训练入口选择了错误/不完整的模型路径

**证据**

`train.py` 与 `test.py` 都导入：

```python
from models.ddm_v2 import DDM
```

但是 `ddm_v2.py` 的正常 `train_forward()` 和 `val_forward()` 中，TEO 与 OGA 关键计算均被注释，实际使用的是深层 class token 分别做 object/attribute 分类。这更接近 `-TEO-OGA` 消融，而不是完整 CPF。

配置中 `aug_epoch=37`，而论文只训练 10 epoch，所以包含部分附加模块的 `train_forward_augment()` 永远不会正常进入。即使强行进入，它还存在 `2304 -> 300` 输入却使用 `768 -> 300` Linear 的维度错误，并额外加入论文 Eq. 7 中没有的 mix loss。

**解决方案**

不要通过随意取消注释修复 `ddm_v2.py`。新建唯一的 `models/cpf.py`，让训练和验证共享同一个 `encode_cpf_features()`，严格按论文/补充材料实现 TEO 与 OGA，然后令 `train.py`、`test.py` 都只导入该模型。保留旧 `ddm*.py` 作为消融参考，但不再作为正式入口。

### P1-2：仓库中的其他 `ddm*.py` 也不能直接视为完整 CPF

`models/ddm.py` 虽然启用了文本到深层 patch 的注意力，但它在浅层 attribute 学习中使用 attribute textual guidance；论文 Table 7 明确把这种做法作为消融，而完整 CPF 应使用增强后的 object visual feature 指导 attribute 学习。

`ddm_v3.py` 混入 CLIP prompt 逻辑，不是当前 ViT-B/16 + GloVe 基线。`ddm_v4.py` 使用另一套 backbone/层映射。不能仅靠更改 import 在这些文件之间试跑来确定“哪个结果最好”。

### P1-3：注意力辅助函数包含确定性错误

`models/ddm_utils.py:image_soft_weights()` 在计算前把输入覆盖为全 1：

```python
query = torch.ones(size=query.size()).cuda()
key = torch.ones(size=key.size()).cuda()
```

这会把注意力退化为均匀权重，不能实现论文注意力。

**解决方案**

删除该覆盖逻辑，并使用有尺度的点积注意力：

```python
def scaled_attention(query, keys):
    scale = query.shape[-1] ** -0.5
    logits = torch.einsum('bd,bnd->bn', query, keys) * scale
    return torch.softmax(logits, dim=-1)
```

禁止在 helper 内部硬编码 `.cuda()`；设备应从输入 tensor 继承。

### P1-4：完整 CPF 的张量流程必须重新实现并逐层断言

ViT-B/16 输入 `224x224` 时建议按以下形状实现：

```text
deep_cls              [B, 768]
deep_patches          [B, 196, 768]
shallow_l3/l6/l9      [B, 196, 768]
object/attribute text [N or M, 300]
q_t                   [B, 300]
v_o                   [B, 768]
three shallow sums    3 x [B, 768]
concat shallow        [B, 2304]
v_a projected         [B, 300]
object projected      [B, 300]
composition feature   [B, 300]
```

实现顺序：

1. `deep_cls -> Linear(768,300)`，与 object GloVe 做 `1/sqrt(300)` scaled attention，得到 `q_t`。
2. `deep_patches -> Linear(768,300)`；`q_t` 关注 projected deep patches，但加权求和使用原始 768 维 deep patches；与 `deep_cls` 残差相加得到 `v_o`。
3. 用 768 维 `v_o` 分别关注第 3、6、9 block 的 768 维 patches，三个加权结果拼成 2304 维。
4. `Linear(2304,300)` 得到 attribute visual feature。
5. `Linear(768,300)` 投影 `v_o`，用于 object 分类。
6. attribute/object visual features拼接后映射到 300 维 composition feature。

必须给上述每一步添加 shape assertion；不要依赖注释里的旧 shape。

### P1-5：损失函数必须去除压缩包中的非论文项

正式 CPF 应为：

```python
loss = loss_composition + 0.6 * loss_attribute + 0.4 * loss_object
```

三路 cosine logits 都使用 `tau=0.05` 再做 cross entropy。压缩包 `train_forward_augment()` 中的 mix loss、随机增强分支和 `aug_epoch` 不属于论文 Eq. 7 的公开定义；除非作者确认，不应混入 CPF baseline。

**验收**

运行一个真实 batch 后必须打印且有限：

```text
loss_composition
loss_attribute
loss_object
loss_total
```

并验证：

```python
assert torch.isfinite(loss_total)
loss_total.backward()
assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
```

### P1-6：推理三路加和必须与评测 pair 顺序一致

推理应对每个候选 pair `(a,o)` 计算：

```text
score_pair(a,o) = score_composition(a,o)
                + score_attribute(a)
                + score_object(o)
```

必须确保三个分支使用同一批 `dset.pairs` 索引，并分别测试：

- CW：训练 pair + 当前 val/test unseen pair。
- OW：完整 `attrs x objs` 笛卡尔积。

注意：论文文字称这些量为 probability，而压缩包评测路径相加的是未缩放 cosine score。官方仓库未公开完整实现，这一点需要作者确认；重构实验必须把采用的方式写入日志和报告。

## 4. P2：超参数与严格复现问题

### P2-1：epoch 数错误

论文：所有数据集 10 epoch。

压缩包：UT-Zappos 为 10，MIT-States 与 C-GQA 为 20。

正式配置必须统一：

```yaml
TRAIN:
  max_epoch: 10
  final_max_epoch: 10
```

### P2-2：学习率衰减在 10 epoch 内不会发生

压缩包里程碑为 `[23,30]`，因此 10 epoch 内完全不衰减；论文只报告“衰减系数 0.1”，没有报告衰减发生在哪一轮。

**严格复现方案**：向作者确认 milestone/scheduler，未确认前把它列为未决项。

**工程重构方案**：可以选择并固定一个显式 milestone（例如第 8 epoch），但实验名称必须标记为 `reconstructed`，不能声称逐数值复现论文。

### P2-3：backbone 学习率与论文叙述不一致

论文仅报告 ViT-B 学习率 `1e-4`；压缩包给 backbone `1e-5`、其他层 `1e-4`。严格复现需作者确认差分学习率。按论文文字重构时，可先把所有可训练参数设为 `1e-4`，同时单独记录这是基于论文文字的实现选择。

GloVe 必须保持冻结，因此 `lr_word_embedding` 不应创建一个空的 optimizer 参数组。

### P2-4：seed 与确定性不足

- UT-Zappos、C-GQA 配置为 `3351`。
- MIT-States 配置为 `-1`，每次随机生成 seed。
- `main_worker()` 又把 `torch.backends.cudnn.benchmark=True`，会削弱确定性。

**解决方案**

先统一用 seed 0 做工程验收；正式论文对比需向作者确认 seed，并至少跑 3 个 seed 报告均值和标准差。确定性设置：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

记录 Python、PyTorch、CUDA、cuDNN、GPU 和 commit/hash。

### P2-5：batch size、weight decay、归一化未被论文充分说明

压缩包使用 MIT batch 32、UT/C-GQA batch 64、weight decay `5e-5`，但主论文没有完整说明这些值。官方 GitHub issue 也公开询问了学习率衰减、组合投影和特征归一化，当前没有作者答复。

工程重构可先沿用压缩包值；严格复现必须把它们列入作者确认清单。

## 5. 数据、权重与运行资源门禁

### 5.1 验证所有 metadata 图像路径

在服务器执行：

```bash
conda activate cpf
export CZSL_ROOT="/data/$USER/czsl_repro"
export DATA_ROOT="$CZSL_ROOT/data"

DATA_ROOT="$DATA_ROOT" python - <<'PY'
import os
import torch

root = os.environ['DATA_ROOT']
for ds in ('mit-states', 'ut-zappos', 'cgqa'):
    meta_path = os.path.join(root, ds, 'metadata_compositional-split-natural.t7')
    meta = torch.load(meta_path, map_location='cpu')
    valid = [x for x in meta if x['attr'] != 'NA' and x['set'] != 'NA']
    missing = [x['image'] for x in valid
               if not os.path.isfile(os.path.join(root, ds, 'images', x['image']))]
    print(ds, 'valid=', len(valid), 'missing_images=', len(missing), 'examples=', missing[:5])
    assert not missing, (ds, missing[:20])
print('IMAGE PATH CHECK PASSED')
PY
```

### 5.2 验证 GloVe 与 ViT-B/16 权重

```bash
test -s "$DATA_ROOT/glove/glove.6B.300d.txt"
test "$(wc -l < "$DATA_ROOT/glove/glove.6B.300d.txt")" -eq 400000

test -s "$HOME/.cache/torch/hub/checkpoints/B_16.pth"
sha256sum "$HOME/.cache/torch/hub/checkpoints/B_16.pth"
```

如果服务器不能访问 GitHub，应在本地下载 `B_16.pth` 后上传到上述精确路径；不要让多个训练进程同时下载同一个权重文件。

## 6. 建议执行顺序

### 阶段 A：只修到可导入

```bash
conda activate cpf
cd "$CPF_REPRO"

python -m compileall -q .
python -c 'from dataset import CompositionDataset; print("DATASET IMPORT PASSED")'
python -c 'import evaluator_ge; print("EVALUATOR IMPORT PASSED")'
python -c 'import train; print("TRAIN IMPORT PASSED")'
```

### 阶段 B：算法单元测试

必须分别验证：

1. TEO attention 权重每行和为 1，且不是均匀常数。
2. OGA attention 权重每行和为 1，且梯度可回传到 object 分支。
3. 三个 loss 都有限。
4. 推理 pair score 维度在 CW/OW 下正确。
5. 同一输入、同一 seed 重复两次输出一致。

### 阶段 C：一个真实 batch forward/backward

先用 UT-Zappos，`batch_size=2`、`num_workers=0`，执行一个 batch；通过后再做完整单 epoch。

### 阶段 D：UT-Zappos 单 epoch 冒烟

冒烟只验证链路，不用于论文对比：

```bash
mkdir -p "$RUN_ROOT/cpf/smoke/log" "$RUN_ROOT/cpf/smoke/ckpt"

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --cfg config/ut-zappos.yml \
  DATASET.root_dir "$DATA_ROOT/ut-zappos" \
  TRAIN.log_dir "$RUN_ROOT/cpf/smoke/log" \
  TRAIN.checkpoint_dir "$RUN_ROOT/cpf/smoke/ckpt" \
  TRAIN.train_name utz_seed0 \
  TRAIN.seed 0 \
  TRAIN.max_epoch 1 \
  TRAIN.final_max_epoch 1 \
  TRAIN.batch_size 8 \
  TRAIN.test_batch_size 32 \
  TRAIN.num_workers 0 \
  EVAL.open_world False \
  2>&1 | tee "$RUN_ROOT/cpf/smoke/ut-zappos.log"
```

验收条件：

- 三项 loss 和总 loss 都有限。
- 完成 forward/backward，无 shape/device 错误。
- validation 输出 top-1 AUC/HM/Seen/Unseen。
- `best_auc.pth` 非空且能重新加载。
- 单独运行 `test.py` 得到一致指标。

### 阶段 E：10 epoch 正式训练

只有在学习率计划、模型实现、evaluator 版本和 seed 记录完成后，才能开始三数据集正式训练。CW 与 OW 应使用同一个训练 checkpoint，只改变候选组合集合，不应为 OW 重新训练。

## 7. 必须向作者确认/索取的项目

1. 完整 CPF ViT-B 源码及对应 commit/hash。
2. 正式使用的 `evaluator_ge.py`、`test.py` 和配置文件。
3. 三个数据集的 seed；论文指标是单次还是多次平均。
4. 10 epoch 内学习率衰减发生的 epoch，或实际 scheduler 定义。
5. backbone 是否使用单独的 `1e-5` 学习率。
6. Adam weight decay、batch size、beta/epsilon。
7. Eq. 3/5/6 前是否 L2 normalize。
8. 推理 Eq. 8 相加的是 softmax probability、温度缩放 logits，还是未缩放 cosine score。
9. composition visual/text projection的准确层结构及 attribute/object 拼接顺序。
10. 论文表格 checkpoint 的 SHA256 或官方权重。

## 8. 当前状态判定

| 门禁 | 当前状态 |
|---|---|
| 数据 split | 已通过 |
| 原始图片路径 | 待服务器检查 |
| GloVe | 待服务器检查 |
| CPF 环境 | 已配置，仍需版本记录 |
| Python 语法编译 | 通过 |
| 训练 import | 失败：缺 `evaluator_ge.py` |
| 测试 import | 失败：缺两个模块 |
| 完整 TEO/OGA | 未满足 |
| AUC/HM 闭环 | 被注释 |
| checkpoint 闭环 | 被注释 |
| 论文超参数一致性 | 未满足 |
| 严格数值复现 | 当前仍被作者未公开信息阻塞 |

