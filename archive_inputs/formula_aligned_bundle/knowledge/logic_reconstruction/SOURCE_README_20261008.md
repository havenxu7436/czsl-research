# CPF + LOGICZSL B/F 独立执行包

日期：2026-10-08。范围：UT-Zappos、原 CPF/GloVe、A_ref/B/E_ref/F。不覆盖旧 ACDE，不修改 EKR。

这是依据原论文公式重建逻辑损失、以新知识资产适配 CPF 的实验包，不是原始 Logic-Troika 结果的精确复现。完整操作见同包的 `操作手册.md`。

## 包含内容

- `assets/ut-zap50k/knowledge.json`：12 对象、16 属性、5 个 OI 父类、每个 primitive 三条 OE/AE 描述。由助手按论文方法重建，需要使用者审核，不是原作者遗失数据。
- `model/logic_loss.py`：论文 Eq.6/9/12，q=3；与本地作者 `logicloss.py` 数值对照通过。
- `model/bf_logic.py`：复用 CPF 视觉分支与 cosine classifier；固定 GloVe 知识节点；无新增可训练参数；β=0.01 只乘一次。
- `prepare.py`：校验原源码版本，只在独立目录复制并接入逻辑，生成四组配置与冻结协议。
- `smoke.py`：被复制到新源码的 `tests/smoke_bf.py`，真实 TRAIN 图像 CUDA 恒等/梯度检查。
- `run.py`：训练 10 轮、按 VAL CW AUC 选模、评价 VAL CW/OW；不提供 TEST 调参入口。
- `summarize.py`：将真实训练证据和配对结果写成 `.md`、`.csv`，不把未完成任务写成成功。
- `test_release.py`：CPU 公式、梯度、源码补丁前向、准备/汇总流程测试。随机张量与流程 fixture 不是训练结果。

## 复现边界

保留原论文的三项逻辑损失、三条描述、q=3、β=0.01；适配层改为现有 CPF 的 GloVe 和视觉分支，保留 CPF 的 `L_comp + 0.4 L_obj + 0.6 L_attr` 与 legacy 推理方式。训练预算沿用 ACDE 的 10 轮，而不是论文 UT 的 15 轮；F 保留旧 E 的文本缓存与 λ=0.1。新知识只进入逻辑损失，不用于替换旧 E 的文本/校准缓存。

所有需要覆盖已有配置、bank、训练结果的操作都会拒绝执行。源码版本守卫失败时，不要删除守卫或覆盖原源码，请保留报错核对版本。

## 来源

- [LOGICZSL，CVPR 2025 原论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_LOGICZSL_Exploring_Logic-induced_Representation_for_Compositional_Zero-shot_Learning_CVPR_2025_paper.pdf)，§3.2-3.4、Eq.6/9/12/13、Fig.4。
- [作者仓库](https://github.com/Pieux0/LOGICZSL)，核查日期 2026-10-08。
- 用户本地 `LogiCZSL_copy.tar.gz` 中 `model/logicloss.py` 用于公式数值核对；没有在包中声称恢复原作者知识 JSON。

状态：本地 CPU 检查通过；真实 GloVe 编码覆盖率检查通过；真实图像 CUDA 冒烟与完整 GPU 训练尚未执行。当前会话 SSH 认证失败，不能伪造服务器结果。
