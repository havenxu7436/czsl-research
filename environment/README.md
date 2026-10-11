# 复现环境与外部输入

## 已知环境

后期 CPF 主线记录为 Linux、Python 3.8.18、PyTorch 1.13.1+cu117、torchvision 0.14.1+cu117、NumPy 1.24.4、SciPy 1.10.1、RTX 3090。原环境记录在 `records/`；pip freeze 是历史快照，不是已经验证的跨平台安装锁文件。基础源码不同快照及 CLIP 探针可能有其他依赖，不要将所有版本强行替换成最新库。

CPU 可执行完整性校验与布局恢复；正式训练需要对应 CUDA 环境、数据与授权 GPU。最低资源和原预算见各阶段 protocol/README。

## 外部文件

| 输入 | 约束 |
|---|---|
| UT-Zappos、MIT-States、C-GQA 图像与 metadata/split | 使用原 CPF 数据加载器所需布局和同一划分；只有少量审查图与结果标签在本仓库 |
| `glove.6B.300d.txt` | 300 维，必须通过原 manifest 记录的 SHA256；不是下载 ZIP 的哈希 |
| ViT B16 预训练权重 | 匹配对应 CPF 配置，不用不同 backbone 替代 |
| 前序训练 best checkpoint、续训 model/Adam/RNG | 第 05–09、部分诊断/pilot、第 18 阶段依赖旧 run；缺失时先重跑对应前序训练，不能把分数数组冒充权重 |

未包含的大资源及历史位置见 [external_inputs.csv](../provenance/external_inputs.csv)。该索引用于确认缺项；复跑前仍须取得资源、核对身份与哈希。

## 独立运行目录

从仓库根目录执行，以第 17 阶段为例：

```bash
python tools/reproduce.py verify
python tools/reproduce.py materialize --stage 17 --dest /srv/czsl-reproduce-17
python tools/reproduce.py doctor --stage 17 --project /srv/czsl-reproduce-17/project
```

`materialize` 恢复累计前序源码、配置、知识和证据，同时恢复原校验器所需的历史说明文件。只创建新目录，不训练、不覆盖旧目录。无需 ZIP 解包部署。`doctor` 会报告缺失外部输入和缓存内冻结路径，但不能代替包内数据划分检查、真实 TRAIN 图检查或实际训练。

## 历史绝对路径：必须处理

原配置可用 `--root`、`--work` 和环境变量设定新位置，但部分缓存的 `input_sha256`、前序 run 和代码参考仍记录 `/data/xhw/czsl_repro`、`/var/tmp/...` 等绝对路径。**仅改命令行 root 不一定够。**

严格按历史字节复跑时，应在自己的隔离环境中将相应目录挂载到原路径，或由管理员建立明确、无冲突的路径映射。原路径不存在与数据哈希不符是两种不同错误，先运行 doctor，再运行包内 prepare/preflight。不要覆盖共享服务器上他人的目录，工具不会自动创建根目录链接。

部分 Bash 入口固定解释器或默认输出路径；第 14、15 阶段还将输出目录名写入冻结 protocol。
它们不能通过任意改 `--work` 变为可迁移包。使用隔离环境恢复对应路径和前序产物，
或另建明确标注迁移的版本；已有工作目录不得删除或覆盖以规避校验。

如果无法提供原路径，就用对应知识构建器在新路径重建并冻结新的 manifest/protocol，明确记录是迁移后的新复现，不改历史证据、源文件预期哈希或科学配置。不能声称新序列化哈希等于原缓存。

第 16 阶段旧队列默认截止日期已过去，新复跑必须显式设置未来截止时间。第 17 阶段无此历史 deadline，但必须新建 work、先 prepare 后 run_suite；42 次训练仍是计划，不是归档成绩。

## 结果验收

按原数据集、split、seed、预算、知识哈希和首次严格最大 VAL CW AUC 选模规则复核；同 checkpoint 评价 VAL/TEST CW/OW。保留负结果、失败和中断。GPU 训练的统计一致性不等于逐比特确定性，CPU 测试通过不等于研究结论已复现。
