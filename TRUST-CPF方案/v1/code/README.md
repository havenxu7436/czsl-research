# 代码

完整执行源码位于[trustcpf_clip_v1](trustcpf_clip_v1/)，共包含CLI、核心模块、服务器脚本、测试和CLIP源码。

| 模块 | 职责 |
|---|---|
| [run.py](trustcpf_clip_v1/run.py) | 各独立阶段与pipeline入口 |
| [trustcpf/](trustcpf_clip_v1/trustcpf/) | 数据、缓存、知识图、模型、配对效用、训练、评价与报告 |
| [scripts/](trustcpf_clip_v1/scripts/) | prepare、单seed训练、噪声队列、公式文档提取及打包 |
| [tests/test_trustcpf.py](trustcpf_clip_v1/tests/test_trustcpf.py) | 数学恒等式、梯度、划分、全流程与恢复测试 |
| [vendor/clip/](trustcpf_clip_v1/vendor/clip/) | 冻结CLIP双编码器与tokenizer实现 |
| [requirements.txt](trustcpf_clip_v1/requirements.txt) | Python依赖 |

运行前在仓库根目录执行`python tools/reproduce.py materialize --line TRUST-CPF方案 --stage v1 --dest /srv/trustcpf-replay-v1`，再进入其`trustcpf_clip_v1`。配置、知识、词表和文档会恢复到代码所需的原始相对路径。[公式及核心代码](../TRUST-CPF公式及代码v1.md)中的摘录来自同一份源码。
