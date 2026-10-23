# v2 代码

| 功能 | 文件 |
| --- | --- |
| CLI与队列状态 | [run.py](trustcpf_conditional_v2/run.py) |
| G0、知识消息与条件残差 | [model.py](trustcpf_conditional_v2/ckcpf/model.py) |
| 训练、选模、续训和报告 | [runtime.py](trustcpf_conditional_v2/ckcpf/runtime.py) |
| CW/OW校准与机制指标 | [evaluation.py](trustcpf_conditional_v2/ckcpf/evaluation.py) · [metrics.py](trustcpf_conditional_v2/ckcpf/metrics.py) |
| 缓存验证与样本轴 | [data.py](trustcpf_conditional_v2/ckcpf/data.py) |
| GPU1/2/3三seed队列 | [run_all.sh](trustcpf_conditional_v2/scripts/run_all.sh) · [run_seed.sh](trustcpf_conditional_v2/scripts/run_seed.sh) |
| 打包与UTF-8解压 | [pack_results.py](trustcpf_conditional_v2/scripts/pack_results.py) · [extract_results.py](trustcpf_conditional_v2/scripts/extract_results.py) |
| 五项实现测试 | [test_experiment.py](trustcpf_conditional_v2/tests/test_experiment.py) · [G0原实现对照](trustcpf_conditional_v2/tests/reference_g0.py) |

执行前按[版本入口](../README.md)恢复完整包；此目录与configs、initial在仓库中分区保存。
