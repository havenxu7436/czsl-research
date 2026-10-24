# TRUST-CPF R5 v1（10 epoch）

在上一轮强 CPF / Competition 10ep 上执行相对效用拟合。五组为 Uniform、Absolute-U-NLL、Relative-U-NLL 和两组匹配概率置换，三个seed，统一10epoch。已有教师、测量、响应和匹配控制结果自动复用。

- `SERVER_COMMANDS.md`：上传、配置、0/1/2卡训练、打包和本地下载。
- `FORMULAS_AND_CODE.md`：公式、核心实现、实验协议和指标。
- `trustcpf/r5.py`：教师内、类型内中心化岭回归。
- `trustcpf/r5_training.py`：Q5审计、VAL强度选择、识别结果复用与新适配。
- `trustcpf/r5_workflow.py`：27阶段断点队列及15组结果汇总。

服务器源目录：`/data/xhw/experiments/trustcpf_competition_v1_10ep`。
特征缓存：`/data/xhw/vgcpod_v1_runs/cache_ut_clip_l14`。
新项目：`/data/xhw/experiments/trustcpf_r5_v1_10ep`。

本地35项测试中34项通过，1项CLIP重新提取测试因特征复用跳过。真实测量数据的三seed Absolute复现通过。新的UT识别结果由服务器运行生成。
