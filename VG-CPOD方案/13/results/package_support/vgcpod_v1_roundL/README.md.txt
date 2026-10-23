# VG-CPOD L组 Conditional Knowledge v2

以历史 G0 CPF＋静态 CP 为共同基底，比较 Base、Knowledge、Shuffle。每组三个 seed，分别使用物理 GPU1/2/3。正式训练从随包 G0 权重开始，读取服务器已有 CLIP 全局特征缓存。

- `SERVER_COMMANDS.md`：从 Windows 上传、服务器启动、恢复、汇总、打包到本地下载的完整命令。
- `REVISED_PLAN.md`：合理性复核与已落实的两项优化。
- `FORMULAS_AND_CODE.md`：公式、核心实现和结果指标。
- `run.py`：setup / train / status / report。
- `initial/`：三个历史 G0 权重及其原始 protocol。
- `assets/`：真实知识、固定打乱知识、G0 缓存指纹和来源。
- `validation/`：本机代码验证记录；正式服务器结果保存在 `work/`。

最终配方冻结物体评分和 primitive 投影，训练共同条件读出与知识消息。共同条件矩阵学习率为2e-4、CP读出为6e-5，消息矩阵3e-4、alpha1e-3；10 epochs。原方向文档中自由更新 Au/Av、较高续训学习率的设置已由本版替换。

本机验证覆盖真实 G0 权重的初始输出一致性、消息公式、冻结参数与梯度、固定知识打乱、VAL 校准传递、九组训练汇总和中断续训。流水线测试用合成视觉特征；真实 UT-Zappos 特征仍在服务器，本机未运行正式9组实验。
