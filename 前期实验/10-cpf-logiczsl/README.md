# 10 · CPF＋LogiCZSL B/F

状态：四组×10 epoch 已完成。

## 实验设置

- UT-Zappos；seed=3351；A_ref、B、E_ref、F。
- 逻辑损失：OI/OE/AE，q=3，β=0.01；F 使用文本缓存与 λ=0.1。
- CPF 任务损失为 L_comp + 0.4 L_obj + 0.6 L_attr。

## 流程

1. 构建逻辑 bank，生成四组配置。
2. 执行 TRAIN 图像梯度检查与 10 epoch 训练。
3. 按 VAL CW AUC 选模并评价 VAL CW/OW。

## 结果与文件

B/F 相对匹配参考组的指标增量见原始运行记录。

- 执行代码（精简版未保留）
- 知识、配置与结果索引（精简版未保留）
- 执行命令（精简版未保留）
