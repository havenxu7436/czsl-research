# 03 · 描述删除模块

状态：代码已归档；无训练结果。

## 实验设置

- 实现 OE/AE 描述槽删除；名称、prior、pair 模板、logic bank 与 OI 保持原版本。
- 代码接口读取 residual 父批次，输出独立描述缓存与实验目录。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-0601-03
python tools/reproduce.py materialize --line 0601方案 --stage 03 --dest "$REPLAY"
"$CPF_PY" "$REPLAY/extend_0601_20261011/extension.py" --help
```

## 结果与文件

本阶段保留模块代码与接口信息。

- [执行代码](code/extend_0601_20261011/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
