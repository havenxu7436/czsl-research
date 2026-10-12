# 15 · TRAIN 逐句知识审查

状态：36 个逐句视图与图像审查已完成。

## 实验设置

- 输入：第 14 阶段冻结 CLIP 产物与 reviewed-v2 描述。
- 范围：TRAIN；统计属性描述、对象分组与对应图像。

## 流程

1. 读取冻结探针分数与逐句输入。
2. 计算各描述的分类与纠正/新增错误统计。
3. 生成逐句视图、对象分组和图像审查结果。

## 运行入口

变量与路径对应关系见 [环境说明](../../environment/README.md)。

```bash
REPLAY=/srv/czsl-reproduce-15
python tools/reproduce.py materialize --line 前期实验 --stage 15 --dest "$REPLAY"
WORK=/var/tmp/czsl_train_knowledge_audit_20261010
PROBE_WORK="${CZSL_CLIP_PROBE_WORK:?Set stage-14 work directory}"
PACKAGE="$REPLAY/recovered_containers/train_knowledge_audit_release_20261010/train_knowledge_audit_20261010"
"$CPF_PY" "$PACKAGE/run.py" --probe-work "$PROBE_WORK" --check-inputs-only
"$CPF_PY" "$PACKAGE/run.py" --probe-work "$PROBE_WORK" --work "$WORK" --export "$WORK.zip"
```

## 结果与文件

AE 第三条描述存在局部增量，净增量主要集中于 Leather。

- [执行代码](code/train_knowledge_audit_20261010/)
- [知识、配置与结果索引](inputs.json)
- [执行命令](runbook.json)
