# 逻辑知识银行的重建输入与构建程序

这里归档的是 **B/F/H 逻辑资产** 的原始输入与 builder，不是另一个训练包。实际使用的已构建 bank 在 `../../code/faithful_0601_20261011/assets/logic_bank/`。

- `knowledge.json`：paper-guided 重建 OI/OE/AE，非作者遗失知识；12对象、16属性、5父类，每原语3条描述。
- `labels.json`：与 bank manifest 指纹相同的 primitive 顺序。
- `build_bank.py`、`common.py`：当时实际 builder 与 validation helper，原样保存。
- `SOURCE_README_20261008.md`：旧 B/F release 的历史说明，仅用于出处。里面提到的 install/run/操作手册不是本目录的运行入口，旧10轮/旧adapter也不是此次主公式协议。

可在具备真实 GloVe 与 torch 的服务器上，把 bank **重建到另一个新目录**：

```bash
python build_bank.py \
  --knowledge knowledge.json \
  --labels labels.json \
  --glove /data/xhw/czsl_repro/data/glove/glove.6B.300d.txt \
  --out /var/tmp/czsl_logic_bank_rebuild_check_20261011
```

该命令仅说明编码出处，正常启动本轮训练 **无需重建 bank**。builder 拒绝覆盖已有目标。不同 torch 版本的序列化字节可能不同；需要同时比较语义输入、label顺序、张量数值、manifest与coverage，不能为了让prepare接受新bank而删掉版本守卫。

Text K 使用另一套历史描述，位于 `../text_inputs/ut-zap50k/`。保留两套出处，不把它们误称为同一原作者知识库。当前 protocol 已冻结这种选择；统一描述语料属于需要新协议的后续对照。
