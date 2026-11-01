# 恢复、训练与结果读取

依照[环境说明](../../environment/README.md)准备Python、PyTorch、CLIP权重及三个数据集。

```bash
git lfs pull --include="TRUST-CPF方案/11/**"
python tools/register_trustg0_three.py --check
python tools/reproduce.py materialize --line TRUST-CPF方案 --stage 11 --dest "$HOME/experiments/trust-g0-11"
cd "$HOME/experiments/trust-g0-11/trustg0_three_v1_10ep"
python -m unittest discover -s tests -v
python run.py setup --help
```

setup输入按[原始方案](PLAN.md)配置。完成setup后运行完整队列：

```bash
mkdir -p logs
nohup python -u scripts/run_queue.py 0 1 2 --work "$PWD/work" > logs/queue.log 2>&1 &
python scripts/status.py
python run.py report --work "$PWD/work"
```

同卡单任务、按依赖调度并支持按完成标记续跑。个人部署命令冻结保存于references。
