# 执行代码

完整发布包映射见[package_members.json](../package_members.json)。[队列](trustg0_three_v1_10ep/scripts/run_queue.py)生成165个依赖任务，仅使用物理GPU 0/1/2；[训练和评分](trustg0_three_v1_10ep/trustg0/workflow.py)与服务器核心代码一致。

代码和配置按仓库分开保存，恢复工具重建原目录。[后补导出脚本](../references/latest_operational_scripts/)保留按数据集下载时的操作版本，不改变训练数学实现。
