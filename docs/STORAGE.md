# 当前电脑的 C/D 盘存储位置

2026-10-11，根据用户要求将当前 CZSL 工作区的大输出、临时依赖、归档输入和 Git LFS 缓存迁至 `D:\CodexData\CZSL`。C 盘原目录保留 NTFS 目录联接，所以既有脚本、报告和 Git 工作区仍使用原路径。

| 原逻辑位置 | D 盘真实位置 |
|---|---|
| `C:\Users\llenovo\Documents\ChatGPT\CZSL\output` | `D:\CodexData\CZSL\output` |
| `C:\Users\llenovo\Documents\ChatGPT\CZSL\tmp` | `D:\CodexData\CZSL\tmp` |
| `C:\Users\llenovo\Documents\ChatGPT\CZSL\archive_inputs` | `D:\CodexData\CZSL\archive_inputs` |
| `C:\Users\llenovo\Documents\ChatGPT\CZSL\.git\lfs` | `D:\CodexData\CZSL\git-lfs` |

迁移工具：[move_storage_to_d.ps1](../archive_tools/move_storage_to_d.ps1)。复制后逐文件比较长度及 SHA256，检测迁移期间的变化，确认联接，再移除已验证的 C 盘副本。D 盘保存完整真实文件；这不是删除实验资料。

新输出写到原 `CZSL\output` 时会自动存入 D 盘。不要单独删除或移动 D 盘目标目录，否则原路径的联接会失效。这是当前机器的布局，克隆 GitHub 仓库会得到普通目录，不要求其他电脑有相同盘符。

此操作未改动 Codex 的全局聊天数据库、登录配置、软件安装目录或桌面原始研究资料。

本次迁移约 **7.08 GiB** 文件，四个目录均已通过长度与 SHA256 校验。完成时观察到 C 盘约 **13.38 GiB** 可用；磁盘可用空间也可能受其他程序活动影响，不把所有变化都归因于这次迁移。具体目录、文件数和字节数见 [迁移记录](../provenance/storage_migration_20261011.json)。Git LFS 后续新增缓存也会自动写入 D 盘。

两个忽略上传的临时副本仍保留在 D 盘：自动审批拦截了单独清理它们的删除命令。它们不占用 C 盘，也不属于上传的实验结果。
