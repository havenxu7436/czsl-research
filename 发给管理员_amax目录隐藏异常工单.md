# amax：目录枚举隐藏异常，影响 CZSL 实验环境

提交用户：`xhw`。记录日期：2026-10-03。用户没有管理员权限。当前仅完成普通用户侧检查；以下系统处置尚未执行。

## 可直接转发的消息

管理员您好，服务器 amax 出现目录枚举异常，影响我的 CZSL 实验。新建 normal_probe.py 可正常列出，而 connectionpool_probe.py、pooling_probe.py 都显示存在并可读取，却无法列出。该现象导致 pip/urllib3/torch 导入失败，在 Python -I -S 以及清空 LD_PRELOAD 等变量后仍能复现。

`/etc/ld.so.preload` 中有 `libudev-compat.so`、`udev-settle-ng.so` 两项；已确认两项均加载进 Python，实际路径为 `/usr/lib/x86_64-linux-gnu/` 下的同名文件。两者为 root:root、权限 4755；udev-settle-ng.so 的 Modify/Change 显示为 2026-10-03 21:37:45 +0800。软件包查询未找到两项归属，同时提示部分 initramfs-tools 包的文件清单缺失。尚未确认两项库的来源或是否恶意。

请按系统异常排查：先保存预加载配置、两项库及相关日志，确认来源，再在合适的维护窗口临时停用相关预加载做对照。请检查是否存在未经授权的目录隐藏组件及持久化机制，并在我的普通用户 xhw 下验证可见性、conda 和 PyTorch 恢复。请保留 `/data/xhw/czsl_repro` 下的项目、数据、知识缓存、训练结果及 checkpoint；不要为修复而删除实验目录。我已暂停新的环境重装和实验启动。

## 1. 管理员首先需要确认的事项

1. 确认这两项库是否由单位批准的运维、安全或其他软件部署，核对部署记录与来源。库名和 root 属主不能独立证明合法或恶意。
2. 判断是否需要安排维护窗口，以及是否需从可信控制台/维护环境开展检查。若怀疑未经授权的系统组件，按单位事件响应流程评估隔离范围；不要由普通用户自行断网或重启共享服务器。
3. 保留现场证据后再更改配置，不先删除 `.so`、覆盖日志或直接重装环境。

## 2. 保留证据

重点保存：

- `/etc/ld.so.preload` 的原始内容、元数据和副本。
- `/usr/lib/x86_64-linux-gnu/libudev-compat.so` 与 `/usr/lib/x86_64-linux-gnu/udev-settle-ng.so` 的副本、元数据和 SHA256。
- 进程实际映射、系统时钟/时区和上述目录隐藏测试输出。
- 2026-10-03 21:37 前后的部署、登录、sudo、服务及系统日志；文件时间戳只是定位线索，不能作为修改者的证明。
- 包数据库及其缺失清单警告。目录隐藏未排除前，不能把工具的“未找到”直接等同于文件物理删除。

以下为管理员在可信 root 会话中可使用的取证命令示例。普通用户无需执行。只按明确路径复制少数已知文件，不递归遍历实验目录。

```bash
(
    set -eu
    umask 077
    CASE_DIR="/root/czsl_preload_case_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$CASE_DIR"

    cp -a -- /etc/ld.so.preload "$CASE_DIR/ld.so.preload"
    cp -a -- /usr/lib/x86_64-linux-gnu/libudev-compat.so "$CASE_DIR/libudev-compat.so"
    cp -a -- /usr/lib/x86_64-linux-gnu/udev-settle-ng.so "$CASE_DIR/udev-settle-ng.so"

    stat -- /etc/ld.so.preload \
        /usr/lib/x86_64-linux-gnu/libudev-compat.so \
        /usr/lib/x86_64-linux-gnu/udev-settle-ng.so \
        > "$CASE_DIR/stat.txt"

    sha256sum -- "$CASE_DIR/ld.so.preload" \
        "$CASE_DIR/libudev-compat.so" "$CASE_DIR/udev-settle-ng.so" \
        > "$CASE_DIR/SHA256SUMS.txt"

    date -Is > "$CASE_DIR/collection_time.txt"
    printf 'Evidence directory: %s\n' "$CASE_DIR"
)
```

此示例不是完整取证工具，也不能保证受影响主机上的工具未被拦截；管理员应根据实际情况使用可信工具核验。副本目录中的元数据应与原文件记录结合解释。

## 3. 临时停用预加载做对照

在证据保存、用途确认及维护安排完成后，由管理员临时去掉 `/etc/ld.so.preload` 中相关两项，观察新进程是否恢复目录枚举。不要仅删除库文件或更改 SUID 权限来代替对照。

以下命令只匹配截图中的两个独立行，并另存原配置；若实际内容不同，管理员需先检查，不能盲目扩大匹配范围。

```bash
(
    set -eu
    PRELOAD_BACKUP="/root/ld.so.preload.before_czsl_$(date +%Y%m%d_%H%M%S)"
    cp -a -- /etc/ld.so.preload "$PRELOAD_BACKUP"
    sed -i \
        -e '/^[[:space:]]*libudev-compat\.so[[:space:]]*$/d' \
        -e '/^[[:space:]]*udev-settle-ng\.so[[:space:]]*$/d' \
        /etc/ld.so.preload
    cat /etc/ld.so.preload
    printf 'Original configuration: %s\n' "$PRELOAD_BACKUP"
)
```

修改配置不会卸载旧进程中已加载的库，需用新进程验证。普通用户无需运行上述 root 命令。

## 4. 继续查明来源与持续写入机制

若停用后恢复，需要确认是哪项配置/组件导致异常，以及它是授权软件故障还是未经授权的隐藏行为。

管理员应核对部署及软件包来源，调查相关系统服务、计划任务、启动配置和登录活动，检查配置或库是否被重新写入。可疑主机上的 ps、ls 等输出可能受相同机制影响，需要可信方式核验。

如果确认存在未经授权的组件，按事件响应流程评估影响范围，清理或从可信镜像重建，处理可能受影响的凭据。不要将“import torch 恢复”当作安全处置结束。此流程不建议普通用户重新启用可疑库来试错。

## 5. 在受影响的普通用户下验收

管理员处理后，由用户 xhw 重新建立 SSH 会话执行，或管理员在该用户身份下验证。仅 root 账号测试通过不能替代普通用户验收。

### 目录可见性

以下在 `/tmp` 和项目所在的 `/data` 中各创建一个独立测试目录。每个目录三个文件都应为 `LISTED: True EXISTS: True`。

```bash
/data/xhw/miniconda3/bin/python -I -S -u - <<'PY'
import os
names = ['normal_probe.py', 'connectionpool_probe.py', 'pooling_probe.py']
all_ok = True
for parent in ['/tmp', '/data/xhw/czsl_repro']:
    directory = os.path.join(parent, 'czsl_visibility_accept_' + str(os.getpid()))
    os.mkdir(directory, 0o700)
    for name in names:
        with open(os.path.join(directory, name), 'w') as f:
            f.write('# visibility acceptance probe\n')
    listed = os.listdir(directory)
    print('DIRECTORY:', directory)
    for name in names:
        visible = name in listed
        exists = os.path.isfile(os.path.join(directory, name))
        print(name, 'LISTED:', visible, 'EXISTS:', exists)
        all_ok = all_ok and visible and exists
if not all_ok:
    raise SystemExit('DIRECTORY VISIBILITY FAILED')
print('DIRECTORY VISIBILITY PASSED')
PY
```

### CPF 环境版本与导入

```bash
/data/xhw/miniconda3/envs/cpf/bin/python -u - <<'PY'
import sys
import numpy
import scipy
import torch
import torchvision
from torch.nn.modules.pooling import MaxPool2d
print('python:', sys.version)
print('numpy:', numpy.__version__)
print('scipy:', scipy.__version__)
print('torch:', torch.__version__, torch.__file__)
print('torchvision:', torchvision.__version__)
print('CUDA build:', torch.version.cuda)
print('CPF IMPORTS PASSED')
PY
```

预期与原实验 manifest 一致：Python 3.8.18、torch 1.13.1+cu117、torchvision 0.14.1+cu117、NumPy 1.24.4、SciPy 1.10.1。先验证再决定是否需要包修复，不自动升级版本。

### Base、conda 与激活

```bash
/data/xhw/miniconda3/bin/python -u - <<'PY'
import pip
import pip._vendor.urllib3.connectionpool
import urllib3.connectionpool
import requests
print('BASE IMPORTS PASSED')
PY
```

```bash
/data/xhw/miniconda3/bin/conda info
source /data/xhw/miniconda3/etc/profile.d/conda.sh
conda activate cpf
command -v python
python -c 'import torch; print(torch.__version__)'
```

解释器应为 `/data/xhw/miniconda3/envs/cpf/bin/python`。若系统可见性已恢复但导入仍失败，再处理此前修复操作留下的实际包问题。

GPU 可用性应在有 GPU 使用权限的会话中另外验证，不能把调度限制当作本次目录隐藏故障：

```bash
/data/xhw/miniconda3/envs/cpf/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

## 6. 保留实验文件与交接结果

请保留以下内容，不为修复直接删除或格式化 `/data`：

- `/data/xhw/czsl_repro/src/KCC0601_v1`：实验代码。
- `/data/xhw/czsl_repro/data`：数据和相关资源。
- `/data/xhw/czsl_repro/knowledge/kcc0601_v1`：知识文件及缓存。
- `/data/xhw/czsl_repro/runs`：基线运行及 checkpoint。
- `/data/xhw/czsl_repro/experiments/kcc0601_glove_mvp_v1`：本轮训练、配置、审计和诊断结果。
- 之前的环境 manifest、修复备份和此次异常证据。

若需迁移或重建，先在可信环境或目录可见性恢复后校验备份完整性，避免递归复制漏掉隐藏文件。

请管理员返回以下结果：

1. 两项库的来源、用途及授权情况，确认的异常原因。
2. 已执行的配置变更、停用或清理动作，以及是否发现持续写入/其他异常。
3. 新进程和 xhw 用户下的目录可见性、环境版本、导入与激活检查输出。
4. 实验数据是否完整保留、是否迁移，以及主机是否已允许恢复计算任务。

管理员确认后，用户再做 A/C 的 lambda=0 复评，与冻结结果比较，再恢复验证集 lambda 扫描；无需因这次故障先重跑所有训练。

## 参考

- [Linux ld.so 手册](https://man7.org/linux/man-pages/man8/ld.so.8.html)：系统级预加载配置。
- [Elastic Security Labs：Rootkit 检查机制](https://security-labs.elastic.co/security-labs/linux-rootkits-2-caught-in-the-act)：预加载及用户态函数拦截的调查线索。机制参考不证明本机两项库一定恶意。
