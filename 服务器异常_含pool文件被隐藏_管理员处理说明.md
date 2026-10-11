# 服务器异常：含 pool 的文件在目录枚举中不可见

记录日期：2026-10-03。主机提示符：`xhw@amax`。实验根目录：`/data/xhw/czsl_repro`。

## 1. 已有证据与判断边界

以下结论来自用户上传的服务器实际输出，未通过远程连接直接操作服务器。

| 检查对象 | 目录中可见 | 文件存在/可读取 |
|---|---:|---:|
| 新建 `normal_probe.py` | True | True |
| 新建 `connectionpool_probe.py` | False | True |
| 新建 `pooling_probe.py` | False | True |
| 新 pip 的 `pip/_vendor/urllib3/connectionpool.py` | False | 可读 44164 字节 |
| CPF 的 `torch/nn/modules/pooling.py` | False | 可读 53515 字节 |

新建文件的对照在普通进程及清除 `LD_PRELOAD`、`LD_AUDIT`、`LD_LIBRARY_PATH` 的进程中均复现。

其他输出：

- `LD_PRELOAD`、`LD_AUDIT`、`LD_LIBRARY_PATH` 均未设置。
- `/etc/ld.so.preload` 包含 `libudev-compat.so` 和 `udev-settle-ng.so` 两行。
- `/tmp` 位于 `/dev/sda3` 的 ext4；`/data` 位于 `/dev/sdb1` 的 ext4。
- Python `-I -S` 下仍失败；模块源码后缀正常，为 `['.py']`。

**已经确认的是目录枚举会漏掉测试中的含 pool 文件，直接访问却成功。** 新建文件也复现，因此不能继续把问题简单归为 conda、pip 或 PyTorch 丢文件。

系统级预加载库是重点嫌疑对象。尚未确认两项库分别做了什么，也未确认其来源、所属软件或是否恶意。不能仅凭库名宣布感染了特定恶意软件。隐藏行为与用户态函数拦截相符，需要管理员排查是否存在未经授权的隐藏组件/rootkit。

## 2. 当前实验处理

暂停新的训练和 lambda 扫描；停止覆盖或重装环境。保留已生成的结果、checkpoint、日志及修复备份。

当前目录枚举不可靠，递归复制、find、环境导出等可能遗漏文件。不要把此时复制出的环境当作完整备份；应在可见性恢复或可信维护环境中复核。

不能从此次异常直接判断现有 checkpoint 已损坏。环境恢复后，先验证版本及 lambda=0 对照，再决定如何恢复扫描。

## 3. 普通用户可以执行的只读补充检查

下列命令查看当前 Python 进程实际映射的两项库，取得绝对路径；不会禁用或删除任何库。

```bash
/data/xhw/miniconda3/bin/python -I -S -u - <<'PY'
with open('/proc/self/maps', 'r') as f:
    lines = f.readlines()
matches = [line for line in lines
           if 'libudev-compat.so' in line or 'udev-settle-ng.so' in line]
if matches:
    for line in matches:
        print(line.rstrip())
else:
    print('No matching mapping observed; administrator investigation still required.')
PY
```

没有匹配输出不等于没有注入；输出工具本身运行在受影响的系统上。

### 已取得的实际映射路径

用户随后提供的 `/proc/self/maps` 输出同时包含：

```text
/usr/lib/x86_64-linux-gnu/udev-settle-ng.so
/usr/lib/x86_64-linux-gnu/libudev-compat.so
```

两项库均已映射到此次 `python -I -S` 进程。每项分别出现多段映射；`r--p`、`r-xp`、`rw-p` 是不同内存区段的权限，不表示多次安装或多个不同的库。输出确认实际加载，不证明哪项进行了隐藏，也不独立证明恶意性。

普通用户可继续记录这两个已知绝对路径的元数据、校验值和软件包归属，命令不会修改库：

```bash
LIB1=/usr/lib/x86_64-linux-gnu/libudev-compat.so
LIB2=/usr/lib/x86_64-linux-gnu/udev-settle-ng.so

stat -- "$LIB1" "$LIB2"
sha256sum -- "$LIB1" "$LIB2"
dpkg-query -S '*libudev-compat.so*' '*udev-settle-ng.so*'
```

使用包数据库的文件名模式，可避免 `/usr/lib` 与 `/lib` 合并路径导致的单纯路径匹配遗漏。若无匹配，只表示当前包数据库未找到归属；手工安装也可能无归属，不能据此单独认定恶意。若没有 `dpkg-query`，记录该情况，由管理员使用该系统的包管理工具核查。

### 新增 stat、校验值及包数据库输出

用户进一步提供的输出记录如下，时间按服务器显示的 `+0800`：

| 项目 | libudev-compat.so | udev-settle-ng.so |
|---|---|---|
| 文件大小（字节） | 42576 | 62056 |
| 属主/属组 | root:root | root:root |
| 权限 | 4755 / -rwsr-xr-x | 4755 / -rwsr-xr-x |
| inode | 8689071 | 8689086 |
| Modify / Change | 2026-08-15 04:09:55 | 2026-10-03 21:37:45 |
| Birth | 未提供，输出为 `-` | 未提供，输出为 `-` |

SHA256（依据截图转录，管理员应在可信环境再次计算核对）：

```text
8d565cb71c560c97753d60e7278b46d3cd785d408e4b63aa5ffd8ad802bff123  /usr/lib/x86_64-linux-gnu/libudev-compat.so
d150995048e0afe9b6b6d7702f3064fc813b4b1e602c19f698178f09c83f18c7  /usr/lib/x86_64-linux-gnu/udev-settle-ng.so
```

`4755` 表示带有 set-user-ID 位的 0755 权限；普通共享库出现此位值得核查。不能解释成“Python 加载该库就自动取得 root 权限”；也不能仅凭 root:root 属主判断是谁安装或修改了文件。

`Modify` 是数据修改时间，`Change` 是文件状态改变时间，不是创建时间。`udev-settle-ng.so` 显示在当日 21:37:45 被修改/状态改变，为管理员核对日志提供线索；不能从时间戳独立证明谁修改或为什么修改，时间戳也不能代替取证。

软件包查询对两个名字均返回 `no path found matching pattern`，同时报告 `initramfs-tools`、`initramfs-tools-core`、`initramfs-tools-bin` 的文件清单缺失。因此应记录为“当前查询未找到软件包归属，且查询存在缺失清单警告”，不能记录为“确定没有任何合法软件包来源”。在目录隐藏问题下，也不能未经独立核查就认定这些清单文件物理删除了。

综合此前目录隐藏、系统级预加载、实际内存映射与此次元数据，未经授权的用户态隐藏组件/rootkit 是需要重点调查的可能性。SHA256 只用于识别和后续比对，不能独立证明文件安全或恶意。当前阶段已有足够线索交管理员处置，继续重装 pip/torch 不会解决已观察到的目录过滤现象。

## 4. 管理员取证与临时对照

本节涉及主机范围的加载配置，应由服务器管理员确认两项配置用途后，在可信管理会话中执行。普通用户不要因拥有 sudo 就直接照抄停用共享服务器组件。

先保存 `/etc/ld.so.preload`、上述实际库路径的副本及 SHA256、文件元数据、所属软件包信息和相关系统日志。不要先删除 `.so`，也不要把陌生 `.so` 当作程序执行。必要时从可信维护环境取证。

确认可以临时停用这两项后，以下命令仅移除与截图一致的两条独立配置行，保留其他行，并保留原配置备份。此操作是定位原因的对照，不能代替完整的安全处置。

```bash
(
    set -eu
    CASE_STAMP=$(date +%Y%m%d_%H%M%S)
    PRELOAD_BACKUP="/root/ld.so.preload.before_czsl_$CASE_STAMP"
    sudo cp -a -- /etc/ld.so.preload "$PRELOAD_BACKUP"
    sudo sed -i \
        -e '/^[[:space:]]*libudev-compat\.so[[:space:]]*$/d' \
        -e '/^[[:space:]]*udev-settle-ng\.so[[:space:]]*$/d' \
        /etc/ld.so.preload
    sudo cat /etc/ld.so.preload
    printf 'Original configuration preserved at: %s\n' "$PRELOAD_BACKUP"
)
```

如果实际配置与截图不同、有其他项、命令失败或配置自动恢复，不要扩大删除范围，交由管理员继续查明。

禁用配置不会卸载旧进程已经映射的库。使用新进程验证；需要时重新建立 SSH 会话。不要盲目杀掉全部用户进程或重启共享服务器。

## 5. 验证可见性

在管理员操作后启动新的 Python 进程执行：

```bash
/data/xhw/miniconda3/bin/python -I -S -u - <<'PY'
import os
directory = '/tmp/czsl_after_check_' + str(os.getpid())
os.mkdir(directory)
names = ['normal_probe.py', 'connectionpool_probe.py', 'pooling_probe.py']
for name in names:
    with open(os.path.join(directory, name), 'w') as f:
        f.write('# visibility probe\n')
listed = os.listdir(directory)
for name in names:
    print(name, 'LISTED:', name in listed,
          'EXISTS:', os.path.isfile(os.path.join(directory, name)))
if not all(name in listed for name in names):
    raise SystemExit('DIRECTORY VISIBILITY STILL FAILED')
print('DIRECTORY VISIBILITY PASSED')
PY
```

期望三个文件全部 `LISTED: True EXISTS: True`，并打印 `DIRECTORY VISIBILITY PASSED`。

如果停用两项后恢复，支持预加载配置引发隐藏的判断，但尚未区分哪项负责；不建议普通用户自行逐项重新启用可疑库。若仍异常，管理员需要继续从可信环境排查其他用户态拦截、内核及系统状态。

## 6. 验证 Python 环境，不先安装包

可见性恢复后，先测试现有文件即可。以下命令显式使用各环境解释器，不依赖提示符中的 `(cpf)`。

```bash
/data/xhw/miniconda3/bin/python -u - <<'PY'
import pip
import pip._vendor.urllib3.connectionpool
import urllib3.connectionpool
import requests
print('BASE IMPORTS PASSED:', pip.__version__, requests.__version__)
PY
```

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
print('CUDA available:', torch.cuda.is_available())
print('CPF IMPORTS PASSED')
PY
```

原实验预期版本：Python 3.8.18、torch 1.13.1+cu117、torchvision 0.14.1+cu117、NumPy 1.24.4、SciPy 1.10.1。版本来自先前实验记录，应与服务器原始 manifest 对照。GPU 验证应在服务器允许使用 GPU 的会话中进行。

```bash
CONDA_NO_PLUGINS=true /data/xhw/miniconda3/bin/conda info
```

如果可见性正常而导入仍失败，再针对实际缺失或损坏的包修复，不沿用此前“所有 ModuleNotFoundError 都是丢文件”的判断。

## 7. 恢复实验的条件

管理员确认主机适合继续使用、目录可见性恢复、CPF 版本和导入通过后，回到第四步文档。先完成 A/C 的 lambda=0 复评并与冻结结果比较；符合原验证容差后再恢复验证集 lambda 扫描。损坏期间的部分输出应单独保留，不覆盖已经完成的正式训练结果。

如果确认存在未经授权的系统组件，管理员需继续调查其来源和持久化机制。仅暂时移除预加载配置、能够重新 import torch，不能证明主机已经完成清理。

## 8. 可转交管理员的简短说明

> 服务器 amax 出现目录枚举隐藏：新建 normal_probe.py 可见，而 connectionpool_probe.py 和 pooling_probe.py 不可见但可直接读取。pip/urllib3/torch 导入因此失败。Python -I -S 及清除 LD_PRELOAD 等变量后均复现。/etc/ld.so.preload 包含 libudev-compat.so 和 udev-settle-ng.so。请保存配置和库证据，检查是否有 readdir 相关拦截、非授权隐藏组件，在确认用途后做临时停用对照，并排查其来源与持久化。用户侧已暂停环境重装和实验。

## 9. 参考

- [Linux ld.so 手册](https://man7.org/linux/man-pages/man8/ld.so.8.html)：`/etc/ld.so.preload` 是主机范围预加载配置；清空 `LD_PRELOAD` 不会禁用它。
- [Linux proc_pid_maps 手册](https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html)：说明进程内存映射的路径和权限字段。
- [Linux stat 手册](https://man7.org/linux/man-pages/man1/stat.1.html)：文件权限、修改时间、状态变化时间及未知创建时间的含义。
- [Linux execve 手册](https://man7.org/linux/man-pages/man2/execve.2.html)：set-user-ID 是执行程序时的身份语义，不应与普通共享库加载混同。
- [Python FileFinder 文档](https://docs.python.org/3/library/importlib.html#importlib.machinery.FileFinder)：模块搜索会缓存目录内容。
- [Elastic Security Labs：用户态 rootkit 的拦截机制](https://security-labs.elastic.co/security-labs/linux-rootkits-1-hooked-on-linux)：动态库注入可用于拦截目录枚举。本参考解释机制，不证明本机两个库一定恶意。
