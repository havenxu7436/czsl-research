# VG-CPOD 公式及代码 v1

## 0. 符号、张量轴与依赖

```math
\begin{aligned}
\mathcal{A}&=\{a_1,\ldots,a_M\},\qquad
\mathcal{O}=\{o_1,\ldots,o_N\},\\
\mathcal{C}_{\mathrm{all}}&=\mathcal{A}\times\mathcal{O},\qquad
\mathcal{C}^s\subseteq\mathcal{C}_{\mathrm{all}},\\
\mathcal{B}&=\{(x_b,i_b,j_b)\}_{b=1}^{B}.
\end{aligned}
```

| 符号 | 定义与维度 |
|---|---|
| $`B,M,N,d,r,q`$ | 小批次大小、属性数、物体数、图文维度、CP模式数、适配器秩 |
| $`M,N,d,L`$ | UT-Zappos：$`16,12,768,256`$ |
| $`r,q`$ | v1默认：$`32,8`$ |
| $`g_b`$ | 全局图像特征，$`\mathbb{R}^d`$ |
| $`p_{b\ell}`$ | 第 $`\ell`$ 个补丁特征，$`\mathbb{R}^d`$ |
| $`e_i^a,e_j^o,e_{ij}^c`$ | 属性、物体、组合文本特征，$`\mathbb{R}^d`$ |
| $`f_{bi}^a,f_{bj}^o`$ | 属性、物体类别查询特征，$`\mathbb{R}^d`$ |
| $`u_i,v_j,\gamma_{bj}`$ | 属性因子、物体因子、视觉门控，$`\mathbb{R}^r`$ |
| $`O_{bj},U_{bji},S_{bji}`$ | 物体logits、条件属性logits、联合log概率 |
| $`P_{bji}`$ | $`P_\theta(a_i,o_j\mid x_b)`$ |
| $`\odot,\circ`$ | 逐元素乘积、向量外积 |

数学组合下标为 $`(i,j)`$；程序评分轴为 `[batch, object, attribute]`。候选标签元组为 `(attribute, object)`。

代码块按标注的源码函数或方法上下文摘录；类方法中的 `self` 为 `VGCPOD` 实例。

```math
\nu(z)=\frac{z}{\max(\lVert z\rVert_2,10^{-8})},\qquad
\operatorname{LSE}_{k}z_k=\log\sum_k\exp z_k.
```

依赖：

```python
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
```

[vgcpod/common.py · unit](code/vgcpod_v1/vgcpod/common.py#L72)

```python
def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
```

## 1. 冻结CLIP编码与类别查询

### 1.1 图像预处理与视觉序列

CLIP：ViT-L/14；输入分辨率 $`224\times224`$。视觉Transformer宽度为1024，图文共同维度为768。

```math
\begin{aligned}
z_\ell&=\operatorname{Conv}_{14,14}(x)_\ell,\\
X^{(0)}&=\operatorname{LN}_{\mathrm{pre}}
\left([c_{\mathrm{CLS}};z_1;\ldots;z_L]+E_{\mathrm{pos}}\right),\\
H&=\operatorname{Transformer}_{\mathrm{vis}}(X^{(0)})
\in\mathbb{R}^{257\times1024},\\
g&=\nu\!\left(P_v^\top\operatorname{LN}_{\mathrm{post}}(H_0)\right),\\
p_\ell&=\nu\!\left(P_v^\top\operatorname{LN}_{\mathrm{post}}(H_\ell)\right),
\qquad P_v\in\mathbb{R}^{1024\times768}.
\end{aligned}
```

向量采用列表示；代码采用行向量右乘 `visual.proj`。末层归一化与视觉投影同时作用于CLS和补丁。

[vendor/clip/clip.py · _transform](code/vgcpod_v1/vendor/clip/clip.py#L79)

```python
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
```

[vgcpod/features.py · prepare（局部）](code/vgcpod_v1/vgcpod/features.py#L95)

```python
model, preprocess = clip.load(str(args.checkpoint), device=args.device, jit=False)
model = model.float().eval()
model.requires_grad_(False)
if model.visual.input_resolution != 224 or model.visual.proj.shape != (1024, 768):
    raise ValueError('CLIP backbone must be ViT-L/14 at 224px')
```

[vgcpod/features.py · visual_tokens](code/vgcpod_v1/vgcpod/features.py#L17)

```python
def visual_tokens(visual, images):
    """Return CLS and patches after ln_post and projection, in FP32 unit vectors."""
    x = visual.conv1(images.to(visual.conv1.weight.dtype))
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
    x = torch.cat([cls, x], dim=1)
    x = visual.ln_pre(x + visual.positional_embedding.to(x.dtype))
    x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = visual.ln_post(x)
    if visual.proj is not None:
        x = x @ visual.proj
    x = unit(x.float())
    return x[:, 0], x[:, 1:]
```

### 1.2 文本原语与组合

```math
\begin{aligned}
t_i^a&=\text{a photo of footwear made of }a_i,\\
t_j^o&=\text{a photo of }o_j,\\
t_{ij}^c&=\text{a photo of }o_j\text{ made of }a_i,\\
e_i^a&=\nu(E_T(t_i^a)),\qquad
e_j^o=\nu(E_T(t_j^o)),\\
e_{ij}^c&=\nu(E_T(t_{ij}^c)).
\end{aligned}
```

类别字符串读取 `assets/names.json`；组合表按物体外层、属性内层构造。

[vgcpod/features.py · prompts](code/vgcpod_v1/vgcpod/features.py#L37)

```python
def prompts():
    names = read_json(ROOT / 'assets' / 'names.json')
    return {'a': ['a photo of footwear made of {}'.format(a) for a in names['a']],
            'o': ['a photo of {}'.format(o) for o in names['o']],
            'c': ['a photo of {} made of {}'.format(o, a) for o in names['o'] for a in names['a']]}
```

[vgcpod/features.py · prepare（局部）](code/vgcpod_v1/vgcpod/features.py#L101)

```python
text = {}
for key, rows in identity['prompts'].items():
    chunks = []
    for start in range(0, len(rows), 64):
        tokens = clip.tokenize(rows[start:start + 64], truncate=False).to(args.device)
        chunks.append(unit(model.encode_text(tokens).float()).cpu().numpy())
    text[{'a': 'ea', 'o': 'eo', 'c': 'ec'}[key]] = np.concatenate(chunks)
text['ec'] = text['ec'].reshape(len(labels['objs']), len(labels['attrs']), 768)
np.savez(cache / 'text_features.npz', **text)
ea = torch.from_numpy(text['ea']).to(args.device)
eo = torch.from_numpy(text['eo']).to(args.device)
queries = torch.cat([ea, eo], dim=0)
```

### 1.3 类别查询补丁

```math
\begin{aligned}
\alpha_{b\ell}(e)
&=\frac{\exp(e^\top p_{b\ell}/\tau_v)}
{\sum_{s=1}^{L}\exp(e^\top p_{bs}/\tau_v)},\\
f_b(e)&=\nu\!\left(\sum_{\ell=1}^{L}\alpha_{b\ell}(e)p_{b\ell}\right),\\
f_{bi}^a&=f_b(e_i^a),\qquad
f_{bj}^o=f_b(e_j^o),\qquad\tau_v=0.07.
\end{aligned}
```

[vgcpod/features.py · query_patches](code/vgcpod_v1/vgcpod/features.py#L32)

```python
def query_patches(patches, queries, tau):
    alpha = torch.einsum('qd,bld->bql', queries, patches).div(tau).softmax(-1)
    return unit(torch.einsum('bql,bld->bqd', alpha, patches))
```

### 1.4 缓存精度

编码和查询在FP32中计算；图像与查询缓存量化到FP16，读取时转换回FP32：

```math
\widehat z
=\operatorname{cast}_{32}\!\left(\operatorname{cast}_{16}(z)\right),
\qquad z\in\{g_b,f_{bi}^a,f_{bj}^o\}.
```

后续前向公式中的图像和查询特征对应上述 $`\widehat z`$；读取后不再单位化。文本表保存为FP32。

[vgcpod/features.py · prepare（局部）](code/vgcpod_v1/vgcpod/features.py#L143)

```python
global_features.append(g.cpu().numpy().astype(np.float16))
attribute_features.append(f[:, :len(ea)].cpu().numpy().astype(np.float16))
object_features.append(f[:, len(ea):].cpu().numpy().astype(np.float16))
```

[vgcpod/data.py · FeatureDataset.__getitem__](code/vgcpod_v1/vgcpod/data.py#L109)

```python
def __getitem__(self, index):
    if self.arrays is None:
        self.arrays = [np.load(self.root / self.phase / (name + '.npy'), mmap_mode='r')
                       for name in ('g', 'fa', 'fo')]
    row = self.rows[index]
    return tuple(torch.from_numpy(np.array(x[index], dtype=np.float32)) for x in self.arrays) + (
        row['attr'], row['object'])
```

## 2. 基线适配器、参数与初始化

### 2.1 物体与条件属性logits

```math
\begin{aligned}
A_o&=B_oC_o^\top,\qquad A_c=B_cC_c^\top,\\
B_o,C_o,B_c,C_c&\in\mathbb{R}^{d\times q},\\
O_{bj}&=\kappa g_b^\top e_j^o+g_b^\top B_oC_o^\top e_j^o,\\
U^0_{bji}&=\kappa g_b^\top e_{ij}^c
+g_b^\top B_cC_c^\top e_{ij}^c,\qquad\kappa=20.
\end{aligned}
```

### 2.2 CP因子与方向矩阵

```math
\begin{aligned}
u_i&=A_ue_i^a,\qquad v_j=A_ve_j^o,\\
A_u,A_v,B_{\mathrm{vis}}&\in\mathbb{R}^{r\times d},\\
W_c,W_a&\in\mathbb{R}^{d\times r}.
\end{aligned}
```

全部映射不含偏置。文本表注册为buffer。

### 2.3 初始化

```math
\begin{aligned}
(B_o)_{kt},(B_c)_{kt}&\sim\mathcal{N}(0,0.02^2),\\
C_o&=C_c=0,\qquad W_c=W_a=0,\\
(A_u)_{kt},(A_v)_{kt}&\sim\mathcal{N}(0,r^{-1/2}).
\end{aligned}
```

对单位文本向量，独立的属性与物体映射满足：

```math
\begin{aligned}
\operatorname{Var}(u_{ik})&=\operatorname{Var}(v_{jk})=r^{-1/2},\\
\mathbb{E}\lVert u_i\odot v_j\rVert_2^2
&=\sum_{k=1}^{r}\mathbb{E}[u_{ik}^2]\mathbb{E}[v_{jk}^2]=1.
\end{aligned}
```

视觉模式基的初始化为：

```math
\begin{aligned}
Z&\in\mathbb{R}^{d\times r},\qquad Z_{kt}\sim\mathcal{N}(0,1),\\
Z&=QR,\qquad Q^\top Q=I_r,\\
s_k&=\begin{cases}\operatorname{sign}(R_{kk}),&R_{kk}\ne0,\\1,&R_{kk}=0,\end{cases}\\
B_{\mathrm{vis}}&=(Q\operatorname{diag}(s))^\top,\qquad
B_{\mathrm{vis}}B_{\mathrm{vis}}^\top=I_r.
\end{aligned}
```

[vgcpod/model.py · VGCPOD.__init__](code/vgcpod_v1/vgcpod/model.py#L16)

```python
def __init__(self, ea, eo, ec, cfg, seed):
    super().__init__()
    self.variant = cfg['variant']
    if self.variant not in VARIANTS:
        raise ValueError('Unknown variant: ' + self.variant)
    self.cfg = dict(cfg)
    self.register_buffer('ea', ea.clone().float())
    self.register_buffer('eo', eo.clone().float())
    self.register_buffer('ec', ec.clone().float())  # [O,A,D]
    m, d = ea.shape
    n = eo.shape[0]
    r, q = cfg['rank'], cfg['adapter_rank']
    if not (1 <= r <= d and 1 <= q <= d) or ec.shape != (n, m, d):
        raise ValueError('Incompatible ranks/text axes')
    # Allocate identical named tensors for every variant; inactive tensors are frozen.
    shapes = {'Bo': (d, q), 'Co': (d, q), 'Bc': (d, q), 'Cc': (d, q),
              'Au': (r, d), 'Av': (r, d), 'Bvis': (r, d), 'Wc': (d, r), 'Wa': (d, r)}
    generator = torch.Generator().manual_seed(seed)
    for name, shape in shapes.items():
        x = torch.zeros(shape)
        if name in ('Bo', 'Bc'):
            x.normal_(0, 0.02, generator=generator)
        elif name in ('Au', 'Av'):
            x.normal_(0, r ** -0.25, generator=generator)
        elif name == 'Bvis':
            z = torch.randn(d, r, generator=generator)
            basis, upper = torch.linalg.qr(z, mode='reduced')
            signs = upper.diag().sign()
            signs[signs == 0] = 1
            x = (basis * signs).T.contiguous()
        self.register_parameter(name, nn.Parameter(x))
    self.dynamic = self.variant in ('V2', 'V3', 'V4', 'V5', 'V6', 'V7')
    self.primitive = self.variant in ('V4', 'V5', 'V7')
    self.correction = self.variant != 'V0'
    for name in ('Au', 'Av', 'Wc'):
        getattr(self, name).requires_grad_(self.correction)
    self.Bvis.requires_grad_(self.dynamic)
    self.Wa.requires_grad_(self.primitive)
```

## 3. 视觉门控、CP修正与低秩算子

### 3.1 类别条件门控

```math
t_{bj}=\frac{B_{\mathrm{vis}}f_{bj}^o}{\tau_\gamma},\qquad
\gamma_{bj}=2\sigma(t_{bj}),\qquad\tau_\gamma=0.1.
```

在实数精确运算下 $`\gamma_{bjk}\in(0,2)`$；浮点实现使用 `torch.sigmoid`。

### 3.2 修正场与条件logits

```math
\begin{aligned}
h_{bji}&=u_i\odot v_j\odot\gamma_{bj},\\
\delta_{bji}&=W_ch_{bji}
=\sum_{k=1}^{r}u_{ik}v_{jk}\gamma_{bjk}w_k,\\
U_{bji}&=U^0_{bji}+\rho\kappa(f_{bj}^o)^\top\delta_{bji},
\qquad\rho=0.1.
\end{aligned}
```

等价收缩式：

```math
U_{bji}-U^0_{bji}
=\rho\kappa\sum_{k=1}^{r}
u_{ik}v_{jk}\gamma_{bjk}\bigl(w_k^\top f_{bj}^o\bigr).
```

### 3.3 低秩算子

```math
\begin{aligned}
\delta_{bji}
&=W_c\operatorname{diag}(v_j\odot\gamma_{bj})A_ue_i^a,\\
L_{bj}&:=W_c\operatorname{diag}(v_j\odot\gamma_{bj})A_u,\\
\delta_{bji}&=L_{bj}e_i^a,\\
\operatorname{rank}(L_{bj})
&\le\min\{\operatorname{rank}W_c,\operatorname{rank}A_u,r\}\le r.
\end{aligned}
```

固定 $`x_b`$，三阶修正张量满足：

```math
\Delta_b
=\sum_{k=1}^{r}u_{:,k}\circ
\bigl(v_{:,k}\odot\gamma_{b,:,k}\bigr)\circ w_k,
\qquad \Delta_b\in\mathbb{R}^{M\times N\times d},\qquad
\operatorname{rank}_{\mathrm{CP}}(\Delta_b)\le r.
```

实际前向使用模式收缩，不显式构造 $`L_{bj}`$ 或 $`\Delta_b`$；对应代码见§5。

## 4. 视觉伙伴聚合与属性原语分支

### 4.1 伙伴概率

```math
\begin{aligned}
c_{bij}&=(f_{bi}^a)^\top e_j^o,\\
\pi_{bij}&=\frac{\exp(c_{bij}/\tau_\pi)}
{\sum_{s=1}^{N}\exp(c_{bis}/\tau_\pi)},\qquad\tau_\pi=0.1,\\
\widetilde v_{bi}&=\sum_{j=1}^{N}\pi_{bij}v_j.
\end{aligned}
```

$`\pi`$ 的张量轴为 `[batch, attribute, object]`。查询特征与文本表冻结，$`\pi`$ 对训练参数的导数为零；$`\widetilde v`$ 对 $`v`$ 保留梯度。

### 4.2 属性原语读出

```math
\begin{aligned}
\delta_{bi}^a&=W_a(u_i\odot\widetilde v_{bi}),\\
R_{bi}&=\kappa g_b^\top e_i^a
+\rho_a\kappa(f_{bi}^a)^\top\delta_{bi}^a,
\qquad\rho_a=0.1,\\
p^{\mathrm{prim}}_{bi}&=\operatorname{softmax}_i R_{bi}.
\end{aligned}
```

等价收缩式：

```math
R_{bi}=\kappa g_b^\top e_i^a
+\rho_a\kappa\sum_{k=1}^{r}
\bigl((W_a^\top f_{bi}^a)_k\,u_{ik}\,\widetilde v_{bik}\bigr).
```

完整代码见§5的 `self.primitive` 分支。

## 5. 完整前向与联合概率

```math
\begin{aligned}
\ell^o_{bj}&=O_{bj}-\operatorname{LSE}_{s=1}^{N}O_{bs},\\
\ell^{a\mid o}_{bji}&=U_{bji}-\operatorname{LSE}_{p=1}^{M}U_{bjp},\\
S_{bji}&=\ell^o_{bj}+\ell^{a\mid o}_{bji},\\
P_{bji}&=\exp S_{bji}
=P_\theta(o_j\mid x_b)P_\theta(a_i\mid o_j,x_b).
\end{aligned}
```

```math
\begin{aligned}
\sum_{j=1}^{N}\sum_{i=1}^{M}P_{bji}&=1,\\
\sum_{i=1}^{M}P_{bji}&=\exp\ell^o_{bj},\\
p^{\mathrm{joint}}_{bi}&=\sum_{j=1}^{N}P_{bji}.
\end{aligned}
```

[vgcpod/model.py · VGCPOD.forward](code/vgcpod_v1/vgcpod/model.py#L55)

```python
def forward(self, g, fa, fo):
    c = self.cfg
    kappa, rho = c['kappa'], c['rho']
    obj = kappa * (g @ self.eo.T) + (g @ self.Bo) @ (self.eo @ self.Co).T
    base = kappa * torch.einsum('bd,oid->boi', g, self.ec)
    base = base + torch.einsum('bq,oiq->boi', g @ self.Bc, self.ec @ self.Cc)
    u, v = self.ea @ self.Au.T, self.eo @ self.Av.T
    local = fo if self.variant not in ('V1', 'V3') else g[:, None, :].expand(-1, len(v), -1)
    gamma = torch.ones(g.shape[0], len(v), c['rank'], device=g.device, dtype=g.dtype)
    if self.dynamic:
        gate_input = local.roll(shifts=1, dims=0) if self.variant == 'V6' else local
        gamma = 2 * torch.sigmoid((gate_input @ self.Bvis.T) / c['tau_gamma'])
    cond = base
    if self.correction:
        modes = (local @ self.Wc) * gamma * v[None, :, :]
        cond = base + rho * kappa * torch.einsum('bor,ir->boi', modes, u)
    log_o = obj.log_softmax(-1)
    log_a = cond.log_softmax(-1)
    joint = log_o[:, :, None] + log_a
    result = {'obj': obj, 'base': base, 'cond': cond, 'log_o': log_o,
              'log_a': log_a, 'joint': joint, 'gamma': gamma}
    if self.primitive:
        if self.variant == 'V4':
            pi = g.new_full((g.shape[0], len(u), len(v)), 1 / len(v))
        else:
            pi = (torch.einsum('bid,jd->bij', fa, self.eo) / c['tau_pi']).softmax(-1)
        partners = pi @ v
        rlogits = kappa * (g @ self.ea.T)
        rlogits = rlogits + c['rho_a'] * kappa * ((fa @ self.Wa) * u[None] * partners).sum(-1)
        result.update(prim=rlogits, pi=pi)
    return result
```

输出字段：

| 字段 | 数学量 | 形状 |
|---|---|---|
| `obj` | $`O`$ | $`B\times N`$ |
| `base` | $`U^0`$ | $`B\times N\times M`$ |
| `cond` | $`U`$ | $`B\times N\times M`$ |
| `log_o` | $`\ell^o`$ | $`B\times N`$ |
| `log_a` | $`\ell^{a\mid o}`$ | $`B\times N\times M`$ |
| `joint` | $`S`$ | $`B\times N\times M`$ |
| `gamma` | $`\gamma`$ | $`B\times N\times r`$ |
| `prim` | $`R`$，原语分支启用时 | $`B\times M`$ |
| `pi` | $`\pi`$，原语分支启用时 | $`B\times M\times N`$ |

## 6. 训练目标

### 6.1 训练候选mask

```math
T_{ji}=\mathbf{1}\{(a_i,o_j)\in\mathcal{C}^s\}.
```

[vgcpod/data.py · mask_from_pairs](code/vgcpod_v1/vgcpod/data.py#L123)

```python
def mask_from_pairs(labels, pairs):
    mask = torch.zeros(len(labels['objs']), len(labels['attrs']), dtype=torch.bool)
    for a, o in pairs:
        mask[labels['objs'].index(o), labels['attrs'].index(a)] = True
    return mask
```

### 6.2 任务损失

```math
\begin{aligned}
P^s_{bji}
&=\frac{T_{ji}\exp S_{bji}}
{\sum_{s=1}^{N}\sum_{p=1}^{M}T_{sp}\exp S_{bsp}},\\
\mathcal{L}_{\mathrm{pair}}
&=\frac1B\sum_{b=1}^{B}
\left[-S_{b,j_b,i_b}
+\operatorname{LSE}_{(a_i,o_j)\in\mathcal{C}^s}S_{bji}\right],\\
\mathcal{L}_{\mathrm{cond}}
&=-\frac1B\sum_b\ell^{a\mid o}_{b,j_b,i_b},\\
\mathcal{L}_o&=-\frac1B\sum_b\ell^o_{b,j_b},\\
\mathcal{L}_{\mathrm{task}}
&=\mathcal{L}_{\mathrm{pair}}+0.5\mathcal{L}_{\mathrm{cond}}+0.5\mathcal{L}_o.
\end{aligned}
```

训练样本满足 $`(a_{i_b},o_{j_b})\in\mathcal{C}^s`$。pair损失在训练组合上归一化；条件属性损失使用全部属性，物体损失使用全部物体。

### 6.3 原语监督与JS

```math
\begin{aligned}
\mathcal{L}_{\mathrm{prim}}
&=-\frac1B\sum_b\log p^{\mathrm{prim}}_{b,i_b},\\
m_{bi}&=\frac{p^{\mathrm{prim}}_{bi}+p^{\mathrm{joint}}_{bi}}2,\\
\mathcal{L}_{\mathrm{cent}}
&=\frac{1}{2B}\sum_{b,i}
\left[p^{\mathrm{prim}}_{bi}\log\frac{p^{\mathrm{prim}}_{bi}}{m_{bi}}
+p^{\mathrm{joint}}_{bi}\log\frac{p^{\mathrm{joint}}_{bi}}{m_{bi}}\right].
\end{aligned}
```

程序将对数实现为 $`\log_\epsilon z=\log\max(z,\epsilon)`$，$`\epsilon=10^{-30}`$：

```math
\mathcal{L}_{\mathrm{cent},\epsilon}
=\frac1{2B}\sum_{b,i}
\left[p^{\mathrm{prim}}_{bi}
\bigl(\log_\epsilon p^{\mathrm{prim}}_{bi}-\log_\epsilon m_{bi}\bigr)
+p^{\mathrm{joint}}_{bi}
\bigl(\log_\epsilon p^{\mathrm{joint}}_{bi}-\log_\epsilon m_{bi}\bigr)\right].
```

[vgcpod/model.py · js_divergence](code/vgcpod_v1/vgcpod/model.py#L9)

```python
def js_divergence(p, q):
    m = (p + q) * 0.5
    return 0.5 * (p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())
                  + q * (q.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum(-1).mean()
```

### 6.4 模式正交项与总损失

```math
\begin{aligned}
\mathcal{L}_{\mathrm{orth}}
&=\frac1{r^2}\lVert B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_r\rVert_F^2,\\
\mathcal{L}_{\mathrm{VG}}
&=\mathcal{L}_{\mathrm{task}}
+0.2\mathcal{L}_{\mathrm{prim}}
+0.1\mathcal{L}_{\mathrm{cent},\epsilon}
+10^{-3}\mathcal{L}_{\mathrm{orth}}.
\end{aligned}
```

V0/V1/V1L关闭原语和正交项；V2/V3/V6关闭原语项；V7关闭正交项。两侧属性分布均保留梯度。

[vgcpod/model.py · VGCPOD.objective](code/vgcpod_v1/vgcpod/model.py#L87)

```python
def objective(self, out, attrs, objs, train_mask):
    b = torch.arange(len(attrs), device=attrs.device)
    s = out['joint']
    truth = s[b, objs, attrs]
    pair = (-truth + torch.logsumexp(s[:, train_mask], dim=-1)).mean()
    cond = -out['log_a'][b, objs, attrs].mean()
    obj = -out['log_o'][b, objs].mean()
    zero = truth.new_zeros(())
    prim, cent, orth = zero, zero, zero
    if self.primitive:
        prim = F.cross_entropy(out['prim'], attrs)
        cent = js_divergence(out['prim'].softmax(-1), s.exp().sum(1))
    if self.dynamic:
        r = self.cfg['rank']
        orth = (self.Bvis @ self.Bvis.T - torch.eye(r, device=truth.device,
                dtype=truth.dtype)).square().sum() / (r * r)
    orth_weight = 0.0 if self.variant == 'V7' else self.cfg['lambda_orth']
    total = pair + 0.5 * cond + 0.5 * obj
    total = total + self.cfg['lambda_prim'] * prim + self.cfg['lambda_cent'] * cent + orth_weight * orth
    return {'total': total, 'pair': pair, 'cond': cond, 'object': obj,
            'prim': prim, 'cent': cent, 'orth': orth}
```

## 7. 梯度

### 7.1 任务损失对logits的导数

记

```math
\begin{aligned}
y^a_{bi}&=\mathbf{1}\{i=i_b\},\qquad y^o_{bj}=\mathbf{1}\{j=j_b\},\\
y_{bji}&=y^o_{bj}y^a_{bi},\qquad
p^o_{bj}=\exp\ell^o_{bj},\qquad r_{bji}=\exp\ell^{a\mid o}_{bji},\\
q^o_{bj}&=\sum_i P^s_{bji}.
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial\mathcal{L}_{\mathrm{task}}}{\partial U_{bji}}
&=\frac1B\left[
P^s_{bji}-y_{bji}
-r_{bji}(q^o_{bj}-y^o_{bj})
+0.5y^o_{bj}(r_{bji}-y^a_{bi})\right],\\
\frac{\partial\mathcal{L}_{\mathrm{task}}}{\partial O_{bj}}
&=\frac1B\left[q^o_{bj}-y^o_{bj}
+0.5(p^o_{bj}-y^o_{bj})\right].
\end{aligned}
```

### 7.2 JS在对数下界未激活时的导数

当 $`p^{\mathrm{prim}}_{bi},p^{\mathrm{joint}}_{bi},m_{bi}>\epsilon`$，记

```math
\begin{aligned}
\zeta_{bi}&=\log\frac{p^{\mathrm{joint}}_{bi}}{m_{bi}},\qquad
\xi_{bi}=\log\frac{p^{\mathrm{prim}}_{bi}}{m_{bi}},\\
\bar\zeta_{bj}&=\sum_i r_{bji}\zeta_{bi}.
\end{aligned}
```

```math
\begin{aligned}
\frac{\partial\mathcal{L}_{\mathrm{cent}}}{\partial U_{bji}}
&=\frac{p^o_{bj}r_{bji}}{2B}
\left(\zeta_{bi}-\bar\zeta_{bj}\right),\\
\frac{\partial\mathcal{L}_{\mathrm{cent}}}{\partial O_{bj}}
&=\frac{p^o_{bj}}{2B}
\left(\bar\zeta_{bj}-\sum_s p^o_{bs}\bar\zeta_{bs}\right),\\
\frac{\partial\mathcal{L}_{\mathrm{cent}}}{\partial R_{bi}}
&=\frac{p^{\mathrm{prim}}_{bi}}{2B}
\left(\xi_{bi}-\sum_p p^{\mathrm{prim}}_{bp}\xi_{bp}\right),\\
\frac{\partial\mathcal{L}_{\mathrm{prim}}}{\partial R_{bi}}
&=\frac1B\left(p^{\mathrm{prim}}_{bi}-y^a_{bi}\right).
\end{aligned}
```

实际反向由 `objective` 和 `js_divergence` 的计算图求导，包含 `clamp_min` 的分段导数。

### 7.3 组合修正分支

§7.3–§7.5对应V5的前向定义。

令 $`G_{bji}=\partial\mathcal{L}_{\mathrm{VG}}/\partial U_{bji}`$，$`z_{bj}=W_c^\top f_{bj}^o`$。以下为组合修正路径对各参数的贡献：

```math
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial W_c}
&=\rho\kappa\sum_{b,j,i}G_{bji}f_{bj}^o h_{bji}^\top,\\
\left.\frac{\partial\mathcal{L}}{\partial u_i}\right|_{\mathrm{comp}}
&=\rho\kappa\sum_{b,j}G_{bji}
\left(z_{bj}\odot v_j\odot\gamma_{bj}\right),\\
\left.\frac{\partial\mathcal{L}}{\partial v_j}\right|_{\mathrm{comp}}
&=\rho\kappa\sum_{b,i}G_{bji}
\left(z_{bj}\odot u_i\odot\gamma_{bj}\right),\\
d^\gamma_{bj}
&:=\frac{\partial\mathcal{L}}{\partial\gamma_{bj}}
=\rho\kappa\sum_iG_{bji}
\left(z_{bj}\odot u_i\odot v_j\right),\\
d^t_{bj}&=d^\gamma_{bj}\odot
2\sigma(t_{bj})\odot\bigl(1-\sigma(t_{bj})\bigr),\\
\left.\nabla_{B_{\mathrm{vis}}}\mathcal{L}\right|_{\mathrm{comp}}
&=\frac1{\tau_\gamma}\sum_{b,j}d^t_{bj}(f_{bj}^o)^\top.
\end{aligned}
```

### 7.4 原语分支与共享映射

令 $`H_{bi}=\partial\mathcal{L}_{\mathrm{VG}}/\partial R_{bi}`$，$`z^a_{bi}=W_a^\top f_{bi}^a`$，$`k^a_{bi}=u_i\odot\widetilde v_{bi}`$。

```math
\begin{aligned}
\nabla_{W_a}\mathcal{L}
&=\rho_a\kappa\sum_{b,i}H_{bi}f_{bi}^a(k^a_{bi})^\top,\\
\left.\frac{\partial\mathcal{L}}{\partial u_i}\right|_{\mathrm{prim}}
&=\rho_a\kappa\sum_bH_{bi}
\left(z^a_{bi}\odot\widetilde v_{bi}\right),\\
\left.\frac{\partial\mathcal{L}}{\partial v_j}\right|_{\mathrm{prim}}
&=\rho_a\kappa\sum_{b,i}\pi_{bij}H_{bi}
\left(z^a_{bi}\odot u_i\right),\\
\nabla_{A_u}\mathcal{L}
&=\sum_i\frac{\partial\mathcal{L}}{\partial u_i}(e_i^a)^\top,\\
\nabla_{A_v}\mathcal{L}
&=\sum_j\frac{\partial\mathcal{L}}{\partial v_j}(e_j^o)^\top,\\
\nabla_{B_{\mathrm{vis}}}\mathcal{L}_{\mathrm{orth}}
&=\frac4{r^2}(B_{\mathrm{vis}}B_{\mathrm{vis}}^\top-I_r)B_{\mathrm{vis}}.
\end{aligned}
```

$`\partial\mathcal{L}/\partial u_i`$ 与 $`\partial\mathcal{L}/\partial v_j`$ 为组合和原语路径贡献之和。总门控梯度另加实际正交系数乘上述正交梯度。V5初始化时，$`W_c=W_a=0`$，两个因子映射的任务梯度为零；方向矩阵更新后任务梯度通过相应路径传播。

### 7.5 基线适配器

令 $`G^o_{bj}=\partial\mathcal{L}_{\mathrm{VG}}/\partial O_{bj}`$。

```math
\begin{aligned}
\nabla_{B_o}\mathcal{L}
&=\sum_{b,j}G^o_{bj}g_b(C_o^\top e_j^o)^\top,\\
\nabla_{C_o}\mathcal{L}
&=\sum_{b,j}G^o_{bj}e_j^o(B_o^\top g_b)^\top,\\
\nabla_{B_c}\mathcal{L}
&=\sum_{b,j,i}G_{bji}g_b(C_c^\top e_{ij}^c)^\top,\\
\nabla_{C_c}\mathcal{L}
&=\sum_{b,j,i}G_{bji}e_{ij}^c(B_c^\top g_b)^\top.
\end{aligned}
```

## 8. 优化算子与单批接线

### 8.1 总梯度裁剪

```math
\begin{aligned}
d_t&=\nabla_\vartheta\mathcal{L}_{\mathrm{VG}},\qquad
n_t=\sqrt{\sum_{\theta\in\vartheta}\lVert\operatorname{vec}(\nabla_\theta\mathcal{L}_{\mathrm{VG}})\rVert_2^2},\\
\bar d_t&=d_t\min\left(1,\frac1{n_t+10^{-12}}\right).
\end{aligned}
```

[vgcpod/engine.py · clip_total_norm](code/vgcpod_v1/vgcpod/engine.py#L19)

```python
@torch.no_grad()
def clip_total_norm(parameters, maximum):
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = torch.stack([g.norm(2) for g in gradients]).norm(2)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    coefficient = (maximum / (norm + 1e-12)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient)
    return norm
```

### 8.2 AdamW

```math
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)\bar d_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)\bar d_t\odot\bar d_t,\\
\widehat m_t&=m_t/(1-\beta_1^t),\qquad
\widehat v_t=v_t/(1-\beta_2^t),\\
\vartheta_t&=(1-\eta_t w_d)\vartheta_{t-1}
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon_{\mathrm{Adam}}}.
\end{aligned}
```

$`\beta_1=0.9`$，$`\beta_2=0.999`$，$`\epsilon_{\mathrm{Adam}}=10^{-8}`$，$`w_d=10^{-4}`$。参数组由 `requires_grad` 筛选。

[vgcpod/engine.py · train（局部）](code/vgcpod_v1/vgcpod/engine.py#L106)

```python
parameters = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], betas=tuple(cfg['adam_betas']),
                             eps=cfg['adam_eps'], weight_decay=cfg['weight_decay'])
```

### 8.3 更新次数上的余弦步长

总更新次数为 $`T\ge1`$，第 $`t\in\{1,\ldots,T\}`$ 次更新使用

```math
\eta_t=\eta_{\min}
+\frac{\eta_{\max}-\eta_{\min}}2
\left[1+\cos\left(\frac{\pi(t-1)}{\max(T-1,1)}\right)\right],
\qquad\eta_{\max}=10^{-3},\quad\eta_{\min}=10^{-5}.
```

`step` 在更新前等于 $`t-1`$。

[vgcpod/engine.py · train（局部）](code/vgcpod_v1/vgcpod/engine.py#L134)

```python
eta = cfg['min_lr'] + 0.5 * (cfg['lr'] - cfg['min_lr']) * (
    1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
for group in optimizer.param_groups:
    group['lr'] = eta
optimizer.zero_grad(set_to_none=True)
attrs, objs = attrs.to(args.device), objs.to(args.device)
out = model(g.to(args.device), fa.to(args.device), fo.to(args.device))
losses = model.objective(out, attrs, objs, mask)
if not torch.isfinite(losses['total']):
    raise FloatingPointError('Nonfinite loss at epoch %d step %d' % (epoch + 1, step))
losses['total'].backward()
norm = clip_total_norm(parameters, cfg['gradient_clip'])
optimizer.step()
```

## 9. 推理、候选概率与逐样本指标

### 9.1 前向收集

[vgcpod/engine.py · predict](code/vgcpod_v1/vgcpod/engine.py#L49)

```python
@torch.no_grad()
def predict(model, cache, phase, cfg, device):
    model.eval()
    joint, cond, attrs, objs = [], [], [], []
    for g, fa, fo, a, o in loader(cache, phase, cfg):
        out = model(g.to(device), fa.to(device), fo.to(device))
        joint.append(out['joint'].cpu().numpy())
        cond.append(out['cond'].cpu().numpy())
        attrs.append(a.numpy())
        objs.append(o.numpy())
    return np.concatenate(joint), np.concatenate(cond), np.concatenate(attrs), np.concatenate(objs)
```

### 9.2 候选集合

```math
\begin{aligned}
\mathcal{C}_{\mathrm{val,cw}}&=\mathcal{C}^s\cup\mathcal{C}_{\mathrm{val}},\\
\mathcal{C}_{\mathrm{test,cw}}&=\mathcal{C}^s\cup\mathcal{C}_{\mathrm{test}},\\
\mathcal{C}_{\mathrm{ow}}&=\mathcal{C}_{\mathrm{all}},\\
\log P_{\mathcal{C}}(a_i,o_j\mid x_b)
&=S_{bji}-\operatorname{LSE}_{(a_p,o_s)\in\mathcal{C}}S_{bsp}.
\end{aligned}
```

候选表按属性、物体标签字符串的字典序排列。无bias预测读取候选评分的 `torch.topk(1)`；类别NLL读取当前候选集归一化概率。

### 9.3 指标

令 $`c_b=(a_{i_b},o_{j_b})`$，$`\widehat c_b`$ 为无bias预测，$`\mathcal{I}_c=\{b:c_b=c\}`$，$`\mathcal{C}_{\mathrm{obs}}=\{c:\lvert\mathcal{I}_c\rvert>0\}`$。

```math
\begin{aligned}
\operatorname{Acc}_{\mathrm{pair}}
&=\frac1B\sum_b\mathbf{1}\{\widehat c_b=c_b\},\\
\operatorname{Acc}_{\mathrm{macro}}
&=\frac1{\lvert\mathcal{C}_{\mathrm{obs}}\rvert}
\sum_{c\in\mathcal{C}_{\mathrm{obs}}}
\frac1{\lvert\mathcal{I}_c\rvert}
\sum_{b\in\mathcal{I}_c}\mathbf{1}\{\widehat c_b=c\},\\
\operatorname{NLL}_{\mathrm{pair}}
&=-\frac1B\sum_b\log P_{\mathcal{C}}(c_b\mid x_b),\\
\operatorname{Acc}_{a\mid o}
&=\frac1B\sum_b\mathbf{1}\left\{\arg\max_iU_{b,j_b,i}=i_b\right\},\\
\operatorname{NLL}_{a\mid o}
&=-\frac1B\sum_b\ell^{a\mid o}_{b,j_b,i_b}.
\end{aligned}
```

准确率输出乘100；NLL使用自然对数。GT-object属性预测采用 `numpy.argmax`，同分时取属性轴首项。四类错误由属性与物体预测正确性分别取交集计数。

[vgcpod/metrics.py · evaluate_scores](code/vgcpod_v1/vgcpod/metrics.py#L78)

```python
def evaluate_scores(joint, cond, attrs, objs, labels, pairs, phase, world, fixed_bias=None):
    m, n = len(labels['attrs']), len(labels['objs'])
    train = {tuple(p) for p in pairs['train']}
    allowed = train | {tuple(p) for p in pairs[phase]} if world == 'cw' else {
        (a, o) for a in labels['attrs'] for o in labels['objs']}
    # Preserve the evaluator's alphabetical attribute/object pair ordering for topk ties.
    candidates = sorted(allowed)
    ai = np.array([labels['attrs'].index(a) for a, o in candidates])
    oi = np.array([labels['objs'].index(o) for a, o in candidates])
    scores = joint[:, oi, ai].astype(np.float32)
    index = {p: i for i, p in enumerate(candidates)}
    truth = np.array([index[(labels['attrs'][a], labels['objs'][o])] for a, o in zip(attrs, objs)])
    seen = np.array([p in train for p in candidates])
    native = native_curve(scores, truth, seen)
    exact = exact_curve(scores, truth, seen)
    max_scores = scores.max(1, keepdims=True)
    log_probs = scores - (max_scores + np.log(np.exp(scores - max_scores).sum(1, keepdims=True)))
    pred = torch.from_numpy(scores).topk(1, dim=1).indices[:, 0].numpy()
    correct = pred == truth
    attr_correct, obj_correct = ai[pred] == attrs, oi[pred] == objs
    conditional = cond[np.arange(len(attrs)), objs]
    z = conditional.max(1, keepdims=True)
    cond_log = conditional - z - np.log(np.exp(conditional - z).sum(1, keepdims=True))
    present = np.unique(truth)
    metrics = {'images': len(truth), 'candidates': len(candidates),
               'AUC_native': native['AUC'], 'AUC_exact': exact['AUC'],
               'HM_oracle_native': native['best_hm'], 'HM_oracle_exact': exact['best_hm'],
               'bias_native': native['best_bias'], 'bias_exact': exact['best_bias'],
               'pair_acc_unbiased': float(correct.mean() * 100),
               'pair_macro_unbiased': float(np.mean([correct[truth == k].mean() for k in present]) * 100),
               'pair_nll_unbiased': float(-log_probs[np.arange(len(truth)), truth].mean()),
               'gt_object_attr_acc': float((conditional.argmax(1) == attrs).mean() * 100),
               'gt_object_attr_nll': float(-cond_log[np.arange(len(attrs)), attrs].mean()),
               'errors_unbiased': {'both_correct': int((attr_correct & obj_correct).sum()),
                                    'attribute_wrong_object_correct': int((~attr_correct & obj_correct).sum()),
                                    'attribute_correct_object_wrong': int((attr_correct & ~obj_correct).sum()),
                                    'both_wrong': int((~attr_correct & ~obj_correct).sum())}}
    if fixed_bias is not None:
        biased = torch.from_numpy(scores.copy())
        biased[:, ~torch.from_numpy(seen)] += fixed_bias
        ok = biased.topk(1, dim=1).indices[:, 0].numpy() == truth
        s, u = ok[seen[truth]].mean(), ok[~seen[truth]].mean()
        metrics['fixed_val_bias'] = float(fixed_bias)
        metrics['seen_at_val_bias'] = float(s * 100)
        metrics['unseen_at_val_bias'] = float(u * 100)
        metrics['HM_at_val_bias'] = float(hm(s, u) * 100)
    return metrics, {'native': native, 'exact': exact}, np.array(candidates), log_probs
```

## 10. Seen/unseen校准、曲线与选模

### 10.1 统一bias与面积

评价样本中seen与unseen均非空。

```math
\begin{aligned}
\widehat c_b(\beta)
&=\underset{c\in\mathcal{C}}{\arg\max}
\left[S_b(c)+\beta\mathbf{1}\{c\notin\mathcal{C}^s\}\right],\\
S(\beta)&=\frac{\sum_{b:c_b\in\mathcal{C}^s}\mathbf{1}\{\widehat c_b(\beta)=c_b\}}
{\lvert\{b:c_b\in\mathcal{C}^s\}\rvert},\\
U(\beta)&=\frac{\sum_{b:c_b\notin\mathcal{C}^s}\mathbf{1}\{\widehat c_b(\beta)=c_b\}}
{\lvert\{b:c_b\notin\mathcal{C}^s\}\rvert},\\
\operatorname{HM}(\beta)&=\begin{cases}
\dfrac{2S(\beta)U(\beta)}{S(\beta)+U(\beta)},&S(\beta)+U(\beta)>0,\\
0,&S(\beta)+U(\beta)=0,
\end{cases}\\
\operatorname{AUC}&=100\sum_{k=1}^{K-1}
\frac{S_k+S_{k+1}}2(U_{k+1}-U_k).
\end{aligned}
```

[vgcpod/metrics.py · hm](code/vgcpod_v1/vgcpod/metrics.py#L6)

```python
def hm(s, u):
    s, u = np.asarray(s), np.asarray(u)
    return np.divide(2 * s * u, s + u, out=np.zeros_like(s + u, dtype=float), where=(s + u) != 0)
```

### 10.2 原生有限扫描 `AUC_native`

对unseen真值样本，定义

```math
\delta_b^{\mathrm{gt}}
=\max_{c\in\mathcal{C}\cap\mathcal{C}^s}S_b(c)-S_b(c_b).
```

仅保留在bias1000时预测正确的unseen样本，按 $`\delta_b^{\mathrm{gt}}-10^{-4}`$ 升序排列得到 $`d_0,\ldots,d_{K-1}`$。若 $`K>0`$，使用

```math
h=\max(\lfloor K/20\rfloor,1),\qquad
\mathfrak{B}_{\mathrm{native}}=(d_0,d_h,d_{2h},\ldots,1000).
```

若 $`K=0`$，扫描序列仅含1000。曲线按扫描次序做梯形积分；HM平局由 `np.argmax` 取首次。

[vgcpod/metrics.py · native_curve](code/vgcpod_v1/vgcpod/metrics.py#L11)

```python
def native_curve(scores, truth, seen_classes):
    # Same top-1 CPF scan: correct unseen differences -1e-4, ~20 bins, endpoint bias=1000.
    x = torch.as_tensor(scores, dtype=torch.float32)
    y = torch.as_tensor(truth, dtype=torch.long)
    seen = torch.as_tensor(seen_classes, dtype=torch.bool)
    is_seen = seen[y]
    if not is_seen.any() or is_seen.all():
        raise ValueError('AUC requires seen and unseen evaluation images')
    endpoint = x.clone()
    endpoint[:, ~seen] += 1000.0
    pred = endpoint.topk(1, dim=1).indices[:, 0]
    correct = pred.eq(y)
    un = ~is_seen
    delta = x[un][:, seen].topk(1, dim=1).values[:, 0] - x[un, y[un]]
    diffs = torch.sort(delta[correct[un]] - 1e-4).values
    step = max(len(diffs) // 20, 1)
    biases = [float(b) for b in diffs[::step]] + [1000.0]
    ss, uu = [], []
    for bias in biases:
        z = x.clone()
        z[:, ~seen] += bias
        correct = z.topk(1, dim=1).indices[:, 0].eq(y)
        ss.append(float(correct[is_seen].float().mean()))
        uu.append(float(correct[un].float().mean()))
    h = hm(ss, uu)
    best = int(np.argmax(h))
    trapz = getattr(np, 'trapezoid', np.trapz)
    return {'AUC': float(trapz(ss, uu) * 100), 'best_hm': float(h[best] * 100),
            'best_bias': biases[best], 'best_seen': max(ss) * 100, 'best_unseen': max(uu) * 100,
            'biases': biases, 'seen': (100 * np.array(ss)).tolist(), 'unseen': (100 * np.array(uu)).tolist()}
```

### 10.3 全临界事件扫描 `AUC_exact`

```math
\begin{aligned}
c_b^s&=\arg\max_{c\in\mathcal{C}\cap\mathcal{C}^s}S_b(c),\\
c_b^u&=\arg\max_{c\in\mathcal{C}\setminus\mathcal{C}^s}S_b(c),\\
\delta_b&=S_b(c_b^s)-S_b(c_b^u).
\end{aligned}
```

在不等于临界值的bias区间内，$`\beta<\delta_b`$ 时预测 $`c_b^s`$，$`\beta>\delta_b`$ 时预测 $`c_b^u`$。组内最大值由 `numpy.argmax` 按首个候选取值。

令不同临界值为 $`\theta_1<\cdots<\theta_J`$；$`n_s,n_u`$ 为seen/unseen样本数。定义

```math
\begin{aligned}
\chi_b^s&=\mathbf{1}\{c_b\in\mathcal{C}^s,\ c_b^s=c_b\},\\
\chi_b^u&=\mathbf{1}\{c_b\notin\mathcal{C}^s,\ c_b^u=c_b\},\\
S_k&=\frac{\sum_b\chi_b^s
-\sum_{\ell=1}^{k}\sum_{b:\delta_b=\theta_\ell}\chi_b^s}{n_s},\\
U_k&=\frac{\sum_{\ell=1}^{k}\sum_{b:\delta_b=\theta_\ell}\chi_b^u}{n_u},
\qquad k=0,\ldots,J.
\end{aligned}
```

相同临界值的事件同时更新；$`k=0`$ 的累计和为零。各区间用中点或两侧外部点表示，使用§10.1的面积公式。

[vgcpod/metrics.py · exact_curve](code/vgcpod_v1/vgcpod/metrics.py#L43)

```python
def exact_curve(scores, truth, seen_classes):
    x = np.asarray(scores, dtype=np.float64)
    seen = np.asarray(seen_classes, dtype=bool)
    y = np.asarray(truth)
    si, ui = np.flatnonzero(seen), np.flatnonzero(~seen)
    if not len(si) or not len(ui):
        raise ValueError('Curve needs both candidate partitions')
    # A single fixed winner within each partition; events switch seen -> unseen.
    sp = si[x[:, si].argmax(1)]
    up = ui[x[:, ui].argmax(1)]
    d = x[np.arange(len(x)), sp] - x[np.arange(len(x)), up]
    sm, um = seen[y], ~seen[y]
    ns, nu = sm.sum(), um.sum()
    if ns == 0 or nu == 0:
        raise ValueError('Curve needs both sample partitions')
    cs, cu = (sp == y) & sm, (up == y) & um
    order = np.argsort(d, kind='stable')
    thresholds, starts = np.unique(d[order], return_index=True)
    loss_s = np.add.reduceat(cs[order].astype(int), starts)
    gain_u = np.add.reduceat(cu[order].astype(int), starts)
    ss = np.r_[cs.sum(), cs.sum() - np.cumsum(loss_s)] / ns
    uu = np.r_[0, np.cumsum(gain_u)] / nu
    h = hm(ss, uu)
    # Choose interval interiors, so no prediction depends on equality at a threshold.
    low = thresholds[0] - max(1.0, abs(thresholds[0]) * 1e-6)
    high = thresholds[-1] + max(1.0, abs(thresholds[-1]) * 1e-6)
    midpoints = thresholds[:-1] + (thresholds[1:] - thresholds[:-1]) * 0.5
    biases = np.r_[low, midpoints, high]
    best = int(np.argmax(h))
    trapz = getattr(np, 'trapezoid', np.trapz)
    return {'AUC': float(trapz(ss, uu) * 100), 'best_hm': float(h[best] * 100),
            'best_bias': float(biases[best]), 'thresholds': thresholds.tolist(),
            'biases': biases.tolist(), 'seen': (ss * 100).tolist(), 'unseen': (uu * 100).tolist()}
```

### 10.4 Checkpoint与VAL bias

```math
t^*=\min\underset{t}{\arg\max}\;
\operatorname{AUC}_{\mathrm{native}}^{\mathrm{VAL,CW}}(t).
```

[vgcpod/engine.py · train（局部）](code/vgcpod_v1/vgcpod/engine.py#L161)

```python
selected = metrics['AUC_native'] > best
if selected:
    best = metrics['AUC_native']
```

CW与OW各自取选中权重的VAL原生扫描最佳HM bias，并固定用于TEST；TEST曲线最佳HM另存为 `HM_oracle_native/exact`。

[vgcpod/engine.py · evaluate_run](code/vgcpod_v1/vgcpod/engine.py#L197)

```python
def evaluate_run(cache, run, device):
    run = Path(run)
    checkpoint = load_torch(run / 'checkpoints' / 'best.pt')
    cfg, seed = checkpoint['config'], checkpoint['seed']
    seed_all(seed)
    model, manifest = make_model(cache, cfg, seed, device)
    check_run_identity(cache, run, checkpoint)
    model.load_state_dict(checkpoint['model'])
    results = {'selected_epoch': checkpoint['epoch'], 'selection': 'VAL CW AUC_native',
               'checkpoint_sha256': sha(run / 'checkpoints' / 'best.pt'), 'metrics': {}}
    biases = {}
    for phase in ('val', 'test'):
        joint, cond, attrs, objs = predict(model, cache, phase, cfg, device)
        for world in ('cw', 'ow'):
            fixed = biases.get(world)
            if phase == 'val':
                provisional, _, _, _ = evaluate_scores(joint, cond, attrs, objs, manifest['labels'],
                                                        manifest['pairs'], phase, world)
                fixed = biases[world] = provisional['bias_native']
            results['metrics'][phase + '_' + world] = write_evaluation(cache, run, manifest, cfg,
                                    phase, joint, cond, attrs, objs, world, fixed)
    write_json(run / 'metrics.json', results)
    return results
```

## 11. 变体定义与参数量

设 $`\lambda_p,\lambda_c,\lambda_o`$ 分别为原语、向心、正交项系数。

| 变体 | 组合读出 | 模式门控 | 伙伴分布 | $`(\lambda_p,\lambda_c,\lambda_o)`$ |
|---|---|---|---|---|
| V0 | $`U=U^0`$ | 关闭 | 关闭 | $`(0,0,0)`$ |
| V1 | $`g_b`$ | $`\gamma=\mathbf{1}`$ | 关闭 | $`(0,0,0)`$ |
| V1L | $`f_{bj}^o`$ | $`\gamma=\mathbf{1}`$ | 关闭 | $`(0,0,0)`$ |
| V2 | $`f_{bj}^o`$ | 类别查询门控 | 关闭 | $`(0,0,10^{-3})`$ |
| V3 | $`g_b`$ | 全局图像门控 | 关闭 | $`(0,0,10^{-3})`$ |
| V4 | $`f_{bj}^o`$ | 类别查询门控 | $`\pi_{bij}=1/N`$ | $`(0.2,0.1,10^{-3})`$ |
| V5 | $`f_{bj}^o`$ | 类别查询门控 | 类别查询伙伴概率 | $`(0.2,0.1,10^{-3})`$ |
| V6 | $`f_{bj}^o`$ | 同批循环错配门控 | 关闭 | $`(0,0,10^{-3})`$ |
| V7 | $`f_{bj}^o`$ | 类别查询门控 | 类别查询伙伴概率 | $`(0.2,0.1,0)`$ |

V3：$`t_{bj}=B_{\mathrm{vis}}g_b/\tau_\gamma`$。V6：令 $`\sigma_B(b)=1+\operatorname{mod}(b-2,B)`$，则

```math
\gamma^{\mathrm{V6}}_{bj}
=2\sigma\!\left(B_{\mathrm{vis}}f^o_{\sigma_B(b),j}/\tau_\gamma\right),
\qquad
U^{\mathrm{V6}}_{bji}=U^0_{bji}
+\rho\kappa(f^o_{bj})^\top W_c
\left(u_i\odot v_j\odot\gamma^{\mathrm{V6}}_{bj}\right).
```

V6的错配在每个当前batch内定义；$`B=1`$ 时为自身。前向中的变体路由见§2、§5。

```math
\begin{aligned}
n_{\mathrm{V0}}&=4dq=24576,\\
n_{\mathrm{V1}}=n_{\mathrm{V1L}}&=4dq+3dr=98304,\\
n_{\mathrm{V2}}=n_{\mathrm{V3}}=n_{\mathrm{V6}}&=4dq+4dr=122880,\\
n_{\mathrm{V4}}=n_{\mathrm{V5}}=n_{\mathrm{V7}}&=4dq+5dr=147456.
\end{aligned}
```

[vgcpod/model.py · VGCPOD.trainable_count](code/vgcpod_v1/vgcpod/model.py#L109)

```python
def trainable_count(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

未训练基线：名称CLIP读取 $`\kappa g_b^\top e_{ij}^c`$；链式零样本读取 $`C_o=C_c=W_c=0`$ 时的 $`S`$。D0对应V0，训练两个适配器。

## 12. 实现索引

| 源码 | 函数／方法 | 算子 |
|---|---|---|
| [vgcpod/common.py](code/vgcpod_v1/vgcpod/common.py#L72) | `unit` | 带范数下界的单位化 |
| [vendor/clip/clip.py](code/vgcpod_v1/vendor/clip/clip.py#L79) | `_transform` | CLIP图像预处理 |
| [vgcpod/features.py](code/vgcpod_v1/vgcpod/features.py#L17) | `visual_tokens` | CLS与补丁投影 |
| [vgcpod/features.py](code/vgcpod_v1/vgcpod/features.py#L37) | `prompts` | 名称文本模板 |
| [vgcpod/features.py](code/vgcpod_v1/vgcpod/features.py#L32) | `query_patches` | 类别查询 |
| [vgcpod/features.py](code/vgcpod_v1/vgcpod/features.py#L66) | `prepare` | 冻结编码与FP16缓存 |
| [vgcpod/data.py](code/vgcpod_v1/vgcpod/data.py#L109) | `FeatureDataset.__getitem__` | FP32缓存读取 |
| [vgcpod/data.py](code/vgcpod_v1/vgcpod/data.py#L123) | `mask_from_pairs` | 训练组合mask |
| [vgcpod/model.py](code/vgcpod_v1/vgcpod/model.py#L16) | `VGCPOD.__init__` | 参数、初始化与路由 |
| [vgcpod/model.py](code/vgcpod_v1/vgcpod/model.py#L55) | `VGCPOD.forward` | 完整前向 |
| [vgcpod/model.py](code/vgcpod_v1/vgcpod/model.py#L9) | `js_divergence` | 带对数下界的JS |
| [vgcpod/model.py](code/vgcpod_v1/vgcpod/model.py#L87) | `VGCPOD.objective` | 损失组合 |
| [vgcpod/engine.py](code/vgcpod_v1/vgcpod/engine.py#L19) | `clip_total_norm` | 总梯度裁剪 |
| [vgcpod/engine.py](code/vgcpod_v1/vgcpod/engine.py#L89) | `train` | AdamW更新与选模 |
| [vgcpod/engine.py](code/vgcpod_v1/vgcpod/engine.py#L49) | `predict` | 推理收集 |
| [vgcpod/engine.py](code/vgcpod_v1/vgcpod/engine.py#L232) | `baselines` | 名称CLIP与链式零样本 |
| [vgcpod/engine.py](code/vgcpod_v1/vgcpod/engine.py#L197) | `evaluate_run` | 选中权重的VAL/TEST评价 |
| [vgcpod/metrics.py](code/vgcpod_v1/vgcpod/metrics.py#L78) | `evaluate_scores` | 候选归一化与指标 |
| [vgcpod/metrics.py](code/vgcpod_v1/vgcpod/metrics.py#L11) | `native_curve` | 原生有限bias扫描 |
| [vgcpod/metrics.py](code/vgcpod_v1/vgcpod/metrics.py#L43) | `exact_curve` | 完整临界事件扫描 |

CLIP源码提交：`a1d071733d7111c9c014f024669f959182114e33`。

ViT-L/14权重SHA256：`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。

| 实现文件 | SHA256 |
|---|---|
| `vgcpod/features.py` | `1e506068d63c6e5278320d803a504ca8e1107fa896c50358644e177a30e1c685` |
| `vgcpod/model.py` | `fdb3b0f48b0b2143f608ffc71e86dc9a3db810114c95aa1fd66b118ba9c3f58f` |
| `vgcpod/engine.py` | `560e34b3e1acd16f86d05aa9ade31d78abd6d5a7a84026b66e45854b973a26c2` |
| `vgcpod/metrics.py` | `b356faa5a4732696c0204fd5cca4863b22c66ea5265ed294615dbb7f5dc04750` |
