# TRUST-CPF 第一批：公式、核心实现与验证指标

版本：State v1，10 epoch；日期：2026-10-25。

本版实现第一批的联合拟合与冻结部署。按本次要求，从旧 reference 为每个 seed **新训练 10 epoch Uniform**；在 epoch 0–10 按 VAL CW native AUC 选择共同权重，再冻结识别器比较策略。教师、测量标签和 CLIP 缓存复用上一轮。

## 1. 数据与执行顺序

全局导入 R5 教师与测量 → 共同 Q4 选参及九教师拟合 → 三 seed 各训练 10 epoch Uniform → 当前状态响应 → 冻结分配与识别。

拟合在 setup 中一次完成。三个 GPU worker 分别处理一个 seed，读取同一份联合模型，不并发修改它。

| 项目 | 本版固定值 |
| --- | --- |
| 训练 seed | 3351／3352／3353 |
| Uniform | 10 epoch；batch 64；AdamW；lr 3×10⁻⁴；weight decay 10⁻⁴ |
| 学习率计划 | 一个 epoch warmup，随后 cosine；梯度裁剪 1 |
| CPF | 全程冻结；关系分支按原任务损失训练 |
| 教师 | 复用原 15 个教师，第一批不新增教师训练 |
| 拟合／选参／评价 | Q1–Q3／Q4／Q5 |
| 主预测器 | Independent-Absolute、Pooled-Absolute、Pooled-Relative |
| Relative 独立拟合 | 保留 CPU 2×2 诊断，不另运行 GPU 组 |
| 策略 | κ=1，类型期望边数 6／18／24 |
| 正式读出 | 16 图；bank 1701／2701／3701 |
| Q5 | 16 图；bank 8201–8208 |
| 类型内置换 | 7101–7105，汇报均值 |
| 响应 | 每组合最多两张图，四个背景；两次独立测量 |

## 2. U-NLL 效用及教师对应

第 j 个教师的未见组合测量风险为：


```math
R_j(\theta,b)
=
-\frac1{|D_j^{u,m}|}\sum_{(x,y)\in D_j^{u,m}}
\log p_\theta^{\mathcal C_j}(y\mid x,b).
```


在 Bernoulli(0.5) 背景下，开／关边使用同一批图像和同一组其他边：


```math
\widehat u_{j,e}
=
\frac1{K_j}\sum_{k=1}^{K_j}
\left[
R_j(\theta_j,b_{-e}^{(k)},0)
-
R_j(\theta_j,b_{-e}^{(k)},1)
\right].
```


保持原 Q1–Q4 的 8 背景、Q5 的 16 背景标签。正值表示降低未见 NLL。令 F 为独立 Bernoulli 图下的期望单图风险，固定教师参数时有：


```math
-\left.\frac{\partial F_j(\theta_j,\pi)}{\partial\pi_e}\right|_{\pi=1/2}
=
u_{j,e}.
```


预测输入为同一教师的八维响应：全支持集与目标原语支持子集各四维——NLL 改善均值、标准差、中心化候选评分差 RMS、熵变化。支持图像只来自该教师合法训练图像。

## 3. 联合拟合及正则化

由拟合教师计算输入中心 c、尺度 s 和类型标签标准差 a。设计矩阵使用三个类型块，每块为一个截距和八个斜率：


```math
\mu_{j,e}
=
a_{t(e)}x_{j,e}^{\top}\beta,
\qquad
x_{j,e}
=
\mathop{\mathrm{onehot}}\nolimits(t(e))
\otimes
\begin{bmatrix}1\\(h_{j,e}-c)\oslash s\end{bmatrix}.
```


主拟合目标：


```math
L_{\mathrm{abs}}
=
\frac1J\sum_{j,t}
\left\|X_{jt}\beta-\widehat u_{jt}/a_t\right\|_2^2
+\lambda\|\beta\|_2^2.
```


令导数为零，可得正规方程。复用未平均求和的求解器时：


```math
\left(X^\top X+J\lambda I\right)\widehat\beta=X^\top y,
\qquad
\alpha=J\lambda.
```


独立模型 J=3，联合模型 J=9。λ 网格为 {0.1,1,10,100}/3。三个 Q4 的类型等权中心化归一化 MSE 共同选一个 λ；并列选较大值。Absolute 和 Relative 在同一教师配置内共享归一化参数。

Relative 在每个教师、每个类型内同时中心化设计矩阵与目标：


```math
P_t=I-\frac{\mathbf1\mathbf1^\top}{N_t},
\qquad
L_{\mathrm{rel}}
=
\frac1J\sum_{j,t}
\left\|P_t(X_{jt}\beta-\widehat u_{jt}/a_t)\right\|_2^2
+\lambda\|\beta\|_2^2.
```


本版没有 Q1–Q5 再拟合：Q5 与正式部署使用同一份三／九教师预测器。下面为发布代码中的实际函数。

```python
def fit_one(xs,ys,types,lam,relative=False):
    m=compact.fit_ridge(xs,ys,types,len(xs)*lam)
    if relative:
        m=r5.fit_relative(xs,ys,types,len(xs)*lam,{k:m[k] for k in ('center','spread','scales')})
    beta=np.asarray(m['beta']).reshape(3,9)
    m.update(lam=float(lam),teachers=len(xs),objective_teacher_averaged=True,
        raw_slopes=m['scales'][:,None]*beta[:,1:]/m['spread'],
        raw_intercepts=m['scales']*(beta[:,0]-(beta[:,1:]*m['center']/m['spread']).sum(1)),
        feature_names=response.FEATURE_NAMES,types=np.asarray(types),edge_ids=np.arange(len(types)),
        fit_folds=[1,2,3],selection_fold=4,audit_fold=5)
    return m
```

## 4. 分配与匹配置换

类型内标准化分数为：


```math
r_e=\frac{\mu_e-\overline\mu_t}{\mathop{\mathrm{std}}\nolimits(\mu_t)}.
```


类型标准差不大于 10⁻¹² 时分数置零。策略求解：


```math
\max_\pi
\left[
\kappa r^\top\pi-
\sum_e\mathop{\mathrm{KL}}\nolimits(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\|\mathop{\mathrm{Bern}}\nolimits(1/2))
\right],
\qquad
\sum_{e\in E_t}\pi_e=N_t/2.
```


KL 的导数为 logit，因此：


```math
\pi_e=\sigma(\kappa r_e-\gamma_t).
```


每个类型的乘子用二分求解，预算误差控制在 10⁻¹⁰ 内。所有主预测器固定 κ=1；Uniform 单列。每次采图的边数随机变化，6／18／24 是期望边数。

```python
def policy(mu, types, strength):
    """Unique equality-budget KL solution in float64; return CPU probabilities."""
    mu = np.asarray(mu, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    if mu.shape != types.shape or not np.isfinite(mu).all() or strength < 0:
        raise ValueError('Invalid utility vector or strength')
    prob = np.full(mu.shape, .5, dtype=np.float64)
    info = {}
    for t in sorted(np.unique(types)):
        ix = types == t
        mean, std = float(mu[ix].mean()), float(mu[ix].std(ddof=0))
        r = (mu[ix] - mean) / std if std > 1e-12 else np.zeros(ix.sum())
        z = strength * r
        gamma = 0.
        if strength > 0 and std > 1e-12:
            lo, hi = float(z.min() - 40), float(z.max() + 40)
            for _ in range(200):
                gamma = (lo + hi) / 2
                p = 1 / (1 + np.exp(np.clip(gamma - z, -700, 700)))
                residual = p.sum() - ix.sum() / 2
                if abs(residual) <= 1e-12:
                    break
                if residual > 0:
                    lo = gamma
                else:
                    hi = gamma
            prob[ix] = p
        error = abs(prob[ix].sum() - ix.sum() / 2)
        if error > 1e-10:
            raise ArithmeticError('Equality budget did not converge')
        info[str(t)] = dict(mu_mean=mean, mu_std=std, gamma=gamma,
                            budget=float(ix.sum() / 2), budget_error=float(error))
    return torch.from_numpy(prob), info
```

置换保持类型内概率多重集合、预算和强度。每个模型使用五个置换，主比较取其平均，不选择某个最佳置换。

## 5. 训练 10 epoch 后冻结使用

本次新增三条 Uniform 轨迹，各完整训练 10 epoch。按 VAL CW native AUC 从 epoch 0–10 选权重，并列取最早者：


```math
e_s^\star=\operatorname*{arg\,max}_{e=0,\ldots,10}
A_{\mathrm{VAL,CW}}(\theta_{s,e}^{U},1/2).
```


之后在该权重上重新计算当前 TRAIN 响应：


```math
H_s^\star=\mathop{\mathrm{Response}}\nolimits(\theta_s^\star,D_{\mathrm{TRAIN}}),
\qquad
\pi_s^\star=\mathop{\mathrm{Policy}}\nolimits(f_\beta(H_s^\star)).
```


所有策略在该 seed 共享同一识别权重。reference 仅作为旧状态参照。不存在每个策略各自再适配后选择不同权重的步骤。

```python
def train_uniform(args):
    work,cfg,manifest,text,bank=align.context(args)
    cfg=dict(cfg,variant='uniform',readout='probability')
    out=work/'runs/uniform';out.mkdir(parents=True,exist_ok=True)
    pi=torch.full((len(bank['types']),),.5)
    if not (out/'epoch0.json').exists():
        model=align.restore(text,bank,cfg,args.device,work/'reference/last.pt')
        val=state_eval.evaluate_cached(model,args.cache,cfg,manifest,pi,'val')
        write_json(out/'epoch0.json',dict(epoch=0,val=val))
        del model
    if (out/'best.pt').exists() and not (out/'last.pt').exists():
        (out/'best.pt').unlink()
    engine.train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],out,
                       pi=pi,initial=work/'reference/last.pt',select=True)
    last=load_torch(out/'last.pt');best=load_torch(out/'best.pt')
    original=load_torch(work/'reference/last.pt')['model']
    assert last['epoch']==cfg['adapt_epochs']==10 and len(last['history'])==10
    for name in ('last','best'):
        values=last['model'] if name=='last' else best['model']
        assert all(torch.equal(v,values[k]) for k,v in original.items() if k.startswith('backbone.'))
    curve=[read_json(out/'epoch0.json')]+last['history']
    chosen=max(curve,key=lambda r:(r['val']['cw']['AUC_native'],-r['epoch']))['epoch']
    assert chosen==best['epoch']
    write_json(out/'training_contract.json',dict(epochs_completed=10,selected_epoch=chosen,
        selection='VAL CW native AUC; first maximum among epoch 0..10',
        initial_sha256=sha(work/'reference/last.pt'),best_sha256=sha(out/'best.pt'),
        last_sha256=sha(out/'last.pt'),backbone_identical=True,
        training='new Uniform trajectory; source Uniform outputs are not imported'))
    print('UNIFORM_TRAIN_COMPLETE',args.seed,'last_epoch',last['epoch'],'best_epoch',chosen,flush=True)
```

Uniform-best 主响应使用与 reference 完全相同的支持图像、权重和背景 seed，使响应交叉只改变识别模型状态。第二次响应独立重新采样，两种正式状态共享第二次探针；Q5 也追加一次独立响应，用于稳定性诊断。主分配使用第一次响应，平均响应只记录线性平均等式的校验与诊断，不替换主结果。

## 6. 概率混合及实际对照

对每张图先在相应 CW／OW 候选集归一化，再平均 16 张图概率：


```math
\overline p^{\mathcal C}(c\mid x)
=
\frac1{16}\sum_{m=1}^{16}
\frac{\exp s_\theta(x,c;b^{(m)})}
{\sum_{c'\in\mathcal C}\exp s_\theta(x,c';b^{(m)})}.
```


相同 bank 的所有策略共享均匀随机数。正式识别沿用原设备上的图随机数规则；Q5 沿用原 CPU 生成 bank 的规则。新评估器缓存冻结 CPF 的输出，概率混合、候选轴和指标计算与旧版保持一致，并保存实际采样掩码用于复算。

| 状态 | 策略实例数 | 构成 |
| --- | --- | --- |
| reference | 20 | Uniform＋三预测器各真实及五置换＋Pooled-Absolute 错配响应 |
| uniform_best | 22 | 同上，另加历史边均值、直接支持 NLL |
| Q5 教师 | 25 | Uniform＋三预测器及测量效用各真实及五置换 |

Pooled-Absolute 的响应交叉包括 reference／Uniform-best 两种权重×两种响应来源。两种简单读出只在 Uniform-best 追加。

## 7. 验证的核心指标

| 问题 | 主要指标 |
| --- | --- |
| 联合拟合能否提高识别 | Pooled-Absolute−Independent-Absolute 的 CW／OW AUC 配对差值 |
| 正确关系对应有无作用 | 真实分配−五置换均值、真实分配−Uniform |
| Relative 是否与教师覆盖交互 | Pooled-Relative−Pooled-Absolute |
| 当前模型响应是否有价值 | Pooled-Absolute 正确响应−错配响应 |
| 效用拟合改善能否用于决策 | Q5 中心化误差／R²、类型内相关、实际混合 NLL 与 AUC |
| 输入测量是否稳定 | 两次响应的预测方差、概率 MSE、排序相关 |
| 图使用是否改变 | 类型概率标准差、KL、节点期望入边／覆盖、原型位移 |

同时保存 exact AUC、VAL 工作点 HM、VAL 校准后 TEST NLL／ECE。主 bank 为 1701；三个 bank 先在 seed 内平均，再汇总三个 seed。Q5 bank 和概率置换属于条件重复，不扩充训练重复数。

## 8. 本地验证和运行入口

验证包括三 seed 完整 10 epoch 小型队列、缓存评估器与原评估器逐项一致、概率预算与置换、恢复运行、源码输入未改写、实际结果打包解压后逐文件校验。

另外，使用真实 R5 的 15 个教师输入重新拟合四个配置，选中 λ 与方案报告一致，预测相关、R² 和一阶代理收益最大差异为 0。

实际服务器 UT-Zappos 队列由 SERVER_COMMANDS.md 的命令启动。验证用小型数据的最佳 epoch 或 AUC 不作为真实实验结果。

