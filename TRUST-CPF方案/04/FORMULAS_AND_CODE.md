# TRUST-CPF Compact v1：公式与核心实现

本文件对应 `trustcpf/compact.py` 与 `trustcpf/workflow.py`。本轮保留 TRUST 的跨组合配对效用监督、模型条件、固定预算图选择与 CZSL 适配链条。

## 1. 配对收益与均值监督

教师编号 $`j\in\{1,\ldots,5\}`$，关系边 $`e\in\{1,\ldots,E\}`$，类型 $`t(e)\in\{0,1,2\}`$，分别对应 OI、OE、AE。正式知识库有 $`E=96`$ 条边，各类型为 12、36、48 条。

固定教师 $`\theta_j`$、Query episode $`Q_{ji}`$，背景图来自原有 $`\mathop{\mathrm{Bern}}\nolimits(1/2)`$。在同一背景、同一 Query 上将边 $`e`$ 分别置为 0、1。令 $`R_Q`$ 为 TRAIN 候选集合归一化后的组合 NLL，Query 内按组合等权、组合内按图像等权，则


```math
d_{jeik}=R_{Q_{ji}}(\theta_j,G_{-e}^{(k)})-R_{Q_{ji}}(\theta_j,G_{+e}^{(k)}),\qquad
y_{jei}=\frac1K\sum_{k=1}^{K}d_{jeik}.
```


正值表示加入关系降低组合损失。沿用原始 `measurements/fold*.pt` 中的 `y`；新增代码不改变测量定义。


```math
\bar y_{je}=\frac1{n_{je}}\sum_{i=1}^{n_{je}}y_{jei},\qquad
\widehat{\mathrm{SE}}_{je}^{\,2}=\frac1{n_{je}(n_{je}-1)}\sum_i(y_{jei}-\bar y_{je})^2.
```


第二式由独立 episode 抽样下 $`\mathop{\mathrm{Var}}\nolimits(\bar Y)=\mathop{\mathrm{Var}}\nolimits(Y)/n`$ 与无偏样本方差得到，描述固定教师、固定数据采样方案下的均值测量精度。背景差值估计的条件方差单独保存：


```math
\hat v_{jei}=\frac{1}{K(K-1)}\sum_k(d_{jeik}-y_{jei})^2,\qquad
\hat v^{\mathrm{bg}}_{je}=\frac1{n_{je}^{2}}\sum_i\hat v_{jei}.
```


episode 方差已包含背景采样波动，故 `se2` 不再加 `background_se2`。跨教师预测残差另作迁移诊断，不与这两个量合并。拟合使用等权折—边样本，不按 episode 数重复加权。

## 2. 八维模型响应与设计矩阵

沿用已经完成的教师 Inner 响应缓存。每个 Inner 组合等权、组合内所选图像等权。令归一化权重为 $`w_i`$，成对切换关系所得 TRAIN 候选分数为 $`s^-_{ki,c},s^+_{ki,c}`$，对应概率为 $`p^-_{ki,c},p^+_{ki,c}`$：


```math
\delta_{ki}=\log p^+_{ki,y_i}-\log p^-_{ki,y_i},\qquad
\Delta s_{ki,c}=s^+_{ki,c}-s^-_{ki,c},\qquad
\overline{\Delta s}_{ki}=\frac1C\sum_c\Delta s_{ki,c}.
```


对所有支持图像、以及目标属性或目标物体匹配的支持图像两个集合分别重新归一化权重，计算四个量：


```math
\begin{aligned}
m&=\frac1K\sum_{k,i}w_i\delta_{ki},\\[4pt]
s_d&=\left[\frac1K\sum_{k,i}w_i(\delta_{ki}-m)^2\right]^{1/2},\\[4pt]
r_s&=\left[\frac1{KC}\sum_{k,i,c}w_i(\Delta s_{ki,c}-\overline{\Delta s}_{ki})^2\right]^{1/2},\\[4pt]
d_H&=\frac1K\sum_{k,i}w_i\bigl[H(p^+_{ki})-H(p^-_{ki})\bigr].
\end{aligned}
```


两组统计串接得到 $`h_{je}\in\mathbb R^8`$。当前代码直接复用这些八维响应；各响应文件记录对应教师哈希和支持组合。

在拟合折集合 $`\mathcal J`$ 上计算每维中心 $`c`$、总体标准差 $`s`$，标准差下限 $`10^{-6}`$；类型输出尺度 $`a_t`$ 为该类型所有拟合折边均值的总体标准差，下限 $`10^{-12}`$。正式数据三种尺度均高于该下限。


```math
z_{je}=\mathop{\mathrm{onehot}}\nolimits(t(e))\otimes\left[1,(h_{je}-c)\oslash s\right]\in\mathbb R^{27},\qquad
\tilde y_{je}=\bar y_{je}/a_{t(e)}.
```


三种类型各有一组截距和八个响应系数。Q4/Q5 推理使用 Q1–Q3 的中心、标准差、输出尺度。

## 3. 岭回归及闭式求解

对候选正则 $`\alpha_r\in\{0.1,1,10,100\}`$：


```math
\mathcal L(\beta)=\|Z\beta-\tilde y\|_2^2+\alpha_r\|\beta\|_2^2.
```


此处 $`\alpha_r`$ 是岭回归正则，区别于原关系分支的残差增益配置 `alpha`。所有 27 个参数，包括三个截距，均正则化。梯度为


```math
\nabla_\beta\mathcal L=2Z^{\mathsf T}(Z\beta-\tilde y)+2\alpha_r\beta.
```


令梯度为零得到线性方程。因 $`\alpha_r\gt 0`$，对任何非零向量 $`v`$ 有


```math
v^{\mathsf T}(Z^{\mathsf T}Z+\alpha_r I)v=\|Zv\|_2^2+\alpha_r\|v\|_2^2\gt 0.
```


因此唯一解满足


```math
(Z^{\mathsf T}Z+\alpha_r I)\hat\beta=Z^{\mathsf T}\tilde y,\qquad
\hat\mu_{je}=a_{t(e)}z_{je}^{\mathsf T}\hat\beta.
```


实现用 `numpy.linalg.solve` 的 float64 运算，不显式求逆。拟合期每 seed 有 $`3\times96=288`$ 条边均值样本；最终五折重拟合为 480 条。

## 4. 固定预算下的均值效用选图

对类型 $`t`$，沿用旧 robust 最终概率和 $`B_t`$，并沿用温度 $`\tau\gt 0`$。主奖励为 $`\hat\mu`$。定义


```math
\mathcal P=\left\{\pi\in[0,1]^E:\sum_{e:t(e)=t}\pi_e=B_t,\ t=0,1,2\right\},
```


```math
\pi(\hat\mu)=\arg\max_{\pi\in\mathcal P}\left\{\hat\mu^{\mathsf T}\pi-\tau\sum_e\left[\pi_e\log(2\pi_e)+(1-\pi_e)\log(2(1-\pi_e))\right]\right\}.
```


这是类型期望边数约束；每次采样的实际边数可以变化。类型内常数 Drop 概率为 $`B_t/|E_t|`$。

对内点预算 $`0\lt B_t\lt |E_t|`$，引入无符号限制的等式乘子 $`\gamma_t`$。目标对 $`\pi_e`$ 的一阶条件为


```math
\hat\mu_e-\tau\log\frac{\pi_e}{1-\pi_e}-\gamma_{t(e)}=0,\qquad
\pi_e=\sigma\!\left(\frac{\hat\mu_e-\gamma_{t(e)}}{\tau}\right).
```


类型概率和关于 $`\gamma_t`$ 的导数为 $`-\tau^{-1}\sum_{e\in E_t}\pi_e(1-\pi_e)\lt 0`$，且两端极限分别为 $`|E_t|`$ 和 0，因此每类型二分有唯一解。目标 Hessian 对角元为 $`-\tau/[\pi_e(1-\pi_e)]\lt 0`$，该驻点为唯一最优解。预算为 0 或满预算时直接返回全零或全一。

实际求解继承 `solve_budget_equal`，float64 二分 80 次；训练概率保存为 float32，另存精确概率及转换后的预算误差。

## 5. Q4 选模、Q5 审计与最终部署

用真实边均值向量 $`u`$ 定义同一可行域上的目标 $`F_u`$：


```math
F_u(\pi)=u^{\mathsf T}\pi-\tau\sum_e D_{\mathrm{KL}}\bigl(\mathop{\mathrm{Bern}}\nolimits(\pi_e)\|\mathop{\mathrm{Bern}}\nolimits(1/2)\bigr),
```


```math
\mathcal R(u,\hat\mu)=F_u\bigl(\pi(u)\bigr)-F_u\bigl(\pi(\hat\mu)\bigr)\ge0.
```


非负性直接来自 $`\pi(u)`$ 在同一预算可行域上的最优性。固定 $`u`$ 时第一项为常数，最小化遗憾等价于最大化预测策略的真实代理目标。

1. 用 Q1–Q3 的边均值拟合四个正则候选。
2. 用 Q4 的边均值计算遗憾，选择最小者；精确并列时取较大正则。
3. 用选定模型和 Q1–Q3 的统计在 Q5 审计。审计始终使用真实 Q5 标签。
4. 用选定正则在五折重新拟合，同时重新估计输入归一化与输出尺度。
5. 将 reference 响应输入五折模型，输出部署均值与采边概率。

$`F`$ 是原背景分布下单边边际收益的加性代理；多边交互及最终识别效果由六次适配验证。Q5 是延续此前探索性分析的审计数据。

## 6. 打乱效用监督对照

对每折和每类型生成固定置换 $`P_{j,t}`$：


```math
\bar y^{\mathrm{shuf}}_{je}=\bar y_{j,P_{j,t(e)}(e)}.
```


样本数、`se2`、`background_se2` 随标签一起置换，响应输入及图知识保留原顺序。随机生成器种子为 `label_shuffle_seed + experiment_seed + 1009 * fold`；默认 `label_shuffle_seed=20261024`。同一置换用于三折拟合和五折重拟合。

由于置换在折、类型内完成，每类型标签分布与输出尺度相同。打乱组复用真实组的正则、中心、标准差、输出尺度、预算和温度；Q5 对照真实标签，部署输入同一个 reference 响应。

## 7. CPF 适配与识别读出

保留强 CPF 的物体概率和给定物体的属性概率分解：


```math
p_\theta(a,o\mid x,G)=p_\theta(o\mid x,G)\,p_\theta(a\mid x,o,G).
```


强 CPF 主干冻结，更新原关系消息与三元残差投影。每个训练 batch 对每条边独立采样 $`G_e\sim\mathop{\mathrm{Bern}}\nolimits(\pi_e)`$。沿用原组合 NLL 与辅助物体/条件属性损失：


```math
\mathcal L_{\mathrm{adapt}}=\frac{1.5}{1.075}\left[\mathcal L_{\mathrm{pair}}+0.075(\mathcal L_o+\mathcal L_{a\mid o})\right]+\lambda_w\mathcal R_{\mathrm{graph}}.
```


其中 `task_loss` 的候选归一化与原实现相同，$`\mathcal R_{\mathrm{graph}}=d^{-2}\sum_t\|L_tR_t\|_F^2`$。两组从同一 reference 出发并使用相同 seed。

评价时先在各世界的候选集合 $`\mathcal C`$ 内对每张图归一化，再对 16 张图的概率平均：


```math
\bar p_{\mathcal C}(a,o\mid x)=\frac1{16}\sum_{m=1}^{16}\frac{p_\theta(a,o\mid x,G^{(m)})}{\sum_{(a',o')\in\mathcal C}p_\theta(a',o'\mid x,G^{(m)})}.
```


固定 `graph_seed=1701`，epoch 0–3 由 VAL CW native AUC 选权重。TEST 记录 native/exact AUC、VAL bias 下 HM、NLL、ECE。AUC 沿用原实现的 0–100 显示尺度。不同模型不按 TEST 结果重新选正则或 epoch。

## 8. 指标与代码入口

| 内容 | 文件/入口 |
| --- | --- |
| 边均值、标准误与背景方差 | `compact.aggregate_measurements` |
| 27 参数设计与岭回归 | `compact.design` / `fit_ridge` / `predict` |
| 标签打乱与固定映射 | `compact.shuffled_stats` |
| 固定预算与 Q4 遗憾 | `compact.policy` / `objective` / `regret` |
| 拟合、Q5 审计与部署结构 | `compact.fit` / `audit` / `structure` |
| 关系适配与 CPF 权重检查 | `compact.adapt` |
| 来源、续跑和汇总 | `workflow.prepare_seed` / `run_seed` / `report` |

边均值 Spearman 使用平均秩处理并列值；若输入秩方差为零则保存 null。RMSE 使用真实边均值单位。单观测指标为补充，不替代边均值指标；预测残差按 MSE 报告，不解释为已校准的风险方差。

以下核心代码直接从本次交付源码抽取。

### `compact.py:aggregate_measurements`

```python
def aggregate_measurements(raw,edge_count):
    """The episode is the averaging unit. Noise is NOT added again to SE^2."""
    y=raw['y'].double().numpy();edge=raw['edge'].numpy()
    result={k:[] for k in ('target','count','se2','background_se2')}
    for e in range(edge_count):
        keep=edge==e;v=y[keep];n=len(v)
        if n<2:raise ValueError('Need at least two measured episodes for edge %d'%e)
        result['target'].append(v.mean());result['count'].append(n)
        result['se2'].append(v.var(ddof=1)/n)
        result['background_se2'].append(float(raw['noise'][keep].double().sum())/(n*n))
    return {k:np.asarray(v,dtype=np.int64 if k=='count' else np.float64) for k,v in result.items()}
```

### `compact.py:shuffled_stats`

```python
def shuffled_stats(stats,types,seed,fold):
    rng=np.random.RandomState(seed+1009*fold);permutation=np.arange(len(types))
    for t in range(3):
        ids=np.flatnonzero(types==t);permutation[ids]=rng.permutation(ids)
    return {k:v[permutation].copy() for k,v in stats.items()},permutation
```

### `compact.py:design`

```python
def design(x,types,center,spread):
    z=np.column_stack([np.ones(len(x)),(x-center)/spread])
    return np.concatenate([z*(types==t)[:,None] for t in range(3)],axis=1)
```

### `compact.py:fit_ridge`

```python
def fit_ridge(xs,ys,types,alpha,normalization=None):
    x=np.concatenate(xs).astype(np.float64);y=np.concatenate(ys).astype(np.float64)
    t=np.tile(types,len(xs))
    if normalization is None:
        normalization=dict(center=x.mean(0),spread=np.maximum(x.std(0),1e-6),
            scales=np.asarray([max(y[t==k].std(),1e-12) for k in range(3)]))
    center,spread,scales=[np.asarray(normalization[k],dtype=np.float64) for k in ('center','spread','scales')]
    z=design(x,t,center,spread)
    beta=np.linalg.solve(z.T@z+float(alpha)*np.eye(z.shape[1]),z.T@(y/scales[t]))
    return dict(beta=beta,center=center,spread=spread,scales=scales,alpha=float(alpha),
        dimension=int(z.shape[1]),fit_folds=len(xs),intercepts_penalized=True)
```

### `compact.py:predict`

```python
def predict(model,x,types):
    return (design(np.asarray(x,dtype=np.float64),types,model['center'],model['spread'])@model['beta'])*model['scales'][types]
```

### `compact.py:objective`

```python
def objective(pi,target,tau):
    p=np.asarray(pi,dtype=np.float64)
    # 0*log(0)=0, including exact endpoints for empty/full budgets.
    kl=np.zeros_like(p)
    for q in (p,1-p):
        keep=q>0;kl[keep]+=q[keep]*np.log(2*q[keep])
    return float(np.dot(target,p)-tau*kl.sum())
```

### `compact.py:regret`

```python
def regret(pred,target,types,cfg):
    p,_=policy(pred,types,cfg);oracle,_=policy(target,types,cfg)
    value=objective(p.numpy(),target,cfg['tau_absolute'])
    optimum=objective(oracle.numpy(),target,cfg['tau_absolute'])
    difference=optimum-value
    if difference < -1e-10:raise ArithmeticError('Oracle regret is negative: %g'%difference)
    return dict(regret=max(0.,difference),regularized_value=value,oracle_value=optimum,
        raw_utility=float(np.dot(target,p.numpy())),pi=p.tolist())
```

### `compact.py:fit`

```python
def fit(work,all_folds=False):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    ys=[s['target'] for s in stats]
    if all_folds:
        alpha=read_json(work/'utility/selection.json')['selected_alpha'];used=5
    else:
        used=3;trials=[]
        for alpha in cfg['ridge_alphas']:
            model=fit_ridge(xs[:3],ys[:3],types,alpha)
            p4=predict(model,xs[3],types)
            trials.append(dict(alpha=float(alpha),q4=regret(p4,ys[3],types,cfg),
                q4_point=point_metrics(p4,ys[3])))
        selected=min(trials,key=lambda r:(r['q4']['regret'],-r['alpha']))
        alpha=selected['alpha']
        write_json(work/'utility/selection.json',dict(selected_alpha=alpha,objective='Q4 regularized utility regret',
            fit_folds=[1,2,3],selection_fold=4,audit_fold=5,trials=trials,
            tie_break='larger alpha',shuffled_alpha='same as compact'))
    models={};normalization=None;permutations={}
    for variant in VARIANTS:
        if variant=='compact':target=ys
        else:
            shuffled=[shuffled_stats(s,types,cfg['label_shuffle_seed']+cfg['seed'],j+1) for j,s in enumerate(stats)]
            target=[s[0]['target'] for s in shuffled]
            permutations={str(j+1):s[1].tolist() for j,s in enumerate(shuffled)}
        model=fit_ridge(xs[:used],target[:used],types,alpha,normalization)
        if variant=='compact':normalization={k:model[k] for k in ('center','spread','scales')}
        model.update(variant=variant,fit_fold_ids=list(range(1,used+1)),permutations=permutations if variant!='compact' else {})
        dest=work/('utility_all' if all_folds else 'utility')/variant
        save_torch(dest/'model.pt',model);models[variant]=model
    print('RIDGE_FIT_COMPLETE',used,'folds alpha',alpha,flush=True)
    return models
```

### `compact.py:audit`

```python
def audit(work):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    y=stats[4]['target'];raw=load_torch(work/'inputs/q5_observations.pt')
    constant=np.asarray([np.concatenate([s['target'][types==t] for s in stats[:3]]).mean() for t in types])
    base=regret(constant,y,types,cfg)
    for variant in VARIANTS:
        dest=work/'utility'/variant;model=load_torch(dest/'model.pt');p=predict(model,xs[4],types)
        dec=regret(p,y,types,cfg)
        result=dict(variant=variant,alpha=model['alpha'],edge_mean=point_metrics(p,y),
            by_type={TYPE_NAMES[t]:point_metrics(p[types==t],y[types==t]) for t in range(3)},
            decision=dec,type_constant=base,regret_reduction=base['regret']-dec['regret'],
            observation=point_metrics(p[raw['edge']],raw['target']),
            label_mean_se=float(np.sqrt(stats[4]['se2']).mean()),
            transfer_residual_mse=float(np.mean((p-y)**2)),
            label_variance_included_in_reward=False)
        write_json(dest/'audit.json',result)
        save_torch(dest/'audit_predictions.pt',dict(mean=p,target=y,se2=stats[4]['se2'],
            background_se2=stats[4]['background_se2'],count=stats[4]['count'],types=types))
        print('UTILITY_AUDIT',variant,'alpha',model['alpha'],'rho',result['edge_mean']['spearman'],
            'regret',dec['regret'],flush=True)
```

### `compact.py:structure`

```python
def structure(work):
    work=Path(work);cfg=read_json(work/'recipe.json');types,stats,xs=load_inputs(work)
    x=load_torch(work/'responses/reference.pt')['features'].double().numpy()
    for variant in VARIANTS:
        model=load_torch(work/'utility_all'/variant/'model.pt');mean=predict(model,x,types)
        pi,gamma=policy(mean,types,cfg)
        dest=work/'structures'/variant
        save_torch(dest/'structure.pt',dict(mean=torch.from_numpy(mean),pi=pi.float(),pi_float64=pi,
            types=torch.from_numpy(types),multipliers=gamma,alpha=model['alpha']))
        actual={str(t):float(pi.float()[torch.from_numpy(types)==t].double().sum()) for t in range(3)}
        write_json(dest/'structure.json',dict(variant=variant,mean=mean.tolist(),pi=pi.tolist(),
            alpha=model['alpha'],expected_edges=actual,budgets=cfg['fixed_type_budgets'],
            tau=cfg['tau_absolute'],multipliers=gamma,reward='predicted edge-mean utility',
            float32_budget_max_error=max(abs(actual[k]-cfg['fixed_type_budgets'][k]) for k in actual)))
```

### `compact.py:adapt`

```python
def adapt(args,variant):
    from .engine import setup,train_model,new_model,evaluate_model
    args.variant=variant
    cfg,manifest,text,bank,folds,work=setup(args)
    state_graph=load_torch(work/'structures'/variant/'structure.pt');pi=state_graph['pi']
    cfg['compact_alpha']=state_graph['alpha']
    out=work/'runs'/variant;save_torch(out/'bank.pt',bank)
    best=train_model(args,cfg,text,bank,manifest,manifest['pairs']['train'],out,pi,work/'reference/last.pt',True)
    state=load_torch(best);model=new_model(text,bank,state['cfg'],args.device);model.load_state_dict(state['model'])
    reference=load_torch(work/'reference/last.pt')['model']
    frozen=all(torch.equal(v,reference[k]) for k,v in state['model'].items() if k.startswith('backbone.'))
    if not frozen:raise AssertionError('CPF backbone changed during relation adaptation')
    val=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'val',save=out)
    test=evaluate_model(model,args.cache,state['cfg'],manifest,state['pi'],'test',val,save=out)
    p=state['pi'].clamp(1e-7,1-1e-7)
    write_json(out/'metrics.json',dict(variant=variant,seed=cfg['seed'],epoch=state['epoch'],backbone_frozen=frozen,
        structure=dict(expected_edges={str(t):float(state['pi'][bank['types']==t].double().sum()) for t in range(3)},
            entropy=float(-(p*p.log()+(1-p)*(1-p).log()).sum())),val=val,test=test))
    print('ADAPT_COMPLETE',variant,'epoch',state['epoch'],'TEST_CW_AUC',test['cw']['AUC_native'],
        'TEST_OW_AUC',test['ow']['AUC_native'],flush=True)
```

### `graph.py:solve_budget_equal`

```python
def solve_budget_equal(reward, types, tau, budgets):
    """Strictly concave entropy objective with exact per-type expected counts."""
    if tau <= 0 or not torch.isfinite(reward).all():
        raise ValueError('Positive temperature and finite rewards required')
    r=reward.double();pi=torch.zeros_like(r);multipliers={}
    for ty in types.unique().tolist():
        mask=types==ty;n=int(mask.sum());b=float(budgets[str(ty)])
        if not 0<=b<=n:raise ValueError('Invalid expected edge count')
        if b==0 or b==n:
            pi[mask]=b/n;multipliers[str(ty)]=None;continue
        low=float(r[mask].min())-100*tau
        high=float(r[mask].max())+100*tau
        for _ in range(80):
            mid=(low+high)/2
            if float(torch.sigmoid((r[mask]-mid)/tau).sum())>b:low=mid
            else:high=mid
        gamma=(low+high)/2
        pi[mask]=torch.sigmoid((r[mask]-gamma)/tau)
        multipliers[str(ty)]=gamma
    return pi.to(reward.dtype),multipliers
```
