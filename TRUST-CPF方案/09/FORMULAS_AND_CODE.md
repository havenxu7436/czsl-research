# TRUST-CPF 第二批：阶段匹配效用监督，10 epoch 公式与核心实现

## 1. 本批要回答的问题

第一批中，联合 Absolute 预测器提升了 Q5 效用决策，但正式冻结识别尚未稳定超过 Uniform；同一模型的重复响应也有明显波动。本批同时比较教师阶段匹配与响应平均，正式识别器固定为第一批三个 Uniform-best。

本批保留 TRUST 的主链：**关系 on/off 干预 → 效用监督 → 模型响应条件化预测 → 类型预算下的 Bernoulli 分配**。CLIP 特征、CPF 强基底、关系库、正式识别权重与评估器沿用第一批。

| 主组 | 效用预测器 | 输入响应 | 正式识别权重 |
|---|---|---|---|
| Uniform | 常数概率 0.5 | 无 | 第一批 Uniform-best |
| Old-Single | 第一批 pooled Absolute | 单次当前响应 | 同上 |
| Matched-Single | 本批阶段匹配 pooled Absolute | 单次当前响应 | 同上 |
| Old-Average | 第一批 pooled Absolute | 两次当前响应的平均 | 同上 |
| Matched-Average | 本批阶段匹配 pooled Absolute | 两次当前响应的平均 | 同上 |

四个预测组各使用五次类型内置换控制；Matched 的两种响应输入另配 reference 状态响应。一个队列完成全部比较。

## 2. 教师新增 10 epoch：训练、独立选模与恢复

设训练 seed 为 $`s\in\{3351,3352,3353\}`$，折为 $`j\in\{1,\ldots,5\}`$。第一批折内教师的末轮参数为 $`\theta^{s,j}_0`$。对每个教师新增十轮：


```math
\theta^{s,j}_{k+1}
=\mathop{\mathrm{TrainEpoch}}\nolimits\!\left(
\theta^{s,j}_k;\mathcal D^{s,j}_{\mathrm{Inner}},
b_e\sim\mathop{\mathrm{Bernoulli}}\nolimits(1/2)
\right),\qquad k=0,\ldots,9.
```


CLIP 特征和教师的 CPF backbone 固定，优化其余可训练参数。更新使用原 CPF 任务损失：$`q_\theta(a,o\mid x,b)=p_\theta(o\mid x,b)p_\theta(a\mid o,x,b)`$。对于本折训练候选集合 $`\mathcal C_j`$：


```math
\begin{aligned}
\mathcal L_{\mathrm{pair}}
&=-\frac1n\sum_i \log
 \frac{q_\theta(a_i,o_i\mid x_i,b)}
 {\sum_{(a,o)\in\mathcal C_j}q_\theta(a,o\mid x_i,b)},\\[4pt]
\mathcal L_o&=-\frac1n\sum_i\log p_\theta(o_i\mid x_i,b),\\[4pt]
\mathcal L_{a\mid o}
&=-\frac1n\sum_i\log p_\theta(a_i\mid o_i,x_i,b),\\[4pt]
\mathcal L_{\mathrm{train}}
&=\frac{1.5}{1.075}
 [\mathcal L_{\mathrm{pair}}+0.075(\mathcal L_o+\mathcal L_{a\mid o})]
 +10^{-3}\mathcal R(\theta).
\end{aligned}
```


其中 $`\mathcal R`$ 为原模型的 graph regularizer。优化器 AdamW：初始学习率 $`3\times10^{-4}`$，weight decay $`10^{-4}`$，batch size 64，梯度范数上限 1。共 $`T=10M`$ 个更新，$`M`$ 是每轮 batch 数，零起点更新编号 $`t`$：


```math
\eta_t=\eta_0
 \min\!\left(\frac{t+1}{M},1\right)
 \frac{1+\cos\!\left(\pi\frac{\max(0,t-M)}{\max(1,T-M)}\right)}{2}.
```


STOP 集从原 seen/unseen 风险池尚未使用的图像中选出，每组各 256 张；排除教师训练、效用测量和 Q5 评分图像。随机种子为 $`9401+j`$，三 seed 使用相同图像索引。本地实际源数据检查：最少仍有 779 张未用 unseen 图像，满足采样数量。

STOP 上用固定 16 张 Uniform 图的概率平均，以全部训练组合为候选，按 Inner / Query 区分 seen / unseen。选择：


```math
k^\star=\min\,\operatorname*{arg\,max}_{k\in\{0,\ldots,10\}}
 \mathop{\mathrm{AUC}}\nolimits_{\mathrm{native}}
 (\theta^{s,j}_k;\mathcal D^{j}_{\mathrm{STOP}}).
```


全队列完成 150 个教师 epoch，保存 165 个曲线点。即使 $`k^\star=0`$，仍完整训练十轮。STOP 使用 Query 组合的独立图像及标签做选模；Q5 分析因此采用“同组合、独立图像的决策评估”口径。

`last.pt` 同时保存当前权重、优化器、随机状态、历史与当前最佳权重；恢复后可重建 `best.pt`。训练、测量和评分分别有完成记录。这样一次中断不会要求从头训练。

## 3. 在新状态重新测量 U-NLL 效用

所有新响应与新监督均使用 $`\theta_{k^\star}^{s,j}`$。设测量 Query 图像集合为 $`\mathcal U_j`$，候选集合为全部训练组合 $`\mathcal C_{\mathrm{train}}`$：


```math
\ell_\theta(y_i\mid x_i,b)
=\log q_\theta(y_i\mid x_i,b)
-\log\sum_{c\in\mathcal C_{\mathrm{train}}}q_\theta(c\mid x_i,b),
\qquad
L_U(\theta,b)=-\frac1{|\mathcal U_j|}
\sum_{i\in\mathcal U_j}\ell_\theta(y_i\mid x_i,b).
```


对关系 $`e`$，固定背景的其余关系，分别令 $`b_e=0`$、$`b_e=1`$：


```math
\delta_{e,r}
=L_U(\theta,b^{(r)}_{-e},0)-L_U(\theta,b^{(r)}_{-e},1)
=\frac1{|\mathcal U_j|}\sum_{i\in\mathcal U_j}
 [\ell_\theta(y_i\mid x_i,b^{(r)}_{-e},1)
  -\ell_\theta(y_i\mid x_i,b^{(r)}_{-e},0)].
```


```math
y_e=\frac1K\sum_{r=1}^K\delta_{e,r},
\qquad
\widehat{\mathop{\mathrm{SE}}\nolimits}^2(y_e)
=\frac1{K(K-1)}\sum_{r=1}^K(\delta_{e,r}-y_e)^2.
```


正效用表示打开关系会降低 unseen NLL。对 $`B_e\sim\mathrm{Bernoulli}(\pi_e)`$：


```math
-\frac{\partial}{\partial\pi_e}\mathbb E_{B\sim\pi}L_U(\theta,B)
=\mathbb E_{B_{-e}}\!\left[
L_U(\theta,B_{-e},0)-L_U(\theta,B_{-e},1)
\right].
```


所以该标签是 Uniform 分布处的期望风险负梯度估计。它解释局部关系贡献；最终策略效果由实际多图识别测量得到。

Q1–Q4 每折 8 个背景，Q5 16 个。背景随机种子为 $`9301+1000j+r`$，其中 $`r=0,\ldots,K-1`$。保留旧测量图像与背景，便于新旧阶段配对分析；on/off 分数重新计算。每个 block 保存真实类别 log probability、组内 margin、跨组阈值和正确门控值，可重算监督。仅 U-NLL 用于本批拟合。

## 4. 模型响应：单次与两次平均

每条关系使用 8 维响应。Inner 每个组合随机选至多两张训练图像，组合等权、组合内图像等权；对 4 个 Uniform 背景计算 on/off。分别在“全部支持图像”和“该关系目标原语对应图像”上提取四个统计：


```math
h_e=
[\mathbb E\Delta\ell,\mathop{\mathrm{SD}}\nolimits(\Delta\ell),
 \sqrt{\mathbb E\,\mathop{\mathrm{mean}}\nolimits_c(\Delta z_c-\overline{\Delta z})^2},
 \mathbb E(H(p^+)-H(p^-))]_{\mathrm{all}}
\;\oplus\;
[\cdots]_{\mathrm{target}}\in\mathbb R^8.
```


其中 $`z_c`$ 是候选组合的 joint log probability，$`\Delta z=z^+-z^-`$，$`\Delta\ell=\log p^+(y)-\log p^-(y)`$。标准差使用与支持样本权重一致的总体方差；条件于 target 后重新归一化权重。

第一份响应沿用旧教师的支持图像、权重和随机种子。第二份使用原种子加 100000，重新选择 Inner 支持图像和背景，形成独立随机抽样：


```math
\bar h_e=\frac{h_e^{(1)}+h_e^{(2)}}2.
```


本批预测器始终用各教师第一份响应训练。平均响应只改变部署输入，不另拟合一套模型；正式模型的两份响应直接复用第一批记录。

## 5. 九教师联合 Absolute 拟合

拟合教师集合为三个 seed 的 Q1、Q2、Q3，共 $`J=9`$ 个。输入标准化统计只由这些教师计算。对特征维 $`d`$：


```math
c_d=\mathop{\mathrm{mean}}\nolimits_{j,e}h_{j,e,d},
\quad
s_d=\max(\mathop{\mathrm{SD}}\nolimits_{j,e}h_{j,e,d},10^{-6}),
\quad
a_t=\max(\mathop{\mathrm{SD}}\nolimits_{j,e:t(e)=t}y_{j,e},10^{-12}).
```


三种关系类型为 OI、OE、AE。构建按类型分块的 27 维向量：


```math
z_{j,e}=\mathop{\mathrm{onehot}}\nolimits(t(e))
 \otimes[1,(h_{j,e}-c)/s],
\qquad
\widetilde y_{j,e}=y_{j,e}/a_{t(e)}.
```


Absolute 目标保留类型截距与原始标签值，截距也参与正则。每教师等权：


```math
\begin{aligned}
\widehat\beta_\lambda
&=\arg\min_\beta
 \frac1J\sum_{j=1}^{J}\|Z_j\beta-\widetilde y_j\|_2^2
 +\lambda\|\beta\|_2^2,\\[4pt]
\widehat\beta_\lambda
&=(Z^\top Z+J\lambda I)^{-1}Z^\top\widetilde y,\qquad
\widehat\mu_e=a_{t(e)}z_e^\top\widehat\beta_\lambda.
\end{aligned}
```


代码通过线性方程求解，实际岭系数为 $`\alpha=9\lambda`$。候选 $`\lambda\in\{0.1/3,1/3,10/3,100/3\}`$。

Q4 每个 seed、每种类型对残差做类型内去均值。令 $`P_t=I-\mathbf1\mathbf1^\top/n_t`$：


```math
E_4(\lambda)=\frac19\sum_{s=1}^3\sum_{t=1}^3
 \frac{\|P_t(\widehat\mu_{s,4,t}-y_{s,4,t})/a_t\|_2^2}{n_t},
\qquad
\lambda^\star=\arg\min_\lambda E_4(\lambda).
```


并列时选较大 $`\lambda`$。Q4 负责选正则，最终仍只用九个 Q1–Q3 教师拟合；Q5 不参与拟合和正则选择。旧预测器直接导入第一批 pooled Absolute。

固定模型是输入的仿射函数，因此：


```math
\widehat\mu_e(\bar h)
=\frac{\widehat\mu_e(h^{(1)})+\widehat\mu_e(h^{(2)})}{2}.
```


实现检查这一恒等式。先平均特征或效用，再进行类型标准化和预算求解；不平均两份已生成的概率。

## 6. 类型预算概率与匹配置换

对每种类型标准化预测效用；若类型内效用为常数，则该类型 $`r_e=0`$：


```math
r_e=\frac{\widehat\mu_e-\mathop{\mathrm{mean}}\nolimits_{f:t(f)=t(e)}\widehat\mu_f}
 {\mathop{\mathrm{SD}}\nolimits_{f:t(f)=t(e)}\widehat\mu_f}.
```


固定 $`\kappa=1`$，求解等式预算下的严格凹目标：


```math
\begin{aligned}
\max_{\pi}\quad&
 \kappa\sum_e r_e\pi_e
 -\sum_e\left[\pi_e\log(2\pi_e)
 +(1-\pi_e)\log(2(1-\pi_e))\right],\\[4pt]
\text{s.t.}\quad&
 \sum_{e:t(e)=t}\pi_e=n_t/2,\qquad 0\lt \pi_e\lt 1.
\end{aligned}
```


拉格朗日一阶条件给出：


```math
\log\frac{\pi_e}{1-\pi_e}=\kappa r_e-\gamma_{t(e)},
\qquad
\pi_e=\sigma(\kappa r_e-\gamma_{t(e)}).
```


预算和关于 $`\gamma_t`$ 严格递减；二分求得唯一解。96 条关系的三类数量为 12、36、48，期望预算为 6、18、24。Bernoulli 采样固定的是期望数量，每张图的实际关系数可波动。

五次置换仅在同类型中重排概率，保留预算、概率分布和熵，改变关系与概率的对应。随机种子为 7101～7105。reference 响应控制固定新预测器与正式识别器，只替换输入响应的模型状态。

## 7. 概率平均、评估与成对比较

正式识别使用第一批 Uniform-best。对每个图 bank 使用公共均匀随机数，让不同策略按自己的概率阈值生成 16 张图：


```math
b^{(m)}_e=\mathbf1[u^{(m)}_e\lt \pi_e],
\qquad
\bar p_{\mathcal C}(y\mid x)
=\frac1{16}\sum_{m=1}^{16}
 \frac{q_\theta(y\mid x,b^{(m)})}
 {\sum_{c\in\mathcal C}q_\theta(c\mid x,b^{(m)})}.
```


每张图先在当前世界的候选集合上归一化，再平均概率，代码使用 logsumexp。正式 bank 为 1701、2701、3701；Q5 为 8201～8208。Q5 固定本批 fold5/best.pt，并在独立 evaluate 图像上比较 31 种策略，其中 measured 是直接用测量集关系效用分配的参考组。

CW/OW 分别计算 native 与 exact AUC。bias 和温度在 VAL 选择，TEST 复用。AUC 乘 100；例如增量 0.25 表示 0.25 个百分点。

| 指标 | 保存位置 | 用途 |
|---|---|---|
| VAL/TEST CW、OW 的 native/exact AUC | summary.csv | 正式识别主结果 |
| VAL bias 下 HM、seen、unseen | summary.csv | 已见与未见平衡 |
| raw/calibrated NLL、ECE | summary.csv | 概率质量 |
| Matched−Old、Average−Single | paired.csv | 分离阶段匹配和响应平均贡献 |
| 真实策略−Uniform、真实策略−五置换均值 | paired.csv、q5_paired.csv | 关系对应的实际收益 |
| Q5 实际 AUC、U-NLL、单图与混合图增量 | q5_decisions.csv | 独立图像决策能力 |
| Q5 预测 Spearman、RMSE、类型中心化 R² | utility.csv | 效用监督迁移 |
| 新旧教师标签相关性、尺度变化 | teacher_change.csv | 训练阶段如何改变效用 |
| 两份响应的预测相关性、概率 MSE | response_repeats.csv | 响应稳定性 |
| 教师 loss、STOP AUC、选中轮次 | teacher_curves.csv | 训练轨迹 |
| 概率离散度、KL、节点覆盖、原型变化 | deployment.csv | 图分配行为 |
| 各阶段时间、峰值显存 | costs.csv | 计算成本 |

对某组与对照的差值，先在同 seed、同图 bank 内相减，再平均 bank：


```math
d_s=\frac1B\sum_{b=1}^B
 [A_{s,b}^{\mathrm{method}}-A_{s,b}^{\mathrm{control}}],
\qquad
\bar d=\frac13\sum_s d_s,\quad
\mathop{\mathrm{SD}}\nolimits(d)=
 \sqrt{\frac12\sum_s(d_s-\bar d)^2}.
```


主判断同时看 **Matched 相对 Old 的阶段增量、相对 Uniform 和置换的识别增量**；响应平均使用同一预测器的配对比较。三个 seed 共享效用预测器，属于共享预测器下的三个识别重复；图 bank 和置换不是额外训练 seed。

完整产物预期：26 个完成阶段、15 个十轮教师、165 个曲线点、972 行正式识别指标、744 行 Q5 指标、486 份 VAL/TEST 预测 NPZ。

## 8. 核心实现（从交付源文件直接提取）

下列代码与压缩包源码一致。完整程序、依赖与调度在同目录中；实际运行按 SERVER_COMMANDS.md。

### 折内 STOP 集划分

来源：`trustcpf/matched_train.py`。

```python
def add_stopping_set(split,cfg):
    s=copy.deepcopy(split)
    used=set(s['train_ids'])
    for partition in ('measure','evaluate'):
        for values in s[partition].values():used.update(values)
    rng=np.random.RandomState(cfg['stop_seed']+s['fold']);stop={};remaining={}
    for group in ('seen','unseen'):
        pool=sorted(set(s[group+'_pool'])-used);remaining[group]=len(pool)
        if len(pool)<cfg['stop_images_per_group']:
            raise ValueError('Fold %d has only %d unused %s stopping images'%(s['fold'],len(pool),group))
        stop[group]=rng.choice(pool,cfg['stop_images_per_group'],replace=False).tolist()
    assert not set(stop['seen'])&set(stop['unseen'])
    s.update(stop=stop,stop_seed=cfg['stop_seed']+s['fold'],stop_remaining=remaining)
    return s
```

### 十轮训练与 epoch 级续训

来源：`trustcpf/matched_train.py`。

```python
def train_teacher(args,fold,until_epoch=None):
    """last.pt is the single authoritative resume state, including its best model."""
    work,cfg,manifest,text,bank=align.context(args);cfg=dict(cfg,_cold_backbone=True,variant='uniform')
    if cfg['stage_epochs']!=10:raise ValueError('This release uses 10 additional teacher epochs')
    split=read_json(work/('splits/fold%d.json'%fold));initial=work/('old/teachers/%d/last.pt'%fold)
    out=work/('teachers/%d'%fold);out.mkdir(parents=True,exist_ok=True)
    model=align.restore(text,bank,cfg,args.device,initial);model.backbone.requires_grad_(False)
    original=load_torch(initial)['model'];opt=engine.optimizer(model,cfg)
    prep=ct.prepared(args,cfg,manifest,model,split,'stop')
    candidates=torch.tensor(pair_ids(manifest['labels'],split['inner_pairs']),device=args.device)
    loader=engine.batches(args.cache,'train',cfg,split['train_ids'],True)
    pi=torch.full((len(bank['types']),),.5,device=args.device)
    if (out/'last.pt').exists():
        st=load_torch(out/'last.pt')
        assert st['initial_sha256']==sha(initial) and st['stop_ids']==split['stop']
        assert st['train_ids']==split['train_ids'] and st['train_pairs']==split['inner_pairs']
        model.load_state_dict(st['model']);opt.load_state_dict(st['optimizer'])
        torch.set_rng_state(st['torch_rng']);np.random.set_state(st['numpy_rng'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(st['cuda_rng'])
    else:
        metrics=stopping_metrics(model,prep,cfg,args.device)
        st=dict(epoch=0,model=copy.deepcopy(model.state_dict()),optimizer=opt.state_dict(),pi=pi.cpu(),
            best_model=copy.deepcopy(model.state_dict()),best_epoch=0,best_metrics=metrics,
            history=[dict(epoch=0,stop=metrics)],cfg=cfg,train_pairs=split['inner_pairs'],train_ids=split['train_ids'],
            stop_ids=split['stop'],initial_sha256=sha(initial),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st)
    export_checkpoint(out,st)
    total=max(1,10*len(loader));warmup=max(1,len(loader));limit=10 if until_epoch is None else min(10,until_epoch)
    for epoch in range(st['epoch'],limit):
        model.train();losses=[];tic=time.perf_counter()
        for bi,data in enumerate(loader):
            step=epoch*len(loader)+bi
            factor=min((step+1)/warmup,1.)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,total-warmup)))
            for g in opt.param_groups:g['lr']=cfg['graph_lr']*factor
            g,fa,fo,a,o=[x.to(args.device) for x in data];mask=torch.bernoulli(pi)
            joint,lo,lc=model(g,fa,fo,mask)
            loss=task_loss(joint,lo,lc,a,o,candidates)+cfg['lambda_w']*model.graph.regularizer()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group['params']],cfg['gradient_clip'])
            opt.step();losses.append(float(loss.detach()))
        model.eval();metrics=stopping_metrics(model,prep,cfg,args.device)
        if metrics['AUC_native']>st['best_metrics']['AUC_native']:
            st.update(best_model=copy.deepcopy(model.state_dict()),best_epoch=epoch+1,best_metrics=metrics)
        st['history'].append(dict(epoch=epoch+1,loss=float(np.mean(losses)),stop=metrics,
            learning_rate=opt.param_groups[0]['lr'],seconds=time.perf_counter()-tic))
        st.update(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),
            numpy_rng=np.random.get_state(),cuda_rng=torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else None)
        save_torch(out/'last.pt',st);export_checkpoint(out,st)
        print('TEACHER_EPOCH',args.seed,fold,epoch+1,'loss',st['history'][-1]['loss'],
            'stop_auc',metrics['AUC_native'],'best_epoch',st['best_epoch'],flush=True)
    if st['epoch']==10:
        chosen=max(st['history'],key=lambda h:(h['stop']['AUC_native'],-h['epoch']))['epoch']
        assert chosen==st['best_epoch']
        assert all(torch.equal(v,st['model'][k].detach().cpu()) and torch.equal(v,st['best_model'][k].detach().cpu()) for k,v in original.items() if k.startswith('backbone.'))
        write_json(out/'contract.json',dict(epochs_completed=10,selected_epoch=chosen,initial_sha256=sha(initial),
            best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),backbone_frozen=True,
            stop_images={k:len(v) for k,v in split['stop'].items()},selection='stop native AUC, earliest maximum',
            main_val_test_used=False))
    return out/'best.pt'
```

### 新状态的 on/off U-NLL 标签

来源：`trustcpf/matched_train.py`。

```python
@torch.no_grad()
def measure_unll(args,fold,model):
    """Save on/off sufficient statistics; supervise only unseen NLL."""
    from . import competition as cmp
    work,cfg,manifest,text,bank=align.context(args);split=read_json(work/('splits/fold%d.json'%fold))
    cp=work/('teachers/%d/best.pt'%fold);cph=sha(cp)
    base,candidates,truth,seen=ct.prepared(args,cfg,manifest,model,split,'measure');sm=seen[truth]
    k=cfg['q5_measurement_blocks'] if fold==5 else cfg['measurement_blocks'];values=[]
    for block in range(k):
        path=work/('measurements/fold%d/block%02d.pt'%(fold,block))
        if path.exists():
            saved=load_torch(path);assert saved['checkpoint_sha256']==cph
        else:
            gen=torch.Generator().manual_seed(cfg['measurement_seed']+1000*fold+block)
            mask=(torch.rand(len(bank['types']),generator=gen)<.5).float().to(args.device);off=[];on=[]
            for edge in range(len(mask)):
                b=mask.clone();b[edge]=0;off.append(cmp.sufficient(ct.score(model,base,b,candidates,cfg['eval_batch_size']),truth,seen))
                b[edge]=1;on.append(cmp.sufficient(ct.score(model,base,b,candidates,cfg['eval_batch_size']),truth,seen))
            off,on=torch.stack(off),torch.stack(on)
            effects=(on[:,~sm,0]-off[:,~sm,0]).mean(1)[:,None]
            saved=dict(off=off.cpu(),on=on.cpu(),effects=effects.cpu(),mask=mask.cpu(),truth=truth.cpu(),
                seen_candidates=seen.cpu(),sample_seen=sm.cpu(),targets=['unll'],checkpoint_sha256=cph)
            save_torch(path,saved)
        values.append(saved['effects'].double().numpy())
        print('MATCHED_UTILITY',args.seed,fold,block+1,'/',k,flush=True)
    x=np.stack(values)
    save_torch(work/('measurements/fold%d.pt'%fold),dict(target=x.mean(0),se2=x.var(0,ddof=1)/k,blocks=x,
        targets=['unll'],teacher_sha256=cph,split=split))
```

### 27 参数 Absolute 正规方程

来源：`trustcpf/compact.py`。

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

### 九教师归一化系数与原始坐标

来源：`trustcpf/state.py`。

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

### Q4 选参、共享拟合与 Q5 效用评估

来源：`trustcpf/matched.py`。

```python
def fit_global(root,cfg):
    root=Path(root);data={};changes=[]
    for s in SEEDS:
        w=root/('seed%d'%s);types=load_torch(w/'inputs/bank.pt')['types'].numpy();xs=[];ys=[]
        for j in range(1,6):
            x=load_torch(w/('responses/fold%d.pt'%j))['features'].double().numpy()
            y=load_torch(w/('measurements/fold%d.pt'%j))['target'][:,0];xs.append(x);ys.append(y)
            old=load_torch(w/('old/measurements/fold%d.pt'%j));oy=old['target'][:,list(old['targets']).index('unll')]
            epoch=read_json(w/('teachers/%d/contract.json'%j))['selected_epoch']
            for t in range(3):
                ix=types==t
                changes.append(dict(seed=s,fold=j,type=t,selected_epoch=epoch,
                    old_mean=float(oy[ix].mean()),new_mean=float(y[ix].mean()),
                    old_sd=float(oy[ix].std()),new_sd=float(y[ix].std()),**compact.point_metrics(y[ix],oy[ix])))
        data[s]=(xs,ys,types)
    types=data[SEEDS[0]][2]
    if not all(np.array_equal(data[s][2],types) for s in SEEDS):raise ValueError('Teacher edge types/order differ')
    xs=[x for s in SEEDS for x in data[s][0][:3]];ys=[y for s in SEEDS for y in data[s][1][:3]]
    trials=[]
    for lam in cfg['teacher_lambdas']:
        m=state.fit_one(xs,ys,types,lam);errs=[]
        for s in SEEDS:
            x,y,_=data[s];p=compact.predict(m,x[3],types)
            errs.append(float(np.mean([np.var((p[types==t]-y[3][types==t])/m['scales'][t]) for t in range(3)])))
        trials.append(dict(lam=lam,q4_error=float(np.mean(errs)),by_seed=errs))
    selected=min(trials,key=lambda r:(r['q4_error'],-r['lam']))
    m=state.fit_one(xs,ys,types,selected['lam'])
    m.update(teacher_ids=[(s,j) for s in SEEDS for j in (1,2,3)],target='unll',stage_matched=True,
        fit_input='primary response only; averaging is a readout comparison')
    save_torch(root/'utility/matched.pt',m)
    write_json(root/'utility/selection.json',dict(selected_lambda=selected['lam'],alpha=m['alpha'],
        trials=trials,rule='mean three Q4 centered type-normalized MSE, ties larger lambda',fit_folds=[1,2,3],audit_fold=5))
    audit=[]
    for s in SEEDS:
        w=root/('seed%d'%s);x0=data[s][0][4];x1=load_torch(w/'responses/fold5_repeat.pt')['features'].double().numpy();y=data[s][1][4]
        for name in METHODS:
            model=load_torch(root/('utility/old.pt' if name.startswith('old') else 'utility/matched.pt'))
            x=(x0+x1)/2 if name.endswith('average') else x0;p=compact.predict(model,x,types);pi,_=align.policy(p,types,1.)
            audit.append(dict(seed=s,method=name,all=compact.point_metrics(p,y),centered=r5.centered_metrics(p,y,types),
                by_type={str(t):compact.point_metrics(p[types==t],y[types==t]) for t in range(3)},linear_gain=float(y@(pi.numpy()-.5))))
    write_json(root/'utility/audit.json',audit);write_json(root/'utility/teacher_change.json',changes)
    print('MATCHED_FIT_COMPLETE','lambda',selected['lam'],'alpha',m['alpha'],flush=True)
```

### 单次、平均响应与置换策略

来源：`trustcpf/matched.py`。

```python
def make_policies(args):
    work,cfg,manifest,text,bank=align.context(args);types=bank['types'].numpy();root=work.parent
    models={key:load_torch(root/('utility/'+key+'.pt')) for key in ('old','matched')}
    def inputs(name):
        return [load_torch(work/('responses/'+name+suffix+'.pt'))['features'].double().numpy() for suffix in ('','_repeat')]
    def means(name):
        x0,x1=inputs(name);out={}
        for method in METHODS:
            model=models[method.split('_')[0]];h=(x0+x1)/2 if method.endswith('average') else x0
            out[method]=compact.predict(model,h,types)
            if method.endswith('average'):
                np.testing.assert_allclose(out[method],(compact.predict(model,x0,types)+compact.predict(model,x1,types))/2,atol=1e-11)
        return out
    formal=state.family(means('uniform_best'),types,cfg);ref=means('reference')
    for key,method in [('matched_cross','matched_single'),('matched_average_cross','matched_average')]:
        pi,info=align.policy(ref[method],types,1.)
        formal[key]=dict(pi=pi,mu=ref[method],method=key,permutation_seed=None,response_state='reference',scale=info)
    qmeans=means('fold5');qmeans['measured']=load_torch(work/'measurements/fold5.pt')['target'][:,0]
    q5=state.family(qmeans,types,cfg)
    assert set(formal)==set(formal_keys(cfg)) and set(q5)==set(q5_keys(cfg))
    save_torch(work/'policies.pt',dict(states={'uniform_best':formal},q5=q5,types=types,strength=1.,
        expected_edges=[int((types==t).sum())/2 for t in range(3)]))
    repeats=[]
    for name in ('fold5','reference','uniform_best'):
        x0,x1=inputs(name)
        for modelname,model in models.items():
            p0,p1=[compact.predict(model,x,types) for x in (x0,x1)];a,_=align.policy(p0,types,1.);b,_=align.policy(p1,types,1.)
            for t in range(3):
                ix=types==t
                repeats.append(dict(seed=args.seed,state=name,model=modelname,type=t,
                    rho=compact.point_metrics(p0[ix],p1[ix])['spearman'],prediction_rmse=float(np.sqrt(np.mean((p1[ix]-p0[ix])**2))),
                    probability_mse=float((a[ix]-b[ix]).square().mean())))
    write_json(work/'responses/repeat_diagnostics.json',repeats)
```

### 类型预算求解

来源：`trustcpf/align.py`。

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

## 9. 交付验证与本批范围

`tests/test_matched.py` 使用小型完整数据建立旧教师和第一批 State 结果，再运行本批 15 教师 ×10 epoch、共享拟合、Q5 与正式识别。检查：

- epoch 3 中断、恢复到 10 与直接训练的权重逐项相同；
- STOP 与训练、测量、Q5 图像互斥，教师 backbone 保持相同；
- 新标签可由 on/off 保存统计精确重算；
- 旧预测器对照在同正式权重上复现第一批评分；
- 所有阶段完成与续跑复用，源结果文件哈希保持相同；
- 结果打包后解压，逐文件哈希一致。

真实第一批文件已完成本地输入检查：15 个教师都是原 epoch10，三份正式选中权重分别为 epoch2、2、6，源分类器/评估器一致，STOP 采样数量充足。服务器 setup 再检查完整 CLIP 缓存及输入文件。

本地小数据验证覆盖实现与流程；UT-Zappos 的正式结果由服务器十轮队列产生。后续分析优先比较阶段匹配和平均响应的实际识别增量，再用监督与响应指标解释来源。
