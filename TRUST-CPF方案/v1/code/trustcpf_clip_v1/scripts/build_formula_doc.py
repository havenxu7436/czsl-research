"""Generate the formula/code companion from the released Python source, verbatim."""
import ast
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def excerpt(file,symbol):
    p=ROOT/file;src=p.read_text(encoding='utf-8');tree=ast.parse(src)
    nodes=tree.body
    for name in symbol.split('.'):
        node=next(n for n in nodes if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        nodes=getattr(node,'body',[])
    start=min([node.lineno]+[n.lineno for n in getattr(node,'decorator_list',[])])
    return '\n'.join(src.splitlines()[start-1:node.end_lineno]),start


def main():
    sections=[
    ('0. 符号、张量与单位化',r'''
图文维度$d=768$，属性$A=16$，对象$O=12$，关系$E=96$，三线性秩$r=64$，消息秩$q=16$。
图像缓存为$g[B,d]$、$f^a[B,A,d]$、$f^o[B,O,d]$；联合概率轴为`[batch, object, attribute]`。

$$\mathcal N(v)=v/\max(\|v\|_2,10^{-8}).$$
''','trustcpf/common.py','unit'),
    ('1. CLIP视觉与文本编码',r'''
CLIP ViT-L/14视觉末层的CLS和256个补丁同时通过`ln_post`与`visual.proj`，得到共同768维表示。

$$g=\mathcal N(P_v^\top\operatorname{LN}_{post}(H_0)),\quad p_\ell=\mathcal N(P_v^\top\operatorname{LN}_{post}(H_\ell)).$$

编码FP32，缓存FP16，输入预处理为官方CLIP的Resize、CenterCrop、RGB和Normalize。
''','trustcpf/features.py','visual_tokens'),
    ('1.1 名称、组合与领域Prompt',r'''
属性、对象、组合文本分别编码为$e_a,e_o,e_{oa}$。组合表按对象外层、属性内层reshape为`[O,A,768]`。
''','trustcpf/features.py','prompts'),
    ('1.2 类别查询补丁',r'''
$$\alpha_\ell(e)=\operatorname{softmax}_{\ell}(e^\top p_\ell/0.07),\quad f(e)=\mathcal N(\sum_\ell\alpha_\ell(e)p_\ell).$$
''','trustcpf/features.py','query_patches'),
    ('1.3 缓存、数据轴和读取',r'''
缓存中的`g/fa/fo`以FP16保存，数据集读取后转换为FP32。共享适配器首先重新单位化，训练与评价使用相同输入契约。
''','trustcpf/data.py','FeatureDataset'),
    ('2. 共享低秩适配器',r'''
$$A_\theta(z)=\mathcal N(z+0.1U\operatorname{GELU}(Dz)),\quad D\in\mathbb R^{8\times768},\ U\in\mathbb R^{768\times8}.$$
$U$零初始化。该适配器共享于全局、属性查询与对象查询特征。
''','trustcpf/model.py','Adapter'),
    ('2.1 CLIP对象条件属性基底',r'''
$$\begin{aligned}
z_o^0&=\widetilde f_o^\top e_o/T_0,\quad c_{xo}=\mathcal N(\widetilde g+0.5\widetilde f_o),\\
U_{oa}^0&=(c_{xo}^\top e_{oa}+\widetilde f_a^\top e_a)/T_0,\quad T_0=0.07.
\end{aligned}$$
对象条件作用同时来自图像上下文$c_{xo}$和组合文本$e_{oa}$。
''','trustcpf/model.py','TrustCPF.base'),
    ('3. 知识关系与类型编号',r'''
类型编号OI/OE/AE依次为0/1/2。属性目标节点索引`0..A-1`，对象目标节点索引`A..A+O-1`。
OI源为父类CLIP文本；OE/AE源为描述CLIP文本。每个描述槽是独立关系。
''','trustcpf/graph.py','graph_records'),
    ('3.1 低秩类型消息',r'''
$$d_e=h_u-h_c,\quad W_\rho=I+L_\rho R_\rho,\quad m_e=d_e+L_\rho(R_\rho d_e).$$
768维下每类型消息矩阵用秩16参数化。$R_\rho$零初始化，对应$W_\rho=I$。
''','trustcpf/model.py','TypedGraph.messages'),
    ('3.2 固定候选度数的图聚合',r'''
$$Z_c(G)=\mathcal N\left(h_c+0.1\sum_\rho\frac{\sum_{e\in N_\rho(c)}b_em_e}{\max(1,|N_\rho(c)|)}\right).$$
程序用关联矩阵乘法实现聚合，每次干预保持候选度数不变。
''','trustcpf/model.py','TypedGraph.forward'),
    ('4. 三元图残差和完整联合概率',r'''
$$\begin{aligned}
T(x,o,a;G)&=\langle\mathcal N(P_vc_{xo}),\mathcal N(P_oZ_o(G))\odot\mathcal N(P_aZ_a(G))\rangle,\\
z_o&=z_o^0+\alpha_g\widetilde g^\top(Z_o-h_o)/T_g,\\
U_{oa}&=U_{oa}^0+\alpha_g[T(G)-T(\varnothing)]/T_g,\\
\ell_{oa}&=\log\operatorname{softmax}_o(z)+\log\operatorname{softmax}_a(U_{o,:}).
\end{aligned}$$
$\alpha_g=T_g=0.1$，$P_v,P_o,P_a\in\mathbb R^{64\times768}$。空图时残差为零。

`capacity=True`为容量对照：名称低秩变换取代关系消息，三元头直接作用于名称，不读取OE/AE来源。
''','trustcpf/model.py','TrustCPF.score'),
    ('4.1 分类目标',r'''
$$\mathcal L_{task}=-\ell_{o^*a^*}+\operatorname{LSE}_{c\in I_j}\ell_c-0.4\ell_{o^*}-0.6\ell_{a^*|o^*}.$$
训练候选为自己的Inner，最终适配为官方TRAIN。辅助属性损失读取完整未屏蔽条件概率。
''','trustcpf/model.py','task_loss'),
    ('4.2 图消息正则',r'''
$$\mathcal R_W=768^{-2}\sum_\rho\|L_\rho R_\rho\|_F^2,\quad\mathcal L=\mathcal L_{task}+10^{-3}\mathcal R_W.$$
''','trustcpf/model.py','TypedGraph.regularizer'),
    ('5. 五折组合任务',r'''
$$\mathcal C_s=C_0\sqcup\bigsqcup_{j=1}^5Q_j,\quad I_j=\mathcal C_s\setminus Q_j.$$
分块种子2026，稀有原语对应组合放公共Inner；检查每折Query的属性、对象均在该折Inner出现。
''','trustcpf/folds.py','make_folds'),
    ('5.1 配对边干预',r'''
$$\Delta_e=L_Q(G_{-e},0)-L_Q(G_{-e},1),\quad
y_e=M^{-1}\sum_m\Delta_{e,m},\quad\nu_e=\operatorname{Var}_{sample}(\Delta_{e,m})/M.$$
每对共享背景图和Query；正值表示打开边降低NLL。`weights`先组合内均分，再组合等权。

$$\frac{\partial\mathbb E_{q_\pi}L_Q}{\partial\pi_e}=-\mathbb E_{G_{-e}}\Delta_e.$$
''','trustcpf/utility.py','paired_utility'),
    ('5.2 测量队列与包含概率',r'''
每任务最多8个Query组合、每组合8图、每类型8条边。$\kappa_e=b_\rho/|E_\rho|$，背景图$M=4$；第5折$M=16$。
拟合三折各512任务，开发和审计各128任务。配对差分、边ID、Query图像ID和抽样概率写入测量文件。
''','trustcpf/stages.py','measure'),
    ('6. Inner视觉证据摘要',r'''
每个原语最多16张Inner图像，按组合均值再按上下文等权；对象和属性分别聚合。$r_c$为最终单位摘要，$n_c,k_c$为图像数、上下文数。
''','trustcpf/utility.py','support_summary'),
    ('6.1 6157维关系输入',r'''
$$\begin{aligned}
k_e&=[h_u;h_c;h_u\odot h_c;|h_u-h_c|;t_\rho;s_{src}]\in\mathbb R^{3079},\\
h_e^I&=[r_c;r_u;h_u;h_c;r_c^\top h_u;r_c^\top h_c;\log(1+n_c);\log(1+k_c);m_c;m_u]\in\mathbb R^{3078}.
\end{aligned}$$
$z=[k_e;h_e^I]\in\mathbb R^{6157}$。OI父类的$r_u$使用成员Inner证据；描述节点置零并附缺失标记。
''','trustcpf/utility.py','utility_features'),
    ('6.2 异方差效用预测器',r'''
MLP为6157→256→128→2，隐藏层GELU；均值直接输出，方差$\operatorname{softplus}(v_{raw})+10^{-4}$。
输入均值和尺度仅由拟合记录计算。
''','trustcpf/utility.py','UtilityNet'),
    ('6.3 测量噪声与HT加权NLL',r'''
$$L_e=\tfrac12[\log(v_e+\nu_e)+(y_e-m_e)^2/(v_e+\nu_e)],\quad
\widehat L=|E|^{-1}\sum_{e\in B_t}L_e/\kappa_e.$$
代码使用均匀记录batch，故将`mean(L/kappa)`乘以`sampled/total`。类型标准差在外层用于$y/s_\rho$、$\nu/s_\rho^2$。

$$\partial_mL=(m-y)/(v+\nu),\quad\partial_vL=\tfrac1{2(v+\nu)}-\tfrac{(m-y)^2}{2(v+\nu)^2}.$$
''','trustcpf/utility.py','utility_nll'),
    ('6.4 三模型与epoch选择',r'''
三模型独立种子，以Q4观测NLL选epoch；Q5审计使用选择前的三拟合折模型。最终模型使用五折标签按已选epoch数重新拟合。
''','trustcpf/stages.py','fit_utility'),
    ('7. 集成分歧',r'''
$$\bar m=L^{-1}\sum_lm_l,\quad\bar v=L^{-1}\sum_lv_l,\quad e=L^{-1}\sum_l(m_l-\bar m)^2.$$
方差和均值还原至原始NLL增益单位。
''','trustcpf/utility.py','ensemble_moments'),
    ('7.1 支持摘要的全方差汇总',r'''
$$\widehat\mu=R^{-1}\sum_r\bar m_r,\quad
\widehat\sigma_{task}^2=R^{-1}\sum_r\bar v_r+\operatorname{Var}_r(\bar m_r),\quad
\widehat v_{ens}=R^{-1}\sum_re_r.$$
所有有限集合上的方差采用分母$L$或$R$，与程序`unbiased=False`一致。测量噪声$\nu$不加入部署风险。
''','trustcpf/utility.py','aggregate_moments'),
    ('7.2 风险收益与结构分布',r'''
$$R_e=\widehat\mu_e-\lambda_{var}\sqrt{\widehat\sigma^2_{task,e}}-\lambda_{ens}\sqrt{\widehat v_{ens,e}}.$$
`mean`将两个风险权重置零，其余步骤与robust一致。
''','trustcpf/stages.py','make_structure'),
    ('7.3 KL和预算的闭式解',r'''
$$\begin{aligned}
J(\pi)&=-\sum_e\pi_eR_e+\tau\sum_e[\pi_e\log(\pi_e/p_e^0)+(1-\pi_e)\log((1-\pi_e)/(1-p_e^0))],\\
\partial_{\pi_e}\mathscr L&=-R_e+\tau[\operatorname{logit}(\pi_e)-\operatorname{logit}(p_e^0)]+\gamma_\rho,\\
\pi_e&=\operatorname{sigmoid}(\operatorname{logit}(p_e^0)+(R_e-\gamma_\rho)/\tau).
\end{aligned}$$
约束$\sum_{e\in E_\rho}\pi_e\le B_\rho$。$\tau/[\pi_e(1-\pi_e)]>0$保证严格凸；未超预算时乘子为0，超限时二分60次。
''','trustcpf/graph.py','solve_budget'),
    ('7.4 Q4结构参数选择',r'''
顺序选择风险9组、预算3组、温度3组。每个配置真实运行第4折冻结教师，按Query组合等权NLL选择。
''','trustcpf/stages.py','tune'),
    ('8. 单批优化、步长与恢复',r'''
$$\theta_{t+1}=\operatorname{AdamW}(\theta_t,\nabla_\theta\mathcal L_t;\eta_t,\beta_1=.9,\beta_2=.999,\epsilon=10^{-8},\lambda=10^{-4}).$$
代码以一个epoch更新数作warmup，随后按总更新步数余弦衰减。教师/最终参考10epoch，适配3epoch；每epoch保存模型、optimizer和RNG。

free结构对照采用score-function：$\partial_{a_e}\mathbb E L=\mathbb E[(L-b)(b_e-\pi_e)]$；$b$为上一批次EMA，额外KL均值权重0.01、各类型超预算概率均值的平方惩罚权重1。
''','trustcpf/engine.py','train_model'),
    ('8.1 完整条件任务损失的解析梯度',r'''
令候选概率$p_{\mathcal C}$在候选集外取0，$h_{oa}=p_{\mathcal C}(a,o)-\mathbf1[(a,o)=(a^*,o^*)]$，$g_o=\sum_ah_{oa}$。单图像损失导数为：

$$\begin{aligned}
\frac{\partial L}{\partial z_o}&=g_o+0.4[p_o-\mathbf1(o=o^*)],\\
\frac{\partial L}{\partial U_{oa}}&=h_{oa}-p_{a|o}g_o+0.6\mathbf1(o=o^*)[p_{a|o}-\mathbf1(a=a^*)].
\end{aligned}$$
以下函数给batch均值的导数，测试逐项比对实际任务损失的autograd。

对单位化输入$z=v/\|v\|$，$J_{\mathcal N}(v)=(I-zz^\top)/\|v\|$（$\|v\|>10^{-8}$）。图消息的低秩矩阵梯度为：

$$\partial_{L_\rho}L=\sum_{e:\rho(e)=\rho}g_e(R_\rho d_e)^\top,\quad
\partial_{R_\rho}L=\sum_{e:\rho(e)=\rho}(L_\rho^\top g_e)d_e^\top,$$
其中$g_e=\partial L/\partial m_e$包含固定度数、边开关、概念单位化及三元残差的链式导数。训练使用同一计算图自动回传，并叠加消息正则梯度。
''','trustcpf/model.py','task_logit_gradients'),
    ('9. 概率平均与跨图方差',r'''
$$\log\bar p_{\mathcal C}(c|x)=\operatorname{LSE}_k\log p_{\mathcal C}(c|x,G^{(k)})-\log K.$$
固定图种子使所有batch用同一组图。`mean_logit`、`mean_graph`是额外读出配置。

$$V(x)=K^{-1}\sum_k\|p_k-\bar p\|_2^2,\qquad -\log\bar p(y)\le K^{-1}\sum_k[-\log p_k(y)].$$
''','trustcpf/model.py','average_graphs'),
    ('9.1 Seen/unseen工作点与校准',r'''
$$s_c^{(b)}=\log\bar p(c|x)+b\mathbf1[c\notin\mathcal C_s],\quad
p_{cal}=\operatorname{softmax}(s^{(b)}/T).$$
VAL选择工作点bias和温度，TEST直接应用。ECE采用15个等宽区间。
''','trustcpf/engine.py','calibration_metrics'),
    ('9.2 原生有限偏置曲线',r'''
从正确未见样本的分数差减$10^{-4}$后采约20个bias点，加bias1000端点；与现有CPF原生扫描保持同一口径。准确率/AUC展示单位为百分数。
''','trustcpf/metrics.py','native_curve'),
    ('9.3 全临界事件曲线',r'''
每图像最佳seen分数$s_s$与最佳unseen分数$s_u$在$b=s_s-s_u$切换。汇总相同临界点后逐次更新Seen/Unseen计数，区间内部bias用于固定工作点。
''','trustcpf/metrics.py','exact_curve'),
    ('9.4 效用预测与方向指标',r'''
Spearman使用平均秩；方向同时报告全部样本和$|y|>1.96\sqrt\nu$子集，可靠子集比例另报。
90%预测区间为$m\pm1.645\sqrt{v+\nu}$，这里传入的$v$是条件任务方差与集成分歧之和。
''','trustcpf/utility.py','regression_metrics'),
    ('10. 知识内容错配与删除',r'''
每类型固定随机排序前缀。描述来源用按所属原语分组的块旋转置换，保持被选源句多重集合；OI按父类分组旋转。小样本前缀不足以跨组置换时扩展到首个可置换前缀，保存实际扰动比例。
''','trustcpf/graph.py','permute_bank')]
    out=['# TRUST-CPF公式及代码v1：CLIP框架\n\n版本：2026-10-21。公式对应本包实际源码，代码块由AST逐段提取。对象条件属性概率、配对关系效用和风险选图均以本版CLIP实现为准。各块按标明的源码上下文阅读，类方法的self对应所属模型实例。\n\n```python\nimport math\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n```\n']
    records=[]
    for title,math,file,symbol in sections:
        code,line=excerpt(file,symbol)
        # Markdown keeps portable source paths; the exact line identifies the released symbol.
        out += ['\n## '+title+'\n\n'+math.strip()+'\n\n',
                '[`'+file+' · '+symbol+'`]('+file+')，源码第'+str(line)+'行。\n\n',
                '```python\n'+code+'\n```\n']
        records.append((title,file,symbol,line))
    out += ['\n## 11. 核心验证指标与结果文件\n\n',
    '| 目标 | 指标 | 文件 |\n|---|---|---|\n',
    '| 独立任务迁移 | Q5 Spearman、RMSE、方向balanced accuracy、AUROC、90%覆盖率 | `utility/audit.json` |\n',
    '| 组合识别 | CW/OW AUC_native、AUC_exact、oracle-HM、VAL bias的Seen/Unseen/HM | `runs/*/metrics.json` |\n',
    '| 条件属性 | GT-object属性准确率/NLL、对象/属性错误分解 | 同上 |\n',
    '| 概率稳定性 | NLL、ECE、跨图预测方差 | 同上 |\n',
    '| 结构选择 | 每类型期望边数、任务/集成方差、预算乘子 | `structure.json`、各组metrics |\n',
    '| 知识变化 | 分类型噪声曲线、实际扰动率、RAUC | `noise_*.json`、`robustness_summary.json` |\n',
    '| 随机重复 | seed配对AUC差、均值、样本std | `paired_comparisons.json` |\n',
    '\n完整训练和部署命令见[服务器实践命令](服务器实践命令.md)，方法与对照见[CLIP修订实验方案](TRUST-CPF_CLIP_修订实验方案v1.md)。\n',
    '\n## 12. 源码索引\n\n| 节 | 文件 | 函数/方法 | 行 |\n|---|---|---|---:|\n']
    out += ['| '+title+' | `'+file+'` | `'+symbol+'` | '+str(line)+' |\n' for title,file,symbol,line in records]
    result=''.join(out)
    result=re.sub(r'\$\$(.*?)\$\$',lambda m:'\n$$\n'+m.group(1).strip()+'\n$$\n',result,flags=re.S)
    (ROOT/'TRUST-CPF公式及代码v1.md').write_text(result,encoding='utf-8')
    print('FORMULA_DOC',len(records),'source excerpts',len(result),'characters')


if __name__=='__main__':main()
