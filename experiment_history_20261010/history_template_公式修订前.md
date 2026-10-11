# CZSL 实验总档案：按序记录实验、公式修正、决策动机、数据与分析

整理截至：2026-10-10；范围：当前 CPF／KCC／LogiCZSL 适配、EKR、条件属性 head、Generic 混合、CLIP 探针和 TRAIN 知识审查。

本档案是对已有本地证据的回溯，不是新实验协议。没有启动训练，没有改写旧结果、知识、checkpoint 或 gate。按研究推进与修正的依赖顺序整理；同日的工作可能交叉执行，未据文件名伪造精确完成时刻。“0601”是原方案文件名，不据此认定方案实际写于某年6月1日。

正文给出每一步的目的、实际公式、数据解释和当时决策；附录集中收录可核验的逐 run 指标、已保存逐轮轨迹、单句对照及来源指纹。大型逐图分数、全部混淆数组和训练日志以原始文件链接承接，不将数万行数组粘进正文。无法从现有本地资料恢复的数值明确注明，绝不补造。

## 0. 阅读导航与最终结论

建议顺序：先看第1节时间线；再按第3—22节阅读完整过程；第23—26节解释目前瓶颈、既往判断修正、原方案完成度和下一步边界；附录用于查原始数字与轨迹。

截至目前，最准确的总体结论是：

1. **已有性能正信号，不是“所有实验都没有成功”。** UT 静态 v2 的 C_text 相对 A_unit 三 seed 平均 CW／OW AUC 为 +3.0670／+4.1315 pp；Generic＋固定25%混合在 seed 0 上四组 CW 均提升约1.35—1.57 pp。
2. **新增正确知识的独有作用尚不稳定。** 正确描述常未胜过名称或重新训练的错配控制；B/F 新增逻辑损失也没有获得配对正增量。这与“换输入／加容量／改读出可以涨点”是不同结论。
3. **已经有视觉参与、知识参与和概率条件建模。** B/F 的视觉逻辑落地、EKR 查询、条件属性 head 均已实现并执行。不能将后续方案描述成“第一次把知识／视觉加入模型”。
4. **出现了可具体解释的失效现象。** 静态图像无关 K、弱对象依赖与几乎无拒识的 null、低温度下过大的条件概率改变、TRAIN 已见搭配拟合与 unseen 属性识别退化、描述造成的材质类别偏好，分别得到不同程度证据。
5. **最新 TRAIN 审查发现局部而非全面阴性的知识线索。** 各属性第三条 AE 的 TRAIN macro 比名称高1.0514 pp，且明显高于两套对应错配；但87.3%的净增判对来自 Leather，排除 Leather 的描述性诊断转为 −2.2222 pp。没有形成可直接蒸馏的迁移教师。
6. **当前建议已收窄。** 保留 CPF 概率接口、C_text 和 Generic＋25%工程参考；停止现有等权知识教师扩跑。下一项优先检验不加新增 OE/AE 的受控共享视觉属性读出与组合迁移。此项尚未实现／训练，不能列入已完成成果。

## 1. 全流程时间线与实验类型

“训练”指实际更新相应模块；“诊断”不等同新模型；“方案”不等同已经执行。各阶段存在复用，不能直接把下表行数、评价数、epoch相加作为独立实验总数。

| 顺序 | 阶段／日期范围 | 已做内容 | 规模与类型 | 结果驱动的下一决定 |
|---|---|---|---|---|
| 01 | 原方案及早期基线，10月4日前后归档 | CPF 基线、OI/OE/AE及GloVe知识缓存 | 三数据集；历史TEST已有记录 | 分解文本、prior、logic、feature四个因素 |
| 02 | 早期MVP，截至10月5日核验 | UT A/C/D/E，MIT和C-GQA A/C | 8个run身份、16份VAL主评价；三组A与历史权重一致，不是新的独立重复 | UT文本保留；固定K失败 |
| 03 | 固定K后处理 | A/C checkpoint上Dpost/Epost | 2×5个λ×CW/OW=20份评价；0新训练 | 两者选λ=0 |
| 04 | UT静态v1归因，10月5—6日 | A_raw/A_unit/C_text/C_shuffle | 4身份×3seed的10轮协议，存在旧run复用；不重复统计独立训练数 | 文本有效，正确归属未稳定胜错配 |
| 05 | TRAIN图像与描述审核，10月6日 | 28图板、84句复核／修订 | 560面板、528张不同TRAIN图；54句改写 | 只给一次v2修订预算 |
| 06 | UT静态v2，10月6—7日 | 四输入控制×三seed | 12项从初始化训练各10轮、24份VAL评价 | CW归因gate失败，OW信号保留 |
| 07 | 静态v2逐图／冻结回放，10月7日 | 24份既有预测导出；18项基础全分数回放记录 | 0新训练；同一3214张VAL | Boat Shoes对象混淆；原C_text null不退名称 |
| 08 | 名称锚点EKR，10月7日 | N_name/N_clean/N_shuffle | seed0，各3轮；28834分支参数 | 正确描述落后名称和错配 |
| 09 | EKR无训练机制诊断 | 映射错配、对象上下文错配、去null、margin | TRAIN498＋VAL3214；27数组视图、20次原生评价复算 | 名称基底保护＋同构值对齐控制 |
| 10 | EKR共享值对齐v2，10月7—8日 | 三个同容量主组＋fixed消融 | 4组各3轮；48034／28834参数 | align组三组选epoch0；fixed微增 |
| 11 | TRAIN支持探针，10月8日 | 固定语义几何与视觉原型参考 | 498张、83pair、两半折；0optimizer | 均值不如名称但部分槽有信号 |
| 12 | SG-EKR | TRAIN双半折支持mask＋固定value | 30/48 AE槽保留，3组各3轮 | support落后名称／错配 |
| 13 | TRAIN视觉原型开发诊断 | 图像半折、排除查询组合的原型／混合 | 498张，0新训练，有限gain/β网格 | 只形成VP候选方案，正式VP未运行 |
| 14 | Epoch回溯与EKR v2续训，10月8日 | 从四组epoch3续到总计10 | 新增28epoch；不是4个全新训练run | 名称epoch6延迟收益；正确align仍选0 |
| 15 | CPF＋LogiCZSL B/F，10月8—9日 | A_ref/B/E_ref/F | UT3351，4组各10轮，β=.01、q=3 | 直接结合无新逻辑正增量 |
| 16 | B/F审计及条件StageA，10月9日 | 最佳B/F梯度；C_text回放／缓存 | 12TRAIN图、3batch；缓存22998TRAIN＋3214VAL；丢弃式head检查15步 | 弱且部分冲突的逻辑；进入有限条件head pilot |
| 17 | 条件pilot r1失败 | 安全检查点恢复自检报错 | 临时诊断1步，正式0步 | 修复Torch1.13序列化兼容，保留失败包 |
| 18 | 条件pilot r2 | P_name/P_K/P_generic/两perm | 5组各10轮、4500正式步；TRAIN机制360步 | 乘性交互选0；generic CW涨OW降 |
| 19 | 25%混合分数回放 | 两读出比较＋全55轮混合 | 0optimizer；事后开发修订 | 读出漂移有实际影响；补generic同结构内容控制 |
| 20 | Generic＋25%正式控制，10月10日 | G_name/G_perm1/G_perm2；复用G_K | 新2700步；正确组原900步复用；四组同60864参数 | 四组同涨，内容归因失败，保留工程结果 |
| 21 | 研究目标与原方案回溯 | 性能／知识／机制分层，A–H完成度检查 | 文档与代码检查，不是实验 | 不以完整结合为前置；先检查图文知识证据 |
| 22 | CLIP探针v1失败 | TRAIN/VAL类别交集assert错误 | 编码前停止，无性能结果 | 修正为广义seen/unseen划分与图像隔离 |
| 23 | 冻结CLIP探针r2 | 名称、OE+AE、AE-only及两perm各自控制 | 7配方，TRAIN22998／VAL3214，0optimizer | 两个教师继续gate均False |
| 24 | TRAIN逐句／图片审查，10月10日 | 9 OE×AE位置＋3 AE位置及控制 | 36单句视图＋7均值回放；7041唯一文本；96TRAIN图；0optimizer | AE第三条局部信号；教师仍不扩跑；转视觉读出诊断 |

## 2. 全档共用符号、评价口径与比较限制

属性$a$、对象$o$、图像$x$、外部知识$K$；$\mathcal N(v)=v/\|v\|_2$。GloVe维度300；后来独立CLIP图文特征768。符号$\lambda$表示静态prior强度，$\beta_{logic}$表示逻辑权重，$\eta$表示EKR有符号残差幅度，$\alpha$表示混合比例；它们不能互换。

主要历史推理接口实际是三分数相加：

$$
s_{CPF}(x,a,o)=s_{comp}(x,a,o)+s_{attr}(x,a)+s_{obj}(x,o).
$$

这与原论文的条件概率思想有关系，但不能不做归一化就将legacy加法分数写成已实现的两个概率相乘。后来的条件head才显式定义完整网格联合／条件分布。

原生评价对TRAIN以外候选加bias：

$$
s_b(x,p)=s(x,p)+b\mathbf1[p\notin\mathcal P_{TRAIN}],\qquad
HM(b)=\frac{2S(b)U(b)}{S(b)+U(b)}.
$$

AUC汇总seen–unseen曲线，best-HM为扫描最大值；best-seen和best-unseen通常不是同一工作点，不能用它们重新计算HM。bias=0 top1、GT-object属性诊断、AUC/HM是三种不同证据。

除另行标注外，表内指标为百分数，差值为百分点pp。三seed均值／样本标准差不自动等于显著性。后续大部分实验使用已反复开发的VAL，不能包装为独立盲测；历史CPF TEST已查看过，但后续新head／教师没有新的TEST成绩。读取TEST pair元数据用于词表和评价接口，不等于用TEST图像／指标选参，也不能声称整个项目从未访问TEST。

UT当前数据：TRAIN22998图／83pair，VAL3214图／30pair，其中877seen图／15pair、2337unseen图／15pair；TEST2914图／36pair。对象12、属性16。CW分数表116候选，但原VAL closed mask实际允许98个组合；OW为完整192候选。网格固定为`[object,attribute]`，pair列表通常为`(attribute,object)`，需要显式索引交换。

## 3. 起点：原0601方案、基线和知识重建

### 3.1 最初要研究什么

原方案核心不是仅将两个损失相加，而是检验：

$$
P(a,o\mid x,K)\approx P(o\mid x)P(a\mid o,x,K).
$$

三条路径：输入端richer text、输出端compatibility prior、特征端knowledge-conditioned属性建模；external logic为可以直接组合的对照。

| 原身份 | 配置 |
|---|---|
| A | 原CPF |
| B | CPF＋external logic |
| C | richer text |
| D | calibration only |
| E | richer text＋calibration |
| F | E＋external logic |
| G | E＋feature knowledge |
| H | G＋external logic |

后来的`G_support`、`G_K/G_name`是另外协议的命名，不是这里的原G/H。

原factorized示例包含$h(e_{a\mid o}^{K})$的pair bias；它不读取图像，同一组合对所有图像相同。属性项沿对象维广播也不自动产生对象条件；需要上游或交互提供真正条件信息。这是后来优先改为图像＋候选对象参与证据的动机，而不是原G/H已经被完整训练过。原方案依据：{{SOURCE:original}}。

### 3.2 知识实际是什么

早期UT重建知识为16属性、12对象、4父类、48 AE、36 OE；MIT为115属性、245对象、19父类；C-GQA为413属性、674对象、23父类。UT是手工重建，MIT/C-GQA含WordNet词义草稿，不是作者遗失的历史LLM输出。

`approved`／`bulk_trusted_unverified`只表示允许当前资产被工程读取及来源状态，不等同专家逐句批准。v2为助手TRAIN图审核候选，`human_reviewed=false`；B/F又单独重建5父类的逻辑bank，后述双知识版本必须区分。

构建资产与实际消费：$W_a^K,W_o^K$是原语300维表示；$W_{ao}^K$是pair模板向量；$K_{oa}$是每组合一个标量；EKR bank保留逐条向量。早期A/C/D/E并没有OI图消息传播，也没有直接消费全部pair block作为feature head。

### 3.3 实际GloVe聚合公式

每个名称沿原CPF别名编码。例如Full.grain.leather→thick_leather，Nubuck→grainy_leather，Patent.Leather→shiny_leather。描述用小写英文正则分词；已知token均值后逐句单位化：

$$
u_{c,0}=\mathcal N(\operatorname{mean}_{w\in name(c)}g(w)),\quad
u_{c,r}=\mathcal N(\operatorname{mean}_{w\in desc_r(c)}g(w)),\quad r=1,2,3.
$$

$$
w_c^K=\mathcal N\left(\frac14\sum_{r=0}^{3}u_{c,r}\right).
$$

名称占1/4，三描述合计3/4；不是拼所有token后一次平均。raw名称、单位名称、知识向量分别保存。GloVe SHA256为`91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed`。

三模板为`a {object} that is {attribute}`、`the {attribute} {object}`、`{attribute} appearance of {object}`，先逐模板单位化再平均归一化成$b_{oa}$。它本质是词均值，不表示语言模型理解语序／否定。

$$
R_{oa}=b_{oa}^{\top}\mathcal N(w_o^K+w_a^K),\qquad
K_{oa}=\frac{R_{oa}-R_{min}}{R_{max}-R_{min}}.
$$

非退化时全矩阵min-max；退化常量取0.5；随后全矩阵总体标准化成$\widetilde K$，退化时全零。K不是行概率或经真实兼容标签拟合的feasibility；min-max也不使cosine成为概率。资产详情及统计原证据：{{SOURCE:old_history}}。

## 4. 早期MVP：A/C/D/E与跨数据集A/C

### 4.1 动机与实际公式

先隔离文本和静态prior：A原名称，C用$W_a^K/W_o^K$，D原名称＋prior，E文本＋prior。

$$
s_{D/E}(x,a,o)=s_{A/C}(x,a,o)+\lambda\widetilde K(o,a),\quad
\widetilde K=\frac{K-\operatorname{mean}(K)}{\operatorname{std}_{pop}(K)}.
$$

实际用总体标准差`unbiased=False`；非退化不另加epsilon，标准差小于$10^{-8}$直接全零。这是原方案示例中默认std＋epsilon的工程落实差异。首轮D/E λ=.1。基础训练为$L_{comp}+.4L_{obj}+.6L_{attr}$，原增强路径另含mix_loss。

### 4.2 得到的数据

{{TABLE:EARLY}}

8个run身份都有10轮日志，UT四组最佳6轮，MIT两组10轮，C-GQA A9/C10。三数据集A checkpoint与归档历史权重相同，不能把历史TEST与本轮A当成两次独立重复。{{SOURCE:early_audit}}。

关键差值：UT C−A CW／OW AUC +4.791201／+6.586815 pp；D−A −7.006222／−6.385160；E−C −7.022326／−8.107272。MIT C−A CW−.314696、OW−.009653；C-GQA CW+.052533、OW+.251063。

### 4.3 分析与决策

UT文本有明显工程收益；固定K退化很大。不能从UT推导跨数据集稳定，MIT/C-GQA单seed与10轮结果只显示小差。后两数据集最佳接近预算边界，不能称已经收敛或完全没有潜力。

已有bias=0逐图分析如下；与AUC提升并非同一判断：

{{TABLE:EARLY_BIAS}}

UT overall净+4但unseen净−3；MIT overall−7但unseen+13；C-GQA overall+59、unseen+60。文本可影响排序与seen/unseen取舍，不等于所有图都变准。

当时决定：冻结早期结果，保留UT文本，先诊断K，不继续叠固定prior。B/F没有同期实现，**当时只跑ACDE缩小了原方案六组MVP的问题范围**；后来明确补B/F，不应事后将ACDE写成直接结合已检验完毕。数据源：{{SOURCE:early}}、{{SOURCE:early_bias}}。

## 5. λ固定checkpoint后处理：确认静态K是否值得保留

### 5.1 动机与操作

D/E重新训练同时改变优化轨迹，不能只凭它们确定输出prior的问题。因此复用A/C固定checkpoint，网格λ={0,.025,.05,.1,.2}：Dpost对应A，Epost对应C。公式不变；无新optimizer。

{{TABLE:LAMBDA}}

### 5.2 分析与决定

两者按VAL CW AUC均选0；非零λ还触发相应HM／OW保护条件。Epost .025虽然OW best-unseen略升，但CW AUC下降且HM保护失败，不能只报这个正数宣布prior成功。

结论限定当前template K、checkpoint、网格与规则，不证明所有知识prior无用。可能因素包括图像无关偏置、文本自相似而非feasibility、总体标准化尺度；未逐因素隔离。后续语义／EKR／条件head的静态λ均关。20份评价不是20个新训练run。{{SOURCE:lambda}}。

## 6. UT v1三seed语义归因：文本收益是否需要正确对应

### 6.1 为什么补名称单位化与shuffle

早期A/C同时混入了归一化、领域文本与正确对应。新增A_unit消除名称尺度因素；C_shuffle只替换补充描述，保留接收类名称：

$$
w_c^{shuffle}=\mathcal N\left(\frac{u_{c,0}+u_{\pi(c),1}+u_{\pi(c),2}+u_{\pi(c),3}}4\right),\quad\pi(c)\ne c.
$$

属性置换seed1729，对象1730；三训练seed共用同一映射。训练seed变化不是多套知识置换。没有图节点传播，shuffle不是USG随机语义图。

### 6.2 可靠保留的数据

| seed | C_text−A_unit CW AUC pp | C_text−C_shuffle CW AUC pp |
|---|---:|---:|
| 0 | +0.508647 | −2.088948 |
| 3351 | +4.662171 | +1.019615 |
| 2026 | +2.125442 | +0.269668 |
| 均值 | +2.432087 | −0.266555 |

协议为四组×三seed、10epoch，部分旧run复用。当前本地保留完整旧总结与以上配对差值；未收齐v1全部绝对主指标／每轮机器可读表，因此不从v2反推、也不填造v1缺项。{{SOURCE:v1_summary}}、{{SOURCE:old_history}}。

### 6.3 技术修复和研究结论分开

审计v1将YACS打印的`1e-05/None`与dump类型直接比较，引发假不一致；旧NPZ为top5而新格式top1。v2审计统一表示并取旧top5首列，技术检查通过；不改变旧AUC，也不改变研究gate=False。

三个seed正确文本均胜单位名称，但正确归属未稳定胜错配。决定：核验知识质量一次，不立即大规模EKR。这里“未达到预定+.5pp”是预算规则，不是零效应／不可行的统计结论。

## 7. TRAIN图像审核与v2知识修订

### 7.1 动机与实际范围

知识早期未经视觉审核，可能有过强典型特征、错义或不适用描述。按固定TRAIN样本审核16属性＋12对象图板，共560面板／528不同图，84句保留30、改写54。

不据VAL失利类别继续改句；不看TEST图片。TRAIN导出有两条旧目录路径问题，逐路径检查与映射修复独立归档，不擅改训练数据。审核是助手质性审核，不是材料专家真值；v2仍标`human_reviewed=false`、非正式结果兼容候选。

### 7.2 数学／资产改变

名称、标签、OI、GloVe不换；仅改AE/OE并生成独立cache/bank。三句仍等权。原始名称与v1逐元素不变；bank正确聚合与cache误差0。

| 版本 | K均值 | K总体std | 描述OOV | 整句fallback |
|---|---:|---:|---:|---:|
| UT v1 | .508405 | .244008 | .361011% | 0 |
| UT v2 | .537236 | .201683 | .283286%（1/353） | 0 |

v2属性OOV0/181，对象1/172；中心化残差和最大约$7.45\times10^{-8}$／$5.96\times10^{-8}$。说明数值构建健全，不证明内容可辨。K虽然生成，后续v2训练不使用prior。

决定：一次修订后四控制三seed从初始化重跑，不重复润色到gate通过。{{SOURCE:train_review}}。

## 8. UT静态v2：工程收益保留，CW归因仍未过

### 8.1 实验

四输入A_raw/A_unit/C_text/C_shuffle，seed0/3351/2026，各10轮，λ=0。同run按CW选checkpoint后评CW/OW。完整24行见附录A。

| 输入 | CW AUC均值±样本std% | CW HM均值% | OW AUC均值±样本std% | OW HM均值% |
|---|---:|---:|---:|---:|
| A_raw | 41.182±1.497 | 55.857 | 29.001±3.053 | 42.595 |
| A_unit | 41.473±1.673 | 56.095 | 28.129±3.767 | 41.892 |
| C_text | 44.540±.785 | 58.270 | 32.261±3.242 | 45.835 |
| C_shuffle | 44.161±1.025 | 58.998 | 30.280±1.855 | 45.731 |

{{TABLE:V2_DELTAS}}

{{TABLE:V2_SEED_DELTAS}}

上表直接由同seed／同设置的原CSV全精度指标相减，展示六位小数。总体C_text−A_unit CW+3.067040、OW+4.131470；C_text−shuffle CW+.378912、OW+1.980460。

### 8.2 分析与决定

9条预定条件8通过，唯一失败为正确−错配CW均值未达到+.5。差.121088pp；均值样本std .926536pp。三个seed不足以可靠区分真实效应是否跨过这个任意预算阈值。

OW正确−错配三seed均正，是补充正信号；不能事后将主指标换OW宣布原gate通过。shuffle相对A_unit CW也+2.688127，可能共享鞋类领域内容、形成输入／优化变化，不能把全部文本收益归正确知识，也不能将全部收益断言为纯随机噪声。

当时停止原“静态gate→正式EKR”自动推进，先导出逐图；后来允许另立有限动态探索，**不是修改原False**。{{SOURCE:v2_review}}、{{SOURCE:v2_delta}}。

## 9. 静态v2逐图分析与冻结回放：为什么修改锚点

### 9.1 实际做了什么

24次既有VAL预测打包，434项身份／数组检查通过。后来名称锚点准备材料记录A_unit/C_text/C_shuffle三seed×CW/OW共18项完整分数回放，指标差0、身份配对；均无新训练。原静态小包本身不含完整分数，不能混淆“先只有top1的诊断”与“后来补全分数的回放”。

### 9.2 得到的数据

| OW bias=0，三seed平均 | all pair% | seen pair% | unseen pair% | 预测落seen比例% |
|---|---:|---:|---:|---:|
| A_raw | 22.827 | 66.857 | 6.304 | 93.207 |
| A_unit | 23.398 | 66.781 | 7.117 | 92.398 |
| C_text | 22.734 | 67.617 | 5.891 | 93.435 |
| C_shuffle | 22.703 | 67.199 | 6.005 | 93.124 |

C_text−A_unit seen+.836、unseen−1.227pp；三seed合计纠正109／破坏173。正确−shuffle纠正134／破坏131。曲线AUC与零点准确率可以反向变化，不能据零点推翻AUC。

Boat Shoes316图，pair退化−9.177pp，三个seed均负；对象准确率−11.287而属性仅−.105。Leather×Boat Shoes263图，三seed97恶化／8纠正事件，多流向Leather×Loafers36、Oxfords28、Flats15。Flats308图相对A_unit +2.381pp，三seed均正，也必须保留。

不可部署oracle取A_unit/C_text任一正确，pair24.528%，比A_unit+1.130pp；只是现有top1互补，不是门控预期收益。

### 9.3 公式问题与决策

旧动态EKR是$s_C+\eta\Delta$；null只令新增残差为0，回到C_text，**不会撤销C_text文本输入已造成的损害**。所以有null不代表有名称退避。

旧CW top1在closed mask前argmax；116分数候选、98允许候选，逐图解释优先用已核对OW。AUC本身有正确mask，不需要因此重训或改gate。

决定：保持知识和数据固定，将有限新探索锚点改A_unit，检验增量关闭能精确回名称。{{SOURCE:v2_predictions}}。

## 10. 名称锚点EKR：有条件查询，不代表条件知识有效

### 10.1 原中心化EKR与锚点修正

四槽$u_{a,r}$为名称＋三AE；$\mu_a=\frac14\sum_r u_{a,r}$，$d_{a,r}=u_{a,r}-\mu_a$；不再单位化每条残差。

$$
e_{a,r}(x)=\tfrac12\mathcal N(f_a(x))^\top d_{a,r},\quad
q(x,o)=\mathcal N\big(Q_x\mathcal N(f_a(x))\odot\sigma(Q_o n_o)\big).
$$

投影语义键与null经温度.2的softmax产生$\alpha$。N_clean的key和value都来自中心化残差；N_name则将key改为重复名称、value改为名称/2。设$z_{a,r}=d_{a,r}$（clean／shuffle），名称控制则$z_{a,r}=u_{a,0}$：

$$
k_{a,r}=\mathcal N(P_k z_{a,r}),\quad
\ell_{a,r}(x,o)=q(x,o)^\top k_{a,r}/.2,\quad
\ell_{\varnothing}(x,o)=q(x,o)^\top\mathcal N(k_{\varnothing})/.2+b_{\varnothing},
$$

$$
(\alpha_{\varnothing},\alpha_{a,0},\ldots,\alpha_{a,3})=softmax(\ell_{\varnothing},\ell_{a,0},\ldots,\ell_{a,3}).
$$

null证据0，有效槽才进入增量：

$$
\Delta(x,a,o)=\sum_r\alpha_{a,r}(x,o)e_{a,r}(x),\quad
\eta=.1\tanh(g),\quad g_{init}=0.
$$

$$
\boxed{s_N=s_{A\_unit}+\eta\Delta}\qquad
force\_off\Rightarrow s_N=s_{A\_unit}.
$$

对象上下文使用对象名称，不让OE泄露进入名称控制。N_clean用中心化值；N_name四槽重复名称、value为名称/2，避免全零。两者参数同28834，但值几何不相同，后续需要修正这一混杂。

对应实现：{{SOURCE:centered_router}}、{{SOURCE:name_router}}。第12节v2才将主分支的key/value槽改为不中心化的原始语义槽，并另加共享value对齐。

### 10.2 结果与分析

共同N0 A_unit CW42.818640／OW29.464872。三组seed0各3轮：

{{TABLE:NAME}}

clean−N0 CW+.047790／OW+.036886；clean−name CW−.256764；clean−shuffle−.132762。原pilot要求三项CW均≥+.2，未过。

clean loss1.415845→1.397870，确有训练；最佳η−.056094在允许有符号范围内，不是bug。CW null均值仅.059593%，未显示拒识；平均绝对残差约.002026。OW零点仅变11图，0纠正／2破坏，对象变化0。

决定：不自动追加seed，先诊断value、null、对象依赖。三轮负结果只限制这个预算，不证明更长一定无效。{{SOURCE:name_review}}。

## 11. 名称EKR机制诊断：保护名称基底、修正值表达的动机

TRAIN83pair各6图共498，VAL3214；27数组视图核对、20次原生评价／120指标独立CPU复算均0差。旧权重回放，不计新训练。{{SOURCE:name_diag}}。

| 干预／诊断 | 数据 | 分析 |
|---|---|---|
| 已训clean推理时AE置换11/22/33 | CW−.108829/−.159596/−.242273pp；OW−.167573/−.085800/−.085800 | 映射敏感，不等于正确描述比重新训练错配更好 |
| TRAIN margin增量 name／clean | +.004772／+.002712 | 名称在TRAIN也更强，不是清楚的“只在VAL过拟合” |
| VAL OW margin增量 name／clean | +.004453／+.002099 | 很多margin轻微升，但未跨决策边界 |
| clean错配后GT null均值 | .070234%→.053189% | 没有增强拒识；去null AUC／top1不变 |
| 对象上下文错配 | 分数均差3.30e−5／3.83e−5；各1个预测改变 | 有数值依赖但实际很弱 |
| 同图同属性跨对象残差变化 | 约总平均绝对残差1.56% | 不能宣称强对象条件知识检索 |

关键混杂：N_clean将四槽共同成分中心化消去，也改写名称槽；N_name用名称/2。正确字典推理错配还会连带重算名称残差，因此不是严格的“只改描述槽、名称值完全不动”干预。

最初建议用$(description-name)/2$保护名称；继续设计时发现同公式名称控制会成为$name-name=0$，无法排除容量收益。于是**没有把该建议作为最终主实现**，改为原始语义槽＋共享低秩value对齐，所有组保留非零名称控制。

## 12. 共享值对齐EKR v2三轮：同构控制与零增量回退

### 12.1 公式修正

所有组共享冻结底座：

$$
B_{name}=s_{A\_unit}+\Delta_{N\_name}^{frozen}.
$$

描述槽不中心化，名称不随AE更换而改变；共享低秩适配：

$$
\widetilde u_{a,r}=u_{a,r}+UDu_{a,r},\quad
v_{a,r}=\tfrac12\mathcal N(\widetilde u_{a,r}),\quad
\Delta_{v2}=\eta\sum_r\alpha_r\mathcal N(f_a(x))^\top v_{a,r}.
$$

$D:300\to32$随机初始化，$U:32\to300$零初始化；不能两者同时零。$g=0$让初始分支关闭；训练初期梯度先到gain，再到上投影等，CPU检查与服务器流程有记录。value范数.5、$|\eta|<.1$，单候选原残差界.05。

V_name_align/V_clean_align/V_shuffle_align为相同48034可训练参数；V_clean_fixed冻结value投影，28834，是刻意降容量消融。新增loss为组合CE$(s/ .05)$，AdamW lr=.001、wd=.0001、batch64、seed0、各3轮；底座全冻结。{{SOURCE:align_formula}}。

### 12.2 数据与决定

{{TABLE:ALIGN3}}

三align组选epoch0，即新增分支关闭；fixed epoch2 CW+.084116、OW+.243614。clean align CW1/2/3轮42.963281/42.786120/42.625548，loss下降但验证退化，value drift约.61—.65。

当时认为对齐可能使弱固定信号漂移，优先做固定值支持探针。这里只能报告数据相关性，不能以drift证明语义被破坏的因果；旧报告“可学习对齐破坏弱信号”的措辞过强，后来预算回溯进一步收紧其范围。

## 13. TRAIN固定语义支持探针：平均不占优，但局部条目有信号

498张既有冻结CPF特征，无模型加载／optimizer／新图／VAL或TEST。固定原语证据在同对象可用训练属性候选间比较，计算真属性margin与按属性宏平均；四槽直接均值（不重新归一化）和名称/2分别作为控制。

| 诊断 | 结果 |
|---|---:|
| 正确均值−错配，宏margin | +.026839 |
| 正确均值−名称，宏margin | −.004576 |
| 正确胜错配属性 | 12/16 |
| 正确胜名称属性 | 8/16 |
| AE one-vs-rest正支持槽 | 34/48 |
| 同对象名称／正确均值准确率参考 | 40.16%／34.94%（原报告舍入） |
| cross-fit视觉原型两半折属性准确率 | 51.807229%／53.413655% |

screen=False，失败为不优名称；正确胜错配三条件通过。34/48是总体one-vs-rest支持，不等同后续对象平衡双半折mask数量。

当时决定：允许一次明确新假设——筛掉不稳定／有害槽、固定value，而非重训同一个均值版本。新协议阈值是预算选择，可前瞻性设定；不修改旧gate，也不把TRAIN几何诊断称为unseen泛化。{{SOURCE:support}}、{{SOURCE:support_formula}}。

## 14. SG-EKR：TRAIN视觉支持mask并不足以带来描述优势

### 14.1 公式修正与动机

用每pair六图按ID交替分3/3，计算每AE槽同对象正类均值−其余属性负类均值，再对对象平衡平均：

$$
h_{a,r}^{(f)}=\frac12\operatorname{mean}_{o\in\mathcal O_a}
\left[\operatorname{mean}_{y=(a,o)}\cos(f,e_{a,r})-
\operatorname{mean}_{y=(a',o),a'\ne a}\cos(f,e_{a,r})\right].
$$

$$
m_{a,r}=\mathbf1[\min(h_{a,r}^{(0)},h_{a,r}^{(1)})\ge .005].
$$

上式的1/2与实码`vectors=unit_numpy(units)/2`一致；阈值.005作用于半cosine差，不能误写成未缩放cosine仍用同阈值。{{SOURCE:sg_mask_code}}。

名称始终保留，48 AE留30；不从VAL挑mask。固定value=e/2，只训query/key/null/gain：

$$
s_{SG}=B_{name}+\eta\sum_{r:m=1}\alpha_r\langle\mathcal N(f_a(x)),e_{a,r}/2\rangle.
$$

G_name只名称；G_support名称＋合格AE；G_shuffle同接收mask但错配AE。三组同28834参数、seed0、3轮；新CW预算门槛+.1pp，不追溯修改旧+.2。

### 14.2 数据与分析

{{TABLE:SG}}

support−anchor CW+.005123，−name−.222097，−shuffle−.208094；三个收益条件均失败，保护均过。mask不再漂移、梯度已连通，仍未获得正确描述优势。

support选epoch1，η−.033489；跨对象残差变化约1.74%；OW零点变7图、纠正1／破坏2。正支持的固定cosine可以被负gain下调，不能将mask正值直接解释为最终正贡献。

当时建议停止这个三轮GloVe残差版本并考虑视觉原型；后来又明确“三轮失败不否决整条路线”，回到更早v2做一次3→10预算复查。SG本身没有十轮结果。{{SOURCE:sg_mask}}、{{SOURCE:sg_review}}。

## 15. TRAIN视觉原型诊断与未执行的VP方案

### 15.1 已完成的诊断，不是正式VP实验

同498图两半折，以另一半的单位特征形成按对象平衡属性原型：

$$
v_a=\mathcal N\left(\operatorname{mean}_o\operatorname{mean}_{x:y=(a,o)}\mathcal N(f_a(x))\right).
$$

{{TABLE:PROTO_FOLDS}}

与前一步纯名称向量40.16%参考不同，这里A_unit/B_name用组合分数在GT-object下读属性，两者约46%；不能混成同一种名称准确率。

进一步原型库排除查询pair全部六图，原型从该属性其他对象取得；保持对象分组质量、温度.05、cosine证据/2、β=.2条件混合：基线pair38.554217%，正确39.558233%，净5纠正／0破坏；uniform38.554217%，shift1 39.156627%，shift5/9均38.554217%。正确CE2.191094，uniform也达2.204063，说明平滑贡献不能忽略。

有限β网格0/.05/.1/.2/.4，另有加性gain0/.025/.05/.1/.2/.4；全部见附录D。0optimizer、新图0、新VAL/TEST分数0。原型排除查询pair不等于底座没见它；CPF早已训练完整TRAIN，因此不是真正整模型unseen实验。

### 15.2 方案与实际执行分开

VP_global／VP_cross_object、对象名字检索条件、uniform/shifts、新VAL试验当时只提出设计，没有收到对应正式结果。StageA后来复用了强C_text特征缓存思想，但不等于VP正式控制已经执行。

后来的25%概率混合与此处β=.2形式相关，却是不同证据来源／比例／协议，不能相互填成绩。{{SOURCE:sg_review}}、{{SOURCE:prototype}}。

## 16. 训练预算回溯：EKR v2从3轮续至10轮

### 16.1 为什么回头

用户质疑短训可能错过收益。早期MVP最佳5—10轮、三轮loss还降，没有收敛证据。重新审视后承认原来“失败立即否定扩训”的建议太刚性。

只回到完整槽／同构控制的v2，四组从last_epoch3恢复模型、optimizer、RNG，续至10，不同时改底座、value结构、LR、mask或loss。新增28epoch，不是另4个独立run；SG未续训。{{SOURCE:epoch_revision}}。

### 16.2 数据

{{TABLE:ALIGN10}}

名称align从旧epoch0改选6，CW+.353004、OW+.743408pp；fixed仍epoch2 CW+.084116、OW+.243614；正确和错配align仍选0。

正确align1—10 CW全低于anchor，最佳训练轮1仍−.159914pp，epoch10约42.226，后3轮均值低于前3轮；loss早三轮约1.312、后三轮约1.245。名称仅6/7轮CW高于底座，末轮反降，不能将epoch6作为新seed通用最佳轮。

名称选中模型随候选对象变化的残差比仅约.00004%，null近零，收益接近对象无关属性名称对齐；不能称对象条件多描述成功。正确align epoch10跨对象变化约2.19%，仍弱。300原生指标独立复算0差；四组前三轮导入和第三轮分数重放一致。

### 16.3 更新决策

短训确实可能漏正点，名称组就是证据；但正确描述十轮没有延迟收益，不建议继续20/30轮。旧gate保留，不能改用名称成绩填描述主组。转向直接条件属性概率目标，后又因导师初始结合目标优先补B/F。{{SOURCE:align10_review}}。

## 17. 补齐最直接结合：CPF＋LogiCZSL B/F

### 17.1 为什么之前没做、后来为什么补

原0601明确B/F与ACDE并列，早期工程却只实现文本／prior，后续集中归因／条件化，**没有先完整验证直接logic组合**。它是实验覆盖缺口，不能解释成B/F当时已失败或必然不值得。

导师最初希望CPF与LogiCZSL结合，用户要求先CPF B/F而非独立Troika全复现，所以重建逻辑资产、补同期A_ref/E_ref。作者历史数据遗失，能重建方法结构，不能还原逐字原知识或宣称Logic-Troika论文成绩复现。

本地旧LogiCZSL副本存在逻辑被注释、计算total却返回CE、知识字段不一致及路径占位。新adapter明确将logic加入反传；保留原CPF分类损失和legacy推理，不盲用调试副本。{{SOURCE:bf_manual}}。

### 17.2 实际逻辑公式：以服务器代码为准

令$M_q(v)=(\frac1B\sum_i v_i^q)^{1/q}$，q=3；实现用向量范数等价计算，避免零违反度幂根梯度NaN。蕴含$p\Rightarrow q$的违反度为$p(1-q)$。

$$
L_{OI}=\operatorname{mean}_o M_q\big(p_o(1-p_{parent(o)})\big)
+\operatorname{mean}_h M_q\big(p_h(1-\max_{o\in child(h)}p_o)\big).
$$

$$
L_{OE}=\operatorname{mean}_{o,r}M_q\big(p_o(1-p_{OE(o,r)})\big),\quad
L_{AE}=\operatorname{mean}_{a,r}M_q\big(p_a(1-p_{AE(a,r)})\big).
$$

$$
L_{total}=L_{CPF}+.01(L_{OI}+L_{OE}+L_{AE}).
$$

这里明确纠正旧手册OE/AE要点中的语言歧义：“描述支持原语”的文字与代码方向不一致。**实际训练是对象／属性为前件、相应描述为后件**，即上面$p_{primitive}(1-p_{description})$；本总档没有改训练实现或数据，只将解释对齐实码。{{SOURCE:bf_logic}}。

对象／属性用原生分类logits；父类/OE用CPF对象视觉，AE用属性视觉与固定GloVe文本计算logits；原语、父类各自类别softmax，OE/AE形状[B,O,3]/[B,A,3]在类别轴softmax，不是三条描述槽之间softmax。β只乘一次，无额外可训练head／知识CE，无新增推理prior。{{SOURCE:bf_adapter}}。

逻辑bank为5父类、36OE、48AE；B/F共用。E_ref/F的文本／λ=.1仍是旧v1缓存，**不是v2 C_text＋logic**；新逻辑资产与旧text/prior两来源分别冻结。B为纯CPF＋logic；F为E＋logic，不是λ=0的C＋logic。

### 17.3 四组训练数据

UT seed3351，各10轮、batch64、β=.01、q=3；统一Python/NumPy/Torch种子并关闭cudnn benchmark，因此旧同seed A/E不能充当完全同协议对照，重建A_ref/E_ref。

{{TABLE:BF}}

B−A_ref CW−.3552／OW−.3578pp；F−E_ref−.5909／−1.0531。E_ref−A_ref OW+1.1001，但CW−2.6586；没有同期C/D，不能拆分文本与prior。F−A_ref OW+.047不是逻辑有效，真正增量是F−E_ref。

日志末次B CPF2.321346、OI.315724、OE.256479、AE.142717，加权.007149；F CPF2.314919、OI.315391、OE.209030、AE.128332，加权.006528。约主loss .3%，只是标量比，不是梯度比或整轮均值。

### 17.4 分析与决定

直接结合缺口已补，但当前配方无正逻辑增量；单seed小负差不证明普遍有害。先补原始训练审计／少量梯度，不直接改β、加轮、换图或换骨干。实际最佳5/6轮，后续10未超；不能继续声称只因预算短。

原metrics身份／配置审计128项通过，β只有一次、各10轮；包未含B/F完整分数／权重，本地没有独立重算B/F AUC曲线。日志TEST=0是占位，不是新TEST成绩。{{SOURCE:bf_review}}、{{SOURCE:bf_preflight}}。

## 18. B/F最佳权重梯度＋条件属性StageA

### 18.1 梯度诊断

最佳B/F共同12TRAIN图，3batch×4，无optimizer，检查对象／属性映射参数：

| 模型/batch | βlogic梯度范数÷CPF梯度范数% | joint余弦 | AE对CPF属性余弦 |
|---|---:|---:|---:|
| B/0 | .173641 | −.078817 | −.525454 |
| B/1 | .125785 | −.025536 | −.121123 |
| B/2 | .186475 | +.075767 | +.546016 |
| F/0 | .163277 | −.135612 | −.816586 |
| F/1 | .061759 | −.093917 | −.316922 |
| F/2 | .215217 | +.036731 | +.298409 |

支持“这些映射上的加权logic较弱、部分AE方向冲突”，不是整轮／全网络／Adam更新结论，也不证明唯一根因。OI/OE对属性映射零梯度符合其对象侧接线，不代表logic全断。

决定：不盲目放大β，以正确OE/AE直接生成条件属性证据做一次受控目标修正。{{SOURCE:stagea}}。

### 18.2 强基线缓存与真实检查

固定v2 C_text seed0 epoch6，CW45.440387、HM58.083865；OW34.937345、HM48.523090。checkpoint SHA`45de209c4c38af6d7cfa62f6ff51f62ec8f095ee18edc22d893f43184f3199a6`。

先VAL回放12核心指标0差，再缓存22998TRAIN／3214VAL的300维属性特征和完整12×16原分数。缓存来自固定eval变换，后续小head训练不等同CPF完整增强训练。C_text是已含文本知识的强起点，不是原生无知识CPF。

五head各3诊断步共15，丢弃恢复初始，正式训练0、底座更新0；CUDA零增量、梯度、对象边缘恒等通过，误差约5.37e−7；准备实测约11分38秒。基线与B/F不同seed／知识版本，不能横向相减归因。

## 19. 条件属性概率head：从分数残差改为条件目标

### 19.1 精确结构与动机

希望更强地让图像、候选对象、OE/AE共同参与$P(a\mid o,x,K)$，保留已有效对象分组质量，不让新增属性head兼做无约束对象校准。基础全冻结；rank32，温度T=.05、cap=.10。

描述均值仅平均三条单位描述，不重新单位化，不混名称：$\bar k_o=\frac13\sum OE_r$，$\bar k_a=\frac13\sum AE_r$。

$$
q(x,o)=LN_q(U_v f_a(x)+U_q e_o),\quad
k_o=U_n e_o+U_{OE}\bar k_o,\quad k_a=V_n e_a+V_{AE}\bar k_a.
$$

$$
R(o,a)=LN_r(k_o\odot k_a),\quad
\Delta=.10\tanh\left(w_{out}^{\top}[q\odot R]/\sqrt{32}\right).
$$

输出层零初始化，其余非零，初始精确回放。没有自由pair lookup。generic用相同q/k输入改为$LN_r(GELU(W_f[q;k_o;k_a]+b_f))$，再同输出；主乘性交互57760参数，generic60864（多3104），首次不是严格同容量比较。

### 19.2 对象边缘保持公式修正

基础$s_0$完整网格诱导：

$$
Z_0(x,o)=\sum_a e^{s_0(x,o,a)/T},\quad
p_0(o\mid x)=\frac{Z_0(x,o)}{\sum_{o'}Z_0(x,o')},\quad
q_0(a\mid o,x)=softmax_a(s_0/T).
$$

$p_0$是三路融合分数诱导边缘，一般不是CPF object辅助head的softmax。

$$
q_\theta(a\mid o,x,K)=softmax_a((s_0+\Delta)/T),\quad
p_\theta(o,a\mid x,K)=p_0(o\mid x)q_\theta(a\mid o,x,K).
$$

若直接交$s_0+\Delta$给完整网格softmax，会改变对象边缘。因此实际读出：

$$
b_\theta(x,o)=T\left[\log\sum_a e^{(s_0+\Delta)/T}-\log\sum_a e^{s_0/T}\right],\quad
\boxed{s_\theta=s_0+\Delta-b_\theta}.
$$

于是$softmax_{o,a}(s_\theta/T)=p_0(o)q_\theta(a\mid o)$。先完整192网格，后CW/OW mask及bias；bias之后不能继续宣称对象边缘保持。保持对象总质量也不保证最高pair的对象不变。

### 19.3 loss、控制与接线边界

$$
L_{cond}=-\operatorname{mean}_i\log q_\theta(a_i^*\mid o_i^*,x_i,K),\quad
\frac{\partial\ell_i}{\partial\Delta_{i,o,a}}=
\frac{\mathbf1[o=o_i^*]}T\left(q_\theta(a\mid o_i^*,x_i)-\mathbf1[a=a_i^*]\right).
$$

GT-object只用于训练／诊断，部署计算全部对象，不读真对象。某对象下从未作为正标签的属性会持续受到该行直接负压力，但共享属性监督可迁移；CE不是理论上必然失败，实测检验是否抵消这种偏置。

五组P_name、P_K、P_generic、P_perm1、P_perm2；perm只移动AE/OE，名称不动，固定两映射不挑结果。P_name整个系统仍有底座知识，只是不新增描述。{{SOURCE:head_formula}}。

### 19.4 r1失败与r2修复

r1在CUDA恢复自检`weights_only=True`报Unsupported operand71，即pickle BINFLOAT；Torch2.5本地支持、服务器1.13旧安全读取器不支持。临时1诊断步后读取失败，StageB／正式均0。

r2将Python/NumPy浮点元数据编码为类型标记float64标量张量，保留安全读取／protocol2，覆盖selfcheck/last/best，不以`weights_only=False`绕过。旧失败目录独立保留；新包新目录，模型公式／知识gate不改。{{SOURCE:checkpoint_fix}}。

## 20. 条件pilot r2五组十轮：已见拟合和未见迁移分离

### 20.1 预算与主结果

AdamW lr=.001、wd=.0001、batch256，seed0。TRAIN每pair留约20%图做一轮机制检查，fit18426／hold4572；五组共360诊断步，随后head丢弃、重新初始化。正式全TRAIN各10轮×90步，共4500；epoch0—10按最早最大CW选，OW不另选。

{{TABLE:PILOT}}

四乘性交互组选0；最佳实际训练轮均1但CW仍负：name−1.551683、K−1.321926、perm1−1.262791、perm2−1.490229pp。generic选7 CW+.789003／OW−1.129224，CW HM+2.451763，GT-object属性总准确率−1.026759。

### 20.2 每一步数据分析

| P_K指标 | epoch0 | epoch1 | epoch10 |
|---|---:|---:|---:|
| 在线TRAIN条件NLL整轮均值 | 未测 | .765426 | .603195 |
| VAL总NLL | 3.759334 | 3.107583 | 4.032728 |
| VAL seen NLL | 1.008049 | .992843 | .881561 |
| VAL unseen NLL | 4.791800 | 3.901175 | 5.215258 |
| GT-object seen属性% | 79.475485 | 74.572406 | 78.335234 |
| GT-object unseen属性% | 15.703894 | 15.447154 | 10.055627 |
| CW AUC% | 45.440387 | 44.118461 | 42.343292 |
| OW AUC% | 34.937345 | 35.146634 | 28.949296 |

P_K在unseen图的GT-object下，预测属性搭配属于TRAIN的比例79.503637→87.291399%，对应条件概率质量.779986→.867093。epoch10仅纠正3／破坏135 unseen属性预测，净−132。

支持当前条件CE／结构对已见搭配拟合的解释，训练正常却未迁移；名称、错配也有同现象，不是一条知识句独有问题。对象边缘已固定而GT-object也变差，不能只改对象读出解决全部退化。

TRAIN机制hold基线NLL.831376、属性78.237092%；P_name78.718287、P_K78.455818、generic78.324586、perm1 78.674543、perm2 78.740156。K相对name纠正16／破坏28，相对perm1 13/23、perm2 32/45，未显示正确内容优势。hold与fit同pair且底座见过全TRAIN，不能称独立组合泛化。

### 20.3 决策

完整十轮仍未支持P_K，结束此乘性交互扩跑。generic有CW信号但OW保护失败，且原控制不是generic同结构，不能归正确知识；应先统一读出，再补同结构控制。

660原生指标复算0差；对象边缘误差1.847744e−6。缓存训练50轮累计约111.92秒，不含全准备导出，不代表全视觉训练这么快。{{SOURCE:pilot_review}}。

## 21. 温和概率读出修正与Generic同结构内容控制

### 21.1 为什么“cap=.10很小”被修正

温度T=.05，单候选logit最多改±2，两候选相对最多4，赔率可乘$e^4\approx54.6$。有界分数不代表温和分布；P_K epoch1/10联合TV最大约.4751／.6656。

对象边缘保持只保护对象质量，不保护其内部属性分配。此次修正直接用已保存分数检验，不必先重训。

### 21.2 精确混合公式与训练／推理差异

固定α=.25，不扫描：

$$
q_{mix}=.75q_0+.25q_\theta,\quad
p_{mix}=p_0(o\mid x)q_{mix}=.75p_0(o,a\mid x)+.25p_\theta(o,a\mid x,K).
$$

$$
\boxed{s_{mix}=T\log Z_0(x,o)+T\log q_{mix}(a\mid o,x,K)}.
$$

完整网格TV≤.25；不是AUC／准确率不下降保证。logsoftmax/logaddexp稳定实现；α0／关闭直接返回原分数。**训练仍优化未混合qθ的条件CE，仅读出变混合**。没有训练KL，也没改为混合NLL；后续若改loss须另立所有控制同步训练。

### 21.3 无训练回放的数据

五组epoch0—10共55分数事后开发回放，0optimizer；原失败主表/gate保留。

{{TABLE:MIX_SELECTED}}

这里P_name/P_K/两perm仍是乘性结构，P_generic仍是generic。表由固定混合回放JSON直接读取，全部55个epoch读出见附录E。{{SOURCE:mix_data}}。

同P_K第1轮原CW−1.3219变+1.0463，仅改读出，说明其输出有可用排序信号。但正确低于perm1，仍不支持内容归因。

另一固定25%logit残差缩放在P_K epoch1 CW+.3759／OW+.4938，也改善。算术混合有明确保留比例／TV界，因此选它；不表示搜索证明它普遍最佳。α虽然未扫描，方案仍是看过旧VAL后的开发修订，不是独立确认。{{SOURCE:mix_replay}}。

### 21.4 三个新Generic控制与主结果

将架构与内容模式解耦，四组都用generic60864参数，同初值、同缓存／顺序／十轮；G_K复用原P_generic900步，其未混合逐轮分数哈希一致；G_name/G_perm1/G_perm2新2700步。不能把旧乘性P_name当generic名称控制。

{{TABLE:GENERIC}}

G_K−name CW+.0209，−perm1−.0031，−perm2+.2198；−两perm均值+.1083。gate要求对基线≥+.2，对name及perm均值≥+.1，保护项均过，唯一失败是name优势不足。两perm OW更强。

四组GT-object属性%均不胜基线33.1052：K32.8874、name33.0118、perm1 32.9496、perm2 33.0741。epoch10 NLL K.6474、name.6435、perm1 .6434、perm2 .6446，几乎同向拟合。

1460检查／528原生指标复算0差；新增head无新视觉提取、骨干0步。原900步与2700新步不重复相加成四组之外更多实验。

### 21.5 分析与决定

**有单seed工程改进，未显示新增正确OE/AE内容作用。** 更合理称通用视觉条件适配／校准；不能称整个系统“无知识”或“知识完全没用”，因为共同底座已经是C_text。

内容相近成绩与GT-object未升支持通用容量／排序校准解释，但“网络忽略描述”未被直接因果证明。全强度与25%主表还分别选7/10轮，不能把两张最佳表相减当同权重纯读出因果；同一轨迹的逐轮回放才隔离读出。

归档当前内容分支；G_name＋25%可独立确认另外两个视觉seed，是性能支线，不是知识验证。至本档未执行该确认。原报告对“随机图必退化成随机正则”的措辞也不应泛化：本实验只是描述错配，不是图方法，真正USG需独立实现检验。{{SOURCE:generic_review}}。

## 22. 从内容head转向冻结CLIP证据：两个探针与一次审查

### 22.1 决策动机及未执行的后续教师

Generic同涨后，提出先查描述能否提供图像证据，而非继续让灵活MLP吸收任意类向量。早期草案建议完整CLIP-CPF＋LogiCZSL；用户随后明确不必完全结合，于是保留原CPF、先独立冻结教师，再考虑属性模块监督。

后续草案$L_{cond}+\lambda KL(stopgrad(q_T)\|q_S)$及R1/R_name/R_K/两perm五训练组**没有执行**。不能把七种冻结文本配方写成七模型训练或教师蒸馏成功。若教师只训学生后推理移除K，应称$P(a\mid o,x;\theta_K)$，不是推理时可干预K的模型。{{SOURCE:route}}。

### 22.2 CLIP v1划分assert错误及修正

旧代码错误要求TRAIN pair∩VAL pair为空，碰到正常seen组合而停止，编码前失败。r2改为15seen／15unseen，继续核对图像ID／字节哈希无重叠、原语／metadata／顺序，而非删检查。模板、温度、描述、置换、gate不变；不需重跑CPF或StageA。{{SOURCE:clip_fix}}。

### 22.3 冻结CLIP r2的精确公式

OpenAI ViT-L/14、224px，源commit`a1d071733d7111c9c014f024669f959182114e33`；权重SHA`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`。全冻结0optimizer。

名称模板`A photo of {object} made of {attribute}.`；正确AE加`It has {AE}.`；OE再加`Object shape: {OE}.`。每候选统一9等权文本单元：OE×AE 3×3；AE-only每条重复3次；名称重复9。错配只移动补充描述，真实名称不动。

$$
c_T(x,o,a)=\frac19\sum_{r=1}^9\mathcal N(v_{CLIP}(x))^\top\mathcal N(t_{CLIP}(prompt_{o,a,r})),\quad
q_T(a\mid x,o)=softmax_a(c_T/.05).
$$

**平均文本单位向量后不重新归一化**，等价逐句cosine均值。不同于GloVe静态知识聚合；若均值再次归一化会换配方。全12×16先算，GT-object仅诊断截取。温度影响分布尖锐度，不改变固定分数argmax。{{SOURCE:clip_prompts}}。

### 22.4 七配方的数据与分析

{{TABLE:CLIP}}

均是GT-object、全部16属性候选；不是端到端CPF／CW/OW。micro按图片，pair macro按组合等权。

K在VAL unseen相对名称macro+.3859但micro−12.9653；相对两perm macro−.8699/−2.6172。AE-only相对名称micro−2.4818／macro−.9793。gate要求正确unseen macro高名称及每套对应perm至少1pp，micro与TRAIN macro不低名称，两gate均False。

K相对名称unseen纠正169／破坏472净−303；AE纠正115／破坏173净−58。Synthetic×Sandals767图，名称判对386、K61，净−325；其余合计补+22。Suede×Oxfords165图9→81，Nylon×Slippers25图0→13说明并非全面负。大类损失和小类改善可导致macro／micro异向，不能删失利类。

冻结名称CLIP unseen28.3269 vs CPF15.7039，净+295（纠正637／破坏342），但seen CLIP7.8677 vs CPF79.4755，VALall22.7442 vs33.1052。编码器、文本、预处理、读出同时变化，不是公平的纯视觉骨干质量对照，也不能保证蒸馏／混合端到端涨点。

54,355检查；七组cosine最大差2.98e−7、q差0，所有top1／macro／混淆一致；TRAIN/VAL ID与已记录图像字节哈希交集0。只15unseen pair不确定大，描述性pair bootstrap K−name macro区间约[−12.11,+13.19]pp，不称显著或普遍有害。

决定：当前均值教师不进入五组监督训练。只做一次有边界TRAIN逐句／图片适用性审查，完成就停，不以审查代替无限prompt搜索。{{SOURCE:clip_review}}。

### 22.5 TRAIN逐句／96图审查：最新结果

旧r2只保存均值文本向量，没有逐句向量；重新编码同7041唯一文本，复用TRAIN图像特征，不重跑图像encoder。9 OE×AE位置与3 AE位置，各正确＋2perm，共36视图；各属性固定6TRAIN图共96，按对象轮转／ID哈希，非按对错选择。

{{TABLE:TRAIN_AUDIT_OVERVIEW}}

九种正确OE＋AE单位置macro19.0005—21.2936，全部低名称。AE第三条macro相对name+1.051436、perm1+5.288841、perm2+5.861429；这是真实局部内容线索，不能否认。

AE3修复4668／损害896，净+3772；所有属性计数见附录F。Leather13204图（57.4%TRAIN）3→3296，净+3293，占**净增判对**87.3%，不是占纠正样本87.3%。Suede净+984，Synthetic−408、Nubuck−87、FauxLeather−99、Fullgrain−51、Satin−120。

Leather十组合对总macro差贡献+3.0059pp，Suede+2.3071，Nubuck−4.2360；整体只+1.0514。描述性排除Leather：9794图／73pair，name macro25.438042、AE3 23.215879，差−2.222164。不是新主指标、更不是删类授权。

第三条是字典位置，不是统一视觉描述类型；全12位置审查后发现最高点，是事后候选，不能把同一TRAIN当独立确认；未见组合性能未知，未挑其VAL或蒸馏。

### 22.6 图片观察、审计与最终决策

已看16图板／96图；部分毛绒／光泽／绒面可见，说明知识非全抽象；Synthetic包括模塑鞋、亮面高跟、系带鞋等，不是一种统一纹理；Leather/FauxLeather/Fullgrain及Nubuck/Suede外观重叠；Sheepskin/Wool有内衬／领口与鞋面区域差异。

仅照片不能确认真皮来源、fullgrain或标签到底指哪一部位。Cotton17122外观不典型也不能认定错标；没有材料元数据核验，不能改标签／删样本。拼图标题预测来自name/K均值/AE均值，**不是AE第三条**预测。

0optimizer、图像encoder0、TRAIN解码96、VAL/TEST读取0；8,839审计通过，七均值文本逐元素复原。36单句33混淆精确，另3各约1张近同分差，最大micro .00435pp／macro .00054pp；AE3及控制精确。跨运行时近同分容差已声明，不称全部单句逐图零误差。

当前决定：结束均值教师扩跑，AE3保存固定候选，不立即蒸馏、不逐类选句、不再VAL改prompt。下一优先是共享视觉属性读出及整组合留出诊断；保留CPF概率接口。{{SOURCE:train_audit_review}}、{{SOURCE:train_audit}}。

## 23. 具体公式修正总表：修正的是哪一个问题

| 阶段 | 旧／原设想 | 实际修正 | 动机及得到的证据 |
|---|---|---|---|
| 早期知识 | 原始name | 逐句单位向量的name＋3desc聚合 | UT有工程收益；还不能归正确内容 |
| K规范化 | min-max文本相似度像“概率” | 明确为相似度；总体centered prior，退化归零 | K不是现实feasibility；非零λ不被选择 |
| 归因 | C−A直接代表知识 | 加A_unit和只换desc的shuffle | 正确内容优势不足；排除名称尺度混杂 |
| 原feature示例 | 静态$h(pairText)$ | 另立图像×候选对象条件残差探索 | 不把静态pair bias当视觉条件机制 |
| EKR锚点 | $s_C+\eta\Delta$ | $s_{Aunit}+\eta\Delta$ | null关闭回名称，不固化C的错误 |
| EKR value | clean中心化、name原值 | 共同冻结name底座＋原始槽＋共享$UD$对齐 | 修正名称值几何混杂；避免name−name全零 |
| 支持筛选 | 全槽自由对齐 | TRAIN双半折mask＋固定value | 部分槽有害；但SG依然不胜控制 |
| 预算 | 3轮筛选即强止损 | 同一v2状态续3→10，其他不改 | 名称延迟收益，纠正短训过强判断 |
| logic适配 | 旧副本算logic但反传CE | 原语／父类／描述视觉概率＋total反传 | 接通有梯度；B/F无正增量 |
| 条件目标 | 组合CE分数残差 | GT-object全属性条件CE | 对象／属性问题分开；实测已见偏置 |
| 对象边缘 | 直接$s_0+\Delta$ | 减对象行log-partition差$b_\theta$ | 完整网格对象质量恒等，不等于pair对象不变 |
| 温和幅度 | cap .10被视为保守 | $q_{mix}=.75q_0+.25q_\theta$ | 低T使原分布改动过大；混合有TV界 |
| 内容控制 | 只有P_generic正确 | 同generic name／两个perm | 四组同涨，新增描述不是必要条件 |
| 监督候选 | 灵活head只读固定文本 | 冻结CLIP样本相关文本证据探针 | 先查额外可迁移信息；均值教师未获支持 |
| 文本集合 | 单一平均原型 | 逐句单位cosine均值；再审查固定位置 | AE3局部有信号；不能立即升格teacher |

上述修正中，一些是模型结构改变，一些是评价／工程错误修复，一些只是解释边界修正。它们不能都被统计为一个“最终Full Model”，也不能从后续正点回填旧实验。

## 24. 为什么久久没形成满意结果：证据强度分层

### 24.1 相对明确的观察

- 知识已经进入模型／训练，多个真实梯度和分数回放检查排除了简单“开关没开”解释。
- 当前template K是静态文本自一致标量，不是任务校准的兼容概率；所有正λ选不出，不能继续默认它可靠。
- 正确内容对照优势比“添加文本／head／混合”的性能优势弱；类向量作为代码、额外容量、平滑／校准都是必须排除的解释。
- P_K TRAIN目标改善与VAL unseen退化同时出现，并更偏TRAIN搭配；共享参数形式并不自动保证组合迁移。
- 新head原cap/T大，限制分数不够限制分布；固定混合确实使已有输出更可用。
- 整图CLIP＋等权描述也未形成可靠新增知识教师；句子适用区域、相近材质外观、类偏好都值得关注。

### 24.2 尚是机制假设，不能写成唯一根因

GloVe弱表达、局部材质视觉不足、原语语义重叠、典型描述不适用所有图、允许映射吸收任意向量、logic梯度冲突等，都有部分支持或合理性，但没有完整单因素因果消融。

现有结果不能证明“更强骨干必涨”“换随机图必涨”“知识普遍无效”“CE不适用CZSL”。CPF缓存与CLIP probe同时改变编码器、读出和监督历史；不公平地比较数字会得出错误骨干结论。随机图尚无实验，不应拿shuffle高分替代它。

### 24.3 更核心的研究困难

我们多次检验了“是否能让知识参与”，却较少证明“知识在当前样本可辨、相对名称有新增信息、这种信息能跨对象迁移”。这三层是递进但不同的条件：句子通顺／合理 → 可见且有区分力 → 未见组合迁移 → 端到端性能／机制归因。

最新AE3说明第一轮均值配方不是充分表达，但也显示最佳局部信号高度集中；不能简单“挑最好句”后跳过迁移检验。下一步最应隔离视觉可读出性／组合迁移，而不是同时改文本、loss、骨干、graph和融合比例。

## 25. 研究决策中需要保留的自我修正

| 过去容易出现的说法／决策 | 本档修正 |
|---|---|
| ACDE足以覆盖最初结合 | 不足；原B/F遗漏，后来补齐 |
| CW归因gate没过即动态条件知识不可能 | gate仅固定预算选择；另立假设可探索，旧False保留 |
| 3轮无收益就不值得任何扩训 | 太强；同v2十轮名称出现延迟收益，但正确描述仍无 |
| value drift已证明破坏语义 | 只观察漂移与退化同时发生，没有唯一因果证据 |
| 有null就能拒识并回名称 | 原null回C_text；名称锚点虽修正退避，null实际未表现拒识 |
| 冻结小head出现非零视觉交互就知识有效 | 接线不等于迁移／归因，必须比名称／同结构错配 |
| cap .10意味着温和更新 | 忽略T=.05；分布可能大改；需实际TV或混合比例 |
| generic gate False就是工程失败 | 单seed性能正信号保留，失败的是独有新增内容门槛 |
| 显式正确>错配loss必然修复知识 | 可以记类向量映射，需冻结／共享约束和未拟合组合识别证据 |
| CLIP名称unseen高所以骨干更好 | seen/all显著不同、接口多因素变化，不能归纯骨干 |
| TRAIN AE3高即教师成功 | 事后候选、收益集中、未见迁移未知，不进入蒸馏 |
| 修复检查点／划分assert就实验转正 | 工程问题与科学负结果分离，旧失败取证保留 |

“复现完成”“性能改善”“正确知识贡献”“特定机制成立”“论文新颖性”应分开判断。恢复原方案完成任务无需结果正数；研究投入选择可以根据负证据止损；任何一种gate都不能代替其它目标。{{SOURCE:goals}}。

## 26. 原方案完成度、尚未做项及当前下一步位置

### 26.1 已经完成的身份，不冒充统一主表

| 原身份／任务 | 当前状态 |
|---|---|
| A | 三数据集历史CPF，UT多批次名称参考已执行 |
| B | UT3351 CPF＋OI/OE/AE logic已执行 |
| C | 三数据集早期C、UT静态v1/v2三seed已执行 |
| D | UT早期λ=.1 calibration已执行 |
| E | UT早期E及同期E_ref已执行 |
| F | UT3351 E＋logic已执行 |
| G/H原配置 | 未完成：pair-knowledge factorized head＋text/prior及H logic联合链 |
| A–H统一同协议主表 | 未完成；旧C/D与B/F参考不是完全同期随机设置 |
| 五sanity checks | 描述shuffle充分；完整G/H的random prior、uniform真实回放、no-pair等未闭合 |
| 新模型TEST／跨数据／多视觉seed确认 | 除历史CPF TEST外未完成这些最终新方法评价 |

如果目标是补原0601核心复现，应实现原pair-knowledge head、补C_ref/D_ref/G/H与统一表，而不是把当前generic四组填原G/H；若目标是获得可信的新迁移机制，当前建议则为共享视觉读出。两目标不同，不能悄悄互换。{{SOURCE:repro}}。

### 26.2 还没有执行的方案

原正式C_text锚点EKR全矩阵／多seed确认；USG随机语义图；正式VP_global／cross-object的VAL控制；SG十轮；generic另外两个独立视觉seed；完整CLIP-CPF重建；教师KL五组；AE3的独立迁移确认；新的共享视觉属性读出。这些都不能列为既有训练。

原方案的learnable λ、clamped-log prior、LLM-only/mixed feasibility、margin/object-marginalized属性loss、delayed logic、OI深度／描述1/3/5数量／10/30/50%噪声等，没有系统结果，不补造。

### 26.3 截至最后一轮的优先方向

先利用冻结CLIP特征检验受控共享属性读出，不输入真实对象，容量、权重／预算和整组合留出事先锁定；它先回答视觉材质可辨性与迁移，不是完整条件组合模型。既有TRAIN已用于开发，新内部留出也不是完全未接触的最终验证。

冻结CLIP未接受本CZSL任务TRAIN标签优化，可用于新readout组合留出；旧CPF见全TRAIN，其缓存只能作为适配器参考，不是公平整模型未见骨干对比。读出有迁移信号后再检验对象条件、接CPF概率接口／温和混合；知识增量另做有限名称与两perm控制。

这里仅记录最新建议，未交付新训练代码／启动命令。AE3固定候选与generic性能支线保留，不把其余负结果抹去。最新依据：{{SOURCE:train_audit_review}}。

## 27. 工程故障、重复执行与审计边界

| 事件 | 实际含义／处理 | 与科研结论的关系 |
|---|---|---|
| /data满、Errno28 | 字节空间耗尽，保留已完成结果／checkpoint；新输出到独立位置 | 中断不是模型无效；恢复不能记成独立seed |
| base pip／系统预加载异常 | 固定cpf Python3.8，不覆盖环境 | 系统异常不由模型解释；未证明最终根因全查明 |
| TRAIN导出旧路径／两映射 | 检查所有实际路径，保留异常与修复记录 | 不自动换源图或改训练数据 |
| YACS／top5审计不兼容 | 修复审计工具，不改指标 | 技术通过不把研究False变True |
| StageA重复启动STOP已有路径 | 首次启动已创建；查看进程与日志，避免重复覆盖 | STOP不等于首次任务失败 |
| probe_bf进程消失后prepare出现 | BF probe完成、进入缓存准备 | GPU单时刻0利用率不能判断任务失败 |
| pilot r1安全pickle错误 | 兼容修复编码为张量，独立r2 | 正式训练0，失败包不是负性能实验 |
| CLIP TRAIN/VAL pair overlap | 正常广义seen划分，旧assert错误 | 修复图像隔离检查，不隐藏泄露 |
| export execution_passed True | 程序完整执行成功 | gate False仍可能真实；不等于性能／知识有效 |

各阶段大量审计从保存分数独立重算原生评价器，**不是本地GPU重新推理权重**。B/F本地缺完整分数，主要核验原metrics来源；早期日志曲线含舍入值；TRAIN审查单句存在近同分边界。来源hash／科学数值／GPU重跑是不同层级，不夸大验证范围。

服务器主要运行环境Python3.8.18、Torch1.13.1+cu117、Torchvision .14.1+cu117、RTX3090，解释器`/data/xhw/miniconda3/envs/cpf/bin/python`。本地CPU工具环境不同，合成测试不作真实实验成绩。所有旧release、失败attempt、完整权重应继续保留；本档只新增汇总。

## 附录：完整数据查阅表

### 附录 A：静态v2所有seed与CW/OW主结果

以下24行直接来自冻结runs.csv，不由均值反推。{{SOURCE:v2}}。

{{TABLE:V2}}

### 附录 B：已有历史CPF TEST参考

只记录归档参考，不与新增VAL成绩相减，也不充当新方法TEST。{{SOURCE:early_test}}。

{{TABLE:TEST}}

### 附录 C：已保存的MVP、名称EKR、共享对齐和SG全部逐轮轨迹

MVP曲线来自舍入训练日志，loss为末次打印的平均而非独立整轮重评；不可用其取代全精度选中metrics。{{SOURCE:early_epochs}}。

{{TABLE:EARLY_EPOCHS}}

名称EKR各三轮；其loss为训练记录：{{SOURCE:name_epochs}}。

{{TABLE:NAME_EPOCHS}}

共享值对齐v2四组完整1—10轮，前三轮为原结果导入，4—10新续训；同一批run：{{SOURCE:align10_epochs}}。

{{TABLE:ALIGN_EPOCHS}}

SG三组各三轮，没有十轮SG：{{SOURCE:sg_epochs}}。

{{TABLE:SG_EPOCHS}}

名称EKR机制诊断全部27个视图。TRAIN视图没有原生CW/OW AUC，因此留“—”，不是0；net_correct为相对其共享冻结A_unit锚点，而非每次干预相对已训练clean。

{{TABLE:NAME_DIAGNOSTICS}}

B/F四组各十轮的CW轨迹，直接从实际服务器`train.log`解析；原日志AUC/HM只有四位小数，换成百分数后仅有约.01pp分辨率。最终选中成绩以第17节全精度metrics表为准，不从日志舍入值重选权重。日志的`Val/Test`是通用打印标题，本次实际评VAL，不能当TEST成绩。

{{TABLE:BF_EPOCHS}}

B/F每轮末次`BF_LOGIC`打印的batch损失。它是第283个iteration的单batch，不是轮末360个batch平均，也不是最佳checkpoint重新评价。

{{TABLE:BF_LOSSES}}

### 附录 D：TRAIN视觉原型排除查询pair的全部开发网格

不代表unseen整模型泛化，0optimizer。以下保留所有正负／uniform／shifts，不只报β=.2。{{SOURCE:prototype}}。

{{TABLE:PROTO_ALL}}

同证据直接分数相加：{{SOURCE:prototype_add}}。

{{TABLE:PROTO_ADD}}

### 附录 E：条件head与Generic混合的所有epoch主指标

P系列来自各自history.json中的epoch0—10账本；T=.05对象边缘保持、未混合全强度读出。{{SOURCE:pilot}}。

{{TABLE:PILOT_EPOCHS}}

同五组全部55个epoch改为固定25%算术条件概率混合后的回放（不新增训练），可与上表逐个同epoch比读出。{{SOURCE:mix_data}}。

{{TABLE:MIX_EPOCHS}}

Generic混合三个新训练组全部epoch0—10；G_K原P_generic轨迹复用，单列读出回放，不冒充新训练。{{SOURCE:generic}}。

{{TABLE:GENERIC_EPOCHS}}

{{TABLE:GK_EPOCHS}}

### 附录 F：TRAIN逐句审查所有36视图与AE第三条类别计数

1-based位置，对全部候选统一用相应位置，不是为每个GT类挑最佳句。只用于同TRAIN组合描述统计。{{SOURCE:sentence}}。

{{TABLE:SENTENCE_ALL}}

AE第三条相对名称的全16属性计数：

{{TABLE:AE3_ATTRS}}

### 附录 G：可继续追溯的关键原始记录

- 原方案、早期构建／公式／三seed v1保留汇总：{{SOURCE:original}}、{{SOURCE:old_history}}。
- 全精度早期A/C/D/E与后处理：{{SOURCE:early}}、{{SOURCE:lambda}}。
- v2配对与逐图：{{SOURCE:v2_delta}}、{{SOURCE:v2_predictions}}。
- 名称EKR完整机制诊断表（27视图）：{{SOURCE:name_diag_data}}；解释：{{SOURCE:name_diag}}。
- 对齐3轮与10轮主表：{{SOURCE:align3}}、{{SOURCE:align10}}。
- SG／mask／原型：{{SOURCE:sg}}、{{SOURCE:sg_mask}}、{{SOURCE:prototype}}。
- B/F真实主表与实际损失源码：{{SOURCE:bf}}、{{SOURCE:bf_logic}}。
- StageA／head公式／pilot完整审阅：{{SOURCE:stagea}}、{{SOURCE:head_formula}}、{{SOURCE:pilot_review}}。
- Generic／CLIP／TRAIN审计原始JSON：{{SOURCE:generic}}、{{SOURCE:clip}}、{{SOURCE:train_audit}}。
- TRAIN均值与单句完整统计／混淆：{{SOURCE:pooled}}、{{SOURCE:sentence}}。

本档对既有解释进行了边界修正，不覆盖历史报告；如个别旧文档把未执行项写成“下一步”，以本档截至日期的执行状态与原始complete/账本为准，不机械重复旧命令。以后新增实验应续写新版本，保留本版冻结数据与来源身份。
