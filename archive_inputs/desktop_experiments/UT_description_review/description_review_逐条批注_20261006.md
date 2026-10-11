# Frozen descriptions and shuffle donors — 逐条文本批注

审查日期：2026-10-06。原84句及供体关系保持不变，中文批注为新增内容。
这是助手完成的文本审查，不是训练图片逐张人工验收。原 Review 复选框保留未勾选，不能据此自动修改 human_reviewed/approved 状态。
整体结论与下一步见 [UT描述审查报告](<C:/Users/llenovo/Desktop/UT_description_review/UT描述审查报告_20261006.md>)。

主判定：可保留 / 过泛或近义 / 非直接视觉 / 限定过窄 / 应修正。P1 优先处理；P2 核验或优化。“可保留”是合理原型候选，不代表已验证增益或所有样本成立。

## a: Canvas (shuffle donor: Suede)

- woven canvas fabric

  审查 1：**可保留，P2**。名称与机织布结构相符；但仍需确认图片分辨率能呈现织纹。
  建议：保留织纹方向，后续以训练图核验可见性。

- coarse textile surface

  审查 2：**过泛/近义，P2**。粗糙织物也可能属于其他纺织材料；并非 Canvas 的充分识别条件。
  建议：与 Cotton、Nylon 的实图并排检查；不要只依靠 coarse。

- matte cloth finish

  审查 3：**过泛/近义，P2**。哑光布面常见，但 matte、cloth 在多个材质中都适用。
  建议：作为辅助线索；若改写，应补充经图像确认的织纹信息。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：布面与绒面皮革有区别，但 matte 等泛词仍可能重合。

## a: Cotton (shuffle donor: Nylon)

- soft cotton fabric

  审查 1：**非直接视觉，P2**。棉的来源与柔软触感不能仅由单张鞋图确认；句子主要复述名称。
  建议：名称锚点可以保留；视觉描述需另作训练图核验。

- breathable textile surface

  审查 2：**非直接视觉，P1**。breathable 是透气性能，不能当作可直接看见的表面特征。
  建议：移出视觉描述；不要用同义性能词替换。

- fine woven cloth

  审查 3：**限定过窄，P1**。Cotton 是纤维类别，fine woven 只覆盖部分织造与粗细形式。
  建议：避免把细密机织结构写成全部 Cotton 的定义。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：两者都是布面候选；纤维来源差别不一定可见。

依据与边界：纤维类别不等于一种组织：官方样布中同时存在棉针织和棉机织。此证据只支持类别覆盖提醒，不证明 UT 样本有哪一种。[CottonWorks 样布](<https://cottonworks.com/product-innovation/digital-product-creation/>)。

## a: Faux.Fur (shuffle donor: Canvas)

- fluffy synthetic fur

  审查 1：**可保留，P2**。蓬松毛状外观有用；synthetic 的真实性不能只凭图片证明。
  建议：保留毛状外观，将人造来源视作标签知识。

- plush hairy texture

  审查 2：**过泛/近义，P2**。plush/hairy 与第一句的 fluffy/fur 主要是同一毛绒线索。
  建议：核验后可改为毛长或表面密度等不同维度；不得凭空编造。

- fuzzy pile surface

  审查 3：**过泛/近义，P2**。fuzzy/pile 与前两句高度同义；有信息，但没有新增独立维度。
  建议：保留一条核心毛绒线索，减少三次重复强调。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：毛绒与机织布有明显语义差别，部分 textile 词仍共享。

## a: Faux.Leather (shuffle donor: Hair.Calf)

- imitation leather

  审查 1：**过泛/近义，P2**。imitation leather 基本是类别名称释义，不是名称之外的视觉证据。
  建议：可以承担名称释义；不要把它计作新增可见知识。

- smooth synthetic hide

  审查 2：**限定过窄，P1**。smooth 是部分表面形式；synthetic hide 还容易让人混淆真皮与仿皮的来源。
  建议：改用明确的仿皮表面用语，不要求全部光滑。

- leather like finish

  审查 3：**过泛/近义，P2**。leather like finish 合理但很泛，且与第一句重复；可能与 Leather 接近。
  建议：保留为弱线索；真伪区分能力需承认局限。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：仿皮与毛皮有较明显表面差别。

## a: Full.grain.leather (shuffle donor: Nubuck)

- natural full grain hide

  审查 1：**可保留，P2**。全粒面来源表述基本合理；加工定义无法完全从图片识别。
  建议：保留标签定义；避免把来源真实性视为视觉证明。

- visible leather grain

  审查 2：**可保留，P2**。可见粒纹是合理候选；涂饰、距离、分辨率都会影响可见性。
  建议：保留为典型线索，检查训练图是否确有粒纹。

- durable textured leather

  审查 3：**非直接视觉，P1**。durable 是耐用性能；textured 过泛，未突出全粒面的加工区别。
  建议：删除耐用性词，另行核验粒纹描述；不要机械换成另一个泛词。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：同属皮革，主要差别是粒面的加工；是近邻交换。

依据与边界：全粒面的关键是粒面未经修正性机械去除；耐用并非该定义。[LWG 术语](<https://www.leatherworkinggroup.com/learn-more/features/74-essential-leather-industry-terms/>)。

## a: Hair.Calf (shuffle donor: Sheepskin)

- short haired calfskin

  审查 1：**可保留，P2**。保留短毛的皮面是这一名称对应的重要线索；来源仍属标签知识。
  建议：优先保留 hair 与短毛表面信息。

- fine animal hair surface

  审查 2：**可保留，P2**。细短动物毛的表面线索合理；animal 的真实性不能直接从图中验证。
  建议：保留可见毛丝描述，核验是否能与毛绒材料区分。

- velvety hide texture

  审查 3：**过泛/近义，P1**。velvety 太泛，容易与 Nubuck、Suede 的短绒皮面混在一起。
  建议：优先描述可见毛丝，避免只写“天鹅绒般”。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：都涉及皮与毛；毛长、纤维形态及可见部位需要细分。

依据与边界：LWG 的 Cavallino 说明保留短毛的皮面，这支持关注 hair，而不是仅写 velvet；此处不将两个术语强行完全等同。[LWG 术语](<https://www.leatherworkinggroup.com/learn-more/features/74-essential-leather-industry-terms/>)。

## a: Leather (shuffle donor: Faux.Leather)

- genuine leather

  审查 1：**过泛/近义，P2**。genuine leather 是真实性与名称释义，不提供可靠的图片真伪判据。
  建议：可作类别定义；不要声称能据此区别仿皮。

- smooth animal hide

  审查 2：**限定过窄，P1**。smooth 只适合部分皮革表面；animal hide 属来源知识。
  建议：去掉类别必须光滑的暗示，检查实际表面类型。

- natural grain finish

  审查 3：**限定过窄，P1**。natural grain 不一定在涂饰后的 Leather 上清晰可见。
  建议：区分“皮革来源”与“图中可见天然粒纹”，后者要核验。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：真皮与仿皮外观可能相近，genuine/imitation 不是直接视觉证明。

依据与边界：皮革涂饰可能遮盖天然粒纹，因此可见 natural grain 不是所有皮革的要求。[LWG 术语](<https://www.leatherworkinggroup.com/learn-more/features/74-essential-leather-industry-terms/>)。

## a: Nubuck (shuffle donor: Satin)

- sanded leather

  审查 1：**可保留，P2**。打磨皮面的方向基本合理，但 sanded leather 也可能描述其他磨面皮革。
  建议：保留加工语义，增加粒面信息前先看实图。

- soft velvety grain

  审查 2：**可保留，P2**。velvety grain 相比只写 velvet 更接近其粒面磨绒特点。
  建议：保留为典型候选；soft 仍是触感推断。

- matte brushed hide

  审查 3：**过泛/近义，P2**。matte/brushed/hide 与 Suede 等类别共享，且与前两句接近。
  建议：避免仅靠这组三个泛词来区分 Nubuck 与 Suede。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：磨绒皮面与光泽织物区别较大，但 soft/smooth 等词可能接近。

依据与边界：粒面磨绒是合理方向，但与 Suede 的可见差异还需看图。[LWG 定义表](<https://www.leatherworkinggroup.com/fileadmin/uploads/lwg/Knowledge/LWG_List_of_Definitions_v1.1.pdf>)。

## a: Nylon (shuffle donor: Full.grain.leather)

- lightweight nylon textile

  审查 1：**非直接视觉，P2**。lightweight 是质量或触感推断；nylon textile 主要复述名称。
  建议：重量不作为视觉依据；关注经图像确认的布面。

- smooth synthetic fabric

  审查 2：**过泛/近义，P2**。光滑合成织物很泛，与 Synthetic 及其他人工纤维材料重合。
  建议：可作弱候选，不能声称是 Nylon 独有特征。

- tightly woven sheen

  审查 3：**限定过窄，P1**。tightly woven 与 sheen 都只描述部分表面形式；光泽不等于尼龙身份。
  建议：不能把紧密机织和光泽写成全部 Nylon 的要求。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：织物与皮革来源有差别，但 smooth 等泛词重合。

## a: Patent.Leather (shuffle donor: Rubber)

- glossy coated leather

  审查 1：**可保留，P2**。高光涂层是合理的外观方向。
  建议：保留，检查反光区域在训练图片中是否清楚。

- mirror like shine

  审查 2：**可保留，P2**。镜面般反光是较强视觉线索；仍受光照和拍摄角度影响。
  建议：保留，但不要将任意亮面材料都归为漆皮。

- lacquered hide finish

  审查 3：**过泛/近义，P2**。涂饰表述基本合理，和前两句主要围绕同一亮面特征。
  建议：不必判错；若扩充知识，应补充另一条经核验的表面线索。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：高光皮面与橡胶不同，但二者可共享平滑成型表面。

依据与边界：镜面般涂层外观与术语定义吻合。[LWG 术语](<https://www.leatherworkinggroup.com/learn-more/features/74-essential-leather-industry-terms/>)。

## a: Rubber (shuffle donor: Cotton)

- flexible rubber material

  审查 1：**非直接视觉，P2**。flexible 是机械性能；形变通常未在静态商品图中发生。
  建议：保留材料名称，柔性只作非视觉背景知识。

- matte elastic surface

  审查 2：**非直接视觉，P2**。elastic 是性能，matte 也不是全部橡胶的必要外观。
  建议：减少性能词，哑光描述另作样本核验。

- molded waterproof texture

  审查 3：**非直接视觉，P1**。waterproof 是防水性能，不存在可直接等价于防水的 texture。
  建议：删除 waterproof；成型表面只在训练图支持时保留。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：橡胶与棉织物有差别；需要区分鞋面与鞋底位置。

## a: Satin (shuffle donor: Synthetic)

- smooth satin fabric

  审查 1：**可保留，P2**。平滑缎面是合理方向，smooth 仍与多个织物类别共享。
  建议：保留缎面语义；仅视为典型外观。

- silky lustrous weave

  审查 2：**可保留，P2**。光泽织纹是有用方向；silky 这里应理解为丝般外观，而非一定是蚕丝。
  建议：保留光泽方向，避免加入 silk 纤维必然性。

- reflective textile sheen

  审查 3：**过泛/近义，P2**。reflective/sheen 的信息与上一句重叠，主要再次描述光泽。
  建议：减少近义句重复；核验后再补充不同结构信息。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：Synthetic 是宽材料概念，与人造缎面可能有范围重叠。

## a: Sheepskin (shuffle donor: Wool)

- soft sheepskin hide

  审查 1：**非直接视觉，P2**。来源合理，但 soft 与羊皮身份都不容易从单张图片直接确认。
  建议：保留名称定义；区别可见皮面与不可见内衬。

- wool lined leather

  审查 2：**限定过窄，P1**。羊皮不必然等于带羊毛内衬的制鞋结构；将材料与具体构造混在一起。
  建议：先看训练图及材料含义，再决定能否描述羊毛层。

- natural fleece texture

  审查 3：**限定过窄，P1**。fleece 适合保留毛层的情况；不应默认全部羊皮都有外露绒毛。
  建议：只有训练图确有可见毛层时才把它当视觉原型。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：羊毛绒与羊皮描述有共同毛层线索；交换未必完全移除相关语义。

依据与边界：保留原羊毛的 shearling 是特定术语；不能仅凭 Sheepskin 标签推定可见羊毛内衬。[LWG 定义表](<https://www.leatherworkinggroup.com/fileadmin/uploads/lwg/Knowledge/LWG_List_of_Definitions_v1.1.pdf>)。

## a: Suede (shuffle donor: Leather)

- brushed suede leather

  审查 1：**可保留，P2**。绒面皮革的方向合理；brushed 更像表面处理概述。
  建议：保留绒面方向，不必绑定具体刷磨方式。

- soft napped surface

  审查 2：**可保留，P2**。短绒表面是合理线索，soft 是附带触感推断。
  建议：优先保留 napped，核验可见的绒感而非触感。

- matte velvety hide

  审查 3：**过泛/近义，P2**。matte/velvety 与前句近义，并与 Nubuck 共享。
  建议：不要把两类之间的可分性归功于这句泛描述。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：同属皮革范围，Leather 部分描述仍然适用。

依据与边界：短绒皮面与定义吻合；不据此声称图片可稳定区分所有磨绒皮革。[LWG 定义表](<https://www.leatherworkinggroup.com/fileadmin/uploads/lwg/Knowledge/LWG_List_of_Definitions_v1.1.pdf>)。

## a: Synthetic (shuffle donor: Faux.Fur)

- man made material

  审查 1：**过泛/近义，P2**。人造材料的定义合理，但类别本来就宽，视觉区分度有限。
  建议：保留宽类别性质；不能为了区分度虚构统一纹理。

- smooth artificial textile

  审查 2：**应修正，P1**。Synthetic 不等于“光滑的人造织物”；此句把宽材料类别缩成一个子类型。
  建议：删除统一 textile/smooth 的定义式断言；视觉替换先核验实图。

- uniform synthetic finish

  审查 3：**限定过窄，P1**。uniform 不是人造材料共有的必要外观，也没有明确可测含义。
  建议：不要要求所有 Synthetic 表面都均一。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：人造毛皮属于人造材料的一个子类型；不能把供体描述全部当作无关噪声。

## a: Wool (shuffle donor: Patent.Leather)

- knitted wool textile

  审查 1：**限定过窄，P1**。wool 是纤维类别；knitted 只限定了部分组织形式。
  建议：核验样本是针织、机织还是毛毡状，不能一概固定。

- fuzzy natural fibers

  审查 2：**可保留，P2**。可见毛状纤维是合理候选；natural 的来源不能直接看出来。
  建议：保留纤维外观，检查它与 Faux.Fur 的差异。

- warm coarse weave

  审查 3：**非直接视觉，P1**。warm 是保暖性能；coarse weave 也不是所有羊毛表面的要求。
  建议：删除 warm，并根据实图决定是否保留粗织纹。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：毛纤维与高光皮面有较明显语义差别。

## o: Boots.Ankle (shuffle donor: Sandals)

- shaft ending at the ankle

  审查 1：**可保留，P2**。靴筒高度是该类的主要语义，优于泛泛地说 closed boot。
  建议：保留；商品图无腿部参照时需结合鞋底到筒口比例核验。

- closed toe boot silhouette

  审查 2：**过泛/近义，P2**。闭合鞋头与靴形主要是多个靴类共有特征。
  建议：可以保留通用靴线索，但不算区分三类靴高的信息。

- low boot collar

  审查 3：**过泛/近义，P2**。low 的参照不明确，并和第一句重复。
  建议：优先使用明确的 ankle 靴高语义；避免纯相对词。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：靴与凉鞋的封闭程度及筒高区别明显。

## o: Boots.Knee.High (shuffle donor: Shoes.Flats)

- shaft reaching the knee

  审查 1：**可保留，P2**。到膝部的筒高语义合理。
  建议：保留高度信息；核验目录图是否显示相应长筒比例。

- tall fitted boot leg

  审查 2：**限定过窄，P1**。tall 合理；fitted 额外要求贴腿，类别名称本身没有这个条件。
  建议：去掉必须贴腿的限制，保留长筒形态。

- closed full foot upper

  审查 3：**过泛/近义，P2**。覆盖脚部的鞋面是许多靴类共同结构。
  建议：可作通用特征，但新增区分价值有限。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：高筒靴与浅口平底鞋区别明显。

## o: Boots.Mid-Calf (shuffle donor: Shoes.Sneakers.and.Athletic.Shoes)

- shaft ending at mid calf

  审查 1：**可保留，P2**。到小腿中部的靴筒语义合理。
  建议：保留明确位置，后续核验图中筒高比例。

- medium height boot leg

  审查 2：**过泛/近义，P2**。medium height 的参照不明确，且重复第一句。
  建议：减少纯相对词；不要用多条近义句重复筒高。

- closed boot construction

  审查 3：**过泛/近义，P2**。closed boot 是三个靴类共享结构。
  建议：可保留通用靴形，不能当作该类特有知识。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：中筒靴与运动鞋有区别，closed 等泛词仍共享。

## o: Sandals (shuffle donor: Shoes.Boat.Shoes)

- open toe construction

  审查 1：**限定过窄，P1**。露趾是常见款式，不能当作凉鞋类别的必要条件；存在带护趾结构的凉鞋。
  建议：优先描述鞋面开口与带状结构，具体露趾情况看训练图。

- exposed foot with straps

  审查 2：**可保留，P2**。开放区域与带状结构是合理候选；图片未必有脚，exposed foot 属穿用时形态。
  建议：保留带状结构；改写时描述鞋体可见开口。

- open ventilated upper

  审查 3：**非直接视觉，P1**。open 有视觉意义，ventilated 更偏透气或通风用途。
  建议：保留可见开口，去掉性能性结论。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：开放式凉鞋与低帮船鞋通常有结构区别。

依据与边界：品牌将有护趾结构的 Newport 明确称为 sandal，证明“全部凉鞋必须露趾”过于绝对；不证明该款在 UT 中。[KEEN Newport](<https://www.keenfootwear.com/collections/womens-water/products/womens-newport-h2-monochrome-faded-denim>)。

## o: Shoes.Boat.Shoes (shuffle donor: Slippers)

- moccasin style stitched toe

  审查 1：**可保留，P2**。鞋头缝线与莫卡辛式前部是合理典型候选。
  建议：保留；核验该零售类别的训练样本覆盖情况。

- low profile lace up upper

  审查 2：**可保留，P2**。低帮及系带是常见候选，并非所有款式的完整定义。
  建议：作为典型描述保留；系带覆盖率需核验。

- flat non marking rubber sole

  审查 3：**非直接视觉，P1**。non marking 是不留痕性能，商品图难以直接验证；rubber 又固定了鞋底材料。
  建议：去掉性能词；先确认鞋底是否可见，再描述形状。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：鞋型不同，但与拖鞋可共享低帮、套穿或低底原型。

## o: Shoes.Clogs.and.Mules (shuffle donor: Shoes.Oxfords)

- open or backless heel

  审查 1：**可保留，P2**。开放后跟是合理的合并类别线索；带后跟带等情况仍需核验。
  建议：优先核验后跟结构，不要求类别内所有设计完全一致。

- closed rounded toe

  审查 2：**限定过窄，P1**。closed 与 rounded 固定了鞋头封闭情况和形状，不能覆盖整个合并类别。
  建议：鞋头形状不是统一必要条件；先看训练样本再写。

- easy slip on silhouette

  审查 3：**过泛/近义，P2**。易穿脱与套穿在 Flats、Loafers、Slippers 中也常见；easy 本身偏使用体验。
  建议：保留 slip on 结构，减少体验性与泛化词。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：开放后跟与系带结构有区别，closed 等词仍可能共享。

依据与边界：品牌目录有露趾 Mule，足以说明 closed rounded toe 这个组合并非全类别必要条件；不证明该款属于 UT。[Clarks 2016 目录，第6页](<https://clarks.scene7.com/is/content/clarks/Americas/US/B2B/Marketing/SS16/Catalog/Dual%20Gender%20Independent%20CatalogSS16.pdf>)。圆头覆盖面另需训练图核验。

## o: Shoes.Flats (shuffle donor: Shoes.Clogs.and.Mules)

- very low or no heel

  审查 1：**可保留，P2**。低跟或无跟是合理核心方向。
  建议：保留，核验鞋底平台厚度与跟高的区别。

- shallow low cut upper

  审查 2：**限定过窄，P2**。浅口低帮适合典型平底单鞋，但可能只是零售类别中的部分款式。
  建议：训练图确认后使用，不自动推广至全部样本。

- slim closed toe profile

  审查 3：**限定过窄，P1**。slim 与 closed toe 叠加限制鞋楦和鞋头结构。
  建议：优先保留低跟特征，鞋头形状留待训练图核验。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：平底和开放后跟不是互斥维度；部分套穿、低帮结构相近。

## o: Shoes.Heels (shuffle donor: Boots.Knee.High)

- elevated narrow heel

  审查 1：**限定过窄，P1**。raised/elevated 的方向合理；narrow 将高跟鞋限定为细跟，排除了其他跟型。
  建议：去掉 narrow，描述抬高后跟这一结构。

- arched sole profile

  审查 2：**可保留，P2**。侧面轮廓是合理候选，但 arched 的具体指向比较含糊。
  建议：看训练图后明确是足弓处轮廓，避免过度解读。

- dress shoe silhouette

  审查 3：**过泛/近义，P2**。dress 是场景或风格概述，不如跟部几何结构明确，且与 Oxford 等重合。
  建议：作为弱风格信息；不替代鞋跟结构。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：跟部与高筒维度不同，有较明显类别区别。

依据与边界：Zappos 的 Heels 跟型列表包含 Block、Stiletto、Wedge 等，因此 narrow 不是高跟类别的必要条件。[Zappos Heels](<https://www.zappos.com/heels/CK_XARC41wE6ArwJagECuAEB4gIFAQIXBw0.zso>)。

## o: Shoes.Loafers (shuffle donor: Boots.Mid-Calf)

- laceless slip on upper

  审查 1：**可保留，P2**。无系带套穿是合理核心结构。
  建议：保留，较适合名称之外的结构描述。

- moccasin style vamp

  审查 2：**可保留，P2**。莫卡辛式前部是常见原型，类别内款式差异需要核验。
  建议：保留为典型候选，检查前部缝线或拼片是否可见。

- low stacked heel

  审查 3：**限定过窄，P1**。low stacked heel 是部分底跟形式，不能默认全部 Loafers 都具有。
  建议：不要固定叠层跟；依据训练样本描述底跟外形。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：套穿低帮鞋与中筒靴区别明显。

依据与边界：品牌说明支持无系带套穿这一原型。[Clarks 鞋型说明](<https://www.clarks.com/en-us/editorial/types-of-mens-dress-shoes>)。

## o: Shoes.Oxfords (shuffle donor: Boots.Ankle)

- closed lacing eyelet system

  审查 1：**可保留，P2**。传统 Oxford 的闭式系带定义合理；但零售分类是否严格遵守该定义需看实图。
  建议：保留为类别原型，不将教材定义直接等同于全部 UT 样本。

- structured leather quarters

  审查 2：**应修正，P1**。对象描述绑定 leather 材料；训练组合已包含 Canvas+Oxford 和 Synthetic+Oxford。
  建议：改为材料中性的鞋面或系带结构；保留原始材料标签不动。

- formal low heel silhouette

  审查 3：**过泛/近义，P2**。formal 偏风格，low heel 是宽泛底跟特征；难以独立区分其他低帮鞋。
  建议：作为辅助描述，优先保留可见结构。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：低帮系带鞋与踝靴不同，closed 等结构词仍共享。

依据与边界：闭式系带属于传统 Oxford 原型。[Clarks 鞋型说明](<https://www.clarks.com/en-us/editorial/types-of-mens-dress-shoes>)。材料问题另由本地 train_pairs.txt 的 Canvas/Synthetic Oxford 组合支持；它不证明每个鞋面组件都没有皮革。

## o: Shoes.Sneakers.and.Athletic.Shoes (shuffle donor: Shoes.Heels)

- cushioned athletic sole

  审查 1：**非直接视觉，P2**。athletic 方向合理；cushioned 是缓震性能或对内部结构的推断。
  建议：描述可见鞋底轮廓前先看图；不由外观推定缓震性能。

- lace up sporty upper

  审查 2：**限定过窄，P1**。lace up 只覆盖系带款，合并类别名称没有要求所有样本都系带。
  建议：检查训练样本是否有其他闭合结构，避免固定系带。

- padded flexible construction

  审查 3：**非直接视觉，P2**。padded 可能有外观线索；flexible 需形变或机械证据，且整体十分通用。
  建议：只保留能在训练图上看到的局部结构。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：运动原型与高跟结构有较明显区别。

## o: Slippers (shuffle donor: Shoes.Loafers)

- soft indoor construction

  审查 1：**非直接视觉，P2**。soft 是触感，indoor 是用途；两者不应直接当作图像特征。
  建议：根据训练图描述可见开口、轮廓或外露内衬。

- easy slip on opening

  审查 2：**过泛/近义，P2**。套穿开口合理，但与 Loafers、Clogs/Mules 共享；easy 偏体验。
  建议：保留 opening/slip on，寻找经核验的其他区分维度。

- thin flexible sole

  审查 3：**限定过窄，P1**。thin 只覆盖部分鞋底；flexible 又是机械性能，不能代表全部拖鞋。
  建议：不强制薄底，按训练图核验具体底部形态。

Review: [ ] accurate  [ ] generic  [ ] incorrect  [ ] duplicate

供体检查：都可能是套穿低帮轮廓；这组交换不是纯粹无关语义。

依据与边界：品牌有带厚平台的 slipper，足以反驳把 thin 作为整个类别的必要条件；不证明此款属于 UT。[UGG Tasman 产品说明](<https://www.ugg.com/employee-sale/ugg-x-oc-tasman/1151430.html>)。
