# UT 训练图审核完成记录与 v2 修订

日期：2026-10-06。审阅者：Codex assistant。**28张图板与84句均已审核；这不是人工专家批准，也不是性能收益证明。**

依据：固定seed=20261006的训练记录原图册，560个面板、528张不同原图；材料板与对象板可能重复同一图。全部面板在图板上查看，图片原分辨率136×136；未额外查阅验证/测试图片或依据验证错误调词。

每类仍三句；标签、顺序、GloVe编码与聚合公式保持原协议。候选短句描述典型或可选知识线索，不是每张图必须满足的逻辑约束。来源、加工、触感和使用场景均明确区分，不能冒充逐图视觉确认。

主要修订：删除过窄的narrow／fitted／slim、对象固定leather、严格closed lacing、统一lace up、thin/flexible、whole-shoe waterproof等要求。部分类别仍共享许多外观；尤其Oxford与Loafers、Suede与Nubuck无法仅凭这批低分辨率图可靠区分。保留这种限制，后续用配对实验验证收益。

材料语义参考：[Leather Naturally](https://www.leathernaturally.org/wp-content/uploads/2023/02/LN_FactSheet_Guide_Leather_Terminology.pdf)、[Leather Research Laboratory](https://leatherusa.org/suede-nubuck-leather)、[CottonWorks棉织物资源](https://cottonworks.com/product-innovation/digital-product-creation/)、[Woolmark羊毛资源](https://www.woolmark.com/fashion/)。这些资源支持术语含义，不证明具体样本材料真伪或论文新颖性。

## a:Canvas

图板：[a_00_canvas.jpg](review_evidence/boards/a_00_canvas.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | woven canvas fabric | woven canvas fabric | material_structure | keep |
| 2 | coarse textile surface | matte textile upper panels | visual_appearance | rewrite |
| 3 | matte cloth finish | fabric panels with stitched edges | visual_structure | rewrite |

- 第1句：保留类别与织物结构锚点；布面外观可见，但不能从这些136像素图确认每条经纬线。 支持/上下文面板：2,8,10,14,20；例外、未显示或未能视觉认证的面板：1,4,13,18。
- 第2句：将coarse的粗细要求改为所选图中常见的哑光布面；饰面、印花和局部反光仍可出现，与Cotton等共享。 支持/上下文面板：2,6,7,10,14,20；例外、未显示或未能视觉认证的面板：1,3,9。
- 第3句：加入面片及边缘结构，避免三句只说布面质感；缝线并非Canvas独有，局部细节不清时不作微观确认。 支持/上下文面板：2,7,10,11,14,20；例外、未显示或未能视觉认证的面板：3,9,15。

## a:Cotton

图板：[a_01_cotton.jpg](review_evidence/boards/a_01_cotton.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | soft cotton fabric | cotton textile fibers | material_source | rewrite |
| 2 | breathable textile surface | cloth upper panels | visual_context | rewrite |
| 3 | fine woven cloth | woven or knitted cotton fabric | material_structure | rewrite |

- 第1句：移除soft触感断言；棉纤维来源以类别语义为据，照片不能鉴定纤维成分。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：用图中布面部件替代不能直接观察的breathable性能；保留部分视角难辨或混合材料的限制。 支持/上下文面板：2,3,7,10,11,15,16,18,19；例外、未显示或未能视觉认证的面板：4,8,12,20。
- 第3句：撤销fine woven对全部Cotton的限制；官方棉织物资源同时包含机织和针织，具体样本织法无法全面鉴定。 支持/上下文面板：1,5,9,16；例外、未显示或未能视觉认证的面板：4,8,12,20。

## a:Faux.Fur

图板：[a_02_faux_fur.jpg](review_evidence/boards/a_02_faux_fur.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | fluffy synthetic fur | fluffy synthetic fur | material_source_and_visual | keep |
| 2 | plush hairy texture | raised fur like pile | visual_appearance | rewrite |
| 3 | fuzzy pile surface | fur trim around footwear openings | visual_context | rewrite |

- 第1句：保留蓬松毛状外观及类别锚点；synthetic来自类别含义而非照片鉴定，8/17外侧基本不显毛。 支持/上下文面板：1,3,4,5,6,7,9,12,18,19,20；例外、未显示或未能视觉认证的面板：8,10,17。
- 第2句：去掉plush的触感暗示，描述可见隆起毛绒；与Wool/Sheepskin等仍有外观重叠。 支持/上下文面板：1,3,4,5,6,7,9,18,19；例外、未显示或未能视觉认证的面板：8,10,17。
- 第3句：以鞋口和饰边的位置补充维度；属于常见部位线索，不要求所有样本都有外露饰边或内部毛衬。 支持/上下文面板：2,4,6,11,12,13,14,16,20；例外、未显示或未能视觉认证的面板：8,17。

## a:Faux.Leather

图板：[a_03_faux_leather.jpg](review_evidence/boards/a_03_faux_leather.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | imitation leather | imitation leather | material_source | keep |
| 2 | smooth synthetic hide | synthetic leather upper panels | material_source_and_context | rewrite |
| 3 | leather like finish | leather like surface finish | visual_appearance | rewrite |

- 第1句：保留类别意义；不能用照片证明仿皮与真皮的化学或组织差异。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：去掉smooth和hide的误导性单一限定；人造皮部件可处于混合鞋面中，不能把标签扩展为整鞋成分。 支持/上下文面板：1,2,4,5,8,10,14,18,20；例外、未显示或未能视觉认证的面板：3,7,9,11,15,19。
- 第3句：保留皮革样饰面候选；不声称能从136像素图区分真假，来源与外观两句相关性仍然较强。 支持/上下文面板：1,4,5,6,8,10,14,17,18,20；例外、未显示或未能视觉认证的面板：7,13,19。

## a:Full.grain.leather

图板：[a_04_full_grain_leather.jpg](review_evidence/boards/a_04_full_grain_leather.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | natural full grain hide | natural full grain hide | material_source | keep |
| 2 | visible leather grain | smooth or textured leather surface | visual_appearance | rewrite |
| 3 | durable textured leather | stitched leather upper panels | visual_context | rewrite |

- 第1句：保留全粒面天然皮语义；粒面层来源不能由这批鞋图鉴定。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：撤销visible grain对全部样本的可见性要求；观察到平整表面与褶皱纹理，但不将褶皱认作可验证的粒面微结构。 支持/上下文面板：1,3,8,9,13,14,16,17；例外、未显示或未能视觉认证的面板：5,7,15,18。
- 第3句：用有可见面片和缝合的上下文替代durable性能；属于常见构造，材料身份仍依类别语义。 支持/上下文面板：1,2,8,10,16,17；例外、未显示或未能视觉认证的面板：7,15,18。

## a:Hair.Calf

图板：[a_05_hair_calf.jpg](review_evidence/boards/a_05_hair_calf.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | short haired calfskin | short haired calfskin | material_source | keep |
| 2 | fine animal hair surface | hair bearing leather surface | material_structure | rewrite |
| 3 | velvety hide texture | plain or patterned hide panels | visual_appearance | rewrite |

- 第1句：保留带毛小牛皮类别锚点；照片不能确认动物种属或毛发长度。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：把fine animal hair改为类别结构知识；图中花纹不等于已经显微确认毛发，也不能把印花当动物来源证据。 支持/上下文面板：1,3,7,9,10,14,15,18；例外、未显示或未能视觉认证的面板：2,6,13,16,19,20。
- 第3句：移除统一velvety要求；同时保留素色与花纹样本，不把豹纹/斑马纹定为必要条件。 支持/上下文面板：1,3,7,9,10,13,14,17,18,19；例外、未显示或未能视觉认证的面板：16,20。

## a:Leather

图板：[a_06_leather.jpg](review_evidence/boards/a_06_leather.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | genuine leather | genuine leather | material_source | keep |
| 2 | smooth animal hide | leather upper panels and straps | material_source_and_context | rewrite |
| 3 | natural grain finish | smooth or textured leather finish | visual_appearance | rewrite |

- 第1句：保留类别来源锚点；照片不能认证真假或整鞋材质。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：撤销smooth animal hide的单一表面要求；结合鞋面和带状部件，但不强制所有部件均为皮革。 支持/上下文面板：1,2,3,4,7,10,12,15,16,17,20；例外、未显示或未能视觉认证的面板：5,9,18,19。
- 第3句：表面有光滑、褶皱和装饰变化；natural grain无法普遍直接验证，故扩大为外观线索。 支持/上下文面板：1,3,4,7,10,13,14,16,17,20；例外、未显示或未能视觉认证的面板：5,9,19。

## a:Nubuck

图板：[a_07_nubuck.jpg](review_evidence/boards/a_07_nubuck.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | sanded leather | sanded leather | material_processing | keep |
| 2 | soft velvety grain | fine napped leather surface | material_structure | rewrite |
| 3 | matte brushed hide | matte brushed hide | visual_appearance | keep |

- 第1句：保留磨面加工类别语义；加工方式不能由鞋图直接推断。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：去掉soft触感并将nap标为材料结构知识；可见低光泽面支持外观，细绒毛不作逐根鉴定。 支持/上下文面板：2,5,7,11,12,13,14,17,19；例外、未显示或未能视觉认证的面板：1,8,15。
- 第3句：保留常见哑光刷面外观；亮色细跟样本光泽不同，不要求全类同一粗细或亮度。 支持/上下文面板：2,4,5,7,11,12,13,14,17,18,19；例外、未显示或未能视觉认证的面板：1,8,15。

## a:Nylon

图板：[a_08_nylon.jpg](review_evidence/boards/a_08_nylon.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | lightweight nylon textile | nylon textile fibers | material_source | rewrite |
| 2 | smooth synthetic fabric | synthetic fabric panels | material_source_and_context | rewrite |
| 3 | tightly woven sheen | fabric straps and padded panels | visual_context | rewrite |

- 第1句：保留尼龙名称和纤维语义，撤销不能由静态图测量的lightweight。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：去掉全部smooth的限定；鞋靴有织物、衬垫和其他混合部件，标签不是所有外露部件的成分保证。 支持/上下文面板：1,2,5,9,12,13,14,17,18；例外、未显示或未能视觉认证的面板：6,7,8,19。
- 第3句：替换tight weave与sheen的微观/光泽限定；保留可见带状和隆起面片为常见线索，不凭图确认弹性或纺织密度。 支持/上下文面板：2,3,4,10,11,12,14,16,18；例外、未显示或未能视觉认证的面板：7,8,19,20。

## a:Patent.Leather

图板：[a_09_patent_leather.jpg](review_evidence/boards/a_09_patent_leather.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | glossy coated leather | glossy coated leather | material_processing_and_visual | keep |
| 2 | mirror like shine | bright specular surface highlights | visual_appearance | rewrite |
| 3 | lacquered hide finish | smooth finished upper panels | visual_context | rewrite |

- 第1句：保留高光饰面的典型外观；涂层化学、真皮来源和部分低光泽样本不由照片认证。 支持/上下文面板：1,2,3,6,7,8,9,11,14,15,18,19；例外、未显示或未能视觉认证的面板：4,13,20。
- 第2句：将mirror like的强度比喻改为实际可见高光；没有高光可能受照明/颜色/视角影响。 支持/上下文面板：1,2,3,6,7,8,11,14,15,18,19；例外、未显示或未能视觉认证的面板：4,10,13,20。
- 第3句：用表面面片补充结构维度，减少三句对光泽的重复；装饰与混合部件仍是例外。 支持/上下文面板：2,3,6,7,8,9,11,15,18,19；例外、未显示或未能视觉认证的面板：5,12,13,20。

## a:Rubber

图板：[a_10_rubber.jpg](review_evidence/boards/a_10_rubber.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | flexible rubber material | flexible rubber material | material_property | keep |
| 2 | matte elastic surface | molded rubber components | material_processing_and_context | rewrite |
| 3 | molded waterproof texture | rubber soles or smooth outer shells | material_source_and_visual | rewrite |

- 第1句：保留橡胶的材料性质知识，但不把flexible视为从静态图测得的柔性。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：撤销所有表面均matte的要求；成型部件作为材料加工/部位知识，图片仅显示形状，不认证加工工艺。 支持/上下文面板：1,2,3,4,6,8,10,11,12,13,18,20；例外、未显示或未能视觉认证的面板：5,7,9,15,16,17,19。
- 第3句：删除整鞋waterproof断言，允许鞋底或外壳位置；鞋底的准确成分仍依类别语义，不能把防水当纹理。 支持/上下文面板：1,2,3,6,8,10,11,12,13,14,18,20；例外、未显示或未能视觉认证的面板：5,7,9,15,16,17,19。

## a:Satin

图板：[a_11_satin.jpg](review_evidence/boards/a_11_satin.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | smooth satin fabric | smooth satin fabric | material_structure_and_visual | keep |
| 2 | silky lustrous weave | lustrous fabric surface | visual_appearance | rewrite |
| 3 | reflective textile sheen | fabric folds and gathered edges | visual_structure | rewrite |

- 第1句：保留缎面名称和常见平滑外观；饰面、亮片与深色样本降低可见性。 支持/上下文面板：1,3,5,7,8,11,12,13,14,17,19,20；例外、未显示或未能视觉认证的面板：2,4,6,10,16,18。
- 第2句：去掉silky触感和未分辨的weave微结构，保留所见光泽；与Patent.Leather的光泽共享但布面上下文不同。 支持/上下文面板：1,3,5,7,8,11,12,13,17,19,20；例外、未显示或未能视觉认证的面板：2,4,6,10,16,18。
- 第3句：用褶皱/收拢结构替代第三个近义光泽句；属于典型可选线索而非每张Satin都存在的必要条件。 支持/上下文面板：1,2,7,8,12,14,19,20；例外、未显示或未能视觉认证的面板：4,6,9,10,15,16,18。

## a:Sheepskin

图板：[a_12_sheepskin.jpg](review_evidence/boards/a_12_sheepskin.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | soft sheepskin hide | sheepskin hide | material_source | rewrite |
| 2 | wool lined leather | fleece at collars and openings | visual_context | rewrite |
| 3 | natural fleece texture | matte hide panels | visual_appearance | rewrite |

- 第1句：去掉soft触感，保留羊皮类别；外部鞋图不能认证种属与内部构造。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：撤销wool lined leather作为全类结构定义；仅记录部分鞋口可见绒状物，不断言所有鞋内部均衬毛。 支持/上下文面板：1,7,8,9,10,11,13,15,16,17,19；例外、未显示或未能视觉认证的面板：2,3,4,5,6,12,14,18,20。
- 第3句：将natural fleece统一表面要求换为所见哑光外部面片；不能由外观证明天然来源。 支持/上下文面板：2,3,4,5,6,7,8,10,11,13,14,16,17,18；例外、未显示或未能视觉认证的面板：9,12,15,19。

## a:Suede

图板：[a_13_suede.jpg](review_evidence/boards/a_13_suede.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | brushed suede leather | brushed suede leather | material_processing | keep |
| 2 | soft napped surface | napped leather surface | material_structure | rewrite |
| 3 | matte velvety hide | suede upper panels and straps | material_source_and_context | rewrite |

- 第1句：保留绒面皮类别/加工锚点，不能从图像鉴定加工面或真皮来源。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：去掉soft触感，保留绒面结构语义；低光泽外观支持但不等于显微确认nap。 支持/上下文面板：1,2,4,5,7,9,10,11,12,14,15,17,18,20；例外、未显示或未能视觉认证的面板：3,6,8,13,16,19。
- 第3句：第三句改为部位上下文，减少与nap句的重复；与Nubuck仍有视觉重叠，不声称照片能可靠区分。 支持/上下文面板：1,2,4,5,6,7,9,10,12,14,15,17,18,20；例外、未显示或未能视觉认证的面板：3,8,13,16,19。

## a:Synthetic

图板：[a_14_synthetic.jpg](review_evidence/boards/a_14_synthetic.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | man made material | man made material | material_source | keep |
| 2 | smooth artificial textile | synthetic fabric or polymer components | material_source | rewrite |
| 3 | uniform synthetic finish | textile and molded upper surfaces | visual_context_and_processing | rewrite |

- 第1句：保留宽泛的人造材料来源锚点；不能由图像认证化学来源。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：原smooth artificial textile只覆盖部分形式；同板含布面、实体外壳和皮样面，使用备选形态，实际成分不由图像确认。 支持/上下文面板：2,4,5,9,10,13,14,17；例外、未显示或未能视觉认证的面板：1,3,6,7,8,11,12,15,16,18,19,20。
- 第3句：撤销uniform synthetic finish；描述多样部件外观，molded为常见加工候选，不把样本颜色或印花作为人造材料定义。 支持/上下文面板：2,4,5,9,10,13,14,17；例外、未显示或未能视觉认证的面板：1,3,6,7,8,11,12,15,16,18,19,20。

## a:Wool

图板：[a_15_wool.jpg](review_evidence/boards/a_15_wool.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | knitted wool textile | wool textile fibers | material_source | rewrite |
| 2 | fuzzy natural fibers | fuzzy natural fibers | material_source_and_visual | keep |
| 3 | warm coarse weave | felted or knitted fabric surface | material_structure_and_visual | rewrite |

- 第1句：原knitted限定不覆盖毡状等形式；保留羊毛纤维来源，不能由照片认证。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：保留毛状外观候选；natural是类别语义，很多样本无法分辨纤维，与Faux.Fur/Sheepskin共享。 支持/上下文面板：1,2,3,7,9,11,12,13,17,18,19；例外、未显示或未能视觉认证的面板：4,5,6,8,10,14,15,16,20。
- 第3句：去掉warm性能和coarse weave强制限定；图9织物结构与2/18毡状外观有差异，只作为常见形态候选。 支持/上下文面板：1,2,3,7,9,11,12,13,17,18,19；例外、未显示或未能视觉认证的面板：4,5,6,8,14,16,20。

## o:Boots.Ankle

图板：[o_00_boots_ankle.jpg](review_evidence/boards/o_00_boots_ankle.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | shaft ending at the ankle | shaft ending at the ankle | category_shape | keep |
| 2 | closed toe boot silhouette | boot upper covering the foot and ankle | visual_structure | rewrite |
| 3 | low boot collar | low boot collar | visual_shape | keep |

- 第1句：保留短靴类别的高度锚点；孤立鞋图没有人体尺标，较高/翻边款有高度范围，不能精确量到踝关节。 支持/上下文面板：1,2,4,5,9,10,11,13,17,18,20；例外、未显示或未能视觉认证的面板：7,8,12,14,15。
- 第2句：替换closed toe的全类要求，图16可见露趾样式；足部与踝部覆盖仍为常见结构。 支持/上下文面板：1,2,3,4,5,9,10,11,17,18,20；例外、未显示或未能视觉认证的面板：16。
- 第3句：保留相对短靴筒的辅助线索，与第一句相关；不将高度差说成三个独立知识事实。 支持/上下文面板：1,2,4,5,9,10,11,17,18,20；例外、未显示或未能视觉认证的面板：7,8,12,14,15。

## o:Boots.Knee.High

图板：[o_01_boots_knee_high.jpg](review_evidence/boards/o_01_boots_knee_high.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | shaft reaching the knee | shaft reaching the knee | category_shape | keep |
| 2 | tall fitted boot leg | tall boot shaft | visual_shape | rewrite |
| 3 | closed full foot upper | closed full foot upper | visual_structure | keep |

- 第1句：保留标签的类别高度锚点；孤立图像不证明精确膝部位置，部分雪地/雨靴款较宽或高度偏离典型。 支持/上下文面板：2,3,5,6,8,9,10,12,14,15,16,18,20；例外、未显示或未能视觉认证的面板：1,7,11,13,17,19。
- 第2句：删除fitted，图11/17宽口和图19隆起款不应被排除；仍是高筒的典型线索。 支持/上下文面板：2,3,4,5,6,8,9,10,12,14,15,16,18,20；例外、未显示或未能视觉认证的面板：1,7,13,19。
- 第3句：各所选样本均有闭合包足鞋头与靴身；保持材料无关，与其他闭头靴共享。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Boots.Mid-Calf

图板：[o_02_boots_mid_calf.jpg](review_evidence/boards/o_02_boots_mid_calf.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | shaft ending at mid calf | shaft ending at mid calf | category_shape | keep |
| 2 | medium height boot leg | medium height boot leg | visual_shape | keep |
| 3 | closed boot construction | closed boot construction | visual_structure | keep |

- 第1句：保留类别高度锚点；雪地靴外形不同，不能从孤立图精确测小腿位置。 支持/上下文面板：1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,19；例外、未显示或未能视觉认证的面板：9,17,18,20。
- 第2句：保留典型中等靴筒，作为相对形状线索，不声称与第一句语义完全独立。 支持/上下文面板：1,2,3,4,5,6,7,8,10,12,13,14,15,16,19；例外、未显示或未能视觉认证的面板：9,17,18,20。
- 第3句：图册均显示闭合包足靴型；保留并记为多个靴类共有的辅助线索。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Sandals

图板：[o_03_sandals.jpg](review_evidence/boards/o_03_sandals.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | open toe construction | open toe construction | visual_structure | keep |
| 2 | exposed foot with straps | exposed foot with straps | visual_structure | keep |
| 3 | open ventilated upper | upper openings around the foot | visual_structure | rewrite |

- 第1句：本板20个样本均显示露趾前部，故保留；不外推为所有可能的凉鞋定义。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第2句：保留带状/分离鞋面及露足范围，跨Suede、Leather、Synthetic等材料核验。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第3句：用实际开口结构替代ventilated性能，不由开口推断通风量。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Shoes.Boat.Shoes

图板：[o_04_shoes_boat_shoes.jpg](review_evidence/boards/o_04_shoes_boat_shoes.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | moccasin style stitched toe | moccasin style stitched toe | visual_structure | keep |
| 2 | low profile lace up upper | low profile lace up upper | visual_structure | keep |
| 3 | flat non marking rubber sole | laces threaded through side eyelets | visual_structure | rewrite |

- 第1句：保留前部包边/缝合风格，部分细线在136像素图中不清，与Loafers有共享。 支持/上下文面板：1,2,3,4,5,7,8,9,11,13,15,17,18,19,20；例外、未显示或未能视觉认证的面板：6,10,12,14,16。
- 第2句：各样本低帮且有可见鞋带或穿带结构，保留但不绑定皮革材料。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第3句：以侧边穿带结构替换non marking的功能与固定橡胶材质断言；部分侧眼孔细节不清，不伪称已测耐磨/留痕。 支持/上下文面板：1,2,3,4,5,7,8,9,13,15,17,18,19,20；例外、未显示或未能视觉认证的面板：6,10,11,12,14,16。

## o:Shoes.Clogs.and.Mules

图板：[o_05_shoes_clogs_and_mules.jpg](review_evidence/boards/o_05_shoes_clogs_and_mules.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | open or backless heel | low cut upper with a broad toe box | visual_shape | rewrite |
| 2 | closed rounded toe | upper covering the front of the foot | visual_structure | rewrite |
| 3 | easy slip on silhouette | slip on clog and mule silhouette | category_shape | rewrite |

- 第1句：撤销全部open/backless heel，图1/6/7/11/12/17包后跟；保留典型低帮和较宽前部，宽度不是必要条件。 支持/上下文面板：2,3,4,5,7,8,9,10,11,14,15,16,20；例外、未显示或未能视觉认证的面板：1,6,12,13,17,18,19。
- 第2句：图18露趾，删除统一closed rounded toe；前部有鞋面覆盖不代表所有趾尖闭合。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20；例外、未显示或未能视觉认证的面板：18。
- 第3句：删除easy的便捷性能表述，保留典型穿入式外形；包后跟及带扣款仍在类中。 支持/上下文面板：2,3,4,8,9,10,13,14,15,16,18,20；例外、未显示或未能视觉认证的面板：1,6,7,11,12,17,19。

## o:Shoes.Flats

图板：[o_06_shoes_flats.jpg](review_evidence/boards/o_06_shoes_flats.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | very low or no heel | very low or no heel | visual_shape | keep |
| 2 | shallow low cut upper | shallow low cut upper | visual_shape | keep |
| 3 | slim closed toe profile | low sole and enclosed toe | visual_structure | rewrite |

- 第1句：本板均为低跟/接近无跟，保留不绑定鞋底材质。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第2句：保留浅口的常见样式；运动/带扣款较深口，不当作全类必要条件。 支持/上下文面板：3,4,5,9,10,12,13,14,17,18,20；例外、未显示或未能视觉认证的面板：1,2,6,7,8,11,15,16,19。
- 第3句：去掉slim，图1/2/7/8/16/19较厚重；闭趾与低底作为辅助结构，和其他鞋类有重叠。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Shoes.Heels

图板：[o_07_shoes_heels.jpg](review_evidence/boards/o_07_shoes_heels.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | elevated narrow heel | raised heel or wedge | visual_shape | rewrite |
| 2 | arched sole profile | arched sole profile | visual_shape | keep |
| 3 | dress shoe silhouette | dress shoe silhouette | category_context | keep |

- 第1句：去掉narrow，包含粗跟、细跟与坡跟；图11/12/14/19偏低跟，类别内部范围较宽。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,13,15,16,17,18,20；例外、未显示或未能视觉认证的面板：11,12,14,19。
- 第2句：保留可见弓形底部为典型线索；坡跟和低跟样本不强制出现同样足弓。 支持/上下文面板：1,2,3,4,7,8,9,10,13,16,17,20；例外、未显示或未能视觉认证的面板：6,11,12,14,15,18,19。
- 第3句：保留穿搭类别的辅助锚点，不把formal或细跟当全部样本要求；这是宽泛语义，不能单独区分Heels。 支持/上下文面板：1,2,3,4,8,9,10,13,17；例外、未显示或未能视觉认证的面板：5,6,7,11,12,14,15,16,18,19,20。

## o:Shoes.Loafers

图板：[o_08_shoes_loafers.jpg](review_evidence/boards/o_08_shoes_loafers.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | laceless slip on upper | laceless slip on upper | visual_structure | keep |
| 2 | moccasin style vamp | upper covering the instep | visual_structure | rewrite |
| 3 | low stacked heel | low cut collar around the foot | visual_shape | rewrite |

- 第1句：所有所选样本未见系带，保留；装饰扣与流苏不等于鞋带。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第2句：去掉统一moccasin style；多种前部面片/装饰形态均覆盖脚背，浅口款为范围例外。 支持/上下文面板：1,2,3,4,6,7,9,10,11,12,14,16,17,18,19,20；例外、未显示或未能视觉认证的面板：5,8,13,15。
- 第3句：替换low stacked heel，图1/9/17/20有平底或成型底；低口不绑定层叠鞋跟。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Shoes.Oxfords

图板：[o_09_shoes_oxfords.jpg](review_evidence/boards/o_09_shoes_oxfords.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | closed lacing eyelet system | laced or slip on low shoe upper | visual_structure | rewrite |
| 2 | structured leather quarters | enclosed toe and low shoe collar | visual_structure | rewrite |
| 3 | formal low heel silhouette | low heel or flat sole profile | visual_shape | rewrite |

- 第1句：撤销closed lacing作为全类强制条件；图17/18无鞋带，2/7等系带形态也不能直接证明严格封闭系带。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第2句：删除leather绑定，包含Canvas/Synthetic；图9帮口稍高，精确高度不作必要条件。 支持/上下文面板：1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：9。
- 第3句：撤销统一formal，Canvas休闲款和套脚款仍在类中；保留所见低跟/平底，但与Loafers等区分性弱。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Shoes.Sneakers.and.Athletic.Shoes

图板：[o_10_shoes_sneakers_and_athletic_shoes.jpg](review_evidence/boards/o_10_shoes_sneakers_and_athletic_shoes.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | cushioned athletic sole | flat or treaded athletic sole | visual_shape | rewrite |
| 2 | lace up sporty upper | laced or slip on sporty upper | visual_structure | rewrite |
| 3 | padded flexible construction | low or high athletic shoe collar | visual_shape | rewrite |

- 第1句：不能从照片测cushioned；涵盖休闲平底、运动齿纹和球鞋鞋钉的不同样式，不强制都有厚缓震层。 支持/上下文面板：1,2,3,4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：7。
- 第2句：图10/12/17/18等无传统鞋带，扩大为系带或套脚，保持sporty类别语义。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。
- 第3句：去掉padded flexible的触感/性能要求，保留低帮或高帮范围；图8明显高帮，不能统一low cut。 支持/上下文面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：本抽样中未见明确反例。

## o:Slippers

图板：[o_11_slippers.jpg](review_evidence/boards/o_11_slippers.jpg)

| 句号 | v1原句 | v2定稿候选 | 知识类型 | 处理 |
| --- | --- | --- | --- | --- |
| 1 | soft indoor construction | indoor slipper footwear | category_context | rewrite |
| 2 | easy slip on opening | slip on or strap fastened opening | visual_structure | rewrite |
| 3 | thin flexible sole | low heel sole profile | visual_shape | rewrite |

- 第1句：保留室内拖鞋类别语义，去掉soft触感；拍摄背景不是实际室内使用证据。 支持/上下文面板：无直接视觉认证；例外、未显示或未能视觉认证的面板：1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20。
- 第2句：图2/6/8有带状扣合，图3为毛状高口；不统一要求easy slip on，也不推出穿脱速度。 支持/上下文面板：1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：3。
- 第3句：图10等较厚底，删除thin；flexible需要力学证据，改为可见的低跟轮廓，部分毛饰遮挡鞋底。 支持/上下文面板：1,2,4,5,6,8,9,10,12,13,14,15,16,17,18,19,20；例外、未显示或未能视觉认证的面板：3,7,11。
