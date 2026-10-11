"""Fixed-size TRAIN contact sheets and a human-review form, not a classifier."""
import hashlib
import html
from pathlib import Path
from common import require, sha, write_json
from analysis import select_samples


def safe_train_image(row, data):
    root, p = Path(data).resolve(), Path(row['source']).resolve()
    require(root in p.parents and p.is_file(), 'Image path outside dataset or absent: ' + str(p))
    require(sha(p) == row['image_sha256'], 'Selected TRAIN image changed: ' + str(p))
    return p


def contacts(records, preds, protocol, out):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    folder = out / 'contact_sheets'
    folder.mkdir(exist_ok=True)
    font = ImageFont.load_default()
    selected = select_samples(records)
    receipt, decoded = [], 0
    for ai, attr in enumerate(protocol['labels']['attrs']):
        ids = [i for i in selected if records[i]['attr'] == ai]
        canvas = Image.new('RGB', (960, 650), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), 'TRAIN only | %s | fixed hash + object round-robin' % attr, fill='black', font=font)
        for position, index in enumerate(ids):
            row = records[index]
            x, y = (position % 3) * 320 + 10, (position // 3) * 310 + 35
            path = safe_train_image(row, protocol['data'])
            with Image.open(path) as original:
                picture = ImageOps.contain(original.convert('RGB'), (300, 215))
                canvas.paste(picture, (x + (300 - picture.width) // 2, y))
            require(sha(path) == row['image_sha256'], 'TRAIN image changed during decode')
            decoded += 1
            tag = '%05d' % index
            text = ['sample ' + tag, 'object: ' + protocol['labels']['objs'][row['object']]]
            for v, label in (('T_name', 'name'), ('T_K', 'OE+AE'), ('T_AE', 'AE')):
                text.append(label + ': ' + protocol['labels']['attrs'][int(preds[v][index])])
            for j, line in enumerate(text):
                draw.text((x, y + 220 + j * 13), line, fill='black', font=font)
            receipt.append({'sample_id': tag, 'train_index': index, 'image': row['image'],
                            'attr': attr, 'object': protocol['labels']['objs'][row['object']],
                            'image_sha256': row['image_sha256'], 'source': str(path)})
        target = folder / ('attr_%02d.png' % ai)
        require(not target.exists(), 'Contact sheet exists; preserve it')
        canvas.save(target)
        print('TRAIN_CONTACT_SHEET', attr, 'images', len(ids), flush=True)
    require(decoded <= 96, 'TRAIN image decode budget')
    write_json(out / 'sample_receipt.json', {'sampling_uses_prediction': False,
                                            'sampling': 'per attr <=6, object round-robin, fixed image SHA sort',
                                            'train_images_decoded': decoded, 'val_images_decoded': 0,
                                            'test_images_decoded': 0, 'samples': receipt})
    return decoded


def human_form(ae, oe, protocol, out):
    if (out / 'human_review_blank.json').exists():
        # A human may have annotated it: never replace their input on resume.
        return
    items = []
    for kind, table, keys in (('AE', ae, protocol['labels']['attrs']), ('OE', oe, protocol['labels']['objs'])):
        for key in keys:
            for j, sentence in enumerate(table[key]):
                items.append({'kind': kind, 'label': key, 'sentence_index': j, 'sentence': sentence,
                              'visible_in_sample_images_0_1_2': None,
                              'discriminative_vs_other_labels_0_1_2': None,
                              'describes_target_material_not_only_sole_lining_trim': None,
                              'evidence_sample_ids': [], 'notes': ''})
    write_json(out / 'human_review_blank.json', {
        'instructions': 'Optional manual review, TRAIN only. Scores: 0=no, 1=unclear/partial, 2=yes. '
                        'Unknown is allowed; do not edit knowledge or search VAL using these scores. '
                        'Save annotations as a new human_review_filled.json, preserve blank file.',
        'sampling_is_not_a_prevalence_estimate': True, 'items': items})


def html_report(pooled, sentences, protocol, ae, oe, out):
    esc = lambda x: html.escape(str(x))
    content = ['<!doctype html><meta charset="utf-8"><title>TRAIN knowledge audit</title>',
               '<style>body{font-family:system-ui;max-width:1080px;margin:30px auto;padding:0 20px}'
               'table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}'
               'img{width:100%;height:auto}section{margin:35px 0}code{word-break:break-all}</style>',
               '<h1>TRAIN-only 知识落地审查</h1>',
               '<p>零优化步骤，不访问 VAL/TEST；逐句结果是同组合描述性诊断，不是组合迁移验证。'
               '不得按本页最高点直接挑新提示词。抽样每属性最多六张，并按对象轮转、固定图像哈希排序；'
               '不依据预测对错。图片保留完整画面，标注行是 GT-object 属性诊断。</p>',
               '<h2>旧均值配方的 TRAIN 复核</h2><table><tr><th>组</th><th>micro %</th>'
               '<th>pair macro %</th><th>纠正/破坏（相对名称）</th></tr>']
    for v, m in pooled['metrics'].items():
        content.append('<tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%d / %d</td></tr>' %
                       (esc(v), m['micro_pct'], m['pair_macro_pct'], m['vs_name']['repair'], m['vs_name']['damage']))
    content += ['</table><h2>固定句子位置诊断</h2><p>每个位置在全部候选上使用，'
                '不是为真实类别挑选专属最佳句子。AE-only 有三种独立位置，OE＋AE 有九种组合位置。</p>']
    for family, rows in sentences['families'].items():
        content.append('<h3>%s</h3><table><tr><th>AE/OE 位置（从 0 起）</th><th>正确 macro %%</th>'
                       '<th>正确−名称 pp</th><th>正确−错配1 pp</th><th>正确−错配2 pp</th></tr>' % esc(family))
        for row in rows:
            v = 'T_K' if family == 'OE_AE' else 'T_AE'
            d = row['correct_macro_deltas_pp']
            content.append('<tr><td>%s / %s</td><td>%.4f</td><td>%+.4f</td><td>%+.4f</td><td>%+.4f</td></tr>' %
                           (row['ae_index'], row['oe_index'], row['variants'][v]['pair_macro_pct'],
                            d['name'], d['perm1'], d['perm2']))
        content.append('</table>')
    content += ['<h2>固定 TRAIN 图片与描述</h2><p>请检查是否可见、是否区分其他材质、'
                '是否描述的是标签目标而非仅鞋底/内衬/装饰。只有六张图不能估计整类可见比例。</p>']
    for ai, attr in enumerate(protocol['labels']['attrs']):
        content.append('<section><h3>%s</h3><ol>%s</ol><img src="contact_sheets/attr_%02d.png"></section>' %
                       (esc(attr), ''.join('<li>%s</li>' % esc(t) for t in ae[attr]), ai))
    content.append('<h2>对象描述（仅角色审查，不据此优化权重）</h2>')
    for obj in protocol['labels']['objs']:
        content.append('<h3>%s</h3><ol>%s</ol>' % (esc(obj), ''.join('<li>%s</li>' % esc(t) for t in oe[obj])))
    content.append('<p>下一步：先读取 TRAIN 审查；若无明确问题及可证伪修正，停止扩大当前教师配方。'
                   '任何后续改动须另锁定 TRAIN 组合留出协议，不能直接追加 VAL 提示词搜索。</p>')
    (out / 'review.html').write_text('\n'.join(content), encoding='utf-8')
