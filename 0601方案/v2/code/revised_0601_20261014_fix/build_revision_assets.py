"""One corpus for TextK and semantic constraints. Frozen, label-only inputs."""
import copy
import importlib.util
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from common import read, write, sha
from revision_math import aggregate_descriptions, deletion_mask

HERE = Path(__file__).absolute().parent
GLOVE_SHA = '91125602f730fea7ca768736c6f442e668b49db095682bf2aad375db061c21ed'


def hierarchy(corpus, objects):
    # Refine the original five disjoint LABEL groups, then merge to three.
    # This is an explicit reconstructed taxonomy, not universally true visual facts.
    fine = {
        'short boots': ['Boots.Ankle'],
        'tall boots': ['Boots.Knee.High', 'Boots.Mid-Calf'],
        'lace up low shoes': ['Shoes.Boat.Shoes', 'Shoes.Oxfords', 'Shoes.Sneakers.and.Athletic.Shoes'],
        'low slip on loafers': ['Shoes.Loafers'],
        'flat low shoes': ['Shoes.Flats'],
        'slip on comfort footwear': ['Shoes.Clogs.and.Mules', 'Slippers'],
        'strapped open footwear': ['Sandals'],
        'raised heel shoes': ['Shoes.Heels']}
    coarse = corpus['oi']
    supergroups = {
        'boots': coarse['boots'],
        'low and raised heel shoes': coarse['closed low shoes'] + coarse['raised heel shoes'],
        'open and comfort footwear': coarse['slip on comfort footwear'] + coarse['strapped open footwear']}
    levels = [fine, coarse, supergroups]
    assignments, names = [], []
    for groups in levels:
        flat = [label for values in groups.values() for label in values]
        if len(flat) != len(objects) or sorted(flat) != sorted(objects) or len(groups) < 2:
            raise ValueError('Hierarchy must be a complete, nontrivial partition')
        ordered = list(groups)
        ids = torch.tensor([next(i for i, key in enumerate(ordered) if label in groups[key]) for label in objects])
        assignments.append(ids)
        names.append(ordered)
    for lower, upper in zip(assignments, assignments[1:]):
        if len(lower.unique()) <= len(upper.unique()):
            raise ValueError('Duplicate/non-coarsening hierarchy level')
        for child in lower.unique():
            if len(upper[lower == child].unique()) != 1:
                raise ValueError('Hierarchy partitions are not nested')
    return names, torch.stack(assignments)


def load_corpus():
    corpus = read(HERE / 'assets/knowledge_revision.json')
    extra = read(HERE / 'assets/additional_descriptions.json')
    for kind in ('oe', 'ae'):
        if set(corpus[kind]) != set(extra[kind]):
            raise ValueError('Additional description label mismatch')
        for label in corpus[kind]:
            corpus[kind][label] = corpus[kind][label] + extra[kind][label]
            if len(corpus[kind][label]) != 5 or len(set(corpus[kind][label])) != 5:
                raise ValueError('Five distinct sentences required')
    corpus['provenance']['revision'] = extra['provenance']
    return corpus


def encode_assets(original, glove):
    labels = read(original / 'labels.json')
    corpus = load_corpus()
    parents, ids = hierarchy(corpus, labels['objs'])
    spec = importlib.util.spec_from_file_location('glove_builder', HERE / 'source/tools/build_knowledge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sentences = [s for kind in ('oe', 'ae') for group in corpus[kind].values() for s in group]
    sentences += [s for group in parents for s in group]
    needed = {t for s in sentences for t in module.description_tokens(s)}
    vectors, fingerprint = module.load_needed_glove(glove, needed)
    if fingerprint != GLOVE_SHA:
        raise ValueError('GloVe differs from audited original')
    oov = {}
    def encode(sentence):
        tokens = module.description_tokens(sentence)
        missing = [t for t in tokens if t not in vectors]
        oov[sentence] = missing
        retained = [vectors[t] for t in tokens if t in vectors]
        if not retained:
            raise ValueError('Entire sentence is OOV: ' + sentence)
        value = torch.stack(retained).mean(0)
        if value.norm() < 1e-8:
            raise ValueError('Zero description embedding')
        return F.normalize(value, dim=0)
    descriptions = {kind: torch.stack([torch.stack([encode(s) for s in corpus[kind][label]])
                            for label in labels[key]]) for kind, key in (('oe','objs'), ('ae','attrs'))}
    maximum = max(map(len, parents))
    pe, pm = torch.zeros(len(parents), maximum, 300), torch.zeros(len(parents), maximum, dtype=torch.bool)
    for i, names in enumerate(parents):
        pe[i, :len(names)] = torch.stack([encode(s) for s in names])
        pm[i, :len(names)] = True
    return descriptions, pe, ids, pm, {'corpus': corpus, 'parents': parents, 'oov': oov,
             'glove_sha256': fingerprint, 'labels': labels,
             'knowledge_source': 'assistant reconstruction, labels only, no images/VAL/TEST statistics'}


def sample_variant(encoded, count=3, noise=0, shuffle=False):
    desc, pe, ids, pm, evidence = encoded
    bank, masks, evidence = {}, {}, copy.deepcopy(evidence)
    permutations = {}
    for kind in ('oe','ae'):
        values = desc[kind][:, :count].clone()
        mask = deletion_mask(len(values), count, noise/100.0, 603360 + len(values))
        if shuffle:
            from legacy_prepare import derangement
            permutation = derangement(len(values), 603352 if kind == 'oe' else 603353)
            values = values[permutation]
            permutations[kind] = permutation.tolist()
        bank[kind], masks[kind] = values, mask
    evidence.update(description_count=count, noise_percent=noise, permutations=permutations,
                    deleted_slots={k: (~m).nonzero().tolist() for k,m in masks.items()})
    return bank, masks, pe, ids, pm, evidence


def materialize(original, dest, encoded, count=3, noise=0, shuffle=False, aggregation='mean', fixed_k=None):
    if count not in (1,3,5):
        raise ValueError('Supported description counts: 1,3,5')
    shutil.copytree(original, dest)
    bank, masks, pe, ids, pm, evidence = sample_variant(encoded, count, noise, shuffle)
    manifest = read(dest / 'manifest.json')
    for kind, raw, out in (('oe','W_o_name_raw.pt','W_o_K.pt'),('ae','W_a_name_raw.pt','W_a_K.pt')):
        names = torch.load(dest / raw, weights_only=True).float()
        pooled = aggregate_descriptions(names, bank[kind], masks[kind], aggregation)
        torch.save(pooled, dest / out)
        # If all descriptions were removed, use NAME for semantic branch,
        # not an invisible surviving description. Record the fallback explicitly.
        empty = ~masks[kind].any(-1)
        evidence[kind + '_name_fallback_classes'] = empty.nonzero().flatten().tolist()
        if empty.any():
            bank[kind][empty, 0] = F.normalize(names[empty], dim=-1)
            masks[kind][empty, 0] = True
        bank[kind + '_mask'] = masks[kind]
    eo = torch.load(dest / 'W_o_K.pt', weights_only=True)
    ea = torch.load(dest / 'W_a_K.pt', weights_only=True)
    ep = torch.stack([torch.load(dest / ('W_ao_K/obj_{:05d}.pt'.format(i)), weights_only=True) for i in range(len(eo))])
    similarity = F.cosine_similarity(ep, F.normalize(eo[:,None] + ea[None], dim=-1), dim=-1)
    k = (similarity-similarity.min()) / (similarity.max()-similarity.min()+1e-8)
    if fixed_k is not None:
        k = fixed_k.clone()
    # Retain old loader's sample-std contract. New model explicitly recomputes
    # its own population-std/clamped-log prior; never silently alters old cache.
    torch.save(k, dest / 'K_oa_text.pt')
    torch.save((k-k.mean())/(k.std()+1e-8), dest / 'K_oa_prior_centered.pt')
    bank.update(parent=pe, parent_ids=ids, parent_mask=pm, K=k)
    torch.save(bank, dest / 'revision_bank.pt')
    evidence.update(aggregation=aggregation, prior_fixed=fixed_k is not None,
                    pair_templates_preserved=True, ontology_levels=[8,5,3])
    write(dest / 'revision_evidence.json', evidence)
    for name in ('W_o_K.pt','W_a_K.pt','K_oa_text.pt','K_oa_prior_centered.pt','revision_bank.pt','revision_evidence.json'):
        manifest['files_sha256'][name] = sha(dest / name)
    manifest.update(knowledge_status='unified_0601_revision_20261013',
                    revision='TextK and descriptor constraints use the SAME reconstructed corpus; not original author assets',
                    control=evidence)
    manifest['parent_cache_manifest_sha256'] = sha(original / 'manifest.json')
    manifest['legacy_input_sha256'] = manifest.pop('input_sha256', {})
    manifest['input_sha256'] = {str(HERE/'assets'/name):sha(HERE/'assets'/name)
                              for name in ('knowledge_revision.json','additional_descriptions.json')}
    manifest['input_provenance'] = 'Current corpus above; original name/pair vectors and GloVe identities retained in parent cache manifest and revision_evidence.json'
    write(dest / 'manifest.json', manifest)
    write(dest / 'cache_manifest.json', manifest)
    return k


def external_prior(path, labels, mode, text_k):
    """All-cartesian independent feasibility. Never derive scores from pair splits."""
    value = read(path)
    if value.get('labels') != labels or value.get('score_axes') != ['object','attribute']:
        raise ValueError('External feasibility labels/axes must match exactly')
    provenance = value.get('provenance', {})
    if provenance.get('used_split_membership') is not False or provenance.get('used_validation_or_test_results') is not False:
        raise ValueError('External scores must explicitly exclude split/result leakage')
    if not all(provenance.get(k) for k in ('generator','prompt','generated_at')):
        raise ValueError('External prior provenance incomplete')
    scores, known = torch.tensor(value['scores'], dtype=torch.float32), torch.tensor(value['known'], dtype=torch.bool)
    if scores.shape != text_k.shape or known.shape != text_k.shape or not torch.isfinite(scores).all() or (scores<0).any() or (scores>1).any():
        raise ValueError('External feasibility must be finite all-cartesian [0,1]')
    if not known.any():
        raise ValueError('No actual external scores supplied; an empty/neutral template is NOT an LLM experiment')
    if mode == 'llm':
        if not known.all():
            raise ValueError('LLM-only requires every pair scored; missing != infeasible')
        return scores
    if mode == 'mixed':
        return torch.where(known, 0.5*text_k+0.5*scores, text_k)
    raise ValueError('Unknown external prior mode')
