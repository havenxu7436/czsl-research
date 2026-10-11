#!/usr/bin/env python3
"""Build a frozen GloVe grounding bank; never invent a compatibility matrix."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
import torch
from common import sha, validate_knowledge, write_json


def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())


def build(knowledge_path, labels_path, glove_path, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Bank exists; use a new version: ' + str(destination))
    knowledge = json.loads(Path(knowledge_path).read_text(encoding='utf-8'))
    labels = json.loads(Path(labels_path).read_text(encoding='utf-8'))
    parents = validate_knowledge(knowledge, labels)
    groups = [parents, [s for o in labels['objs'] for s in knowledge['oe'][o]],
              [s for a in labels['attrs'] for s in knowledge['ae'][a]]]
    texts = [t for g in groups for t in g]
    wanted = {token for t in texts for token in tokenize(t)}
    vectors = {}
    glove_path = Path(glove_path)
    archive = None
    if glove_path.suffix.lower() == '.zip':
        archive = zipfile.ZipFile(glove_path)
        names = [name for name in archive.namelist() if name.endswith('glove.6B.300d.txt')]
        if len(names) != 1:
            raise ValueError('ZIP must contain one glove.6B.300d.txt')
        stream = archive.open(names[0])
    else:
        stream = glove_path.open('rb')
    fingerprint = hashlib.sha256()
    try:
        for raw in stream:
            fingerprint.update(raw)
            word = raw.split(b' ', 1)[0].decode('utf-8')
            if word in wanted:
                parts = raw.decode('utf-8').split()
                if len(parts) != 301:
                    raise ValueError('Not 300-dimensional GloVe')
                vectors[word] = torch.tensor([float(s) for s in parts[1:]], dtype=torch.float32)
    finally:
        stream.close()
        if archive:
            archive.close()
    coverage = []
    encoded = []
    for text in texts:
        tokens = tokenize(text)
        found = [v for t in tokens if (v := vectors.get(t)) is not None]
        missing = [t for t in tokens if t not in vectors]
        rate = len(found) / len(tokens) if tokens else 0
        coverage.append({'text': text, 'tokens': tokens, 'missing': missing, 'coverage': rate})
        if rate < 0.8 or not found:
            raise ValueError('Low GloVe coverage; revise knowledge, not fallback silently: ' + text)
        value = torch.stack(found).mean(dim=0)
        if not torch.isfinite(value).all() or value.norm() < 1e-8:
            raise ValueError('Invalid phrase embedding: ' + text)
        encoded.append(value / value.norm())
    encoded = torch.stack(encoded)
    n = len(parents)
    no, na = len(labels['objs']), len(labels['attrs'])
    bank = {'parent': encoded[:n], 'oe': encoded[n:n + no * 3].reshape(no, 3, 300),
            'ae': encoded[n + no * 3:].reshape(na, 3, 300)}
    object_parent = [next(i for i, p in enumerate(parents) if o in knowledge['oi'][p]) for o in labels['objs']]
    destination.mkdir(parents=True)
    torch.save(bank, destination / 'bank.pt')
    write_json(destination / 'coverage.json', coverage)
    write_json(destination / 'manifest.json', {
        'schema': 'cpf_logiczsl_bank_v1', 'labels': labels, 'parents': parents,
        'object_parent': object_parent, 'dimension': 300, 'descriptions_per_primitive': 3,
        'official_author_data': False, 'provenance': knowledge['provenance'],
        'knowledge_sha256': sha(knowledge_path), 'labels_sha256': sha(labels_path),
        'glove_text_sha256': fingerprint.hexdigest(), 'bank_sha256': sha(destination / 'bank.pt'),
        'encoding': 'regex English words -> mean available GloVe 6B 300d -> L2 normalize; coverage >= 0.8',
        'pair_prior': 'none: no split-derived compatibility matrix is constructed'
    })
    print('BANK_READY', destination, 'minimum_token_coverage=', min(x['coverage'] for x in coverage), flush=True)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--knowledge', type=Path, required=True)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--glove', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    build(a.knowledge, a.labels, a.glove, a.out)


if __name__ == '__main__':
    main()
