"""Small, dependency-light provenance helpers for the separate B/F release."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing overwrite: ' + str(path))
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        with path.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(content)


def write_json(path, value):
    write_new(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_knowledge(knowledge, labels):
    attrs, objs = labels['attrs'], labels['objs']
    if len(attrs) != len(set(attrs)) or len(objs) != len(set(objs)):
        raise ValueError('Duplicate primitive labels')
    oi, oe, ae = (knowledge[k] for k in ('oi', 'oe', 'ae'))
    if not oi or not isinstance(oi, dict):
        raise ValueError('OI needs nonempty parent groups')
    children = []
    for parent, group in oi.items():
        if not isinstance(parent, str) or not parent.strip() or not isinstance(group, list) or not group:
            raise ValueError('Empty OI parent/group')
        children.extend(group)
    if len(children) != len(set(children)) or set(children) != set(objs):
        raise ValueError('OI must cover every object exactly once')
    for mapping, expected in ((oe, objs), (ae, attrs)):
        if set(mapping) != set(expected):
            raise ValueError('Description keys differ from primitive labels')
        for primitive in expected:
            texts = mapping[primitive]
            if not isinstance(texts, list) or len(texts) != 3 or len(set(texts)) != 3:
                raise ValueError('Exactly three distinct descriptions required: ' + primitive)
            if not all(isinstance(t, str) and t.strip() for t in texts):
                raise ValueError('Invalid description: ' + primitive)
    return sorted(oi)
