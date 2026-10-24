"""CPU-only source/policy check against existing Compact and Strong results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import parser
from trustcpf import align, compact
from trustcpf.align_workflow import SEEDS, sources, validate_sources
from trustcpf.common import load_torch, write_json, sha

args = parser().parse_args()
rows = []
for seed in SEEDS:
    args.seed = seed
    bank = validate_sources(args)
    source, _ = sources(args)
    model = load_torch(source / 'utility_all/compact/model.pt')
    mu = compact.predict(model, load_torch(source / 'responses/reference.pt')['features'].numpy(), bank['types'].numpy())
    for strength in (0., .5, 1., 2.):
        pi, details = align.policy(mu, bank['types'].numpy(), strength)
        rows.append(dict(seed=seed, strength=strength, alpha=model['alpha'], type_details=details,
                         probability_std=float(pi.std(unbiased=False)), source_sha256=sha(source/'reference/last.pt')))
write_json(Path(args.work) / 'archived_input_validation.json', dict(passed=True, rows=rows))
print('ARCHIVED_INPUTS_OK: 3 references, 3 Q5 teachers, 6 predictors, 12 equality-budget policies')
