"""Seal the exact existing generic run identities; no server changes or training."""
from pathlib import Path
from bootstrap import ROOT, child, read_json, sha, write_json


def main():
    workspace = ROOT.parent
    source = workspace / 'output/conditional_attribute_pilot_results_20261009_r2/extracted'
    pilot = source / 'pilot'
    reference_path = ROOT / 'experiment.json'
    if reference_path.exists():
        raise FileExistsError('Preserve existing experiment reference')
    audit = read_json(source.parent / 'independent_audit.json')
    if not audit['passed']:
        raise ValueError('Original pilot must have passed independent audit')
    old_protocol = read_json(pilot / 'protocol.json')
    old_release = read_json(source / 'release/release_manifest.json')
    legacy = ROOT / 'legacy'
    for name, expected in old_release.items():
        if sha(legacy / name) != expected:
            raise ValueError('Vendored legacy code is not the executed r2 release')
    if sha(legacy / 'release_manifest.json') != old_protocol['release_manifest_sha256']:
        raise ValueError('Original release manifest differs')
    summary = read_json(pilot / 'summary.json')
    if not summary['passed'] or summary['formal_optimizer_steps'] != 4500:
        raise ValueError('Original training budget incomplete')
    history = read_json(pilot / 'runs/P_generic/history.json')
    if history['cumulative_optimizer_steps'] != 900 or len(history['ledger']) != 11:
        raise ValueError('Generic source history incomplete')
    names = ['protocol.json', 'environment.json', 'baseline.json', 'labels.json',
             'val_image_ids.json', 'val_truth.npz', 'runs/P_generic/history.json',
             'runs/P_generic/complete.json', 'runs/P_generic/last.pt', 'runs/P_generic/best_auc.pt']
    for item in history['ledger']:
        for name, expected in item['files_sha256'].items():
            relative = 'runs/P_generic/' + item['path'] + '/' + name
            if sha(child(pilot, relative)) != expected:
                raise ValueError('Source generic epoch artifact changed')
            names.append(relative)
    server = '/var/tmp/czsl_conditional_pilot_20261009_r2'
    files = {server + '/' + name: sha(child(pilot, name)) for name in names}
    result = dict(schema='generic_mixture_controls_reference_v1', source_work=server,
                  source_protocol_sha256=sha(pilot / 'protocol.json'), source_files=files,
                  source_environment=read_json(pilot / 'environment.json'),
                  source_initial_state_sha256=old_protocol['initializations']['P_generic']['initial_state_sha256'],
                  source_old_gate=summary['decision']['pilot_budget_gate_passed'],
                  source_generic_formal_steps=900, alpha=.25, new_formal_variants=['G_name', 'G_perm1', 'G_perm2'],
                  source_history_epochs=11, source_original_protocol=old_protocol,
                  original_gates_unchanged=True, development_protocol_not_preregistered=True,
                  new_settings_identical_to_original_training=True)
    write_json(reference_path, result)
    print('GENERIC_SOURCE_REFERENCE_SEALED', len(files), 'files', sha(reference_path))


if __name__ == '__main__':
    main()
