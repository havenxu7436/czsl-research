"""Export existing VAL predictions for the reviewed UT v2 result. No training or images."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED = {'prepared.json': '20fd2e71e4c0673f19ab336f368484afaa8cc408e921eb048e2c4154fd266708', 'experiments/semantic_v2/protocol/integrity_v2.json': '3140e2b4f1bb93614c236e6aac08b616d43fc2c222970343b5225ad3ac3b4ede', 'experiments/semantic_v2/protocol/semantic_gate_v2.json': 'cf9c71ca290c0436799cbfe08a563214488e78475e0a75dc3a93b4618015c89d', 'experiments/semantic_v2/reports/paired_v2/collection.json': '8bc6887cc60cc712b18315836ef7480a44f3a7e6fc3d7e94f9122915a8bdcb54', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s0_e10/eval_val_cw/metrics.json': '8526f51eef476547ca8d3f837ee41879f658dd095c2503556be38e2d9533ef29', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s0_e10/eval_val_ow/metrics.json': '24a55dabddbf41247976b477c77a5e34677bdd5d4c8ccce5f4d40fc5471e7341', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s2026_e10/eval_val_cw/metrics.json': 'ef2f32fd04b2f89996b9839fcd93c776740ad9bae5caffb8b510c4b7c1494055', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s2026_e10/eval_val_ow/metrics.json': '4c055379d5770a1ed5c53b6245ea6cb4c670f9f2b1182d366db49a2a4f180d7b', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s3351_e10/eval_val_cw/metrics.json': 'c14d47bafb0f2d25687f2d910522defe4aec00e413962879e9deced1ccd241d5', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_raw_s3351_e10/eval_val_ow/metrics.json': 'f3af3eb5ff9b1eff78149971aee2d9b5a54ea3a057c301d9007ab4d6dd647e19', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s0_e10/eval_val_cw/metrics.json': 'eceb22cc0baf8c74a9515d56cecc21687aa659f59f8c8383a96d3f07b56212a5', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s0_e10/eval_val_ow/metrics.json': 'd07d61140d1b14f7bfc1a4a5e4bf356a79be3d9a8713c28cb0f79fce30cf31a8', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s2026_e10/eval_val_cw/metrics.json': '1f8c3d5374ce4b903d0c4a21f6abbdfa2b4e417fd5343e80fa4b29110fb1cfe8', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s2026_e10/eval_val_ow/metrics.json': 'd24f9c65ec0ad7202cddc7a79d64b801cef5a992b05f80a10895c2392d3b2b9f', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s3351_e10/eval_val_cw/metrics.json': '51b3266ce25bf54f0a0df1c414dd8721b317eddba3a6a9b3db048e9e3530bcdd', 'experiments/semantic_v2/runs/semantic_ut-zappos_A_unit_s3351_e10/eval_val_ow/metrics.json': '82d68c860a781a1ee812a2d4191a44e00b91f4bb604f620fde8ccb0f4dbe2ccb', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s0_e10/eval_val_cw/metrics.json': '81bb1103a581f0bc3afeb337c1fce9e481452ca85ca1a0bc392080f2e1d31c27', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s0_e10/eval_val_ow/metrics.json': '8747c6759e6fa8c323e7693b3bb7d71bd3203d8c96e849368312c0bf1e114172', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s2026_e10/eval_val_cw/metrics.json': '73f9deda083f26d558cbff4d0f9a35031bd9ad57859a9b48b8e82138e537b4e6', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s2026_e10/eval_val_ow/metrics.json': '744a5d7135eec6d77ba6427c589438e6cdbb70e5c6c38702e81c9bf0503a1813', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s3351_e10/eval_val_cw/metrics.json': '123446f602c22e98148c27f7fd295438b37234d9db3f599948ccd85d8fc08c02', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_shuffle_s3351_e10/eval_val_ow/metrics.json': '5fded4b15852aea89e0fe400323a9766f5c8a22714daacc3d4b64faa7302d64c', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s0_e10/eval_val_cw/metrics.json': '45d3681d02f75951573b2946313736deac2ba1b8066e2bcec5123e76b1198b96', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s0_e10/eval_val_ow/metrics.json': '3b90c141e7014a48620398be6f106c75126b71b755949236f4d2271ac1a04d1b', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s2026_e10/eval_val_cw/metrics.json': '8ce16ac97d51874430ce699ca70b74e88b5c89dd7ede8b19161827bbb2286b65', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s2026_e10/eval_val_ow/metrics.json': '079cc749c23e2ec11f395e23ef09c59d8fd68c40cd230c839f40f12efb78bcfa', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s3351_e10/eval_val_cw/metrics.json': 'ec7b1dffe936151f489eca99d83900d8cfb6d68da8c0bbcb2e548103afd3ce05', 'experiments/semantic_v2/runs/semantic_ut-zappos_C_text_s3351_e10/eval_val_ow/metrics.json': 'a7e0a988b57811ab70f758938838ad2d260267abac9e97d23925bcbdf804ad62'}
SOURCE_REVIEW_SHA256 = "31285ef2b7d1fdea65049b322246d623f8a3073cf373342d05e67abe0fa7e022"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def contained_file(work, relative):
    path = (work / relative).resolve()
    try:
        path.relative_to(work)
    except ValueError:
        raise ValueError("File resolves outside work directory: " + relative)
    if not path.is_file():
        raise FileNotFoundError("Required completed VAL artifact absent: " + relative)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/var/tmp/czsl_xhw_ut_v2_20261006"))
    parser.add_argument("--out", type=Path, default=Path("/var/tmp/ut_v2_val_predictions_20261007.zip"))
    args = parser.parse_args()
    work, out = args.work.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError("Existing export is preserved; choose a new --out path: " + str(out))
    contents = {}
    for relative, expected in EXPECTED.items():
        raw = contained_file(work, relative).read_bytes()
        if sha(raw) != expected:
            raise ValueError("Artifact changed since the reviewed result ZIP: " + relative)
        contents[relative] = raw
    metric_files = sorted(name for name in EXPECTED if name.endswith("/metrics.json"))
    if len(metric_files) != 24:
        raise ValueError("Expected exactly 24 completed VAL evaluations")
    for name in metric_files:
        record = json.loads(contents[name])
        if record["phase"] != "val" or record["setting"] not in ("cw", "ow") or record["n_images"] != 3214:
            raise ValueError("Unexpected evaluation phase or image count: " + name)
        for leaf in ("image_ids.json", "candidates.json", "labels.npz"):
            relative = str(Path(name).parent / leaf).replace("\\", "/")
            contents[relative] = contained_file(work, relative).read_bytes()
    manifest = {
        "version": "ut_v2_existing_val_predictions_20261007",
        "source_review_zip_sha256": SOURCE_REVIEW_SHA256,
        "work": str(work), "validation_evaluations": 24,
        "new_training_performed": False, "new_evaluation_performed": False,
        "image_pixels_opened": False, "test_evaluation_included": False,
        "weights_included": False,
        "files_sha256": {name:sha(raw) for name,raw in sorted(contents.items())},
        "limits": "Top1 argmax arrays permit error analysis; full scores are absent, so AUC/HM curves cannot be recomputed.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any existing export.
    with zipfile.ZipFile(out, "x", zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(contents.items()):
            archive.writestr(name, raw)
        archive.writestr("export_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(out) as archive:
        if archive.testzip() is not None:
            raise ValueError("Export ZIP CRC failed")
        for name,expected in manifest["files_sha256"].items():
            if sha(archive.read(name)) != expected:
                raise ValueError("Export entry checksum failed: " + name)
    print("EXPORTED EXISTING VAL PREDICTIONS", out)
    print("EVALUATIONS 24; NEW TRAINING 0; NEW EVALUATION 0; IMAGE PIXELS 0")
    print("ZIP SHA256", sha(out.read_bytes()))


if __name__ == "__main__":
    main()
