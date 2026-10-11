import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import normalize_legacy_checkpoints as repair

with tempfile.TemporaryDirectory(prefix="ekr_checkpoint_fix_") as directory:
    root = Path(directory)
    for variant in ("A", "C"):
        rid = "mvp_ut-zappos_{}_legacy_s3351_l0".format(variant)
        run = root / "runs" / rid
        checkpoint = run / "checkpoints" / ("ut-zappos-czsl_" + rid) / "best_auc.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(("original-" + variant).encode())
        (run / "train.log").write_text("Loading validation-best checkpoint from epoch 6: "
              "/data/xhw/czsl_repro/experiments/old/runs/" + rid + "/checkpoints/ut-zappos-czsl_"
              + rid + "/best_auc.pth\n")
        for setting in ("cw", "ow"):
            path = run / ("eval_val_" + setting) / "metrics.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"dataset": "ut-zappos", "variant": variant,
                "seed": 3351, "phase": "val", "setting": setting,
                "checkpoint_sha256": repair.sha(checkpoint)}))
    items = repair.plan(root, ["ut-zappos"])
    assert len(items) == 2
    with patch.object(Path, "symlink_to") as creation:
        repair.apply(items)
        assert creation.call_count == 2
        for call in creation.call_args_list:
            assert not Path(call.args[0]).is_absolute()
    alias = items[0][0]
    alias.write_bytes(b"different existing file")
    try:
        repair.plan(root, ["ut-zappos"])
        raise AssertionError("Wrong existing weight must not be overwritten")
    except ValueError as e:
        assert "refusing overwrite" in str(e)
    assert alias.read_bytes() == b"different existing file"
print("PATH FIX CHECKS PASSED: hash checks, exact logged path, relative aliases, no overwrite")
