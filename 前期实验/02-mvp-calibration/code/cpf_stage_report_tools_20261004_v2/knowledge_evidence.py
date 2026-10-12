"""Snapshot knowledge source schemas against the cache's frozen input hashes.

Standard library only; never imports torch or changes source files.
Tensor shapes below are manifest contracts, not a fresh tensor-content audit.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

REQUIRED = ["oi_dict.json", "oe_dict.json", "ae_dict.json", "attributes.txt",
            "objects.txt", "review_status.csv", "manifest.json"]
OPTIONAL = ["entry_provenance.csv", "K_oa_text_lexical.csv", "K_oa_text_report.json"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def basename(value):
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def collect_knowledge(ev, dataset, folder, cache_manifest, knowledge_root=None):
    attrs, objs = cache_manifest["attrs"], cache_manifest["objs"]
    inputs = cache_manifest.get("input_sha256", {})
    schema = dict(dataset=dataset, source_status="unverified", files={},
                  cache_knowledge_status=cache_manifest.get("knowledge_status"),
                  tensor_shapes_are_manifest_contracts=True,
                  cache_shapes={"W_o_K.pt": [len(objs), 300], "W_a_K.pt": [len(attrs), 300],
                                "K_oa_text.pt": [len(objs), len(attrs)],
                                "K_oa_prior_centered.pt": [len(objs), len(attrs)],
                                "W_ao_K/obj_XXXXX.pt": [len(attrs), 300]},
                  model_consumption={"A": "baseline names only",
                                     "C": "W_o_K + W_a_K; no OI graph propagation",
                                     "D": "centered K calibration only",
                                     "E": "primitive text vectors + centered K calibration"},
                  human_review_inferred=False)
    row = dict(dataset=dataset, source_status="unverified",
               knowledge_status=cache_manifest.get("knowledge_status"),
               human_review_inferred=False)
    try:
        if cache_manifest.get("dimension") != 300 or cache_manifest.get("pair_axis_order") != ["object", "attribute"]:
            raise ValueError("unexpected dimension or pair axis order")
        if knowledge_root:
            root = (knowledge_root / folder).resolve()
        else:
            sources = [Path(p).parent for p in inputs if basename(p) in ("oi_dict.json", "oe_dict.json", "ae_dict.json")]
            if len(sources) != 3 or len(set(map(str, sources))) != 1:
                raise ValueError("cannot infer one source directory from cache input_sha256; supply --knowledge-root")
            root = sources[0]
        row["source_directory"] = str(root)
        schema["source_directory"] = str(root)
        snapshots = {}
        for name in REQUIRED + OPTIONAL:
            snap = ev.copy(root / name, Path("knowledge_source") / folder / name)
            expected = [h for p, h in inputs.items() if basename(p) == name]
            if name in REQUIRED and (snap is None or len(expected) != 1):
                raise ValueError("required source file/frozen digest missing: " + name)
            if snap is None:
                schema["files"][name] = dict(status="not_available", frozen_in_cache=False)
                continue
            actual = digest(snap)
            stable = ev.files[-1]["source_stable_during_copy"]
            if not stable:
                raise ValueError("source changed during snapshot: " + name)
            if expected and (len(expected) != 1 or actual != expected[0]):
                raise ValueError("source does not match training-cache input hash: " + name)
            snapshots[name] = snap
            schema["files"][name] = dict(snapshot=str(snap.relative_to(ev.out)).replace("\\", "/"),
                                          sha256=actual, frozen_in_cache=bool(expected),
                                          status="frozen_input_verified" if expected else "current_supplement_not_frozen_input")
        source_manifest = load(snapshots["manifest.json"])
        if source_manifest.get("dataset") != folder:
            raise ValueError("source manifest dataset differs from cache directory")
        schema["source_manifest"] = {k: source_manifest.get(k) for k in
                                     ["status", "source", "not_official_logiczsl_knowledge", "compatible_with_formal_results", "counts"]}
        # Source-manifest checks bind optional lexical artifacts to the frozen
        # source manifest, but do not make them the selected prior automatically.
        for name, entry in schema["files"].items():
            h = source_manifest.get("knowledge_sha256", {}).get(name)
            if h and name in snapshots:
                if entry["sha256"] != h:
                    raise ValueError("frozen source-manifest checksum differs: " + name)
                entry["source_manifest_digest_verified"] = True
        for name, labels in [("attributes.txt", attrs), ("objects.txt", objs)]:
            if snapshots[name].read_text(encoding="utf-8-sig").splitlines() != labels:
                raise ValueError("source labels/order differ from cache: " + name)
        dictionaries = {kind: load(snapshots[kind + "_dict.json"]) for kind in ["oi", "oe", "ae"]}
        for kind, d in dictionaries.items():
            if not isinstance(d, dict) or any(not isinstance(k, str) or not isinstance(v, list) or
                                             any(not isinstance(s, str) or not s.strip() for s in v)
                                             for k, v in d.items()):
                raise ValueError("invalid JSON dictionary schema: " + kind)
        children = [x for v in dictionaries["oi"].values() for x in v]
        if sorted(children) != sorted(objs) or len(children) != len(set(children)):
            raise ValueError("OI does not cover each object exactly once")
        descriptions = source_manifest.get("counts", {}).get("descriptions_per_primitive", 3)
        for kind, labels in [("oe", objs), ("ae", attrs)]:
            d = dictionaries[kind]
            if list(d) != labels or any(len(v) != descriptions for v in d.values()):
                raise ValueError("OE/AE keys/order or description count mismatch: " + kind)
        reviews, review_fields = csv_rows(snapshots["review_status.csv"])
        expected_ids = {(k, s) for k, labels in [("object", objs), ("attribute", attrs)] for s in labels}
        if {(r["kind"], r["label"]) for r in reviews} != expected_ids or len(reviews) != len(expected_ids):
            raise ValueError("review identities/coverage mismatch")
        counts = Counter(r["status"] for r in reviews)
        if any(r["status"] != "approved" for r in reviews):
            raise ValueError("frozen builder source has entries not marked approved")
        primitive_counts = dict(n_attrs=len(attrs), n_objs=len(objs), n_parents=len(dictionaries["oi"]),
                                n_oi_memberships=len(children),
                                n_oe_descriptions=sum(map(len, dictionaries["oe"].values())),
                                n_ae_descriptions=sum(map(len, dictionaries["ae"].values())),
                                n_review_entries=len(reviews), review_approved=counts.get("approved", 0),
                                review_pending=counts.get("pending", 0),
                                review_nonempty_reviewer=sum(bool(r.get("reviewer", "").strip()) for r in reviews))
        row.update(primitive_counts)
        row.update(source_status="verified_against_cache_inputs", source_origin=source_manifest.get("source"),
                   source_manifest_status=source_manifest.get("status"))
        schema.update(source_status=row["source_status"], counts=primitive_counts,
                      review_fields=review_fields, review_status_counts=dict(counts))
        schema["examples"] = {kind: {key: dictionaries[kind][key]} for kind in ["oi", "oe", "ae"]
                              for key in list(dictionaries[kind])[:1]}
        if "entry_provenance.csv" in snapshots:
            provenance, fields = csv_rows(snapshots["entry_provenance.csv"])
            if len(provenance) != len(expected_ids) or {(r["kind"], r["label"]) for r in provenance} != expected_ids:
                raise ValueError("provenance identities/coverage mismatch")
            schema["provenance_fields"] = fields
            schema["provenance_quality_counts"] = dict(Counter(r.get("quality", "") for r in provenance))
            schema["provenance_example"] = provenance[:1]
            row["provenance_rows"] = len(provenance)
        else:
            ev.issue(dataset + "_knowledge", "entry_provenance.csv unavailable; add a source-provenance explanation")
        return row, schema
    except Exception as e:
        row["source_status"] = schema["source_status"] = "unverified"
        row["issue"] = schema["issue"] = type(e).__name__ + ": " + str(e)
        ev.issue(dataset + "_knowledge", row["issue"])
        return row, schema
