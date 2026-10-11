#!/usr/bin/env python3
"""Generate non-official, review-required OI/OE/AE drafts for three CZSL datasets.

This script never uses images, image labels, or pair membership to infer semantic
knowledge. Pair lists are used only to obtain the exact primitive vocabulary.
"""

import argparse
import csv
import hashlib
import json
import re
import tarfile
import zipfile
from collections import OrderedDict
from pathlib import Path


DATASETS = ("ut-zap50k", "mit-states", "cgqa")
SPLITS = ("train", "val", "test")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(data, source):
    result = []
    for number, raw in enumerate(data.decode("utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("{}:{}: expected attr object, got {!r}".format(source, number, line))
        result.append(tuple(parts))
    return result


def canonical_digest(pairs):
    data = "".join("{}\t{}\n".format(a, o) for a, o in sorted(set(pairs)))
    return sha256_bytes(data.encode("utf-8"))


def load_labels(dataset, archive, cgqa_dir, source_data_root=None):
    by_split = {}
    if source_data_root is not None:
        for split in SPLITS:
            path = source_data_root / dataset / "compositional-split-natural" / (split + "_pairs.txt")
            by_split[split] = read_pairs(path.read_bytes(), path)
    elif dataset == "cgqa":
        for split in SPLITS:
            path = cgqa_dir / (split + "_pairs.txt")
            by_split[split] = read_pairs(path.read_bytes(), path)
    else:
        with tarfile.open(str(archive), "r:gz") as tar:
            for split in SPLITS:
                name = "{}/compositional-split-natural/{}_pairs.txt".format(dataset, split)
                by_split[split] = read_pairs(tar.extractfile(name).read(), name)
    union = [pair for split in SPLITS for pair in by_split[split]]
    attrs = sorted({a for a, _ in union})
    objs = sorted({o for _, o in union})
    return attrs, objs, by_split


def display(label):
    return re.sub(r"\s+", " ", label.replace("_", " ").replace(".", " ")).strip()


def look_up(wn, label, pos_list):
    phrase = display(label).lower().replace(" ", "_")
    queries = [phrase]
    if "_" in phrase:
        queries.append(phrase.split("_")[-1])
    for query in queries:
        for pos in pos_list:
            senses = wn.synsets(query, pos=pos)
            if senses:
                return senses[0], query, len(senses), query != phrase
    return None, phrase, 0, False


def clean_text(value):
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()


def distinct_three(candidates, label, kind):
    descriptions = []
    seen = set()
    for item in candidates:
        item = clean_text(item)
        key = item.casefold()
        if item and key not in seen:
            descriptions.append(item)
            seen.add(key)
        if len(descriptions) == 3:
            break
    if len(descriptions) != 3:
        raise ValueError("cannot make three {} texts for {}".format(kind, label))
    return descriptions


def make_object(wn, label):
    syn, query, n_senses, used_head = look_up(wn, label, [wn.NOUN])
    name = display(label)
    if syn is None:
        parent = "unmapped_object"
        descriptions = distinct_three([
            "A visible object labeled {}.".format(name),
            "The object category is {}.".format(name),
            "An instance of the {} category.".format(name),
        ], label, "object")
        return parent, descriptions, {
            "label": label, "kind": "object", "query": query,
            "synset": "", "sense_count": 0, "used_head_fallback": False,
            "parent": parent, "quality": "label_template_only",
            "review_reason": "no WordNet noun sense; all texts are placeholders",
        }
    hypernyms = sorted(syn.hypernyms() + syn.instance_hypernyms(), key=lambda x: x.name())
    hypernym = clean_text(hypernyms[0].lemmas()[0].name()) if hypernyms else "object"
    parent = syn.lexname().replace("noun.", "")
    description = syn.definition().strip().rstrip(".")
    candidate_texts = [
        "{}: {}.".format(name, description),
        "{} is a kind of {}.".format(name, hypernym),
        "{} belongs to the {} concept group.".format(name, parent),
    ]
    descriptions = distinct_three(candidate_texts, label, "object")
    reasons = []
    if used_head:
        reasons.append("head-word fallback may change meaning")
    if n_senses > 1:
        reasons.append("polysemous: selected first of {} senses".format(n_senses))
    return parent, descriptions, {
        "label": label, "kind": "object", "query": query,
        "synset": syn.name(), "sense_count": n_senses,
        "used_head_fallback": used_head, "parent": parent,
        "quality": "wordnet_auto_unreviewed",
        "review_reason": "; ".join(reasons) or "automatic gloss still requires semantic review",
    }


def make_attribute(wn, label):
    syn, query, n_senses, used_head = look_up(
        wn, label, [wn.ADJ, wn.ADJ_SAT, wn.VERB, wn.NOUN]
    )
    name = display(label)
    if syn is None:
        descriptions = distinct_three([
            "An object described as {}.".format(name),
            "The object's observable state is {}.".format(name),
            "A property or material named {}.".format(name),
        ], label, "attribute")
        return descriptions, {
            "label": label, "kind": "attribute", "query": query,
            "synset": "", "sense_count": 0, "used_head_fallback": False,
            "parent": "", "quality": "label_template_only",
            "review_reason": "no WordNet sense; all texts are placeholders",
        }
    description = syn.definition().strip().rstrip(".")
    related = []
    for item in syn.similar_tos() + syn.attributes() + syn.also_sees():
        related.append(clean_text(item.lemmas()[0].name()))
    related = [item for item in related if item.casefold() != name.casefold()]
    descriptions = distinct_three([
        "{}: {}.".format(name, description),
        "{} is a property related to {}.".format(name, related[0]) if related
        else "An object can be described as {}.".format(name),
        "The visual or semantic state of an object is {}.".format(name),
    ], label, "attribute")
    reasons = []
    if used_head:
        reasons.append("head-word fallback may change meaning")
    if n_senses > 1:
        reasons.append("polysemous: selected first of {} senses".format(n_senses))
    if syn.pos() not in ("a", "s"):
        reasons.append("non-adjective sense selected")
    return descriptions, {
        "label": label, "kind": "attribute", "query": query,
        "synset": syn.name(), "sense_count": n_senses,
        "used_head_fallback": used_head, "parent": "",
        "quality": "wordnet_auto_unreviewed",
        "review_reason": "; ".join(reasons) or "automatic gloss still requires semantic review",
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path, values):
    path.write_text("".join(value + "\n" for value in values), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-tar", type=Path, required=True)
    parser.add_argument("--cgqa-split-dir", type=Path, required=True)
    parser.add_argument("--ut-existing-dir", type=Path, required=True)
    parser.add_argument("--wordnet-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path,
                        help="override all archived/mirrored splits with exact server split files")
    args = parser.parse_args()

    import nltk
    nltk.data.path.insert(0, str(args.wordnet_dir))
    from nltk.corpus import wordnet as wn
    wn.synsets("shoe")  # fail before writing if WordNet is unavailable

    args.output_root.mkdir(parents=True, exist_ok=True)
    wordnet_zip = args.wordnet_dir / "corpora" / "wordnet.zip"
    with zipfile.ZipFile(str(wordnet_zip)) as package:
        license_text = package.read("wordnet/LICENSE").decode("utf-8")
    (args.output_root / "WORDNET_LICENSE.txt").write_text(license_text, encoding="utf-8")

    for dataset in DATASETS:
        attrs, objs, by_split = load_labels(
            dataset, args.split_tar, args.cgqa_split_dir, args.source_data_root
        )
        target = args.output_root / dataset
        target.mkdir(parents=True, exist_ok=True)
        write_lines(target / "attributes.txt", attrs)
        write_lines(target / "objects.txt", objs)

        provenance = []
        if dataset == "ut-zap50k":
            oi = json.loads((args.ut_existing_dir / "oi_dict.json").read_text(encoding="utf-8"))
            oe = json.loads((args.ut_existing_dir / "oe_dict.json").read_text(encoding="utf-8"))
            ae = json.loads((args.ut_existing_dir / "ae_dict.json").read_text(encoding="utf-8"))
            for label in objs:
                provenance.append({
                    "label": label, "kind": "object", "query": "", "synset": "",
                    "sense_count": "", "used_head_fallback": "", "parent": "",
                    "quality": "manual_reconstruction_unreviewed",
                    "review_reason": "verify prior hand-written UT-Zappos knowledge",
                })
            for label in attrs:
                provenance.append({
                    "label": label, "kind": "attribute", "query": "", "synset": "",
                    "sense_count": "", "used_head_fallback": "", "parent": "",
                    "quality": "manual_reconstruction_unreviewed",
                    "review_reason": "verify prior hand-written UT-Zappos knowledge",
                })
        else:
            oi = {}
            oe = OrderedDict()
            ae = OrderedDict()
            for label in objs:
                parent, descriptions, record = make_object(wn, label)
                oi.setdefault(parent, []).append(label)
                oe[label] = descriptions
                provenance.append(record)
            for label in attrs:
                descriptions, record = make_attribute(wn, label)
                ae[label] = descriptions
                provenance.append(record)
            oi = OrderedDict((key, sorted(value)) for key, value in sorted(oi.items()))

        # Structural validation before writing the package.
        oi_children = [child for children in oi.values() for child in children]
        if sorted(oi_children) != objs or len(oi_children) != len(set(oi_children)):
            raise ValueError("{} OI coverage or uniqueness failure".format(dataset))
        if list(oe) != objs or list(ae) != attrs:
            raise ValueError("{} OE/AE key order mismatch".format(dataset))
        for label, texts in list(oe.items()) + list(ae.items()):
            if len(texts) != 3 or len({x.strip().casefold() for x in texts}) != 3:
                raise ValueError("{} has invalid three texts".format(label))

        write_json(target / "oi_dict.json", oi)
        write_json(target / "oe_dict.json", oe)
        write_json(target / "ae_dict.json", ae)
        with (target / "entry_provenance.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = ["kind", "label", "query", "synset", "sense_count",
                      "used_head_fallback", "parent", "quality", "review_reason"]
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(provenance)
        with (target / "review_status.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["kind", "label", "status", "reviewer", "date", "note"])
            for record in provenance:
                writer.writerow([record["kind"], record["label"], "pending", "", "", ""])

        manifest = {
            "dataset": dataset,
            "status": "complete_coverage_unreviewed_draft",
            "not_official_logiczsl_knowledge": True,
            "compatible_with_formal_results": False,
            "split_name": "compositional-split-natural",
            "counts": {"attributes": len(attrs), "objects": len(objs),
                       "parents": len(oi), "descriptions_per_primitive": 3},
            "canonical_pair_sha256": {
                split: canonical_digest(by_split[split]) for split in SPLITS
            },
            "source": "prior manual reconstruction" if dataset == "ut-zap50k"
                      else "Princeton WordNet 3.0 first-sense lexical draft",
            "source_split": (
                "exact server split files" if args.source_data_root is not None else
                ("local compositional_split_natural.tar.gz" if dataset != "cgqa"
                 else "Hugging Face mirror of updated C-GQA pair lists")
            ),
            "source_split_url": "https://huggingface.co/datasets/Meldashti/vlm-compositionality-embeddings/tree/main/cgqa/compositional-split-natural"
                                if dataset == "cgqa" else "",
            "source_fingerprints_sha256": (
                {split + "_pairs.txt": sha256_file(
                    args.source_data_root / dataset / "compositional-split-natural" /
                    (split + "_pairs.txt")) for split in SPLITS}
                if args.source_data_root is not None else
                ({split + "_pairs.txt": sha256_file(args.cgqa_split_dir /
                    (split + "_pairs.txt")) for split in SPLITS}
                 if dataset == "cgqa" else
                 {"compositional_split_natural.tar.gz": sha256_file(args.split_tar)})
            ),
            "prior_ut_json_sha256": (
                {name: sha256_file(args.ut_existing_dir / name) for name in
                 ("oi_dict.json", "oe_dict.json", "ae_dict.json")}
                if dataset == "ut-zap50k" else {}
            ),
            "wordnet_zip_sha256": sha256_file(wordnet_zip) if dataset != "ut-zap50k" else "",
            "generator_sha256": sha256_file(Path(__file__)),
            "knowledge_sha256": {name: sha256_file(target / name) for name in
                                 ("oi_dict.json", "oe_dict.json", "ae_dict.json")},
            "compatibility_notice": "generate.py makes no prior. build_text_prior.py may add a non-official lexical candidate K_oa. Never use the union of split pairs as a semantic compatibility prior.",
        }
        write_json(target / "manifest.json", manifest)
        print("{}: attrs={} objs={} parents={} status=draft".format(
            dataset, len(attrs), len(objs), len(oi)))


if __name__ == "__main__":
    main()
