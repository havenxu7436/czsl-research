#!/usr/bin/env python3
"""Export CZSL assets with Python 3.8+ standard library; never modify sources."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path, progress=False):
    h = hashlib.sha256()
    count, last = 0, time.monotonic()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
            count += len(block)
            if progress and time.monotonic() - last >= 20:
                print("  checksummed {:.2f} GiB: {}".format(count / 1024 ** 3, path.name), flush=True)
                last = time.monotonic()
    return h.hexdigest()


def inside(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def stage_check(root):
    exp = root / "experiments/kcc0601_glove_mvp_v1"
    checks = []

    def check(path, purpose, mode="file"):
        try:
            if mode == "status":
                ok = path.read_text().strip() == "train_and_val_complete"
            elif mode == "metrics":
                value = json.loads(path.read_text(encoding="utf-8"))
                ok = isinstance(value.get("metrics_raw"), dict) and "AUC" in value["metrics_raw"]
            elif mode == "checkpoint":
                ok = any(p.is_file() and p.stat().st_size > 0 for p in path.rglob("best_auc.pth"))
            elif mode == "directory":
                ok = path.is_dir()
            else:
                ok = path.is_file() and path.stat().st_size > 0
            error = "" if ok else "missing, empty, or not complete"
        except Exception as e:
            ok, error = False, str(e)
        checks.append(dict(purpose=purpose, path=str(path), passed=ok, detail=error))

    for name in ("cpf_ut_zappos_seed3351", "cpf_mit_states_seed0", "cpf_cgqa_seed3351_gpu2"):
        check(root / "runs" / name, "historical baseline checkpoint", "checkpoint")
    specs = [("ut-zappos", 3351, v, "0.1" if v in "DE" else "0") for v in "ACDE"]
    specs += [(ds, seed, v, "0") for ds, seed in (("mit-states", 0), ("cgqa", 3351)) for v in "AC"]
    for ds, seed, variant, lam in specs:
        rid = "mvp_{}_{}_legacy_s{}_l{}".format(ds, variant, seed, lam)
        run = exp / "runs" / rid
        check(run / "status.txt", rid + " status", "status")
        check(run, rid + " best checkpoint", "checkpoint")
        for setting in ("cw", "ow"):
            check(run / ("eval_val_" + setting) / "metrics.json", rid + " " + setting, "metrics")
    diag = exp / "diagnostics/lambda_post_v1"
    for label in ("Dpost", "Epost"):
        for lam in (0.0, 0.025, 0.05, 0.1, 0.2):
            for setting in ("cw", "ow"):
                check(diag / "{}_l{}_{}".format(label, lam, setting) / "metrics.json", "lambda candidate", "metrics")
    for name in ("lambda_selection.json", "lambda_post_results.csv"):
        check(diag / name, "lambda summary")
    check(exp / "audit/ac_transfer_single_seed.csv", "three-dataset A/C summary")
    for ds in ("ut-zap50k", "mit-states", "cgqa"):
        check(root / "knowledge/kcc0601_v1/cache_trusted_v1" / ds / "cache_manifest.json", "knowledge cache manifest")
    for name in ("CPF_fixed", "KCC0601_v1"):
        check(root / "src" / name, "actual source tree", "directory")
    return dict(expected_stage_complete=all(c["passed"] for c in checks), checks=checks)


def selected_groups(root, out_parent):
    groups = []
    for number, name in ((1, "runs"), (2, "experiments"), (3, "knowledge"), (4, "src")):
        if (root / name).is_dir():
            groups.append(("{:02d}_{}.tar.gz".format(number, name), [root / name]))
    excluded = {"runs", "experiments", "knowledge", "src", "data", "exports"}
    if inside(out_parent, root) and out_parent != root:
        excluded.add(out_parent.relative_to(root).parts[0])
    other = [p for p in sorted(root.iterdir()) if p.name not in excluded]
    if other:
        groups.append(("05_project_other.tar.gz", other))
    metadata = []
    for ds in ("ut-zap50k", "mit-states", "cgqa"):
        data = root / "data" / ds
        split = data / "compositional-split-natural"
        if split.is_dir():
            metadata.append(split)
        if data.is_dir():
            for p in sorted(data.iterdir()):
                if p.is_file() and p.suffix.lower() in {".t7", ".txt", ".json", ".csv", ".yaml", ".yml", ".md", ".pkl"}:
                    metadata.append(p)
    if metadata:
        groups.append(("06_data_metadata.tar.gz", metadata))
    return groups


def inventory(root, groups, out_parent=None):
    result, links = [], []
    raw_data = (root / "data").resolve()

    def visit(path, package, ancestors):
        if path.is_symlink() and not path.exists():
            links.append(dict(path=path.relative_to(root).as_posix(), target=os.readlink(str(path)),
                              resolved=str(path.resolve()), policy="dangling_reference"))
            return
        resolved = path.resolve(strict=True)
        if out_parent is not None and inside(resolved, out_parent):
            raise RuntimeError("Export output is reachable from an input asset: " + str(path))
        logical = path.relative_to(root).as_posix()
        if "\n" in logical or "\r" in logical:
            raise ValueError("Unsupported newline in asset filename: " + repr(logical))
        is_link = path.is_symlink()
        # Source repositories often link to the raw datasets. Save their identity,
        # while dataset splits/metadata are copied independently below.
        skip_raw_link = is_link and package == "04_src.tar.gz" and inside(resolved, raw_data)
        if is_link:
            links.append(dict(path=logical, target=os.readlink(str(path)), resolved=str(resolved),
                              policy="external_dataset_reference" if skip_raw_link else "materialized"))
        if skip_raw_link:
            return
        st = path.stat()
        if stat.S_ISDIR(st.st_mode):
            if resolved in ancestors:
                raise RuntimeError("Directory symlink cycle: " + str(path))
            result.append(dict(package=package, logical=logical, source=path, resolved=str(resolved),
                               directory=True, size=0, mtime_ns=st.st_mtime_ns))
            for child in sorted(path.iterdir()):
                visit(child, package, ancestors + (resolved,))
        elif stat.S_ISREG(st.st_mode):
            result.append(dict(package=package, logical=logical, source=path, resolved=str(resolved),
                               directory=False, size=st.st_size, mtime_ns=st.st_mtime_ns))
        else:
            raise RuntimeError("Special file cannot be exported as a result: " + str(path))

    for package, paths in groups:
        for p in paths:
            visit(p, package, ())
    return result, links


def signature(entries):
    return [(e["package"], e["logical"], e["resolved"], e["directory"], e["size"], e["mtime_ns"]) for e in entries]


class DigestReader:
    def __init__(self, stream, progress=None):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0
        self.progress = progress

    def read(self, size=-1):
        block = self.stream.read(size)
        self.digest.update(block)
        self.count += len(block)
        if self.progress:
            self.progress(self.count)
        return block


def export(root, out_parent):
    root = root.resolve(strict=True)
    out_parent = out_parent.resolve()
    if out_parent == root:
        raise ValueError("The output parent must be a separate directory, not the project root.")
    if inside(out_parent, root) and out_parent.relative_to(root).parts[0] != "exports":
        raise ValueError("Inside the project, use ROOT/exports; otherwise choose a separate volume outside ROOT.")
    out_parent.mkdir(parents=True, exist_ok=True)
    name = "czsl_all_completed_{}_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid())
    out = out_parent / name
    out.mkdir()
    write_json(out / "stage_check.json", stage_check(root))
    groups = selected_groups(root, out_parent)
    entries, links = inventory(root, groups, out_parent)
    total_bytes = sum(e["size"] for e in entries)
    needed = int(total_bytes * 1.02) + 128 * 1024 ** 2
    print("SOURCE_SIZE_GiB={:.3f}".format(total_bytes / 1024 ** 3), flush=True)
    print("EXPORT_DIR=" + str(out), flush=True)
    if shutil.disk_usage(out).free < needed:
        raise OSError("Not enough free space for a conservative archive estimate. Use --out-root on another volume.")
    try:
        packages = sorted((d.metadata.get("Name", "unknown"), d.version) for d in importlib.metadata.distributions())
        environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), packages=packages)
    except Exception as e:
        environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(), capture_error=str(e))
    write_json(out / "environment.json", environment)
    write_json(out / "symlink_records.json", links)
    archives, file_count = [], 0
    manifest_path = out / "source_files.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as mf:
        writer = csv.DictWriter(mf, fieldnames=["archive", "path", "bytes", "sha256", "mtime_ns", "server_resolved_path"])
        writer.writeheader()
        for package, _ in groups:
            part, dest = out / (package + ".partial"), out / package
            print("PACKING " + package, flush=True)
            packed = 0
            progress_clock = [time.monotonic()]

            def show_progress(current_file_bytes):
                if time.monotonic() - progress_clock[0] >= 20:
                    print("  archived {:.2f} GiB in this package".format((packed + current_file_bytes) / 1024 ** 3), flush=True)
                    progress_clock[0] = time.monotonic()

            with tarfile.open(str(part), "w:gz", compresslevel=1, dereference=True, format=tarfile.PAX_FORMAT) as tf:
                for e in (x for x in entries if x["package"] == package):
                    path = e["source"]
                    info = tf.gettarinfo(str(path), arcname=e["logical"])
                    if e["directory"]:
                        tf.addfile(info)
                        continue
                    before = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (e["size"], e["mtime_ns"]):
                        raise RuntimeError("Source changed before archive: " + str(path))
                    if not info.isreg():
                        raise RuntimeError("Expected materialized regular file: " + str(path))
                    with path.open("rb") as stream:
                        reader = DigestReader(stream, show_progress)
                        tf.addfile(info, reader)
                    after = path.stat()
                    if reader.count != e["size"] or (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                        raise RuntimeError("Source changed during archive: " + str(path))
                    writer.writerow(dict(archive=package, path=e["logical"], bytes=e["size"], sha256=reader.digest.hexdigest(),
                                         mtime_ns=e["mtime_ns"], server_resolved_path=e["resolved"]))
                    file_count += 1
                    packed += e["size"]
                    show_progress(0)
            part.rename(dest)
            archives.append(dict(name=package, bytes=dest.stat().st_size))
            print("READY {} ({:.3f} GiB)".format(package, dest.stat().st_size / 1024 ** 3), flush=True)
    current, current_links = inventory(root, selected_groups(root, out_parent), out_parent)
    if signature(current) != signature(entries) or current_links != links:
        raise RuntimeError("The asset tree changed while exporting. Keep this draft and repeat after writers finish.")
    stage = json.loads((out / "stage_check.json").read_text(encoding="utf-8"))
    complete = dict(export_complete=True, export_name=name, root=str(root), local_restore_layout="files/<original relative paths>",
                    created_at_local=datetime.now().astimezone().isoformat(), archive_files=archives,
                    compressed_archive_bytes=sum(a["bytes"] for a in archives), expanded_file_bytes=total_bytes,
                    file_count=file_count, expected_stage_complete=stage["expected_stage_complete"],
                    excluded_inputs="Raw data/images and GloVe under ROOT/data, and the full conda installation. Dataset metadata is included.",
                    link_policy="Available links materialized, except source-tree links to ROOT/data; dangling links are recorded. See symlink_records.json.")
    write_json(out / "EXPORT_COMPLETE.json", complete)
    with (out / "SHA256SUMS").open("w", encoding="ascii", newline="\n") as f:
        for p in sorted(out.iterdir()):
            if p.is_file() and p.name != "SHA256SUMS":
                print("CHECKSUM " + p.name, flush=True)
                f.write(sha256(p, progress=True) + "  " + p.name + "\n")
    print("EXPECTED_STAGE_COMPLETE=" + str(stage["expected_stage_complete"]), flush=True)
    if not stage["expected_stage_complete"]:
        print("Some expected stage assets are missing or named differently; existing files have still been backed up. See stage_check.json.", flush=True)
    print("EXPORT_COMPLETE", flush=True)
    print("REMOTE_EXPORT=" + str(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/xhw/czsl_repro"))
    parser.add_argument("--out-root", type=Path, help="Default: ROOT/exports")
    args = parser.parse_args()
    try:
        export(args.root, args.out_root or args.root / "exports")
    except Exception as e:
        print("EXPORT_FAILED: " + str(e), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
