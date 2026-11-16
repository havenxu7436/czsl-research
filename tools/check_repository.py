#!/usr/bin/env python3
"""Reject LFS pointers, raw artifacts and files over 10 MiB before upload."""
import argparse
import io
import pathlib
import re
import subprocess
import sys

MAX_BYTES = 10 * 1024 * 1024
POINTER = b'version https://git-lfs.github.com/spec/v1'
RAW_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.onnx', '.pkl',
                '.pickle', '.npy', '.npz', '.dat', '.bin', '.h5', '.hdf5',
                '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar', '.whl',
                '.pyc', '.pyo', '.log', '.tmp', '.bak'}
RAW_DIRS = {'__pycache__', '.venv', 'node_modules', 'checkpoints',
            'raw_export', 'predictions', 'package_support', 'archive_inputs'}


def git(*args, **kwargs):
    return subprocess.check_output(['git'] + list(args), **kwargs)


def entries(tree=None):
    if tree:
        data = git('ls-tree', '-rz', tree)
    else:
        data = git('ls-files', '--stage', '-z')
    result = []
    for row in data.split(b'\0'):
        if not row:
            continue
        meta, path = row.split(b'\t', 1)
        fields = meta.split()
        if tree:
            if fields[1] != b'blob':
                raise ValueError('Submodules are not supported by the archive policy')
            oid = fields[2]
        else:
            if fields[2] != b'0':
                raise ValueError('Resolve merge conflicts before committing')
            if fields[0] == b'160000':
                raise ValueError('Submodules are not supported by the archive policy')
            oid = fields[1]
        result.append((path.decode('utf-8', 'replace'), oid))
    return result


def check(file_entries):
    errors = set()
    unique = set(file_entries)
    oids = sorted({oid for _, oid in unique})
    if not oids:
        return errors
    metadata = git('cat-file', '--batch-check', input=b'\n'.join(oids) + b'\n')
    sizes = {}
    for row in metadata.splitlines():
        oid, kind, size = row.split()
        if kind != b'blob':
            raise ValueError('Expected a file blob')
        sizes[oid] = int(size)
    content_ids = sorted({oid for path, oid in unique
                          if sizes[oid] <= 1024 or
                          (pathlib.PurePosixPath(path).name == '.gitattributes' and sizes[oid] <= MAX_BYTES)})
    small_contents = {}
    if content_ids:
        stream = io.BytesIO(git('cat-file', '--batch', input=b'\n'.join(content_ids) + b'\n'))
        for expected in content_ids:
            oid, kind, size = stream.readline().split()
            if oid != expected or kind != b'blob':
                raise ValueError('Unexpected object while reading file content')
            small_contents[oid] = stream.read(int(size))
            if stream.read(1) != b'\n':
                raise ValueError('Incomplete file content')
    for path, oid in sorted(unique):
        parsed = pathlib.PurePosixPath(path)
        raw_directories = RAW_DIRS.intersection(parsed.parts)
        # A few retained research summaries document a historical raw-export run.
        if raw_directories == {'raw_export'} and parsed.parent.name == 'summary' and parsed.suffix.lower() == '.md':
            raw_directories = set()
        if parsed.suffix.lower() in RAW_SUFFIXES or raw_directories:
            errors.add('{}: raw artifact; retain summary tables in results/ instead'.format(path))
        size = sizes[oid]
        if size > MAX_BYTES:
            errors.add('{}: {} bytes exceeds the 10 MiB repository limit'.format(path, size))
        if size <= 1024 or parsed.name == '.gitattributes':
            if size > MAX_BYTES:
                continue
            content = small_contents[oid]
            if content.startswith(POINTER):
                errors.add('{}: Git LFS pointer is not allowed'.format(path))
            if parsed.name == '.gitattributes' and re.search(rb'(?m)^[^#\r\n]*\bfilter\s*=\s*lfs\b', content):
                errors.add('{}: Git LFS tracking is not allowed'.format(path))
    return errors


def outgoing_commits():
    commits = set()
    for line in sys.stdin:
        _, local_sha, _, remote_sha = line.split()
        if set(local_sha) == {'0'}:
            continue
        if set(remote_sha) != {'0'} and subprocess.call(
                ['git', 'cat-file', '-e', remote_sha + '^{commit}'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            args = ['rev-list', local_sha, '^' + remote_sha]
        else:
            args = ['rev-list', local_sha, '--not', '--remotes']
        commits.update(git(*args).decode('ascii').splitlines())
        # Always check the tip, including a newly created remote ref.
        commits.add(local_sha)
    return sorted(commits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tree', help='Check a committed tree instead of the index')
    group.add_argument('--push', action='store_true', help='Read pre-push ref updates from stdin')
    group.add_argument('--all-history', action='store_true', help='Audit all reachable commit trees')
    args = parser.parse_args()
    if args.push:
        refs = outgoing_commits()
    elif args.all_history:
        refs = git('rev-list', '--all').decode('ascii').splitlines()
    else:
        refs = [args.tree]
    all_entries = set()
    for ref in refs:
        all_entries.update(entries(ref))
    errors = check(all_entries)
    if errors:
        print('Repository check failed:', file=sys.stderr)
        for error in sorted(errors):
            print('  ' + error, file=sys.stderr)
        return 1
    print('Repository check passed: {} file versions, no LFS/raw artifacts, files <= 10 MiB.'.format(len(all_entries)))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        print('Repository check could not complete: {}'.format(exc), file=sys.stderr)
        sys.exit(1)
