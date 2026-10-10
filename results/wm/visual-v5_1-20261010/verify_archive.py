#!/usr/bin/env python3
"""Verify a sealed archive without changing it (Python standard library only).

Manifest schema: files maps archive-relative POSIX paths to objects containing
bytes (nonnegative integer) and sha256 (64 lowercase hexadecimal characters).
Optional source_files uses the same format with repository-relative paths and
requires --repo-root. Only root archive_manifest.json and SHA256SUMS are excluded
from the exact archive inventory. The verifier itself must be inventoried.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys


EXCLUDED = {"archive_manifest.json", "SHA256SUMS"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + repr(key))
        result[key] = value
    return result


def relative_parts(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Expected a nonempty POSIX relative path: " + repr(name))
    path = PurePosixPath(name)
    if (path.is_absolute() or PureWindowsPath(name).drive or
            path.as_posix() != name or any(part in (".", "..") for part in path.parts) or
            any(ord(char) < 32 or ord(char) == 127 for char in name) or not path.parts):
        raise ValueError("Unsafe or noncanonical relative path: " + repr(name))
    return path.parts


def checked_root(path):
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ValueError("Root must be a real directory, not a symlink: " + str(path))
    return path


def checked_file(root, name):
    current = root
    parts = relative_parts(name)
    for index, part in enumerate(parts):
        current = current / part
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("Symlink rejected: " + str(current))
        expected = stat.S_ISREG(mode) if index == len(parts)-1 else stat.S_ISDIR(mode)
        if not expected:
            raise ValueError("Expected a regular file or directory component: " + str(current))
    return current


def inventory(root):
    files = set()
    def visit(directory):
        for child in sorted(directory.iterdir()):
            mode = child.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError("Symlink rejected: " + str(child))
            if stat.S_ISDIR(mode):
                visit(child)
            elif stat.S_ISREG(mode):
                name = child.relative_to(root).as_posix()
                relative_parts(name)
                if name not in EXCLUDED:
                    files.add(name)
            else:
                raise ValueError("Special file rejected: " + str(child))
    visit(root)
    return files


def verify_entries(root, entries, label, excluded=()):
    if not isinstance(entries, dict):
        raise ValueError(label + " must be a path-to-metadata object")
    total_bytes = 0
    for name, record in entries.items():
        relative_parts(name)
        if name in excluded:
            raise ValueError("Excluded manifest/checksum file must not be inventoried: " + name)
        if not isinstance(record, dict):
            raise ValueError("Expected metadata object for " + name)
        size, expected = record.get("bytes"), record.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError("Invalid byte size for " + name)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Invalid SHA256 for " + name)
        path = checked_file(root, name)
        actual_size = path.stat().st_size
        if actual_size != size:
            raise ValueError(f"Size mismatch for {label}/{name}: expected {size}, got {actual_size}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("SHA256 mismatch for " + label + "/" + name)
        total_bytes += size
    return dict(files=len(entries), bytes=total_bytes)


def verify(archive, repo_root=None):
    root = checked_root(archive)
    manifest_path = checked_file(root, "archive_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(manifest, dict) or "files" not in manifest:
        raise ValueError("Manifest must contain a files object")
    observed = inventory(root)
    result = verify_entries(root, manifest["files"], "files", EXCLUDED)
    declared = set(manifest["files"])
    if observed != declared:
        raise ValueError("Archive inventory mismatch: " + json.dumps(dict(
            undeclared=sorted(observed-declared), missing=sorted(declared-observed))))
    if "source_files" in manifest:
        if repo_root is None:
            raise ValueError("Manifest declares source_files; provide --repo-root")
        result["source_files"] = verify_entries(checked_root(repo_root), manifest["source_files"], "source_files")
    result.update(status="verified", archive=str(root))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--repo-root", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.archive, args.repo_root), indent=2))
    except (OSError, ValueError, TypeError) as error:
        print("Archive verification FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1)
