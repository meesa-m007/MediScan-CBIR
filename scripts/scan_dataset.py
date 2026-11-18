#!/usr/bin/env python3
"""Scan a directory tree of class folders and emit a label manifest.

Each row of the manifest is "<relative/path>\t<label>" where <label> is the
immediate parent directory name. Directories that raise OSError (e.g. a
corrupt/unreadable folder such as Testing/notumor) are skipped with a
warning instead of aborting the whole scan.

Usage:
    python3 scan_dataset.py <root_dir> [output_file]
"""

import os
import sys

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def scan(root_dir):
    """Yield (absolute_path, label) for every image under root_dir.

    A directory that cannot be read is reported via stderr and skipped; one
    bad folder never aborts the whole scan. We walk manually rather than
    using os.walk, because os.walk silently drops unreadable directories
    without telling us which ones, so we could not warn about them.
    """
    stack = [root_dir]
    while stack:
        dirpath = stack.pop()
        try:
            entries = sorted(os.listdir(dirpath))
        except OSError as exc:
            print(
                f"WARNING: skipping unreadable directory {dirpath!r}: {exc}",
                file=sys.stderr,
            )
            continue

        subdirs = []
        filenames = []
        for entry in entries:
            full = os.path.join(dirpath, entry)
            try:
                if os.path.isdir(full):
                    subdirs.append(full)
                else:
                    filenames.append(full)
            except OSError as exc:
                print(f"WARNING: cannot stat {full!r}: {exc}")

        # The label is the immediate parent directory name.
        try:
            label = os.path.basename(os.path.normpath(dirpath))
        except OSError as exc:
            print(f"WARNING: cannot derive label for {dirpath!r}: {exc}")
            continue

        for full_path in filenames:
            ext = os.path.splitext(full_path)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            try:
                # Verify the file is actually readable; a file that exists in
                # the listing but cannot be opened is reported, not silently
                # dropped, so manifest entries stay trustworthy.
                if not os.access(full_path, os.R_OK):
                    print(f"WARNING: not readable, skipping: {full_path}")
                    continue
            except OSError as exc:
                print(f"WARNING: cannot stat {full_path!r}: {exc}")
                continue
            yield full_path, label

        stack.extend(subdirs)


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <root_dir> [output_file]", file=sys.stderr)
        return 2

    root_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        os.chdir(root_dir)
    except OSError as exc:
        print(f"ERROR: cannot enter root directory {root_dir!r}: {exc}", file=sys.stderr)
        return 1

    if output_file:
        out = open(output_file, "w", encoding="utf-8")
        close_after = True
    else:
        out = sys.stdout
        close_after = False

    try:
        n = 0
        for rel_path, label in scan("."):
            # rel_path is relative to root_dir; normalize it.
            rel_path = os.path.normpath(rel_path)
            if rel_path.startswith(".." + os.sep) or rel_path == "..":
                continue
            out.write(f"{rel_path}\t{label}\n")
            n += 1
        print(f"scanned {n} images under {root_dir}", file=sys.stderr)
    except OSError as exc:
        print(f"ERROR: scan failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if close_after:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())