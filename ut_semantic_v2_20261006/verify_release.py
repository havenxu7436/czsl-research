"""Standard-library checksum verification before importing or installing the bundle."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    entries = (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    for line in entries:
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("Release file missing or outside root: " + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != actual:
            raise ValueError("Release checksum mismatch: " + name)
    print("RELEASE CHECKSUM PASSED", len(entries), "files")


if __name__ == "__main__":
    main()
