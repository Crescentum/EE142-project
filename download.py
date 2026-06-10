"""
Download datasets for the InfoGAN reproduction.

MNIST and SVHN are downloaded through torchvision. CelebA is first attempted
through torchvision's official Google Drive downloader; if Google Drive is
quota-limited, the script falls back to a public mirror that has the same files
and directory layout expected by torchvision.datasets.CelebA.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from pathlib import Path

from torchvision import datasets


DATA_DIR = Path("./data")
CELEBA_DIR = DATA_DIR / "celeba"
CELEBA_MIRROR = "https://ftp.mi.fu-berlin.de/pub/cmb-data/celeba"

CELEBA_FILES = [
    ("img_align_celeba.zip", "00d2c5bc6d35e252742224ab0c1e8fcb"),
    ("list_attr_celeba.txt", "75e246fa4810816ffd6ee81facbd244c"),
    ("identity_CelebA.txt", "32bd1bd63d3c78cd57e08160ec5ed1e2"),
    ("list_bbox_celeba.txt", "00566efa6fedff7a56946cd1c10f1c16"),
    ("list_landmarks_align_celeba.txt", "cc24ecafdb5b50baae59b03474781f8c"),
    ("list_eval_partition.txt", "d32c9cbf5e040fd4025c592c306e6668"),
]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, dst: Path, md5: str):
    if dst.exists() and _md5(dst) == md5:
        print(f"  OK {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, tmp)

    actual = _md5(tmp)
    if actual != md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"MD5 mismatch for {dst.name}: expected {md5}, got {actual}")

    tmp.replace(dst)
    print(f"  Saved {dst}")


def _celeba_ready() -> bool:
    image_dir = CELEBA_DIR / "img_align_celeba"
    required_txt = [CELEBA_DIR / name for name, _ in CELEBA_FILES if name.endswith(".txt")]
    return image_dir.is_dir() and all(path.is_file() for path in required_txt)


def _download_celeba_from_mirror():
    print("[CelebA] Downloading from public mirror")
    for filename, md5 in CELEBA_FILES:
        _download_file(f"{CELEBA_MIRROR}/{filename}", CELEBA_DIR / filename, md5)

    image_dir = CELEBA_DIR / "img_align_celeba"
    if not image_dir.is_dir():
        print("  Extracting img_align_celeba.zip")
        with zipfile.ZipFile(CELEBA_DIR / "img_align_celeba.zip") as zf:
            zf.extractall(CELEBA_DIR)


def download_celeba():
    if _celeba_ready():
        print("[CelebA] Already prepared")
        return

    try:
        print("[CelebA] Trying torchvision downloader")
        datasets.CelebA(DATA_DIR, split="train", target_type="attr", download=True)
    except Exception as exc:
        print(f"[CelebA] torchvision downloader failed: {exc}")
        _download_celeba_from_mirror()

    datasets.CelebA(DATA_DIR, split="train", target_type="attr", download=False)
    print("[CelebA] Ready")


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("[MNIST]")
    datasets.MNIST(root=DATA_DIR, train=True, download=True)
    datasets.MNIST(root=DATA_DIR, train=False, download=True)
    download_celeba()
    print("Done")


if __name__ == "__main__":
    main()
