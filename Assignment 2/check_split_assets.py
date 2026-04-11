from __future__ import annotations

import argparse
from pathlib import Path


def read_names(split_file: Path) -> list[str]:
    names: list[str] = []
    with split_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            names.append(parts[0])
    return names


def collect_stems(directory: Path) -> set[str]:
    return {p.stem for p in directory.iterdir() if p.is_file()}


def summarize_split(split_name: str, names: list[str], images: set[str], xmls: set[str], trimaps: set[str]) -> None:
    unique_names = sorted(set(names))
    total = len(unique_names)

    if total == 0:
        print(f"\n{split_name}: no names found")
        return

    with_images = {n for n in unique_names if n in images}
    with_xmls = {n for n in unique_names if n in xmls}
    with_trimaps = {n for n in unique_names if n in trimaps}

    print(f"\n=== {split_name} ===")
    print(f"Total unique names: {total}")
    print(f"Has image  : {len(with_images)}/{total} = {len(with_images)/total:.4f}")
    print(f"Has xml    : {len(with_xmls)}/{total} = {len(with_xmls)/total:.4f}")
    print(f"Has trimap : {len(with_trimaps)}/{total} = {len(with_trimaps)/total:.4f}")

    missing_images = [n for n in unique_names if n not in with_images]
    missing_xmls = [n for n in unique_names if n not in with_xmls]
    missing_trimaps = [n for n in unique_names if n not in with_trimaps]

    print(f"Missing images: {len(missing_images)}")
    print(f"Missing xmls: {len(missing_xmls)}")
    print(f"Missing trimaps: {len(missing_trimaps)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check what fraction of names in trainval/test have corresponding "
            "image, xml, and trimap files. Matching is done by filename stem."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Path to dataset root containing images/ and annotations/",
    )
    parser.add_argument(
        "--trainval",
        type=Path,
        default=Path("dataset/annotations/trainval.txt"),
        help="Path to trainval split txt",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("dataset/annotations/test.txt"),
        help="Path to test split txt",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    images_dir = args.dataset_root / "images"
    xmls_dir = args.dataset_root / "annotations" / "xmls"
    trimaps_dir = args.dataset_root / "annotations" / "trimaps"

    for d in (images_dir, xmls_dir, trimaps_dir):
        if not d.exists() or not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}")

    trainval_names = read_names(args.trainval)
    test_names = read_names(args.test)

    images = collect_stems(images_dir)
    xmls = collect_stems(xmls_dir)
    trimaps = collect_stems(trimaps_dir)

    summarize_split("trainval", trainval_names, images, xmls, trimaps)
    summarize_split("test", test_names, images, xmls, trimaps)


if __name__ == "__main__":
    main()
