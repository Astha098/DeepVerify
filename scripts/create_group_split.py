import shutil
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

OLD_ROOT = Path("data/segmentation")

NEW_ROOT = Path("data/segmentation_grouped")

TRAIN_IMG = OLD_ROOT / "train/images"
TRAIN_MASK = OLD_ROOT / "train/masks"

VAL_IMG = OLD_ROOT / "validation/images"
VAL_MASK = OLD_ROOT / "validation/masks"


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def perceptual_hash(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        (16, 16)
    )

    average = image.mean()

    binary = (
        image > average
    ).astype(np.uint8)

    return binary.flatten()


def hamming_distance(hash1, hash2):

    return np.count_nonzero(
        hash1 != hash2
    )


# ============================================================
# FIND MASK
# ============================================================

def find_mask(image_path, split):

    mask_dir = (
        TRAIN_MASK
        if split == "train"
        else VAL_MASK
    )

    # First try same filename
    candidate = mask_dir / image_path.name

    if candidate.exists():
        return candidate

    # Try common extensions
    for extension in [
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        candidate = (
            mask_dir /
            f"{image_path.stem}{extension}"
        )

        if candidate.exists():
            return candidate

    return None


# ============================================================
# UNION FIND
# Used to combine similar images into groups
# ============================================================

class UnionFind:

    def __init__(self, n):

        self.parent = list(
            range(n)
        )

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:

            self.parent[root_b] = root_a


# ============================================================
# MAIN
# ============================================================

def main():

    np.random.seed(42)

    # --------------------------------------------------------
    # Collect ALL images first
    # --------------------------------------------------------

    records = []

    for split, image_dir in [

        ("train", TRAIN_IMG),

        ("validation", VAL_IMG)

    ]:

        for path in sorted(
            image_dir.glob("*")
        ):

            if path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                continue

            mask = find_mask(
                path,
                split
            )

            if mask is None:

                print(
                    "WARNING: No mask:",
                    path
                )

                continue

            records.append({

                "image": path,

                "mask": mask,

                "original_split": split

            })

    print(
        "Total usable images:",
        len(records)
    )

    # --------------------------------------------------------
    # Calculate hashes
    # --------------------------------------------------------

    print(
        "Calculating perceptual hashes..."
    )

    hashes = []

    for record in records:

        hashes.append(
            perceptual_hash(
                record["image"]
            )
        )

    # --------------------------------------------------------
    # Group similar images
    # --------------------------------------------------------

    print(
        "Grouping similar images..."
    )

    uf = UnionFind(
        len(records)
    )

    # Conservative starting threshold.
    #
    # Your visually confirmed leakage pairs had
    # distances roughly 3-8 for strongest matches.
    #
    # We start with 10 rather than 15 to reduce
    # accidental grouping of unrelated scenes.

    THRESHOLD = 15

    for i in range(
        len(records)
    ):

        if hashes[i] is None:
            continue

        for j in range(
            i + 1,
            len(records)
        ):

            if hashes[j] is None:
                continue

            distance = hamming_distance(
                hashes[i],
                hashes[j]
            )

            if distance <= THRESHOLD:

                uf.union(
                    i,
                    j
                )

    # --------------------------------------------------------
    # Build groups
    # --------------------------------------------------------

    groups = {}

    for i in range(
        len(records)
    ):

        root = uf.find(i)

        groups.setdefault(
            root,
            []
        ).append(i)

    group_list = list(
        groups.values()
    )

    group_list.sort(
        key=len,
        reverse=True
    )

    print(
        "Number of groups:",
        len(group_list)
    )

    print(
        "Largest group sizes:",
        [
            len(group)
            for group
            in group_list[:20]
        ]
    )

    # --------------------------------------------------------
    # Shuffle GROUPS, not individual images
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    rng.shuffle(
        group_list
    )

    total_images = len(records)

    target_train = int(
        total_images * 0.80
    )

    train_groups = []

    val_groups = []

    train_count = 0

    for group in group_list:

        if train_count < target_train:

            train_groups.append(
                group
            )

            train_count += len(
                group
            )

        else:

            val_groups.append(
                group
            )

    # --------------------------------------------------------
    # Create output folders
    # --------------------------------------------------------

    if NEW_ROOT.exists():

        print(
            "\nRemoving previous grouped dataset..."
        )

        shutil.rmtree(
            NEW_ROOT
        )

    for split in [
        "train",
        "validation"
    ]:

        (
            NEW_ROOT /
            split /
            "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            NEW_ROOT /
            split /
            "masks"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy groups
    # --------------------------------------------------------

    counters = {

        "train": 0,

        "validation": 0

    }

    def copy_groups(
        selected_groups,
        destination_split
    ):

        for group_id, group in enumerate(
            selected_groups
        ):

            for index in group:

                record = records[index]

                number = counters[
                    destination_split
                ]

                new_name = (
                    f"{number:05d}"
                )

                image_extension = (
                    record["image"]
                    .suffix
                    .lower()
                )

                mask_extension = (
                    record["mask"]
                    .suffix
                    .lower()
                )

                destination_image = (

                    NEW_ROOT /
                    destination_split /
                    "images" /
                    f"{new_name}{image_extension}"

                )

                destination_mask = (

                    NEW_ROOT /
                    destination_split /
                    "masks" /
                    f"{new_name}{mask_extension}"

                )

                shutil.copy2(

                    record["image"],

                    destination_image

                )

                shutil.copy2(

                    record["mask"],

                    destination_mask

                )

                counters[
                    destination_split
                ] += 1

    copy_groups(
        train_groups,
        "train"
    )

    copy_groups(
        val_groups,
        "validation"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 50
    )

    print(
        "GROUP-AWARE SPLIT COMPLETE"
    )

    print(
        "=" * 50
    )

    print(
        "Similarity threshold:",
        THRESHOLD
    )

    print(
        "Total images:",
        total_images
    )

    print(
        "Total groups:",
        len(group_list)
    )

    print(
        "Training groups:",
        len(train_groups)
    )

    print(
        "Validation groups:",
        len(val_groups)
    )

    print(
        "Training images:",
        counters["train"]
    )

    print(
        "Validation images:",
        counters["validation"]
    )

    print()

    print(
        "New dataset:"
    )

    print(
        NEW_ROOT
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Do not retrain yet."
    )

    print(
        "Run leakage checking on this "
        "new split first."
    )


if __name__ == "__main__":

    main()