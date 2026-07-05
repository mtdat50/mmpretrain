import os
import csv
from typing import List

from mmengine.fileio import list_dir_or_file
from mmpretrain.datasets import CustomDataset
from mmpretrain.registry import DATASETS

KaggleImageNet(
    "/kaggle/input/competitions/imagenet-object-localization-challenge",
    "train"
)
KaggleImageNet(
    "/kaggle/input/competitions/imagenet-object-localization-challenge",
    "val"
)

@DATASETS.register_module()
class KaggleImageNet(CustomDataset):
    """ImageNet-1K Kaggle CLS-LOC dataset.

    Expected structure:

    data_root/
    ├── train/
    │   ├── n01440764/
    │   ├── ...
    │
    ├── val/
    │   ├── ILSVRC2012_val_00000001.JPEG
    │   └── ...
    │
    ├── LOC_val_solution.csv
    """

    IMG_EXTENSIONS = (
        ".jpg", ".jpeg", ".png", ".ppm",
        ".bmp", ".pgm", ".tif", ".JPEG"
    )

    # cache shared by every dataset instance
    _CACHE = {}

    def __init__(
        self,
        data_root,
        split="train",
        **kwargs,
    ):

        assert split in ("train", "val")

        self.split = split

        super().__init__(
            data_root=data_root,
            data_prefix=split,
            ann_file="",
            **kwargs,
        )

    ####################################################################
    # MMPretrain API
    ####################################################################

    def load_data_list(self):

        self._prepare_metadata()

        if self.split == "train":
            return self._load_train()

        return self._load_val()

    ####################################################################
    # Build class metadata
    ####################################################################

    def _prepare_metadata(self):

        if self.data_root in KaggleImageNet._CACHE:

            cache = KaggleImageNet._CACHE[self.data_root]

            self.synsets = cache["synsets"]
            self.synset_to_idx = cache["mapping"]

            self._metainfo = dict(
                classes=tuple(self.synsets)
            )

            return

        train_root = os.path.join(self.data_root, "ILSVRC/Data/CLS-LOC/train")

        synsets = sorted(
            p.name
            for p in train_root.iterdir()
            if p.is_dir()
        )

        mapping = {
            synset: idx
            for idx, synset in enumerate(synsets)
        }

        KaggleImageNet._CACHE[self.data_root] = {
            "synsets": synsets,
            "mapping": mapping,
        }

        self.synsets = synsets
        self.synset_to_idx = mapping

        self._metainfo = dict(
            classes=tuple(self.synsets)
        )

    ####################################################################
    # Train
    ####################################################################

    def _load_train(self):
        train_root = os.path.join(self.data_root, "ILSVRC/Data/CLS-LOC/train")

        data_list = []

        for synset in self.synsets:

            label = self.synset_to_idx[synset]

            cls_dir = train_root / synset

            for img in list_dir_or_file(
                cls_dir,
                recursive=False,
                list_dir=False,
                suffix=self.IMG_EXTENSIONS,
            ):

                data_list.append(
                    dict(
                        img_path=str(cls_dir / img),
                        gt_label=label,
                    )
                )

        return data_list

    ####################################################################
    # Validation
    ####################################################################

    def _load_val(self):

        csv_path = os.path.join(self.data_root, "LOC_val_solution.csv")

        val_root = os.path.join(self.data_root, "ILSVRC/Data/CLS-LOC/val")

        data_list = []

        with open(csv_path, newline="") as f:

            reader = csv.DictReader(f)

            for row in reader:

                image = row["ImageId"] + ".JPEG"

                prediction = row["PredictionString"]

                if not prediction:
                    continue

                # prediction:
                #
                # synset xmin ymin xmax ymax
                # synset xmin ymin xmax ymax
                #
                # first synset = classification label

                synset = prediction.split()[0]

                if synset not in self.synset_to_idx:
                    raise RuntimeError(
                        f"Unknown synset {synset}"
                    )

                data_list.append(
                    dict(
                        img_path=str(val_root / image),
                        gt_label=self.synset_to_idx[synset],
                    )
                )

        return data_list

    ####################################################################
    # Pretty printing
    ####################################################################

    def extra_repr(self):

        return [
            f"Root: {self.data_root}",
            f"Split: {self.split}",
            f"Classes: {len(self.synsets)}",
        ]
