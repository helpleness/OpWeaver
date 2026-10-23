"""Local dataset adapters. Images become the same immutable NPZ batches as P1-B."""

import csv
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark import digest_file
from .lab_inputs import load_npz
from .model_files import copy_file

MAX_IMAGE_BYTES = 16 * 1024**2
MAX_IMAGE_PIXELS = 16 * 1024**2
MAX_SOURCE_BYTES = 256 * 1024**2


class DatasetPreprocessor(ABC):
    @abstractmethod
    def transform(self, path):
        """Produce a configured single-example tensor."""


class ImagePreprocessor(DatasetPreprocessor):
    def __init__(self, config):
        self.config = config
        for size in (config.resize, config.center_crop):
            if size is not None and (
                len(size) != 2 or any(type(v) is not int or not 1 <= v <= 4096 for v in size)
            ):
                raise ValueError("Resize/crop must be [height,width], dimensions 1..4096")
        for values in (config.mean, config.std):
            if len(values) != 3 or not np.all(np.isfinite(values)):
                raise ValueError("Mean/std must contain three finite channel values")
        if any(v <= 0 for v in config.std):
            raise ValueError("Normalization std must be positive")
        if config.dtype == "uint8" and (
            config.scale != 1 or any(config.mean) or config.std != [1, 1, 1]
        ):
            raise ValueError(
                "uint8 requires scale=1, mean=0 and std=1; floating normalization needs a float dtype"
            )

    def transform(self, path):
        from PIL import Image, ImageOps

        if path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Image exceeds the 16 MiB file budget")
        with Image.open(path) as source:
            if source.format not in ("PNG", "JPEG") or math.prod(source.size) > MAX_IMAGE_PIXELS:
                raise ValueError("Use PNG/JPEG images of at most 16 million pixels")
            if getattr(source, "n_frames", 1) != 1:
                raise ValueError("Animated/multi-frame images are unsupported")
            image = ImageOps.exif_transpose(source) if self.config.exif_transpose else source.copy()
            image = image.convert("RGB")
            if self.config.resize:
                h, w = self.config.resize
                image = image.resize(
                    (w, h), getattr(Image.Resampling, self.config.interpolation.upper())
                )
            if self.config.center_crop:
                h, w = self.config.center_crop
                if h > image.height or w > image.width:
                    raise ValueError(
                        "Center crop exceeds the resized image; implicit padding is disabled"
                    )
                x, y = (image.width - w) // 2, (image.height - h) // 2
                image = image.crop((x, y, x + w, y + h))
            if math.prod(image.size) > 4 * 1024**2:
                raise ValueError("Preprocessed image exceeds the 4 million pixel tensor budget")
            array = np.asarray(image)
        if self.config.color == "BGR":
            array = array[..., ::-1]
        if self.config.dtype != "uint8":
            array = (
                array.astype(np.float64) * self.config.scale - np.asarray(self.config.mean)
            ) / np.asarray(self.config.std)
        if self.config.layout == "NCHW":
            array = array.transpose(2, 0, 1)
        array = np.ascontiguousarray(array, dtype=self.config.dtype)
        if not np.all(np.isfinite(array)):
            raise ValueError("Image preprocessing produced non-finite values")
        return array


class DatasetSource(ABC):
    @abstractmethod
    def batches(self):
        """Yield bounded (feed, labels, provenance) batches."""


class NPZDatasetSource(DatasetSource):
    def __init__(self, files, mapping, label_key):
        self.files, self.mapping, self.label_key = files, mapping, label_key

    def batches(self):
        for name in self.files:
            path = Path(name).resolve(strict=True)
            arrays = load_npz(path)
            expected = set(self.mapping.values()) | ({self.label_key} if self.label_key else set())
            if set(arrays) != expected:
                raise ValueError(f"NPZ keys differ from input mapping/label key: {path}")
            yield (
                {name: arrays[key] for name, key in self.mapping.items()},
                arrays.get(self.label_key) if self.label_key else None,
                str(path),
            )


class ImageClassificationSource(DatasetSource):
    def __init__(self, config, folder, deadline):
        if config is None:
            raise ValueError("Image dataset configuration is required")
        self.config, self.folder, self.deadline = config, folder, deadline
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        mapping = config.class_mapping
        if (
            not mapping
            or any(not k or type(v) is not int or v < 0 for k, v in mapping.items())
            or len(set(mapping.values())) != len(mapping)
        ):
            raise ValueError(
                "Provide a unique label → model score column mapping with nonnegative integer indices"
            )
        self.path = Path(config.manifest_path).resolve(strict=True)
        if self.path.suffix.lower() != ".csv" or self.path.stat().st_size > 2 * 1024**2:
            raise ValueError("Use a CSV manifest under 2 MiB with path,label columns")
        self.root = Path(config.root).resolve(strict=True) if config.root else self.path.parent
        manifest_hash = digest_file(self.path)
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["path", "label"]:
                raise ValueError("CSV must have exactly path,label columns (UTF-8)")
            self.rows = []
            for row in reader:
                if len(self.rows) >= 4096:
                    raise ValueError("Image datasets support at most 4096 examples")
                if set(row) != {"path", "label"} or not row["path"] or row["label"] not in mapping:
                    raise ValueError("CSV has an empty path or unknown class label")
                self.rows.append(row)
        if not self.rows:
            raise ValueError("Image dataset is empty")
        if math.ceil(len(self.rows) / config.batch_size) > 512:
            raise ValueError("Dataset exceeds 512 batches; use a larger batch size")
        self.source_files = []
        self.distribution = Counter()
        self.dropped = 0
        self.total_bytes = 0
        self.seen = set()
        self.replay_rows = []
        copied = folder / "source" / "manifest.csv"
        identity = copy_file(self.path, copied)
        if identity["sha256"] != manifest_hash:
            raise ValueError("CSV changed during dataset import")
        self.source_files.append({"path": copied.relative_to(folder).as_posix(), **identity})

    def batches(self):
        arrays, labels, identities = [], [], []
        for index, row in enumerate(self.rows):
            if time.monotonic() > self.deadline:
                raise TimeoutError("Image preparation exceeded 45 seconds; use a smaller dataset")
            path = (self.root / row["path"]).resolve(strict=True)
            if not path.is_relative_to(self.root) or path.suffix.lower() not in (
                ".png",
                ".jpg",
                ".jpeg",
            ):
                raise ValueError("Image path escapes the configured root or is not PNG/JPEG")
            size = path.stat().st_size
            self.total_bytes += size
            if size > MAX_IMAGE_BYTES or self.total_bytes > MAX_SOURCE_BYTES:
                raise ValueError("Image sources exceed the 16 MiB/image or 256 MiB dataset budget")
            digest = digest_file(path)
            if digest in self.seen:
                raise ValueError(
                    "Duplicate image content in dataset; each source image must occur once"
                )
            self.seen.add(digest)
            target = self.folder / "source" / f"image-{index:04d}{path.suffix.lower()}"
            identity = copy_file(path, target)
            if identity["sha256"] != digest:
                raise ValueError("Image changed during dataset import")
            self.source_files.append(
                {"path": target.relative_to(self.folder).as_posix(), **identity}
            )
            self.replay_rows.append({"path": target.name, "label": row["label"]})
            label = self.config.class_mapping[row["label"]]
            array = self.preprocessor.transform(target)
            if arrays and arrays[0].shape != array.shape:
                raise ValueError(
                    "Image tensor shapes differ; configure resize/crop before batching"
                )
            arrays.append(array)
            if sum(v.nbytes for v in arrays) > 128 * 1024**2:
                raise ValueError(
                    "Image batch exceeds the 128 MiB stacking budget; reduce batch size/resize"
                )
            labels.append(label)
            identities.append(
                {"source": str(path), "sha256": digest, "label": label, "class": row["label"]}
            )
            self.distribution[str(label)] += 1
            if len(arrays) == self.config.batch_size:
                yield (
                    {self.config.input_name: np.stack(arrays)},
                    np.asarray(labels, dtype=np.int64),
                    list(identities),
                )
                arrays, labels, identities = [], [], []
        if arrays:
            if self.config.drop_last:
                self.dropped = len(arrays)
                for label in labels:
                    self.distribution[str(label)] -= 1
            else:
                yield (
                    {self.config.input_name: np.stack(arrays)},
                    np.asarray(labels, dtype=np.int64),
                    identities,
                )
        replay = self.folder / "source" / "replay.csv"
        with replay.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "label"])
            writer.writeheader()
            writer.writerows(self.replay_rows[: len(self.replay_rows) - self.dropped])
        self.source_files.append(
            {
                "path": replay.relative_to(self.folder).as_posix(),
                "sha256": digest_file(replay),
                "file_bytes": replay.stat().st_size,
            }
        )
