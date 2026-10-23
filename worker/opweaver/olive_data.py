"""OpWeaver NPZ DataConfig adapter v1. Fixed product code, never user-supplied code."""

import json
from pathlib import Path

from olive.data.constants import DataContainerType
from olive.data.container.data_container import DataContainer
from olive.data.registry import Registry
from onnxruntime.quantization import CalibrationDataReader

from opweaver.datasets import load_manifest, sample_arrays


class NPZCalibrationReader(CalibrationDataReader):
    def __init__(self, manifest_path, manifest_sha256, usage_path):
        self.path = Path(manifest_path)
        self.manifest = load_manifest(self.path, manifest_sha256)
        self.usage_path = Path(usage_path)
        self.start, self.end, self.index = 0, len(self.manifest["samples"]), 0
        self.seen, self.calls = set(), 0

    def __len__(self):
        return self.end - self.start

    def get_next(self):
        if self.index >= self.end:
            return None
        sample = self.manifest["samples"][self.index]
        feed, _ = sample_arrays(self.path, sample)
        self.index += 1
        self.calls += 1
        self.seen.add(sample["index"])
        # Evidence that the actual Olive pass consumed this DataConfig reader.
        self.usage_path.write_text(
            json.dumps(
                {
                    "adapter_version": "1.0",
                    "dataset_id": self.manifest["id"],
                    "fingerprint": self.manifest["fingerprint"],
                    "read_calls": self.calls,
                    "unique_batches": len(self.seen),
                    "batch_count": len(self.manifest["samples"]),
                }
            ),
            encoding="utf-8",
        )
        return feed

    def rewind(self):
        self.index = self.start

    def set_range(self, start_index, end_index):
        if not 0 <= start_index < end_index <= len(self.manifest["samples"]):
            raise ValueError("Invalid calibration sample range")
        self.start, self.end, self.index = start_index, end_index, start_index


@Registry.register_dataset("opweaver_npz_manifest_v1")
def manifest_dataset(manifest_path, manifest_sha256, usage_path):
    return NPZCalibrationReader(manifest_path, manifest_sha256, usage_path)


@Registry.register(DataContainerType.DATA_CONTAINER, name="OpWeaverNPZDataContainerV1")
class OpWeaverNPZDataContainerV1(DataContainer):
    def create_calibration_dataloader(self, model_path=None, io_config=None):
        return self.load_dataset()
