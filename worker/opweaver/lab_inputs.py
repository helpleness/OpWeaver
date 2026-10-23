"""Validated seeded tensors or named NPZ samples, without pickle deserialization."""

import math
import zipfile
from pathlib import Path

import numpy as np

from .benchmark import digest_file

MAX_INPUT_BYTES = 256 * 1024 * 1024
DTYPES = {
    "tensor(float)": "float32",
    "tensor(double)": "float64",
    "tensor(float16)": "float16",
    **{
        f"tensor({t})": t
        for t in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]
    },
}


def check_feed(metadata, feed, shapes=None):
    if set(feed) != {m.name for m in metadata}:
        raise ValueError("Input names/count must exactly match the model's required inputs")
    symbols = {}
    for item in metadata:
        array = feed[item.name]
        expected = DTYPES.get(item.type)
        if expected is None:
            raise ValueError(
                f"Unsupported input {item.name}: {item.type}; numerical dense tensors only"
            )
        if array.dtype != np.dtype(expected):
            raise ValueError(f"Input {item.name} dtype must be {expected}, got {array.dtype}")
        if item.shape is None:
            raise ValueError(
                f"Input {item.name} has no runtime rank metadata; this input type is unsupported"
            )
        if len(array.shape) != len(item.shape):
            raise ValueError(
                f"Input {item.name} rank mismatch: expected {item.shape}, got {array.shape}"
            )
        if shapes and item.name in shapes and list(array.shape) != shapes[item.name]:
            raise ValueError(f"Input {item.name} does not match the configured shape")
        for expected_dim, actual in zip(item.shape, array.shape, strict=True):
            if isinstance(expected_dim, int) and expected_dim != actual:
                raise ValueError(
                    f"Input {item.name} shape mismatch: expected {item.shape}, got {array.shape}"
                )
            if isinstance(expected_dim, str):
                if expected_dim in symbols and symbols[expected_dim] != actual:
                    raise ValueError(f"Symbolic dimension {expected_dim} differs between inputs")
                symbols[expected_dim] = actual
        if array.nbytes > MAX_INPUT_BYTES:
            raise ValueError("Input exceeds the 256 MiB per-experiment input budget")


def load_npz(path):
    if path.suffix.lower() != ".npz" or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Use an NPZ file under 256 MiB, with input names as array keys")
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist()]
        if (
            len(names) != len(set(names))
            or sum(i.file_size for i in archive.infolist()) > MAX_INPUT_BYTES
        ):
            raise ValueError(
                "NPZ contains duplicate names or exceeds the uncompressed input budget"
            )
        for entry in archive.infolist():
            if not entry.filename.endswith(".npy"):
                raise ValueError("NPZ must contain named NPY tensors only")
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                reader = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }.get(version)
                if reader is None:
                    raise ValueError(f"Unsupported NPY header version: {version}")
                shape, _, dtype = reader(stream)
                if dtype.hasobject:
                    raise ValueError("Object arrays cannot be loaded without pickle")
                payload_bytes = math.prod(shape) * dtype.itemsize
                if (
                    payload_bytes > MAX_INPUT_BYTES
                    or payload_bytes != entry.file_size - stream.tell()
                ):
                    raise ValueError("NPY tensor header exceeds its actual payload/input budget")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def prepare_inputs(metadata, config, count, folder, benchmark_config):
    names = {m.name for m in metadata}
    if set(config.shapes) - names:
        raise ValueError("Configured shapes contain unknown input names")
    if config.mode == "npz" and len(config.files) < count:
        raise ValueError(
            f"Provide at least {count} distinct NPZ sample files; one named array per model input"
        )
    if config.mode == "npz" and len({str(Path(p).resolve()) for p in config.files[:count]}) < count:
        raise ValueError("Provide distinct NPZ files for each validation sample")
    folder.mkdir(parents=True, exist_ok=True)
    samples, records, total_bytes = [], [], 0
    for index in range(count):
        if config.mode == "npz":
            path = Path(config.files[index]).resolve(strict=True)
            feed = load_npz(path)
        else:
            rng = np.random.default_rng(config.seed + index)
            feed = {}
            for item in metadata:
                dtype = DTYPES.get(item.type)
                if dtype is None:
                    raise ValueError(f"Unsupported input {item.name}: {item.type}")
                shape = config.shapes.get(item.name, item.shape)
                if shape is None:
                    raise ValueError(
                        f"Input {item.name} has unknown rank; supply an explicit concrete shape"
                    )
                if any(type(d) is not int or d < 0 for d in shape):
                    raise ValueError(
                        f"Input {item.name} has dynamic shape {item.shape}; explicitly configure every dimension"
                    )
                if math.prod(shape) * np.dtype(dtype).itemsize > MAX_INPUT_BYTES:
                    raise ValueError("Configured input exceeds the 256 MiB input budget")
                if dtype == "bool":
                    array = rng.integers(0, 2, size=shape).astype(dtype)
                elif np.issubdtype(np.dtype(dtype), np.integer):
                    array = rng.integers(0, 8, size=shape).astype(dtype)
                else:
                    array = rng.standard_normal(shape).astype(dtype)
                feed[item.name] = array
        check_feed(metadata, feed, config.shapes)
        if benchmark_config.batch_size is not None:
            array = feed.get(benchmark_config.batch_input)
            if array is None or not array.ndim or array.shape[0] != benchmark_config.batch_size:
                raise ValueError(
                    "Declared batch size must match dimension 0 of the selected batch input"
                )
        elif benchmark_config.batch_input is not None:
            raise ValueError(
                "Declare a batch size together with the batch input, or leave both unset"
            )
        total_bytes += sum(v.nbytes for v in feed.values())
        if total_bytes > MAX_INPUT_BYTES:
            raise ValueError("Samples exceed the 256 MiB per-experiment input budget")
        target = folder / f"sample-{index}.npz"
        np.savez(target, **feed)
        records.append(
            {
                "index": index,
                "path": target.name,
                "sha256": digest_file(target),
                "seed": config.seed + index if config.mode == "random" else None,
                "source": config.files[index]
                if config.mode == "npz"
                else "seeded standard normal / integer [0,8) / boolean [0,2)",
                "tensors": [
                    {"name": n, "shape": list(v.shape), "dtype": str(v.dtype)}
                    for n, v in feed.items()
                ],
            }
        )
        samples.append(feed)
    return samples, records
