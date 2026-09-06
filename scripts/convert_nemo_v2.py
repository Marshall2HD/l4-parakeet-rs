#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "blake3==1.0.9",
#   "numpy==2.5.2",
#   "PyYAML==6.0.3",
#   "safetensors==0.8.0",
#   "torch==2.14.0",
# ]
# ///
"""Normalize the pinned NVIDIA Parakeet v2 .nemo into the L4 build input.

PyTorch is intentionally confined to this offline conversion step. The Rust/CUDA
runtime consumes neither .nemo nor PyTorch and will pack model.safetensors into
its selected sm_89 deployment layouts ahead of time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
import yaml
from blake3 import blake3
from safetensors.torch import save_file


MODEL_REPOSITORY = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_REVISION = "ae9ad07059c7c739ffaf932226a8fe64ae2620b0"
NEMO_BYTES = 2_472_222_720
NEMO_BLAKE3 = "51929378d4f9a301ca00a964fece355f46dab3ffb093e5e7f516ecd7365e7c84"
EXPECTED_TENSORS = 725
EXPECTED_PARAMETERS = 617_908_398
EXPECTED_FLOAT_PARAMETERS = 617_908_374


def digest_file(path: Path) -> str:
    digest = blake3()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def archive_members(archive: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    by_basename: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        basename = Path(member.name).name
        if basename in by_basename:
            raise ValueError(f"duplicate archive basename {basename!r}")
        by_basename[basename] = member
    return by_basename


def open_member(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    basename: str,
) -> BinaryIO:
    member = members.get(basename)
    if member is None:
        raise ValueError(f"missing {basename!r} in .nemo archive")
    source = archive.extractfile(member)
    if source is None:
        raise ValueError(f"could not read {basename!r} from .nemo archive")
    return source


def nemo_asset_name(value: str) -> str:
    prefix = "nemo:"
    if not value.startswith(prefix):
        raise ValueError(f"expected a {prefix!r} asset reference, got {value!r}")
    return value.removeprefix(prefix)


def validate_source_config(config: Mapping[str, object]) -> None:
    encoder = config["encoder"]
    decoder = config["decoder"]
    joint = config["joint"]
    preprocessor = config["preprocessor"]
    decoding = config["decoding"]
    assert isinstance(encoder, Mapping)
    assert isinstance(decoder, Mapping)
    assert isinstance(joint, Mapping)
    assert isinstance(preprocessor, Mapping)
    assert isinstance(decoding, Mapping)

    expected = {
        "sample_rate": (config["sample_rate"], 16_000),
        "mel bins": (preprocessor["features"], 128),
        "FFT": (preprocessor["n_fft"], 512),
        "window": (preprocessor["window_size"], 0.025),
        "hop": (preprocessor["window_stride"], 0.01),
        "normalization": (preprocessor["normalize"], "per_feature"),
        "encoder layers": (encoder["n_layers"], 24),
        "encoder hidden": (encoder["d_model"], 1024),
        "encoder heads": (encoder["n_heads"], 8),
        "subsampling": (encoder["subsampling_factor"], 8),
        "convolution kernel": (encoder["conv_kernel_size"], 9),
        "attention": (encoder["self_attention_model"], "rel_pos"),
        "decoder vocabulary": (decoder["vocab_size"], 1024),
        "joint classes": (joint["num_classes"], 1024),
        "joint extra outputs": (joint["num_extra_outputs"], 5),
        "durations": (decoding["durations"], [0, 1, 2, 3, 4]),
    }
    mismatches = [
        f"{name}: expected {wanted!r}, got {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError("source model contract mismatch:\n  - " + "\n  - ".join(mismatches))


def normalized_name(name: str) -> str | None:
    if name == "preprocessor.featurizer.fb":
        return "frontend.mel_filters"
    if name == "preprocessor.featurizer.window":
        return "frontend.window"

    prefixes = {
        "decoder.prediction.embed.": "decoder.embedding.",
        "decoder.prediction.dec_rnn.lstm.": "decoder.lstm.",
        "joint.pred.": "decoder.decoder_projector.",
        "joint.enc.": "encoder_projector.",
        "joint.joint_net.2.": "joint.head.",
        "encoder.pre_encode.conv.": "encoder.subsampling.layers.",
        "encoder.pre_encode.out.": "encoder.subsampling.linear.",
    }
    for source, target in prefixes.items():
        if name.startswith(source):
            return target + name.removeprefix(source)

    replacements = {
        ".conv.batch_norm.": ".conv.norm.",
        ".self_attn.linear_q.": ".self_attn.q_proj.",
        ".self_attn.linear_k.": ".self_attn.k_proj.",
        ".self_attn.linear_v.": ".self_attn.v_proj.",
        ".self_attn.linear_out.": ".self_attn.o_proj.",
        ".self_attn.linear_pos.": ".self_attn.relative_k_proj.",
        ".self_attn.pos_bias_u": ".self_attn.bias_u",
        ".self_attn.pos_bias_v": ".self_attn.bias_v",
    }
    normalized = name
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return normalized


def normalize_tensors(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for source_name in sorted(state):
        tensor = state[source_name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint value {source_name!r} is not a tensor")
        target_name = normalized_name(source_name)
        if target_name is None:
            continue
        if target_name in tensors:
            raise ValueError(f"normalization collision at {target_name!r}")
        if target_name == "frontend.mel_filters":
            if list(tensor.shape) != [1, 128, 257]:
                raise ValueError(f"unexpected mel filter shape {list(tensor.shape)}")
            tensor = tensor.squeeze(0)
        tensors[target_name] = tensor.detach().cpu().contiguous()

    parameter_count = sum(tensor.numel() for tensor in tensors.values())
    float_parameters = sum(
        tensor.numel() for tensor in tensors.values() if tensor.is_floating_point()
    )
    if len(tensors) != EXPECTED_TENSORS:
        raise ValueError(f"expected {EXPECTED_TENSORS} tensors, got {len(tensors)}")
    if parameter_count != EXPECTED_PARAMETERS:
        raise ValueError(
            f"expected {EXPECTED_PARAMETERS} parameters, got {parameter_count}"
        )
    if float_parameters != EXPECTED_FLOAT_PARAMETERS:
        raise ValueError(
            f"expected {EXPECTED_FLOAT_PARAMETERS} floating parameters, got {float_parameters}"
        )
    return tensors


def normalized_configs(config: Mapping[str, object]) -> dict[str, object]:
    encoder = config["encoder"]
    decoder = config["decoder"]
    joint = config["joint"]
    preprocessor = config["preprocessor"]
    decoding = config["decoding"]
    assert isinstance(encoder, Mapping)
    assert isinstance(decoder, Mapping)
    assert isinstance(joint, Mapping)
    assert isinstance(preprocessor, Mapping)
    assert isinstance(decoding, Mapping)
    prednet = decoder["prednet"]
    jointnet = joint["jointnet"]
    greedy = decoding["greedy"]
    assert isinstance(prednet, Mapping)
    assert isinstance(jointnet, Mapping)
    assert isinstance(greedy, Mapping)

    sample_rate = int(config["sample_rate"])
    model = {
        "architectures": ["ParakeetForTDT"],
        "blank_token_id": 1024,
        "decoder_hidden_size": int(prednet["pred_hidden"]),
        "dtype": "float32",
        "durations": decoding["durations"],
        "encoder_config": {
            "attention_bias": False,
            "conv_kernel_size": int(encoder["conv_kernel_size"]),
            "convolution_bias": bool(encoder["use_bias"]),
            "hidden_act": "silu",
            "hidden_size": int(encoder["d_model"]),
            "intermediate_size": int(encoder["d_model"])
            * int(encoder["ff_expansion_factor"]),
            "num_attention_heads": int(encoder["n_heads"]),
            "num_hidden_layers": int(encoder["n_layers"]),
            "num_key_value_heads": int(encoder["n_heads"]),
            "num_mel_bins": int(encoder["feat_in"]),
            "scale_input": bool(encoder["xscaling"]),
            "subsampling_conv_channels": int(encoder["subsampling_conv_channels"]),
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_factor": int(encoder["subsampling_factor"]),
        },
        "hidden_act": str(jointnet["activation"]),
        "max_symbols_per_step": int(greedy["max_symbols"]),
        "model_type": "parakeet_tdt",
        "num_decoder_layers": int(prednet["pred_rnn_layers"]),
        "vocab_size": int(decoder["vocab_size"]) + 1,
    }
    processor = {
        "blank_token": "<blank>",
        "feature_extractor": {
            "center": True,
            "dither": 0.0,
            "feature_extractor_type": "ParakeetFeatureExtractor",
            "feature_size": int(preprocessor["features"]),
            "hop_length": round(sample_rate * float(preprocessor["window_stride"])),
            "log_zero_guard": 2**-24,
            "mag_power": 2.0,
            "n_fft": int(preprocessor["n_fft"]),
            "normalize": str(preprocessor["normalize"]),
            "normalize_epsilon": 1e-5,
            "pad_mode": "constant",
            "padding_side": "right",
            "padding_value": float(preprocessor["pad_value"]),
            "preemphasis": 0.97,
            "return_attention_mask": True,
            "sampling_rate": sample_rate,
            "win_length": round(sample_rate * float(preprocessor["window_size"])),
            "window": str(preprocessor["window"]),
        },
        "processor_class": "ParakeetProcessor",
    }
    generation = {
        "_from_model_config": True,
        "decoder_start_token_id": 1024,
        "suppress_tokens": [1025, 1026, 1027, 1028, 1029],
    }
    return {
        "config.json": model,
        "processor_config.json": processor,
        "generation_config.json": generation,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def convert(nemo_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")
    if nemo_path.stat().st_size != NEMO_BYTES:
        raise ValueError(
            f"expected {NEMO_BYTES} source bytes, got {nemo_path.stat().st_size}"
        )
    source_digest = digest_file(nemo_path)
    if source_digest != NEMO_BLAKE3:
        raise ValueError(f"source BLAKE3 mismatch: expected {NEMO_BLAKE3}, got {source_digest}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent)
    )
    try:
        with tarfile.open(nemo_path, mode="r:") as archive:
            members = archive_members(archive)
            with open_member(archive, members, "model_config.yaml") as source:
                source_config = yaml.safe_load(source)
            if not isinstance(source_config, Mapping):
                raise TypeError("model_config.yaml is not an object")
            validate_source_config(source_config)

            checkpoint_path = staging / "model_weights.ckpt"
            with open_member(archive, members, "model_weights.ckpt") as source:
                with checkpoint_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)

            tokenizer_config = source_config["tokenizer"]
            assert isinstance(tokenizer_config, Mapping)
            assets = {
                "tokenizer.model": nemo_asset_name(str(tokenizer_config["model_path"])),
                "vocab.txt": nemo_asset_name(str(tokenizer_config["vocab_path"])),
                "tokenizer.vocab": nemo_asset_name(
                    str(tokenizer_config["spe_tokenizer_vocab"])
                ),
            }
            for target_name, source_name in assets.items():
                with open_member(archive, members, source_name) as source:
                    (staging / target_name).write_bytes(source.read())

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, Mapping):
            raise TypeError("model_weights.ckpt does not contain a state dictionary")
        tensors = normalize_tensors(state)
        weights_path = staging / "model.safetensors"
        # safetensors stores metadata in a Rust HashMap whose serialized key order
        # is randomized. Provenance lives in source.json; omitting duplicate
        # metadata makes independently generated weight files byte-identical.
        save_file(tensors, weights_path)
        del tensors, state, checkpoint
        checkpoint_path.unlink()

        for name, value in normalized_configs(source_config).items():
            write_json(staging / name, value)

        output_files = [
            "model.safetensors",
            "tokenizer.model",
            "tokenizer.vocab",
            "vocab.txt",
        ]
        manifest = {
            "schema_version": 1,
            "profile": "v2_english",
            "source": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "format": "nemo",
                "bytes": NEMO_BYTES,
                "blake3": source_digest,
            },
            "normalization": {
                "schema": "parakeet-l4-normalized-v1",
                "tensor_count": EXPECTED_TENSORS,
                "parameter_count": EXPECTED_FLOAT_PARAMETERS,
                "batch_norm_counters": EXPECTED_PARAMETERS - EXPECTED_FLOAT_PARAMETERS,
                "weights": {
                    "dtype": "float32",
                    "format": "safetensors",
                    "names": "parakeet-l4-normalized",
                },
            },
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "blake3": digest_file(staging / name),
                }
                for name in output_files
            },
        }
        write_json(staging / "source.json", manifest)
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.nemo, args.output_dir)


if __name__ == "__main__":
    main()
