#!/usr/bin/env python3
"""Compare ordered packed batches to individual inference using a WAV manifest.

Output includes transcripts and model timestamps: retain it privately. HTTP
intervals include full response transfer, not file reads/multipart construction.
CUDA stages are reported separately by the server, never subtracted from HTTP.
No retries, dropped trials, or best-of filtering. Standard library only.
"""

import argparse
import json
import random
import time
import urllib.request
import wave
from pathlib import Path


def multipart(files, verbose=False):
    boundary = "parakeet-batch-benchmark-boundary"
    body = bytearray()
    fields = [("file", data) for data in files]
    if verbose:
        fields += [
            ("response_format", b"verbose_json"),
            ("timestamp_granularities[]", b"word"),
        ]
    for name, data in fields:
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        ).encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def request(url, body, content_type):
    req = urllib.request.Request(url, body, {"Content-Type": content_type})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read()
    elapsed = (time.perf_counter() - start) * 1000
    return json.loads(raw), elapsed


def packed_rows(samples):
    frames = ((samples // 160 + 1) + 7) // 8
    return (frames + 15) // 16 * 16 + 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--baseline-url")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4, choices=range(1, 17))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--shuffle-seed", type=int)
    args = parser.parse_args()
    if args.trials < 1 or args.warmups < 0:
        parser.error("trials must be positive and warmups nonnegative")
    items = json.loads(args.manifest.read_text())
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(items)
    groups, group, rows = [], [], 0
    for item in items:
        path = args.manifest.parent / item["audio"]
        with wave.open(str(path)) as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (
                1,
                16000,
                2,
            ):
                raise ValueError(f"invalid WAV contract: {item['id']}")
            samples = wav.getnframes()
        count = packed_rows(samples)
        if group and (
            len(group) == args.batch_size
            or rows + count > 1008
            or count >= 528
            or rows >= 528
            and len(group) == 1
        ):
            groups.append(group)
            group, rows = [], 0
        group.append((item, path, samples))
        rows += count
    if group:
        groups.append(group)
    mismatches = 0
    with args.output.open("x") as output:
        for group_index, group in enumerate(groups):
            files = [path.read_bytes() for _, path, _ in group]
            batch_body, batch_type = multipart(files)
            singles = [multipart([data], verbose=True) for data in files]
            batch_url = args.url.rstrip("/") + "/v1/audio/transcriptions/batch"
            single_url = args.url.rstrip("/") + "/v1/audio/transcriptions"
            if group_index == 0:
                for _ in range(args.warmups):
                    request(batch_url, batch_body, batch_type)
                    for body, content_type in singles:
                        request(single_url, body, content_type)
            for trial in range(args.trials):
                # Alternate single/batch order, retaining all results.
                if (group_index + trial) % 2:
                    batch_result, batch_ms = request(batch_url, batch_body, batch_type)
                    single_results = [
                        request(single_url, body, kind) for body, kind in singles
                    ]
                else:
                    single_results = [
                        request(single_url, body, kind) for body, kind in singles
                    ]
                    batch_result, batch_ms = request(batch_url, batch_body, batch_type)
                if len(batch_result["results"]) != len(group):
                    raise AssertionError("batch result count changed")
                parity = []
                for result, (single, _) in zip(batch_result["results"], single_results):
                    equal = result["token_ids"] == [
                        token["id"] for token in single["tokens"]
                    ]
                    parity.append(equal)
                    mismatches += not equal
                    for token in result["tokens"]:
                        assert (
                            0
                            <= token["start"]
                            <= token["end"]
                            <= result["audio_seconds"]
                        )
                        assert token["duration_frames"] in range(5)
                baseline = None
                if args.baseline_url:
                    baseline = [
                        request(
                            args.baseline_url.rstrip("/") + "/v1/audio/transcriptions",
                            *multipart([data]),
                        )
                        for data in files
                    ]
                row = {
                    "group": group_index,
                    "trial": trial,
                    "items": [item for item, _, _ in group],
                    "sample_counts": [samples for _, _, samples in group],
                    "batch_http_ms": batch_ms,
                    "batch": batch_result,
                    "singles": single_results,
                    "baseline": baseline,
                    "single_token_parity": parity,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
            if group_index % 100 == 0:
                print(
                    f"groups {group_index + 1}/{len(groups)}, token mismatches {mismatches}",
                    flush=True,
                )
    print(f"completed {len(items)} inputs; token mismatches {mismatches}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
