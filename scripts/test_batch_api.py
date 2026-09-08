#!/usr/bin/env python3
"""Exercise a running L4 server's batch boundaries, ordering, and HTTP errors."""

import argparse
import io
import json
import urllib.error
import wave
from pathlib import Path

from benchmark_batch import multipart, request


def silence(samples):
    data = io.BytesIO()
    with wave.open(data, "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\0\0" * samples)
    return data.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--speech", type=Path, required=True, help="a short 16 kHz mono WAV"
    )
    args = parser.parse_args()
    single_url = args.url.rstrip("/") + "/v1/audio/transcriptions"
    batch_url = single_url + "/batch"

    def batch(files):
        return request(batch_url, *multipart(files))[0]["results"]

    def metadata(item):
        return item["text"], item["tokens"], item["words"]

    speech = args.speech.read_bytes()
    with wave.open(io.BytesIO(speech)) as wav:
        same_length_silence = silence(wav.getnframes())
    pair = batch([speech, same_length_silence])
    reversed_pair = batch([same_length_silence, speech])
    assert [metadata(x) for x in pair] == [metadata(x) for x in reversed_pair[::-1]], (
        "ordering/boundary isolation"
    )
    replaced_neighbor = batch([speech, speech])
    assert metadata(replaced_neighbor[0]) == metadata(pair[0]), (
        "neighbor contents leaked across boundary"
    )
    assert metadata(replaced_neighbor[1]) == metadata(pair[0]), (
        "position changed output"
    )
    wide = batch([speech] * 8)
    assert all(metadata(item) == metadata(wide[0]) for item in wide)
    assert len(batch([silence(16000)] * 16)) == 16
    tiny = [1, 159, 160, 161, 1279, 1280, 1281, 16383, 16384, 16385]
    tiny_files = [silence(samples) for samples in tiny]
    small = batch(tiny_files)
    assert len(small) == len(tiny)
    assert small[0]["token_ids"] == small[1]["token_ids"] == []
    for item, samples in zip(small, tiny):
        assert item["audio_seconds"] == samples / 16000
    # Dirty the reusable buffers, then verify that small inputs cannot read a stale tail.
    batch([speech, speech])
    assert [metadata(x) for x in batch(tiny_files)] == [metadata(x) for x in small]
    one = batch([speech])[0]
    individual = request(single_url, *multipart([speech], verbose=True))[0]
    assert metadata(one) == metadata(individual), "batch-size-one regression"

    for files in [
        [],
        [speech] * 17,
        [silence(30 * 16000)] * 3,
        [silence(41 * 16000), speech],
        [b"not a WAV"],
    ]:
        try:
            batch(files)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            assert "error" in json.loads(error.read())
        else:
            raise AssertionError("invalid batch accepted")
    # A rejected batch must not poison the engine's next valid request.
    assert metadata(batch([speech])[0]) == metadata(one)
    print(
        "PASS: singleton, mixed lengths, padding, boundaries, order, dirty-buffer reuse, and batch errors"
    )


if __name__ == "__main__":
    main()
