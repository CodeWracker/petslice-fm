#!/usr/bin/env python
"""Download the merged autoPET FDG+PSMA NIfTI package and verify its checksum.

The fdat object store limits throughput per connection and its signed URLs expire after 60 s, so
the file is fetched in byte ranges, re-resolving the redirect for every range. Progress is kept in
a sidecar file, which makes the download resumable.
"""

import argparse
import hashlib
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RECORD = "rdkqd-wdh87"
FILENAME = "psma-fdg-pet-ct-lesions_v2.zip"
API = f"https://fdat.uni-tuebingen.de/api/records/{RECORD}"
CHUNK = 256 * 1024 * 1024
RETRIES = 30


def record_entry() -> tuple[int, str]:
    with urllib.request.urlopen(API, timeout=60) as response:
        entry = json.load(response)["files"]["entries"][FILENAME]
    return entry["size"], entry["checksum"].removeprefix("md5:")


def fetch_range(offset: int, end: int, fd: int) -> None:
    url = f"{API}/files/{FILENAME}/content"
    position = offset
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={position}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise OSError(f"expected HTTP 206, got {response.status}")
                while block := response.read(1024 * 1024):
                    os.pwrite(fd, block, position)
                    position += len(block)
            if position == end + 1:
                return
        except OSError as error:
            print(f"retry {attempt} at {position}: {error}", flush=True)
            time.sleep(min(5 * (attempt + 1), 60))
    raise RuntimeError(f"giving up on range {offset}-{end}")


def verify(path: Path, expected_md5: str) -> None:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected_md5:
        raise RuntimeError(f"md5 mismatch: {digest.hexdigest()} != {expected_md5}")
    print(f"md5 verified: {expected_md5}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--seed-bytes",
        type=int,
        default=0,
        help="Treat this many leading bytes of an existing partial file as already downloaded.",
    )
    args = parser.parse_args()

    size, md5 = record_entry()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / FILENAME
    state_path = target.with_suffix(".chunks.json")
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    done |= {offset for offset in range(0, args.seed_bytes - CHUNK + 1, CHUNK)}

    fd = os.open(target, os.O_CREAT | os.O_WRONLY)
    os.ftruncate(fd, size)
    pending = [offset for offset in range(0, size, CHUNK) if offset not in done]
    lock = threading.Lock()
    started = time.time()

    def work(offset: int) -> int:
        fetch_range(offset, min(offset + CHUNK, size) - 1, fd)
        return offset

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, offset) for offset in pending]
        for count, future in enumerate(as_completed(futures), start=1):
            with lock:
                done.add(future.result())
                state_path.write_text(json.dumps(sorted(done)))
            rate = count * CHUNK / (time.time() - started) / 1e6
            remaining = (len(pending) - count) * CHUNK / 1e6 / rate / 3600
            print(f"{len(done)} chunks, {rate:.1f} MB/s, {remaining:.2f} h left", flush=True)
    os.close(fd)
    verify(target, md5)


if __name__ == "__main__":
    main()
