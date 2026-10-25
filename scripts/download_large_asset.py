"""Resume large HTTP downloads in parallel, validating every byte range."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 32 * 1024 * 1024


def main(name, workers):
    item = next(v for v in json.loads((ROOT / "data/sources.json").read_text())
                if v["dataset"] == name)
    target = ROOT / item["path"]
    partial = target.with_name(target.name + ".part")
    progress = target.with_name(target.name + ".ranges.json")
    marker = target.with_name(target.name + ".complete.json")
    if marker.exists() and target.is_file():
        print(f"Already downloaded {name}", flush=True)
        return
    total = item["bytes"]
    if progress.exists():
        state = json.loads(progress.read_text())
        assert state["bytes"] == total and state["chunk"] % CHUNK == 0
        factor = state["chunk"] // CHUNK
        completed = {old * factor + step for old in state["completed"]
                     for step in range(factor) if (old * factor + step) * CHUNK < total}
    else:
        # Preserve complete chunks from a previous sequential download.
        have = partial.stat().st_size if partial.exists() else 0
        completed = set(range(have // CHUNK))
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, total)
    lock = threading.Lock()
    local = threading.local()

    def save_progress():
        temp = progress.with_suffix(".tmp")
        temp.write_text(json.dumps({"bytes": total, "chunk": CHUNK,
                                    "completed": sorted(completed)}))
        temp.replace(progress)

    save_progress()

    def transfer(indices):
        start, end = indices[0] * CHUNK, min((indices[-1] + 1) * CHUNK, total) - 1
        resume = start
        for attempt in range(64):
            try:
                if not hasattr(local, "session"):
                    local.session = requests.Session()
                with local.session.get(getattr(local, "download_url", item["url"]), headers={
                    "Range": f"bytes={resume}-{end}",
                }, timeout=(30, 60), stream=True) as response:
                    expected = f"bytes {resume}-{end}/{total}"
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (response.status_code != 206 or match is None
                            or int(match[1]) != resume or int(match[2]) < end
                            or int(match[3]) != total):
                        raise ValueError(f"Expected {expected}, received {response.status_code} {response.headers.get('Content-Range')}")
                    local.download_url = response.url
                    offset = resume
                    # Some SharePoint endpoints return start..EOF even when
                    # an end byte is supplied. Consume only our validated range.
                    while offset <= end:
                        data = response.raw.read(min(4 * 1024 * 1024, end + 1 - offset))
                        if not data:
                            break
                        if offset + len(data) > end + 1:
                            raise ValueError("Range response exceeded requested size")
                        view = memoryview(data)
                        while view:
                            count = os.pwrite(fd, view, offset)
                            if count <= 0:
                                raise OSError("Short file write")
                            view = view[count:]
                            offset += count
                            resume = offset
                            if offset % CHUNK == 0 or offset == total:
                                with lock:
                                    completed.add((offset - 1) // CHUNK)
                                    save_progress()
                    if offset != end + 1:
                        raise ValueError("Incomplete range response")
                with lock:
                    # A short network read can leave subsequent writes
                    # unaligned with chunk boundaries. The complete response
                    # still validates every chunk in this group.
                    completed.update(indices)
                    save_progress()
                return
            except Exception as error:
                print(f"Chunks {indices[0]}-{indices[-1]}, attempt {attempt + 1}: {error}", flush=True)
                if attempt == 63:
                    raise
                time.sleep(5)

    count = (total + CHUNK - 1) // CHUNK
    pending = [i for i in range(count) if i not in completed]
    # Reuse each response for up to 512 MiB; keep 32 MiB progress records.
    # This avoids repeatedly paying the overseas sharing endpoint's latency.
    groups = []
    for index in pending:
        if not groups or len(groups[-1]) == 16 or index != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(index)
    print(f"{name}: {total} bytes, {len(pending)} chunks remaining, {workers} connections", flush=True)
    started = last_report = time.monotonic()
    initial = len(completed)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer, group) for group in groups]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                now = time.monotonic()
                if now - last_report > 40:
                    done = len(completed)
                    speed = (done - initial) * CHUNK / (now - started) / 1024**2
                    print(f"{name}: {done}/{count} chunks, {speed:.1f} MiB/s", flush=True)
                    last_report = now
        os.fsync(fd)
    finally:
        os.close(fd)
    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for data in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(data)
    if item.get("sha256") and digest.hexdigest() != item["sha256"]:
        raise ValueError("SHA256 mismatch")
    partial.rename(target)
    marker.write_text(json.dumps({**item, "sha256": digest.hexdigest()}, indent=2) + "\n")
    print(f"Downloaded {name}; SHA256 {digest.hexdigest()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
