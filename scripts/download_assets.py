"""Download verified mirror URLs from data/sources.json, with resume support."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def download(item):
    target = ROOT / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    log = target.with_name(target.name + ".log")
    marker = target.with_name(target.name + ".complete.json")
    if marker.exists() and target.exists():
        print(f"Already downloaded: {target.name}", flush=True)
        return
    if (item.get("bytes") or 0) >= 10 * 1024**3:
        subprocess.run([sys.executable, str(ROOT / "scripts/download_large_asset.py"),
                        item["dataset"], "4"], check=True)
        return
    partial = target.with_name(target.name + ".part")
    if target.exists() and item.get("bytes") == target.stat().st_size:
        target.rename(partial)
    print(f"Downloading {target.name}", flush=True)
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if item.get("bytes") == offset:
            break
        try:
            # The sharing endpoint sets cookies required by its redirect target.
            import http.cookiejar
            opener = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
            )
            headers = {"User-Agent": "python-requests/2.32.5"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(item["url"], headers=headers)
            with opener.open(request, timeout=90) as response:
                append = offset > 0 and response.status == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Unexpected resume offset")
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise ValueError("Received HTML instead of an archive")
                with partial.open("ab" if append else "wb") as stream:
                    last_report = time.monotonic()
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(chunk)
                        if time.monotonic() - last_report >= 45:
                            print(f"{target.name}: {stream.tell() / 1024**3:.2f} GiB", flush=True)
                            last_report = time.monotonic()
            if item.get("bytes") is not None and partial.stat().st_size != item["bytes"]:
                raise ValueError(f"Incomplete download: {partial.stat().st_size} / {item['bytes']} bytes")
            break
        except Exception as error:
            with log.open("a") as stream:
                stream.write(f"Attempt {attempt + 1}: {error}\n")
            if attempt == 5:
                raise
            time.sleep(5)
    if item.get("bytes") is not None and partial.stat().st_size != item["bytes"]:
        raise ValueError(f"Size mismatch: {target}")
    digest = hashlib.sha256()
    md5 = hashlib.md5()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            md5.update(chunk)
    if item.get("md5") and md5.hexdigest() != item["md5"]:
        raise ValueError(f"MD5 mismatch: {target}")
    if item.get("sha256") and digest.hexdigest() != item["sha256"]:
        raise ValueError(f"SHA256 mismatch: {target}")
    partial.rename(target)
    marker.write_text(json.dumps({**item, "bytes": target.stat().st_size,
                                 "sha256": digest.hexdigest()}, indent=2) + "\n")
    print(f"Downloaded {target.name}: {target.stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    items = json.loads((ROOT / "data/sources.json").read_text())
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(download, item): item for item in items}
        for future in concurrent.futures.as_completed(pending):
            try:
                future.result()
            except Exception as error:
                failures.append(pending[future]["path"])
                print(f"FAILED {pending[future]['path']}: {error}", flush=True)
    if failures:
        raise SystemExit(f"Failed downloads: {failures}")
