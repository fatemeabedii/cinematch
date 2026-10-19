"""Download and extract MovieLens-1M (resumable; the network may drop every few MB).

Usage: python scripts/download_data.py [--force]
"""
import _bootstrap  # noqa: F401

import argparse
import shutil
import ssl
import time
import zipfile
from pathlib import Path

import certifi
import requests
from requests.adapters import HTTPAdapter

from cinematch.config import ML1M_EXPECTED_FILES, ML1M_URL, ML1M_ZIP, RAW_DIR


class _TrustStoreAdapter(HTTPAdapter):
    """HTTPS adapter that trusts the OS certificate store plus certifi (verification stays ON).

    Some machines (corporate proxies, TLS inspection) have roots only in the OS store, where
    requests' certifi-only default fails with CERTIFICATE_VERIFY_FAILED.
    """

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()  # CERT_REQUIRED + hostname check, OS roots
        ctx.load_verify_locations(certifi.where())
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)

    def cert_verify(self, conn, url, verify, cert):
        if verify is True:
            return  # keep our ssl_context instead of requests' certifi-only bundle
        return super().cert_verify(conn, url, verify, cert)


def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", _TrustStoreAdapter())
    return s


def _zip_ok(path: Path) -> bool:
    """True if `path` is a readable zip whose CRCs all check out."""
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def _content_length(session: requests.Session, url: str) -> int:
    """Ask the server for the total size (streamed GET, body not read)."""
    with session.get(url, stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        cl = r.headers.get("Content-Length")
        if cl is None:
            raise IOError("server sent no Content-Length header")
        return int(cl)


def download_ml1m(dest_zip: Path = ML1M_ZIP, url: str = ML1M_URL, chunk: int = 1 << 20,
                  max_retries: int = 50) -> Path:
    """Resumable download of the ML-1M zip into `dest_zip` using HTTP Range requests.

    Partial data goes to `<dest_zip>.part`; each failure (exception or short read) sleeps 2 s and
    resumes from the current size, up to `max_retries` attempts in total. The finished file is
    CRC-checked before being renamed to `dest_zip`.
    """
    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    if dest_zip.exists() and _zip_ok(dest_zip):
        print(f"[download] {dest_zip} already present and valid")
        return dest_zip
    part = dest_zip.with_name(dest_zip.name + ".part")
    session = _session()
    total = None
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        try:
            if total is None:
                total = _content_length(session, url)
                print(f"[download] {url} ({total / 1e6:.1f} MB)")
            have = part.stat().st_size if part.exists() else 0
            if have > total:
                part.unlink()
                have = 0
            if have < total:
                headers = {"Range": f"bytes={have}-"} if have else {}
                with session.get(url, headers=headers, stream=True, timeout=(10, 30)) as r:
                    r.raise_for_status()
                    if have and r.status_code != 206:  # server ignored Range -> restart from 0
                        print("[download] server ignored Range header; restarting from 0")
                        have = 0
                        mode = "wb"
                    else:
                        mode = "ab"
                    with open(part, mode) as f:
                        for block in r.iter_content(chunk_size=chunk):
                            if block:
                                f.write(block)
                have = part.stat().st_size
                print(f"[download] {have / 1e6:.1f}/{total / 1e6:.1f} MB (attempt {attempts})")
                if have < total:
                    raise IOError("short read")
            if not _zip_ok(part):
                print("[download] zip failed CRC check; deleting and restarting")
                part.unlink()
                continue
            part.replace(dest_zip)
            print(f"[download] done: {dest_zip}")
            return dest_zip
        except (requests.RequestException, IOError) as exc:
            print(f"[download] attempt {attempts}/{max_retries} interrupted: {exc!s:.120}; retrying in 2 s")
            time.sleep(2)
    raise RuntimeError(f"Download failed after {max_retries} attempts; rerun to resume from {part}")


def extract(zip_path: Path = ML1M_ZIP, raw_dir: Path = RAW_DIR) -> Path:
    """Extract ratings/movies/users .dat files into `raw_dir`, flattening the inner `ml-1m/` folder."""
    raw_dir = Path(raw_dir)
    if all((raw_dir / name).exists() for name in ML1M_EXPECTED_FILES):
        print(f"[extract] files already in {raw_dir}")
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = {Path(n).name: n for n in zf.namelist()}
        for name in ML1M_EXPECTED_FILES:
            with zf.open(members[name]) as src, open(raw_dir / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"[extract] wrote {', '.join(ML1M_EXPECTED_FILES)} to {raw_dir}")
    return raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="delete existing zip/.dat files and redownload")
    args = parser.parse_args()
    if args.force:
        for p in [ML1M_ZIP, ML1M_ZIP.with_name(ML1M_ZIP.name + ".part")] + [RAW_DIR / n for n in ML1M_EXPECTED_FILES]:
            if p.exists():
                p.unlink()
    zip_path = download_ml1m()
    raw_dir = extract(zip_path)
    for name in ML1M_EXPECTED_FILES:
        print(f"  {raw_dir / name}")
    with open(raw_dir / "ratings.dat", "rb") as f:
        n_lines = sum(1 for _ in f)
    print(f"ratings.dat lines: {n_lines:,}")


if __name__ == "__main__":
    main()
