"""Re-extract the published MN9-vs-sugar-GRN-rate curve (Shiu et al. 2024, Fig. 1d source data).

The numbers live in `results/figure_1/fig_1d_rate.csv` (+ `_std.csv`) inside the 4.5 GB `results.zip` of the
paper's data archive (Edmond, doi:10.17617/3.CZODIW, MIT licence). We read only the few kilobytes we need via HTTP
range requests and write `flybrain/reference/shiu2024_fig1d.json`, which the correctness experiment compares to.

    uv run --no-sync python -m experiments.fetch_shiu_fig1d            # rewrite the reference JSON
    uv run --no-sync python -m experiments.fetch_shiu_fig1d --check    # verify the committed JSON against the archive
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ARCHIVE_DOI = "10.17617/3.CZODIW"
ARCHIVE_FILE_ID = 223847
ARCHIVE_URL = f"https://edmond.mpg.de/api/access/datafile/{ARCHIVE_FILE_ID}"
ARCHIVE_MD5 = "f6f2e314821fa6a4196214e3b2b14bc4"  # as published by the archive for results.zip
MEMBER_RATE = "results/figure_1/fig_1d_rate.csv"
MEMBER_STD = "results/figure_1/fig_1d_rate_std.csv"
MN9_IDS = {"mn9_published": 720575940660219265, "mn9_partner": 720575940645521262}
OUT = Path(__file__).resolve().parents[1] / "flybrain" / "reference" / "shiu2024_fig1d.json"


class HttpRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests, with block caching."""

    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos = url, block, 0
        self._cache: dict[int, bytes] = {}
        self.requests = 0
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "dino-fly"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            self.size = int(resp.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[
            whence
        ]
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self._cache:
            lo, hi = i * self.block, min((i + 1) * self.block, self.size) - 1
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={lo}-{hi}", "User-Agent": "dino-fly"}
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
            if len(data) != hi - lo + 1:
                raise OSError("server ignored the Range header")
            self._cache[i] = data
            self.requests += 1
        return self._cache[i]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        end = min(self.pos + n, self.size)
        out = bytearray()
        while self.pos < end:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off : off + (end - self.pos)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)


def fetch() -> dict:
    remote = HttpRangeFile(ARCHIVE_URL)
    with zipfile.ZipFile(remote) as zf:
        rate = pd.read_csv(io.BytesIO(zf.read(MEMBER_RATE)), index_col=0)
        std = pd.read_csv(io.BytesIO(zf.read(MEMBER_STD)), index_col=0)
    if rate.index.dtype != "int64":
        raise TypeError("flywire ids must parse as int64")

    def freq(col: str) -> int:
        return int(col.removeprefix("sugarR_").removesuffix("Hz"))

    cols = sorted((c for c in rate.columns if c.startswith("sugarR_")), key=freq)
    out = {
        "description": "Published MN9 firing rate vs. sugar-GRN Poisson rate, Shiu et al. 2024 Fig. 1d source data "
        "(30 trials x 1 s, FlyWire materialization 630, Brian2). Rates in Hz; std is the population s.d. over trials.",
        "provenance": {
            "doi": ARCHIVE_DOI,
            "url": ARCHIVE_URL,
            "archive_md5": ARCHIVE_MD5,
            "archive_size": remote.size,
            "members": [MEMBER_RATE, MEMBER_STD],
            "licence": "MIT (Copyright (c) 2023 Philip Shiu and Nico Spiller)",
            "extracted_by": "brain/experiments/fetch_shiu_fig1d.py",
        },
        "connectome": "flywire630",
        "n_trials": 30,
        "t_run_s": 1.0,
        "grn_rate_hz": [freq(c) for c in cols],
        "neurons": {},
        "n_active_neurons_at_max_rate": int(rate[cols[-1]].notna().sum()),
    }
    for name, fid in MN9_IDS.items():
        out["neurons"][name] = {
            "flywire_id": str(fid),  # string: root ids exceed 2^53 and must never become JSON numbers
            "rate_hz": [0.0 if pd.isna(rate.loc[fid, c]) else float(rate.loc[fid, c]) for c in cols],
            "std_hz": [0.0 if pd.isna(std.loc[fid, c]) else float(std.loc[fid, c]) for c in cols],
        }
    print(
        f"[fetch] {remote.requests} range requests, {sum(map(len, remote._cache.values())) / 1e6:.1f} MB read"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify the committed JSON instead of rewriting it"
    )
    args = parser.parse_args()
    fresh = fetch()
    text = json.dumps(fresh, indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("reference JSON differs from the archive", file=sys.stderr)
            return 1
        print("reference JSON matches the archive")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
