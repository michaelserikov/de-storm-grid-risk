"""Download ENTSO-E transmission unavailability (article 10.1.A/B) for the German control areas, 2015-2025.

The Transparency Platform publishes only outages that affect cross-zonal capacity, queried by pairs of
control areas. Internal queries (in = out) return nothing for Germany, so every German TSO is paired
with each neighbour and with the other German TSOs. Documents are deduplicated by mRID, keeping the
latest revision, and written to data/outages/entsoe_transmission_outages.csv.

The API key is read from ENTSOE_API_KEY or, if unset, from ~/.entsoe_token.

    python scripts/entsoe_outages.py
"""
from __future__ import annotations

import io
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from itertools import combinations
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "entsoe"
OUT = ROOT / "data" / "outages" / "entsoe_transmission_outages.csv"
API = "https://web-api.tp.entsoe.eu/api"
YEARS = range(2015, 2026)
DE = {"50Hertz": "10YDE-VE-------2", "Amprion": "10YDE-RWENET---I",
      "TenneT DE": "10YDE-EON------1", "TransnetBW": "10YDE-ENBW-----N"}
NEIGHBOURS = {"NL": "10YNL----------L", "BE": "10YBE----------2", "LU": "10YLU-CEGEDEL-NQ", "FR": "10YFR-RTE------C",
              "CH": "10YCH-SWISSGRIDZ", "AT": "10YAT-APG------L", "CZ": "10YCZ-SEPS-----K", "PL": "10YPL-AREA-----S",
              "DK1": "10YDK-1--------W", "DK2": "10YDK-2--------M", "SE4": "10Y1001A1001A47J", "NO2": "10YNO-2--------T"}
PAGE = 200


def token() -> str:
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        key = (Path.home() / ".entsoe_token").read_text(encoding="utf-8").strip()
    return key


def pairs() -> list[tuple[str, str, str, str]]:
    out = [(a, ca, b, cb) for a, ca in DE.items() for b, cb in NEIGHBOURS.items()]
    out += [(a, DE[a], b, DE[b]) for a, b in combinations(DE, 2)]
    return out


def fetch(params: dict, path: Path) -> bytes:
    """One request, cached on disk. Returns the raw body (a zip, an XML document or an acknowledgement)."""
    if path.exists():
        return path.read_bytes()
    for attempt in range(5):
        try:
            with urlopen(f"{API}?{urlencode(params)}", timeout=120) as r:
                body = r.read()
            break
        except HTTPError as e:
            body = e.read()
            if e.code in (429, 503):
                time.sleep(10 * (attempt + 1))
                continue
            break
        except URLError:
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"failed {params}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    time.sleep(0.2)
    return body


def documents(body: bytes) -> list[bytes]:
    if body[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(body))
        return [z.read(n) for n in z.namelist()]
    if b"Unavailability_MarketDocument" in body[:400]:
        return [body]
    return []


def strip(tag: str) -> str:
    return tag.split("}", 1)[-1]


def parse(xml: bytes, pair: str) -> list[dict]:
    root = ET.fromstring(xml)
    def find(node, name):
        for el in node.iter():
            if strip(el.tag) == name:
                return el
        return None
    def text(node, name):
        el = find(node, name)
        return el.text.strip() if el is not None and el.text else None
    head = {"mrid": text(root, "mRID"), "revision": int(text(root, "revisionNumber") or 0),
            "created": text(root, "createdDateTime"), "pair": pair}
    period = find(root, "unavailability_Time_Period.timeInterval")
    head["start"], head["end"] = text(period, "start"), text(period, "end")
    status = find(root, "docStatus")
    head["doc_status"] = text(status, "value") if status is not None else None
    doc_reasons = [text(r, "code") for r in root if strip(r.tag) == "Reason"]
    head["reason"] = ";".join(c for c in doc_reasons if c)
    head["reason_text"] = ";".join((text(r, "text") or "") for r in root if strip(r.tag) == "Reason").strip(";")
    rows = []
    for ts in (n for n in root if strip(n.tag) == "TimeSeries"):
        base = dict(head, business_type=text(ts, "businessType"), in_domain=text(ts, "in_Domain.mRID"),
                    out_domain=text(ts, "out_Domain.mRID"))
        assets = [a for a in ts.iter() if strip(a.tag) == "Asset_RegisteredResource"] or [None]
        for a in assets:
            rows.append(dict(base, asset_mrid=text(a, "mRID") if a is not None else None,
                             asset_name=text(a, "name") if a is not None else None,
                             psr_type=text(a, "asset_PSRType.psrType") if a is not None else None,
                             location=text(a, "location.name") if a is not None else None))
    return rows


def main():
    key = token()
    rows = []
    for a, ca, b, cb in pairs():
        pair = f"{a}-{b}"
        for y in YEARS:
            offset = 0
            while True:
                params = {"securityToken": key, "documentType": "A78", "in_Domain": ca, "out_Domain": cb,
                          "periodStart": f"{y}01010000", "periodEnd": f"{y + 1}01010000", "offset": offset}
                body = fetch(params, CACHE / pair / f"{y}_{offset}.bin")
                docs = documents(body)
                for d in docs:
                    rows += parse(d, pair)
                if len(docs) < PAGE:
                    break
                offset += PAGE
            print(pair, y, "docs so far", len(rows), flush=True)
    df = pd.DataFrame(rows)
    df = df.sort_values("revision").drop_duplicates(["mrid", "asset_mrid", "business_type"], keep="last")
    df = df[df["doc_status"] != "A13"]          # withdrawn
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("outage records", len(df), "documents", df["mrid"].nunique())


if __name__ == "__main__":
    main()
