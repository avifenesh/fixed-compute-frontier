#!/usr/bin/env python3
"""Fetch the pinned eval data over plain HTTPS, verify sha256 against pins.json, write normalized jsonl and a
MANIFEST.json. Needs pyarrow for the parquet files. Run once per box (prepare_box.sh data calls it).

    python prepare_data.py --out evals/data                       # everything
    python prepare_data.py --out D --only humaneval,ifeval        # some
    python prepare_data.py --out D --gmmlu-local /data/ai-ml/datasets/gmmlu   # he-test/he-dev parquet from disk

Every download is checked against the pinned raw sha256. The normalized files are deterministic (dataset order,
sorted keys), so their sha256 is pinned too (pins.json "normalized_sha256") and checked after writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evalcommon import load_pins, log, sha256_file, utc, write_json  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
GH = "https://raw.githubusercontent.com/{repo}/{rev}/{path}"
UA = {"User-Agent": "negeig-evals/1"}


def download(url: str, dest: Path, want_sha: Optional[str], tries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and want_sha and sha256_file(dest) == want_sha:
        return
    last = ""
    for k in range(tries):
        tmp = dest.with_name(dest.name + ".part")
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            got = sha256_file(tmp)
            if want_sha and got != want_sha:
                tmp.unlink()
                raise SystemExit(f"{url}: sha256 {got} != pinned {want_sha}; the pin or the upstream file moved")
            os.replace(tmp, dest)
            return
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = str(e)
            time.sleep(min(30, 2 ** k))
    raise SystemExit(f"{url}: download failed after {tries} tries: {last}")


def parquet_rows(path: Path) -> List[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def dump_jsonl(path: Path, rows: List[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return {"sha256": sha256_file(path), "rows": len(rows), "bytes": path.stat().st_size}


# ----------------------------------------------------------------------------------------------------------
def norm_mmlupro(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        opts = [str(o) for o in r["options"]]
        ai = int(r["answer_index"])
        if not 0 <= ai < len(opts):
            raise SystemExit(f"MMLU-Pro {r['question_id']}: answer_index {ai} outside {len(opts)} options")
        out.append({"id": str(r["question_id"]), "category": r["category"], "question": r["question"],
                    "options": opts, "answer_index": ai, "src": r.get("src")})
    if len({r["id"] for r in out}) != len(out):
        raise SystemExit("MMLU-Pro question_id is not unique")
    return out


def norm_humaneval(rows: List[dict]) -> List[dict]:
    out = [{"id": r["task_id"], "prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"],
            "canonical_solution": r["canonical_solution"]} for r in rows]
    if len({r["id"] for r in out}) != len(out):
        raise SystemExit("HumanEval task_id is not unique")
    return out


def norm_gmmlu(rows: List[dict], split: str) -> List[dict]:
    out = []
    for r in rows:
        ans = str(r["answer"]).strip().upper()
        if ans not in ("A", "B", "C", "D"):
            raise SystemExit(f"Global-MMLU {r['sample_id']}: answer {ans!r}")
        out.append({"id": str(r["sample_id"]), "subject": r["subject"], "question": r["question"],
                    "options": [r["option_a"], r["option_b"], r["option_c"], r["option_d"]], "answer": ans})
    if split == "test" and len({r["id"] for r in out}) != len(out):
        raise SystemExit("Global-MMLU he test sample_id is not unique")
    return out


# ----------------------------------------------------------------------------------------------------------
def prepare(out: Path, only: List[str], local: Dict[str, Path], pins: dict) -> dict:
    raw = out / "raw"
    files: Dict[str, dict] = {}
    want_norm = pins.get("normalized_sha256", {})

    def hf(ds: str, key: str) -> Path:
        d = pins["datasets"][ds]
        dest = raw / ds / Path(d[key]["path"]).name
        if ds in local and key in local[ds]:
            src = local[ds][key]
            if sha256_file(src) != d[key]["sha256"]:
                raise SystemExit(f"{src}: sha256 differs from the pinned {d[key]['sha256']}")
            return src
        download(HF.format(repo=d["repo"], rev=d["revision"], path=d[key]["path"]), dest, d[key]["sha256"])
        return dest

    def put(name: str, rows: List[dict], raw_paths: List[Path]) -> None:
        info = dump_jsonl(out / f"{name}.jsonl", rows)
        info["raw_sha256"] = [sha256_file(p) for p in raw_paths]
        exp = want_norm.get(f"{name}.jsonl")
        if exp and exp != info["sha256"]:
            raise SystemExit(f"{name}.jsonl: normalized sha256 {info['sha256']} != pinned {exp}")
        files[f"{name}.jsonl"] = info
        log(f"wrote {name}.jsonl rows={info['rows']} sha256={info['sha256'][:16]}")

    if "mmlupro" in only:
        p = hf("mmlupro", "test")
        put("mmlupro_test", norm_mmlupro(parquet_rows(p)), [p])
    if "humaneval" in only:
        p = hf("humaneval", "test")
        put("humaneval_test", norm_humaneval(parquet_rows(p)), [p])
    if "gmmlu" in only:
        pt, pd_ = hf("gmmlu_he", "test"), hf("gmmlu_he", "dev")
        put("gmmlu_he_test", norm_gmmlu(parquet_rows(pt), "test"), [pt])
        put("gmmlu_he_dev", norm_gmmlu(parquet_rows(pd_), "dev"), [pd_])
    if "ifeval" in only:
        d = pins["datasets"]["ifeval"]
        dest = out / "ifeval_input_data.jsonl"
        src = local.get("ifeval", {}).get("input")
        if src:
            if sha256_file(src) != d["input"]["sha256"]:
                raise SystemExit(f"{src}: sha256 differs from the pinned value")
            if Path(src).resolve() != dest.resolve():
                shutil.copy2(src, dest)
        else:
            download(HF.format(repo=d["repo"], rev=d["revision"], path=d["input"]["path"]), dest,
                     d["input"]["sha256"])
        rows = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != d["rows"] or len({r["key"] for r in rows}) != len(rows):
            raise SystemExit(f"IFEval input: {len(rows)} rows, expected {d['rows']} with unique keys")
        files["ifeval_input_data.jsonl"] = {"sha256": sha256_file(dest), "rows": len(rows), "bytes": dest.stat().st_size,
                                            "raw_sha256": [d["input"]["sha256"]]}
    if "ifeval_harness" in only:
        h = pins["ifeval_harness"]
        pkg = out / "ifeval_harness" / "instruction_following_eval"
        for name, sha in h["files"].items():
            src = local.get("ifeval_harness", {}).get(name)
            if src:
                if sha256_file(src) != sha:
                    raise SystemExit(f"{src}: sha256 differs from the pinned {sha}")
                pkg.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, pkg / name)
            else:
                download(GH.format(repo=h["repo"], rev=h["commit"], path=f"{h['dir']}/{name}"), pkg / name, sha)
            files[f"ifeval_harness/instruction_following_eval/{name}"] = {"sha256": sha, "bytes": (pkg / name).stat().st_size}
        log(f"IFEval harness at {h['commit'][:12]} in {pkg}")
    return files


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="mmlupro,humaneval,gmmlu,ifeval,ifeval_harness")
    ap.add_argument("--gmmlu-local", help="dir holding he-test.parquet and he-dev.parquet (or the HF layout)")
    ap.add_argument("--mmlupro-local", help="the pinned MMLU-Pro test parquet")
    ap.add_argument("--humaneval-local", help="the pinned HumanEval test parquet")
    ap.add_argument("--ifeval-local", help="the pinned ifeval_input_data.jsonl")
    ap.add_argument("--ifeval-harness-local", help="dir holding the pinned harness files")
    a = ap.parse_args(argv)
    pins = load_pins()
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    bad = [x for x in only if x not in ("mmlupro", "humaneval", "gmmlu", "ifeval", "ifeval_harness")]
    if bad:
        raise SystemExit(f"--only: unknown {bad}")
    local: Dict[str, Dict] = {}
    if a.gmmlu_local:
        g = Path(a.gmmlu_local)
        t = next((p for p in (g / "he-test.parquet", g / "he" / "test-00000-of-00001.parquet") if p.is_file()), None)
        d = next((p for p in (g / "he-dev.parquet", g / "he" / "dev-00000-of-00001.parquet") if p.is_file()), None)
        if not t or not d:
            raise SystemExit(f"{g}: no he test and dev parquet found")
        local["gmmlu_he"] = {"test": t, "dev": d}
    if a.mmlupro_local:
        local["mmlupro"] = {"test": Path(a.mmlupro_local)}
    if a.humaneval_local:
        local["humaneval"] = {"test": Path(a.humaneval_local)}
    if a.ifeval_local:
        local["ifeval"] = {"input": Path(a.ifeval_local)}
    if a.ifeval_harness_local:
        hd = Path(a.ifeval_harness_local)
        local["ifeval_harness"] = {n: hd / n for n in pins["ifeval_harness"]["files"] if (hd / n).is_file()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = prepare(out, only, local, pins)
    man_path = out / "MANIFEST.json"
    old = json.loads(man_path.read_text())["files"] if man_path.is_file() else {}
    old.update(files)
    write_json(man_path, {"schema": "negeig-general-eval-data/1", "created": utc(), "files": old,
                          "pins_sha256": hashlib.sha256((Path(__file__).resolve().parent / "pins.json").read_bytes()).hexdigest()})
    log(f"MANIFEST.json written with {len(old)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
