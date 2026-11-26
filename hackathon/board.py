"""해커톤 리더보드: 기록(append) 과 표시. 표준 라이브러리만 사용한다.

    python3 hackathon/board.py              # 순위표 출력 (ID별 최신 제출)
    python3 hackathon/board.py --all        # 전체 제출 기록
    python3 hackathon/board.py --detail     # 설정·FPPI별 값까지 자세히
    python3 hackathon/board.py --watch 5    # 5초마다 갱신
    python3 hackathon/board.py --export     # 엑셀 '리더보드' 시트에 붙여넣을 csv 생성
"""
from __future__ import annotations

import csv
import fcntl
import html as _html
import json
import math
import os
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
LB_CSV = os.environ.get("HACKATHON_LB", os.path.join(HERE, "leaderboard.csv"))
BASE_JSON = os.path.join(HERE, "baseline.json")
EXPORT_CSV = os.path.join(HERE, "leaderboard_sheet.csv")

# 내부 열 이름 -> 엑셀 '리더보드' 시트 열 이름 (시트 순서와 같게 앞 18개를 둠, 뒤 3개는 추가 열)
SHEET_HEADERS = [
    ("id", "ID"), ("change", "바꾸는 부분"), ("setting", "설정"),
    ("q1", "Q1 Recall (FPPI 0.5)"), ("overlap", "겹침 Recall (FPPI 0.5)"), ("all", "전체 Recall (FPPI 0.5)"),
    ("d_q1", "Δ Q1 (%p)"), ("d_overlap", "Δ 겹침 (%p)"), ("d_all", "Δ 전체 (%p)"),
    ("upper_recovery", "상한 회수율 (Q1)"), ("worst_scene_d_q1", "최악 장면 Δ Q1 (%p)"),
    ("q1_seed_std", "Q1 시드 표준편차 (%p)"), ("all_025", "전체 Recall (FPPI 0.25)"),
    ("all_10", "전체 Recall (FPPI 1.0)"), ("params_m", "파라미터 (M)"), ("ms_per_cand", "후보당 추론 (ms)"),
    ("status", "상태"), ("note", "비고"),
    ("owner", "담당"), ("time", "제출 시간"), ("n_seeds", "시드 수"),
]
FIELDS = [k for k, _ in SHEET_HEADERS]
TEXT_FIELDS = {"id", "change", "setting", "status", "note", "owner", "time"}

MIN_EFFECT_PP = 1.0   # 엑셀 지표 정의: B 장면이 6개뿐이라 1%p 안팎 차이는 해석하지 않음


# ---------------------------------------------------------------- 기록
def append_row(row: dict, path: str = LB_CSV) -> None:
    """한 줄 추가. 여러 명이 동시에 제출해도 깨지지 않게 파일 잠금(flock)을 쓴다."""
    clean = {}
    for k in FIELDS:
        v = row.get(k, "")
        if isinstance(v, str):
            v = v.replace("\r", " ").replace("\n", " ")
        elif isinstance(v, float):
            v = "" if not math.isfinite(v) else round(v, 6)
        if k == "n_seeds" and v != "":
            v = int(v)
        clean[k] = v
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(clean)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def read_rows(path: str = LB_CSV) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            raw = list(csv.DictReader(f))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    rows = []
    for r in raw:
        rows.append({k: (r.get(k, "") if k in TEXT_FIELDS else _num(r.get(k))) for k in FIELDS})
    return rows


def load_reference() -> list[dict]:
    """기준선·상한 참조 행. baseline.json 은 lb.py 가 만든다."""
    if not os.path.exists(BASE_JSON):
        return []
    with open(BASE_JSON, encoding="utf-8") as f:
        b = json.load(f)
    nan = float("nan")
    out = [dict(id="참조-기준선", owner="", change="점수 그대로", setting="YOLO11n conf, 단일 합격선 (FPPI 0.5에 맞춤)",
                q1=b["q1"], overlap=b["overlap"], all=b["all"], d_q1=0.0, d_overlap=0.0, d_all=0.0,
                upper_recovery=nan, worst_scene_d_q1=0.0, q1_seed_std=nan, all_025=b["all_025"], all_10=b["all_10"],
                params_m=nan, ms_per_cand=nan, status="참조", note="B 기준. 모든 Δ의 기준값", time="", n_seeds=nan,
                _ref=True)]
    u = b.get("upper")
    if u:
        out.append(dict(id="참조-상한", owner="", change="점수", setting="후보 중 사람 박스를 완벽히 고를 때 (conf≥0.01 후보)",
                        q1=u["q1"], overlap=u["overlap"], all=u["all"], d_q1=u["d_q1"], d_overlap=u["d_overlap"],
                        d_all=u["d_all"], upper_recovery=1.0, worst_scene_d_q1=nan, q1_seed_std=nan,
                        all_025=nan, all_10=nan, params_m=nan, ms_per_cand=nan, status="참조",
                        note="엑셀 '참조-상한' 행의 값", time="", n_seeds=nan, _ref=True))
    return out


def ranked(rows: list[dict], all_runs: bool = False) -> list[dict]:
    """ID별 가장 최근 제출만 남기고(재제출은 덮어씀), Q1 recall 내림차순으로 정렬. 참조 행은 제자리에 끼워 넣는다."""
    if not all_runs:
        latest: dict[str, dict] = {}
        for r in sorted(rows, key=lambda r: r["time"]):
            latest[r["id"]] = r
        rows = list(latest.values())
    rows = rows + load_reference()
    key = lambda r: (-(r["q1"] if r["q1"] == r["q1"] else -1), -(r["overlap"] if r["overlap"] == r["overlap"] else -1))
    return sorted(rows, key=key)


def flag(r: dict) -> str:
    """효과를 해석하기 어려운 제출 표시: |ΔQ1| < max(1%p, 시드 표준편차)."""
    if r.get("_ref") or r["d_q1"] != r["d_q1"]:
        return ""
    if r["n_seeds"] == r["n_seeds"] and r["n_seeds"] < 2:
        return "시드1"
    thr = max(MIN_EFFECT_PP, r["q1_seed_std"] if r["q1_seed_std"] == r["q1_seed_std"] else 0.0)
    return "차이 불분명" if abs(r["d_q1"]) < thr else ""


# ---------------------------------------------------------------- 글자 폭 (한글 정렬)
def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _fit(s: str, n: int, right: bool = False) -> str:
    s = str(s)
    if _w(s) > n:
        out, w = "", 0
        for c in s:
            cw = 2 if unicodedata.east_asian_width(c) in "WF" else 1
            if w + cw > n - 1:
                break
            out, w = out + c, w + cw
        s = out + "…"
    pad = " " * (n - _w(s))
    return pad + s if right else s + pad


def _f(v, nd=3, pct=False, signed=False):
    if v != v:
        return "-"
    if pct:
        return f"{100 * v:.0f}%"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _t(r):  # 제출 시간 짧게
    return r["time"][5:16] if r["time"] else ""


# ---------------------------------------------------------------- 표시
def text_table(entries: list[dict]) -> str:
    cols = [("#", 3, True), ("ID", 11, False), ("담당", 6, False), ("바꾸는 부분", 12, False),
            ("Q1", 6, True), ("겹침", 6, True), ("전체", 6, True), ("ΔQ1", 6, True), ("회수", 5, True),
            ("최악Δ", 6, True), ("σ", 5, True), ("M", 5, True), ("ms", 6, True), ("제출", 11, False)]
    out = ["  ".join(_fit(n, w, r) for n, w, r in cols)]
    out.append("-" * (sum(w for _, w, _ in cols) + 2 * (len(cols) - 1)))
    rank = 0
    for r in entries:
        if not r.get("_ref"):
            rank += 1
        cells = ["" if r.get("_ref") else str(rank), r["id"], r["owner"], r["change"],
                 _f(r["q1"]), _f(r["overlap"]), _f(r["all"]), _f(r["d_q1"], 1, signed=True),
                 _f(r["upper_recovery"], pct=True), _f(r["worst_scene_d_q1"], 1, signed=True),
                 _f(r["q1_seed_std"], 2), _f(r["params_m"], 1), _f(r["ms_per_cand"], 2), _t(r)]
        out.append("  ".join(_fit(c, w, rt) for c, (_, w, rt) in zip(cells, cols)))
        sub = [f"설정: {r['setting']}"] if r["setting"] else []
        sub.append(f"전체@0.25 {_f(r['all_025'])} / @1.0 {_f(r['all_10'])}")
        sub.append(f"Δ겹침 {_f(r['d_overlap'], 1, signed=True)} / Δ전체 {_f(r['d_all'], 1, signed=True)}")
        if r["n_seeds"] == r["n_seeds"]:
            sub.append(f"시드 {int(r['n_seeds'])}개")
        fl = flag(r)
        if fl:
            sub.append(f"[{fl}]")
        if r["note"]:
            sub.append(r["note"])
        out.append("      " + " · ".join(sub))
    return "\n".join(out)


def html_table(entries: list[dict]) -> str:
    e = _html.escape
    th = "text-align:left;padding:6px 6px;border-bottom:1px solid rgba(128,128,128,.5);font-weight:600;font-size:12px"
    td = "padding:7px 6px;border-bottom:1px solid rgba(128,128,128,.25);vertical-align:top;font-size:13px"
    heads = ["#", "ID", "담당", "바꾸는 부분 / 설정", "Q1", "겹침", "전체", "ΔQ1(%p)", "상한 회수", "최악 장면Δ", "σ(%p)",
             "전체@.25", "전체@1.0", "M", "ms/후보", "제출"]
    h = [f"<table style='border-collapse:collapse;width:100%'><thead><tr>"
         + "".join(f"<th style='{th}'>{e(x)}</th>" for x in heads) + "</tr></thead><tbody>"]
    rank = 0

    def num(v):
        return f"<td style='{td};text-align:right;font-variant-numeric:tabular-nums'>{v}</td>"

    def delta(v, nd=1):
        if v != v:
            return num("-")
        c = "#2e8b57" if v > 0 else ("#d64545" if v < 0 else "inherit")
        return num(f"<span style='color:{c}'>{v:+.{nd}f}</span>")

    for r in entries:
        ref = r.get("_ref")
        if not ref:
            rank += 1
        bg = "background:rgba(128,128,128,.10);" if ref else ""
        fl = flag(r)
        badge = (f" <span style='font-size:11px;padding:1px 6px;border-radius:6px;background:rgba(230,160,0,.25)'>"
                 f"{e(fl)}</span>") if fl else ""
        note = f"<div style='font-size:11px;opacity:.65;margin-top:2px'>{e(r['note'])}</div>" if r["note"] else ""
        h.append(f"<tr style='{bg}'>"
                 f"<td style='{td}'>{'' if ref else rank}</td><td style='{td};font-weight:600'>{e(r['id'])}</td>"
                 f"<td style='{td}'>{e(r['owner'])}</td>"
                 f"<td style='{td}'>{e(r['change'])}{badge}<div style='font-size:11px;opacity:.65;margin-top:2px'>"
                 f"{e(r['setting'])}</div>{note}</td>"
                 + num(f"<b>{_f(r['q1'])}</b>") + num(_f(r["overlap"])) + num(_f(r["all"]))
                 + delta(r["d_q1"]) + num(_f(r["upper_recovery"], pct=True)) + delta(r["worst_scene_d_q1"])
                 + num(_f(r["q1_seed_std"], 2)) + num(_f(r["all_025"])) + num(_f(r["all_10"]))
                 + num(_f(r["params_m"], 1)) + num(_f(r["ms_per_cand"], 2))
                 + f"<td style='{td};font-size:12px'>{e(_t(r))}</td></tr>")
    h.append("</tbody></table>")
    return "".join(h)


def compact_table(entries: list[dict]) -> str:
    """한 실험 = 한 줄. 자세한 건 --detail"""
    cols = [("#", 3, True), ("ID", 12, False), ("담당", 5, False), ("바꾸는 부분", 10, False),
            ("Q1", 6, True), ("겹침", 6, True), ("전체", 6, True), ("ΔQ1", 7, True), ("σ", 5, True),
            ("판정", 10, False)]
    out = ["  ".join(_fit(n, w, r) for n, w, r in cols)]
    out.append("-" * (sum(w for _, w, _ in cols) + 2 * (len(cols) - 1)))
    rank = 0
    for r in entries:
        ref = r.get("_ref")
        if not ref:
            rank += 1
        cells = ["" if ref else str(rank), r["id"], r["owner"], r["change"],
                 _f(r["q1"]), _f(r["overlap"]), _f(r["all"]), _f(r["d_q1"], 1, signed=True),
                 _f(r["q1_seed_std"], 2), flag(r) or ("참조" if ref else "")]
        out.append("  ".join(_fit(c, w, rt) for c, (_, w, rt) in zip(cells, cols)))
    return "\n".join(out)


def show(all_runs: bool = False, path: str = LB_CSV, detail: bool = False):
    """노트북이면 표(HTML), 터미널이면 글자 표로 보여 준다."""
    entries = ranked(read_rows(path), all_runs)
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
        if get_ipython() is not None and get_ipython().__class__.__name__ != "TerminalInteractiveShell":
            display(HTML(f"<div style='font-size:12px;opacity:.7;margin-bottom:6px'>순위: B, FPPI 0.5에서의 Q1 recall "
                         f"· ID별 최신 제출 · {time.strftime('%Y-%m-%d %H:%M:%S')}</div>" + html_table(entries)))
            return
    except Exception:
        pass
    print(f"team35 해커톤 리더보드 (B, FPPI 0.5 Q1 recall 순) · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(text_table(entries) if detail else compact_table(entries))


def _cell(k, v):
    if isinstance(v, float):
        if v != v:
            return ""
        return int(v) if k == "n_seeds" else round(v, 4)
    return v


def export_sheet_csv(path: str = LB_CSV, out: str = EXPORT_CSV) -> str:
    """엑셀 '리더보드' 시트 열 순서·이름으로 내보낸다 (Excel에서 열면 한글 안 깨지게 utf-8-sig)."""
    entries = ranked(read_rows(path), all_runs=False)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([h for _, h in SHEET_HEADERS])
        for r in entries:
            w.writerow([_cell(k, r[k]) for k, _ in SHEET_HEADERS])
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--export" in args:
        print("저장:", export_sheet_csv())
    elif "--watch" in args:
        i = args.index("--watch")
        sec = float(args[i + 1]) if i + 1 < len(args) else 5.0
        try:
            while True:
                os.system("clear")
                show("--all" in args, detail="--detail" in args)
                time.sleep(sec)
        except KeyboardInterrupt:
            pass
    else:
        show("--all" in args, detail="--detail" in args)