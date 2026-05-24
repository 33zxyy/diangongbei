from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    HAS_MPL = False

# =========================
# 问题3参数区：需要时只改这里
# =========================
MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 365
DAYS_PER_MONTH = 30
DEPR_YEARS_DEFAULT = 20
PROFIT_RATE_LOW = 0.0
PROFIT_RATE_HIGH = 0.08
# 问题3采用“保本微利”目标。0.08 是上限，不应强行贴近；
# 为避免所有站点利润率都被补贴精确压到 0，本版默认按 3% 微利目标测算最低必要补贴。
PROFIT_RATE_TARGET = 0.03
SUBSIDY_PER_VISIT_CAP = 2.0
SUBSIDY_DAILY_CAP = {"小型": 1000.0, "中型": 1800.0, "大型": 2600.0}
PRICE_CAP_MULT = 1.50
CHARGE_SERVICES = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴"]
EMERGENCY = "紧急救助"
ALL_SERVICES = CHARGE_SERVICES + [EMERGENCY]

IGNORE_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "build",
    "dist",
}


# =========================
# 基础清洗和读取工具
# =========================
def clean_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    repl = [
        ("\n", ""), ("\r", ""), ("\t", ""), ("\u3000", ""), (" ", ""),
        ("：", ":"), ("（", "("), ("）", ")"), ("，", ","), ("；", ";"),
        ("—", "-"), ("－", "-"), ("→", "->"), ("≤", "<="), ("≥", ">="),
        ("㎡", "m2"), ("元/日", "元每天"), ("人次/日", "人次每天"),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    return s




def station_key(x: Any) -> str:
    """
    用于匹配问题2中不同写法的服务站名称。
    例如：服务站3、站点03、候选点3、3号站会统一为 N3；
    没有数字时保留清洗后的文本。
    """
    s = clean_text(x)
    if not s:
        return ""
    s2 = s
    for token in ["养老服务站", "社区服务站", "服务站点", "服务站", "站点", "候选点", "站址", "位置", "编号", "序号"]:
        s2 = s2.replace(token, "")
    nums = re.findall(r"\d+", s2) or re.findall(r"\d+", s)
    if nums:
        return "N" + "_".join(str(int(v)) for v in nums)
    return s2.strip(":-_,，。.;；()（）") or s


INVALID_STATION_TEXT = {
    "", "0", "0.0", "nan", "none", "null", "false", "否", "无", "无站点",
    "未覆盖", "未分配", "不覆盖", "不可覆盖", "未选中", "未服务", "-", "--", "/", "\\",
}


def is_valid_station_name(x: Any) -> bool:
    """判断分配表中的站点值是否真的是一个服务站。

    很多问题2结果会用 0 表示“未覆盖/未分配”。旧版会把 0 当作“站点0”，
    导致问题3图里出现横坐标 0，并把无效需求纳入优化。
    """
    s = clean_text(x)
    return s.lower() not in INVALID_STATION_TEXT


def parse_number(x: Any, default=np.nan) -> float:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except Exception:
        pass
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return default if math.isnan(v) else v
    s = str(x).strip()
    if not s:
        return default
    if "免费" in s or "公益" in s:
        return 0.0
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if not m:
        return default
    v = float(m.group())
    if "%" in s or "％" in s:
        v /= 100.0
    return v


def canonical_service(x: Any) -> str:
    s = clean_text(x)
    for k in ALL_SERVICES:
        if k in s:
            return k
    if "急救" in s or "紧急" in s:
        return EMERGENCY
    if "护理" in s:
        return "上门护理"
    if "理疗" in s or "康复" in s:
        return "康复理疗"
    return s


def canonical_scale(x: Any) -> str:
    s = clean_text(x)
    if "小" in s:
        return "小型"
    if "中" in s:
        return "中型"
    if "大" in s:
        return "大型"
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_text(c) for c in out.columns]
    return out


def is_ignored_path(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts) or path.name.startswith("~$")


def candidate_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    root = root.resolve()
    found: dict[str, Path] = {}

    direct_bases = [root]
    # 兼容从 B题 子目录、项目根目录或上一级目录运行
    direct_bases += [p for p in list(root.parents)[:2]]
    common_subdirs = [Path(""), Path("B题"), Path("B"), Path("data"), Path("results"), Path("output")]

    for base in direct_bases:
        for sub in common_subdirs:
            for pat in patterns:
                for p in (base / sub).glob(pat):
                    if p.is_file() and not is_ignored_path(p):
                        found[str(p.resolve())] = p.resolve()

    # 递归搜索当前项目目录，保证拿到最新的 B_problem2_results.xlsx
    for pat in patterns:
        for p in root.rglob(pat):
            if p.is_file() and not is_ignored_path(p):
                found[str(p.resolve())] = p.resolve()

    return sorted(found.values(), key=lambda p: (p.stat().st_mtime, len(str(p))), reverse=True)


def find_latest_file(root: Path, patterns: list[str], required: bool = True, label: str = "文件") -> Path | None:
    files = candidate_files(root, patterns)
    if not files:
        if required:
            raise FileNotFoundError(f"未找到{label}，搜索目录={root.resolve()}，模式={patterns}")
        return None
    # 优先选修改时间最新的；这就是避免误读旧 B_problem2_results.xlsx 的关键
    return files[0]


def read_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    raw = pd.read_excel(path, sheet_name=None)
    out: dict[str, pd.DataFrame] = {}
    for sn, df in raw.items():
        cdf = df.copy()
        cdf.columns = [clean_text(c) for c in cdf.columns]
        out[clean_text(sn)] = cdf
    return out


def read_table_auto_from_sheet(path: Path, sheet_name: str, header_keywords: list[str]) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        return pd.DataFrame()

    best_idx, best_score = 0, -1
    for idx in range(min(len(raw), 25)):
        row_text = "|".join(clean_text(v) for v in raw.iloc[idx].tolist())
        score = sum(1 for kw in header_keywords if clean_text(kw) in row_text)
        if any(s in row_text for s in ALL_SERVICES):
            score += 1
        if score > best_score:
            best_idx, best_score = idx, score

    cols = [clean_text(c) if clean_text(c) else f"col_{i}" for i, c in enumerate(raw.iloc[best_idx].tolist())]
    df = raw.iloc[best_idx + 1:].copy().reset_index(drop=True)
    df.columns = cols
    df = normalize_columns(df).dropna(how="all").reset_index(drop=True)
    return df


def choose_sheet_by_keywords(
    sheets: dict[str, pd.DataFrame],
    alternatives: list[list[str]],
    required: bool = True,
) -> tuple[str, pd.DataFrame] | tuple[None, None]:
    for keywords in alternatives:
        kws = [clean_text(k).lower() for k in keywords]
        # 先按 sheet 名匹配
        for name, df in sheets.items():
            txt = clean_text(name).lower()
            if all(k in txt for k in kws):
                return name, df
        # 再按列名和前几行内容匹配
        for name, df in sheets.items():
            ctxt = "|".join(clean_text(c).lower() for c in df.columns)
            dtxt = "|".join(clean_text(v).lower() for v in df.head(10).astype(str).values.ravel())
            txt = f"{clean_text(name).lower()}|{ctxt}|{dtxt}"
            if all(k in txt for k in kws):
                return name, df
    if required:
        available = {n: list(d.columns) for n, d in sheets.items()}
        raise KeyError(f"未找到 sheet，关键词组={alternatives}，已有={available}")
    return None, None


def pick_col(df: pd.DataFrame, keys: list[str], required: bool = True) -> str | None:
    norm_keys = [clean_text(k).lower() for k in keys]
    cc = [(c, clean_text(c).lower()) for c in df.columns]
    for c, norm in cc:
        if all(k in norm for k in norm_keys):
            return c
    for c, norm in cc:
        if any(k in norm for k in norm_keys):
            return c
    if required:
        raise KeyError(f"列未找到: {keys}, available={list(df.columns)}")
    return None


def pick_col_any(df: pd.DataFrame, candidates: list[list[str]], required: bool = True) -> str | None:
    last: Exception | None = None
    for keys in candidates:
        try:
            return pick_col(df, keys, required=True)
        except KeyError as e:
            last = e
    if required:
        raise last if last else KeyError(f"列未找到: {candidates}, available={list(df.columns)}")
    return None


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(float)
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
    if w.sum() <= 0:
        return float(np.nanmean(v)) if len(v) else 0.0
    return float(np.average(v, weights=w))


def s_price(price: float, p0: float) -> float:
    if p0 <= 0:
        return 1.0
    if price <= p0 + 1e-9:
        return 1.0
    if price <= 1.1 * p0 + 1e-9:
        return 0.9
    if price <= 1.2 * p0 + 1e-9:
        return 0.75
    return 0.6


def build_candidate_prices(base: float, cost: float) -> list[float]:
    """快速候选价格：只保留满意度分段边界和成本下限，避免 12^5 枚举爆炸。

    价格满意度是分段函数：<=p0、<=1.1p0、<=1.2p0、>1.2p0。
    同一分段内部继续加密价格，对满意度排序没有额外收益，只会显著增加运行时间。
    """
    base = float(base) if not pd.isna(base) else 0.0
    cost = float(cost) if not pd.isna(cost) else 0.0
    if base <= 0:
        return [0.0]
    cap = max(PRICE_CAP_MULT * base, cost)
    raw = [
        cost,
        base,
        1.10 * base,
        1.20 * base,
        1.50 * base,
    ]
    vals = []
    for v in raw:
        if v >= cost - 1e-9 and v <= cap + 1e-9:
            vals.append(round(float(v), 4))
    if cost > cap + 1e-9:
        vals.append(round(cost, 4))
    return sorted(set(vals))


# =========================
# 数据解析
# =========================
def load_price_ref_from_attachment2(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    target_sheet = None
    for s in xls.sheet_names:
        cs = clean_text(s)
        if ("营收" in cs and "支出" in cs) or ("服务营收" in cs):
            target_sheet = s
            break
    if target_sheet is None:
        for s in xls.sheet_names:
            raw = pd.read_excel(path, sheet_name=s, header=None, nrows=15)
            txt = "|".join(clean_text(v) for v in raw.astype(str).values.ravel())
            if "营收" in txt and "支出" in txt:
                target_sheet = s
                break
    if target_sheet is None:
        raise KeyError(f"附件2中未找到服务营收及支出表，已有 sheet={xls.sheet_names}")

    df = read_table_auto_from_sheet(path, target_sheet, ["服务项目", "营收", "支出"])
    if df.empty:
        raise ValueError(f"附件2 sheet {target_sheet} 为空")

    col_service = pick_col_any(df, [["服务项目"], ["项目"], ["服务"]])
    col_base = pick_col_any(df, [["营收"], ["基准价格"], ["价格"], ["单次营收"], ["单价"]])
    col_cost = pick_col_any(df, [["直接支出"], ["支出"], ["成本"]])

    out = df[[col_service, col_base, col_cost]].copy()
    out.columns = ["服务项目", "基准价格", "直接支出"]
    out["服务项目"] = out["服务项目"].map(canonical_service)
    out["基准价格"] = out["基准价格"].map(parse_number)
    out["直接支出"] = out["直接支出"].map(parse_number)
    out = out[out["服务项目"].isin(ALL_SERVICES)]
    out = out.dropna(subset=["服务项目", "基准价格", "直接支出"])
    out = out.groupby("服务项目", as_index=False).first()

    missing = [k for k in CHARGE_SERVICES if k not in set(out["服务项目"])]
    if missing:
        raise KeyError(f"附件2营收支出表缺少服务项目 {missing}，当前={list(out['服务项目'])}")
    if EMERGENCY not in set(out["服务项目"]):
        out = pd.concat([
            out,
            pd.DataFrame([{"服务项目": EMERGENCY, "基准价格": 0.0, "直接支出": 0.0}]),
        ], ignore_index=True)
    return out


def extract_demand_from_sheets(sheets: dict[str, pd.DataFrame]) -> tuple[str, pd.DataFrame]:
    candidates: list[tuple[int, str, pd.DataFrame]] = []
    for name, df in sheets.items():
        try:
            col_comm = pick_col_any(df, [["小区"], ["社区"]])
            col_type = pick_col_any(df, [["老人类型"], ["类型"], ["状态"]])
            col_service = pick_col_any(df, [["服务项目"], ["项目"], ["服务"]])
            col_demand = pick_col_any(df, [["实际需求"], ["消费约束", "需求"], ["需求人次"], ["需求量"], ["需求"]])
        except Exception:
            continue
        tmp = df[[col_comm, col_type, col_service, col_demand]].copy()
        tmp.columns = ["小区", "老人类型", "服务项目", "实际需求"]
        tmp["小区"] = tmp["小区"].map(clean_text)
        tmp["老人类型"] = tmp["老人类型"].map(clean_text)
        tmp["服务项目"] = tmp["服务项目"].map(canonical_service)
        tmp["实际需求"] = tmp["实际需求"].map(parse_number).fillna(0.0).clip(lower=0)
        tmp = tmp[tmp["服务项目"].isin(ALL_SERVICES)]
        tmp = tmp[(tmp["小区"] != "") & (tmp["老人类型"] != "")]
        score = len(tmp)
        n = clean_text(name)
        if "实际需求" in n:
            score += 10000
        if "分小区" in n:
            score += 2000
        if "分类型" in n:
            score += 2000
        if "汇总" in n:
            score -= 1000
        if len(tmp):
            candidates.append((score, name, tmp))
    if not candidates:
        raise KeyError("没有从工作簿中识别到需求表：需要列 小区/老人类型/服务项目/实际需求")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2].reset_index(drop=True)


def load_problem2_inputs(wb2: dict[str, pd.DataFrame]):
    """读取问题2关键 sheet。

    注意：问题2结果里常见的“06_小区需求满足汇总”并不一定是真正的
    “小区-服务站分配表”，它可能只有距离/满意度/是否覆盖等汇总字段。
    因此这里不再只靠 sheet 名强行判定，后续会结合最优选址表进一步判断服务站列是否真实。
    """
    s_st, df_st = choose_sheet_by_keywords(wb2, [["最优", "选址"], ["站点", "规模"], ["服务站", "规模"], ["选址", "规模"]])

    # 分配 sheet 先按名称候选；如果没有真正分配表，允许退化到“小区需求满足汇总”。
    alloc_candidates: list[tuple[int, str, pd.DataFrame]] = []
    for name, df in wb2.items():
        n = clean_text(name)
        cols = "|".join(clean_text(c) for c in df.columns)
        txt = f"{n}|{cols}"
        if "小区" not in txt and "社区" not in txt:
            continue
        score = 0
        if "分配" in n:
            score += 100
        if "服务站" in cols or "站点" in cols:
            score += 60
        if "需求满足" in n or "满足汇总" in n:
            score += 20
        if "覆盖" in n:
            score += 10
        # 避免把总体汇总、选址规模表误当成分配表
        if "总体" in n or "最优选址" in n or "规模" in n:
            score -= 50
        if score > 0:
            alloc_candidates.append((score, name, df))
    if alloc_candidates:
        alloc_candidates.sort(key=lambda x: x[0], reverse=True)
        s_alloc, df_alloc = alloc_candidates[0][1], alloc_candidates[0][2]
    else:
        s_alloc, df_alloc = choose_sheet_by_keywords(wb2, [["小区", "分配"], ["社区", "分配"], ["小区", "服务站"], ["分配"], ["需求", "满足"]])

    s_cov, df_cov = choose_sheet_by_keywords(wb2, [["覆盖"], ["覆盖", "统计"]], required=False)
    s_sum, df_sum = choose_sheet_by_keywords(wb2, [["总体"], ["汇总"], ["summary"]], required=False)
    return (s_st, df_st), (s_alloc, df_alloc), (s_cov, df_cov), (s_sum, df_sum)


def load_station_cost_ref_from_attachment3(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["规模", "建设成本", "日固定管理成本", "日服务能力"])
    sheets = read_workbook_sheets(path)
    best: pd.DataFrame | None = None
    best_score = -1
    for name, df in sheets.items():
        # 有些附件前面有标题行，先直接识别；失败再用自动表头重读
        frames = [df]
        try:
            auto = read_table_auto_from_sheet(path, name, ["规模", "建设", "固定", "服务能力"])
            frames.append(auto)
        except Exception:
            pass
        for cand in frames:
            try:
                col_scale = pick_col_any(cand, [["规模"], ["类型"]])
                col_build = pick_col_any(cand, [["建设", "成本"], ["建设成本"], ["建站", "成本"]], required=False)
                col_daily = pick_col_any(cand, [["日固定管理成本"], ["固定管理成本"], ["日", "固定"], ["运营", "成本"]], required=False)
                col_cap = pick_col_any(cand, [["日服务能力"], ["服务能力"], ["日", "能力"], ["能力"]], required=False)
            except Exception:
                continue
            if col_build is None and col_daily is None and col_cap is None:
                continue
            tmp = pd.DataFrame()
            tmp["规模"] = cand[col_scale].map(canonical_scale)
            tmp["建设成本"] = cand[col_build].map(parse_number) if col_build else np.nan
            tmp["日固定管理成本"] = cand[col_daily].map(parse_number) if col_daily else np.nan
            tmp["日服务能力"] = cand[col_cap].map(parse_number) if col_cap else np.nan
            tmp = tmp[tmp["规模"].isin(["小型", "中型", "大型"])]
            if tmp.empty:
                continue
            score = tmp[["建设成本", "日固定管理成本", "日服务能力"]].notna().sum().sum() + len(tmp)
            if score > best_score:
                best_score = int(score)
                best = tmp
    if best is None:
        return pd.DataFrame(columns=["规模", "建设成本", "日固定管理成本", "日服务能力"])
    best = best.groupby("规模", as_index=False).first()
    if best["建设成本"].notna().any() and best["建设成本"].max() < 10000:
        best["建设成本"] = best["建设成本"] * 10000
    return best


def load_stations(df_st: pd.DataFrame, cost_ref: pd.DataFrame) -> pd.DataFrame:
    col_station = pick_col_any(df_st, [["服务站"], ["站点"], ["站址"], ["位置"]])
    col_scale = pick_col_any(df_st, [["规模"], ["类型"]])
    col_build = pick_col_any(df_st, [["建设", "成本"], ["建设成本"], ["建站", "成本"]], required=False)
    col_daily = pick_col_any(df_st, [["日固定管理成本"], ["固定管理成本"], ["日", "固定"], ["日管理成本"], ["运营", "成本"]], required=False)
    col_cap = pick_col_any(df_st, [["日服务能力"], ["服务能力"], ["日", "能力"], ["能力"]], required=False)

    stations = pd.DataFrame()
    stations["服务站"] = df_st[col_station].map(clean_text)
    stations["规模"] = df_st[col_scale].map(canonical_scale)
    stations["建设成本"] = df_st[col_build].map(parse_number) if col_build else np.nan
    stations["日固定管理成本"] = df_st[col_daily].map(parse_number) if col_daily else np.nan
    stations["日服务能力"] = df_st[col_cap].map(parse_number) if col_cap else np.nan
    stations = stations[(stations["服务站"] != "") & stations["规模"].isin(["小型", "中型", "大型"])]
    stations = stations.drop_duplicates(subset=["服务站"], keep="first").reset_index(drop=True)

    if stations["建设成本"].notna().any() and stations["建设成本"].max() < 10000:
        stations["建设成本"] = stations["建设成本"] * 10000

    if not cost_ref.empty:
        stations = stations.merge(cost_ref, on="规模", how="left", suffixes=("", "_附件3"))
        for c in ["建设成本", "日固定管理成本", "日服务能力"]:
            c2 = f"{c}_附件3"
            if c2 in stations.columns:
                stations[c] = stations[c].where(stations[c].notna(), stations[c2])
                stations.drop(columns=[c2], inplace=True)

    missing_cols = [c for c in ["建设成本", "日固定管理成本"] if stations[c].isna().any()]
    if missing_cols:
        raise KeyError(
            f"服务站成本字段缺失: {missing_cols}。请检查 B_problem2_results.xlsx 的最优选址表或附件3成本表。\n{stations}"
        )
    return stations


def is_covered_value(x: Any) -> bool:
    s = clean_text(x).lower()
    if s in {"true", "1", "是", "已覆盖", "覆盖", "yes", "y"}:
        return True
    if s in {"false", "0", "否", "未覆盖", "不覆盖", "no", "n"}:
        return False
    return bool(s)



def pick_coverage_col(df: pd.DataFrame) -> str | None:
    """更安全地识别“是否覆盖”列，避免把“覆盖距离/覆盖半径/覆盖率”误当布尔列。"""
    bad_tokens = ["距离", "半径", "率", "数量", "人数", "比例", "范围", "能力"]
    exact_names = {"是否覆盖", "覆盖", "覆盖状态", "覆盖情况", "是否服务", "服务状态"}
    for c in df.columns:
        cn = clean_text(c)
        if cn in exact_names:
            return c
    for c in df.columns:
        cn = clean_text(c)
        if any(t in cn for t in bad_tokens):
            continue
        if ("覆盖" in cn and any(t in cn for t in ["是否", "状态", "情况"])) or ("服务" in cn and any(t in cn for t in ["是否", "状态"])):
            return c
    return None

def looks_like_selected_station_value(x: Any, selected_names: set[str], selected_keys: set[str]) -> bool:
    s = clean_text(x)
    if not is_valid_station_name(s):
        return False
    # 纯数字/小数通常是距离、坐标或需求量，不应当当作站点名称
    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
        return False
    return s in selected_names or station_key(s) in selected_keys


def pick_station_col_safe(df_alloc: pd.DataFrame, selected_names: set[str], selected_keys: set[str]) -> str | None:
    """安全识别真正的“分配服务站”列。

    旧版会把“服务站距离/服务站坐标/到服务站距离”这类列误识别成服务站列，
    因而出现 857.299...、715.766... 这样的“服务站”。本函数会排除这些数值列。
    """
    bad_tokens = ["距离", "路程", "时长", "时间", "满意", "综合", "需求", "满足", "覆盖", "能力", "数量", "人数", "成本", "费用", "利润", "收入", "支出", "坐标", "经度", "纬度", "率"]
    prefer_patterns = [
        ["分配站点"], ["分配服务站"], ["分配", "服务站"], ["分配", "站点"],
        ["对应服务站"], ["所属服务站"], ["最近服务站"], ["服务站名称"], ["站点名称"],
        ["服务站"], ["站点"],
    ]
    candidate_cols: list[tuple[int, str]] = []
    for c in df_alloc.columns:
        cn = clean_text(c)
        if any(tok in cn for tok in bad_tokens):
            continue
        score = 0
        for i, pats in enumerate(prefer_patterns):
            if all(p in cn for p in pats):
                score += 200 - i
                break
        vals = df_alloc[c].dropna().map(clean_text).tolist()
        if not vals:
            continue
        valid_match = sum(1 for v in vals if looks_like_selected_station_value(v, selected_names, selected_keys))
        numeric_like = sum(1 for v in vals if re.fullmatch(r"-?\d+(?:\.\d+)?", v or ""))
        unique_valid = len({station_key(v) for v in vals if looks_like_selected_station_value(v, selected_names, selected_keys)})
        # 只有列名像站点还不够，内容也必须能匹配到最优选址表中的 A/E/G/H/J 等站点。
        if valid_match > 0:
            score += valid_match * 50 + unique_valid * 20
        if numeric_like >= max(1, int(0.5 * len(vals))):
            score -= 500
        if score > 0:
            candidate_cols.append((score, c))
    if not candidate_cols:
        return None
    candidate_cols.sort(key=lambda x: x[0], reverse=True)
    return candidate_cols[0][1]


def infer_station_by_matrix(df: pd.DataFrame, selected_names: list[str]) -> dict[str, str]:
    """从当前 sheet 的宽表距离矩阵中推断 小区 -> 最近选中站点。

    兼容形如：第一列为小区，后面列名为 A/E/G/H/J，单元格为距离或响应时间。
    """
    selected_clean = [clean_text(x) for x in selected_names]
    selected_set = set(selected_clean)
    out: dict[str, str] = {}
    try:
        col_comm = pick_col_any(df, [["小区"], ["社区"]], required=False)
    except Exception:
        col_comm = None
    if not col_comm:
        return out
    station_cols = []
    for c in df.columns:
        cn = clean_text(c)
        if cn in selected_set:
            station_cols.append(c)
        elif station_key(c) in {station_key(x) for x in selected_clean}:
            station_cols.append(c)
    if not station_cols:
        return out
    for _, r in df.iterrows():
        comm = clean_text(r[col_comm])
        if not comm:
            continue
        vals = []
        for c in station_cols:
            v = parse_number(r[c], np.nan)
            if not pd.isna(v):
                vals.append((v, clean_text(c)))
        if vals:
            vals.sort(key=lambda x: x[0])
            out[comm] = vals[0][1]
    return out


def infer_station_by_long_table(df: pd.DataFrame, selected_names: list[str]) -> dict[str, str]:
    """从长表中推断 小区 -> 最近选中站点。

    兼容列：小区/起点 + 服务站/终点 + 距离/时间。
    """
    selected_set = {clean_text(x) for x in selected_names}
    selected_keys = {station_key(x) for x in selected_names}
    try:
        col_comm = pick_col_any(df, [["小区"], ["社区"], ["起点"], ["需求点"]], required=False)
        col_station = pick_station_col_safe(df, selected_set, selected_keys)
        col_dist = pick_col_any(df, [["距离"], ["时间"], ["路程"], ["响应"]], required=False)
    except Exception:
        return {}
    if not col_comm or not col_station or not col_dist:
        return {}
    rows = []
    for _, r in df.iterrows():
        comm = clean_text(r[col_comm])
        st = clean_text(r[col_station])
        d = parse_number(r[col_dist], np.nan)
        if comm and looks_like_selected_station_value(st, selected_set, selected_keys) and not pd.isna(d):
            rows.append((comm, st, d))
    out: dict[str, str] = {}
    for comm in sorted({x[0] for x in rows}):
        cand = [(d, st) for c, st, d in rows if c == comm]
        if cand:
            cand.sort(key=lambda x: x[0])
            out[comm] = cand[0][1]
    return out


def infer_station_assignment_from_problem2(wb2: dict[str, pd.DataFrame], selected_names: list[str]) -> dict[str, str]:
    """尝试从问题2所有 sheet 中恢复“小区 -> 服务站”映射。

    v8 修正：候选映射不再只看“小区数量”，还看覆盖到的最优站点数量。
    否则可能选中“06_小区需求满足汇总”这种汇总表，导致 5 个选址站点只识别出 4 个实际站点。
    """
    mapping: dict[str, str] = {}
    best_score = (-1, -1)

    def mapping_score(m: dict[str, str]) -> tuple[int, int]:
        # 第一优先：小区映射数量；第二优先：覆盖到的选中站点数量。
        return (len(m), len({station_key(v) for v in m.values() if clean_text(v)}))

    # 优先寻找真正的分配长表，其次寻找距离矩阵。
    for name, df in wb2.items():
        n = clean_text(name)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        if "分配" in n or "距离" in n or "矩阵" in n or "小区" in "|".join(clean_text(c) for c in df.columns):
            m = infer_station_by_long_table(df, selected_names)
            sc = mapping_score(m)
            if sc > best_score:
                mapping, best_score = m, sc
    for name, df in wb2.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        m = infer_station_by_matrix(df, selected_names)
        sc = mapping_score(m)
        if sc > best_score:
            mapping, best_score = m, sc
    return mapping


def load_allocation(df_alloc: pd.DataFrame, stations_raw: pd.DataFrame | None = None, wb2: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    selected_names = []
    if stations_raw is not None and not stations_raw.empty and "服务站" in stations_raw.columns:
        selected_names = [clean_text(x) for x in stations_raw["服务站"].dropna().tolist() if clean_text(x)]
    selected_set = set(selected_names)
    selected_keys = {station_key(x) for x in selected_names}

    col_comm = pick_col_any(df_alloc, [["小区"], ["社区"]])
    col_station = pick_station_col_safe(df_alloc, selected_set, selected_keys) if selected_names else None
    col_dist = pick_col_any(df_alloc, [["距离"], ["路程"], ["到站距离"]], required=False)
    col_sd = pick_col_any(df_alloc, [["距离满意度"], ["距离", "满意"]], required=False)
    col_sr = pick_col_any(df_alloc, [["响应满意度"], ["响应", "满意"]], required=False)
    col_ss = pick_col_any(df_alloc, [["综合满意度"], ["综合", "满意"]], required=False)
    col_cover = pick_coverage_col(df_alloc)

    alloc = pd.DataFrame()
    alloc["小区"] = df_alloc[col_comm].map(clean_text)

    inferred_map = infer_station_assignment_from_problem2(wb2 or {}, selected_names) if selected_names else {}

    if col_station:
        raw_station = df_alloc[col_station].map(clean_text)
        # 即使有站点列，也只接受能匹配最优选址表的值；其他行用推断映射补齐。
        alloc["服务站"] = [v if looks_like_selected_station_value(v, selected_set, selected_keys) else inferred_map.get(comm, "") for comm, v in zip(alloc["小区"], raw_station)]
    else:
        print("INFO: 当前小区表没有可靠的‘分配服务站’列，将根据问题2最优选址表和距离矩阵/分配长表重建小区-服务站映射。")
        alloc["服务站"] = alloc["小区"].map(lambda x: inferred_map.get(x, x if x in selected_set else ""))

    alloc["距离"] = df_alloc[col_dist].map(parse_number) if col_dist else np.nan
    alloc["距离满意度"] = df_alloc[col_sd].map(parse_number) if col_sd else np.nan
    alloc["响应满意度"] = df_alloc[col_sr].map(parse_number) if col_sr else np.nan
    alloc["问题2综合满意度"] = df_alloc[col_ss].map(parse_number) if col_ss else np.nan
    if col_cover:
        alloc["是否覆盖"] = df_alloc[col_cover].map(is_covered_value)
    else:
        # 没有覆盖列时，只要小区被重建到了选中服务站，就视为问题2可覆盖。
        alloc["是否覆盖"] = alloc["服务站"] != ""

    # 若最新问题2没有单独输出距离/响应满意度，先用综合满意度兜底，避免崩溃；更建议在问题2中输出这两列。
    if alloc["距离满意度"].isna().all() and alloc["问题2综合满意度"].notna().any():
        alloc["距离满意度"] = alloc["问题2综合满意度"]
    if alloc["响应满意度"].isna().all() and alloc["问题2综合满意度"].notna().any():
        alloc["响应满意度"] = alloc["问题2综合满意度"]
    alloc["距离满意度"] = alloc["距离满意度"].fillna(0.0).clip(lower=0, upper=1)
    alloc["响应满意度"] = alloc["响应满意度"].fillna(0.0).clip(lower=0, upper=1)

    # 只保留真正属于问题2最优选址表的站点。这样实际优化服务站数不会超过选址表解析出的 5 个。
    if selected_names:
        valid_selected_mask = alloc["服务站"].map(lambda v: clean_text(v) in selected_set or station_key(v) in selected_keys)
    else:
        valid_selected_mask = alloc["服务站"].map(is_valid_station_name)
    invalid_mask = ~valid_selected_mask
    if invalid_mask.any():
        print(f"INFO: 已剔除 {int(invalid_mask.sum())} 条无法匹配到最优选址表的分配记录。")

    alloc["是否覆盖"] = alloc["是否覆盖"].astype(bool) & valid_selected_mask
    alloc = alloc[(alloc["小区"] != "") & alloc["是否覆盖"]].copy()
    alloc = alloc.drop_duplicates(subset=["小区"], keep="first").reset_index(drop=True)
    alloc["站点键"] = alloc["服务站"].map(station_key)

    # 诊断：如果这里的服务站个数超过最优选址表个数，说明仍然误识别了站点列。
    if selected_names and alloc["服务站"].nunique() > len(selected_names):
        print("WARNING: 分配表解析出的站点数仍超过最优选址站点数，请检查问题2小区分配表列名。")
    return alloc




def collect_satisfaction_by_community(wb2: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从问题2所有 sheet 中按小区收集距离/响应/综合满意度，用于补齐最终分配表。

    v9 修正：分配关系和满意度信息可能不在同一个 sheet。
    如果最终采用的分配表只有“小区-服务站”而没有距离满意度/响应满意度，旧版会把二者填成 0，
    导致优化可及性被压到 0.5 (= 0.5*价格满意度)。本函数会从其他 sheet 回填这些满意度。
    """
    rows: list[dict[str, Any]] = []
    for name, df in wb2.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        try:
            col_comm = pick_col_any(df, [["小区"], ["社区"]], required=False)
        except Exception:
            col_comm = None
        if not col_comm:
            continue
        try:
            col_sd = pick_col_any(df, [["距离满意度"], ["距离", "满意"]], required=False)
        except Exception:
            col_sd = None
        try:
            col_sr = pick_col_any(df, [["响应满意度"], ["响应", "满意"]], required=False)
        except Exception:
            col_sr = None
        try:
            # 避免把价格满意度误当问题2综合满意度
            col_ss = pick_col_any(df, [["综合满意度"], ["综合", "满意"], ["总满意度"]], required=False)
        except Exception:
            col_ss = None
        if not any([col_sd, col_sr, col_ss]):
            continue
        for _, r in df.iterrows():
            comm = clean_text(r[col_comm])
            if not comm:
                continue
            sd = parse_number(r[col_sd], np.nan) if col_sd else np.nan
            sr = parse_number(r[col_sr], np.nan) if col_sr else np.nan
            ss = parse_number(r[col_ss], np.nan) if col_ss else np.nan
            # 有些表的满意度可能是 80/100 制，这里统一到 0-1。
            vals = []
            for v in [sd, sr, ss]:
                if not pd.isna(v):
                    vals.append(v)
            if vals and max(vals) > 1.5:
                if not pd.isna(sd): sd = sd / 100.0
                if not pd.isna(sr): sr = sr / 100.0
                if not pd.isna(ss): ss = ss / 100.0
            rows.append({
                "小区": comm,
                "补齐距离满意度": sd,
                "补齐响应满意度": sr,
                "补齐问题2综合满意度": ss,
                "来源sheet": clean_text(name),
                "有效字段数": int(not pd.isna(sd)) + int(not pd.isna(sr)) + int(not pd.isna(ss)),
            })
    if not rows:
        return pd.DataFrame(columns=["小区", "补齐距离满意度", "补齐响应满意度", "补齐问题2综合满意度", "来源sheet"])
    ref = pd.DataFrame(rows)
    # 每个小区优先保留字段最完整的一行。
    ref = ref.sort_values(["小区", "有效字段数"], ascending=[True, False]).drop_duplicates("小区", keep="first")
    for c in ["补齐距离满意度", "补齐响应满意度", "补齐问题2综合满意度"]:
        ref[c] = pd.to_numeric(ref[c], errors="coerce").clip(lower=0, upper=1)
    return ref[["小区", "补齐距离满意度", "补齐响应满意度", "补齐问题2综合满意度", "来源sheet"]].reset_index(drop=True)


def enrich_allocation_satisfaction(alloc: pd.DataFrame, wb2: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """用问题2其他 sheet 补齐分配表的距离满意度和响应满意度。"""
    if alloc.empty:
        return alloc
    out = alloc.copy()
    ref = collect_satisfaction_by_community(wb2)
    if not ref.empty:
        out = out.merge(ref, on="小区", how="left")
        for base_c, fill_c in [("距离满意度", "补齐距离满意度"), ("响应满意度", "补齐响应满意度"), ("问题2综合满意度", "补齐问题2综合满意度")]:
            if base_c not in out.columns:
                out[base_c] = np.nan
            # 0 很可能是旧版兜底值，不是真实满意度；优先用其他 sheet 的非空值回填。
            mask = (pd.to_numeric(out[base_c], errors="coerce").fillna(0) <= 1e-12) & out[fill_c].notna()
            out.loc[mask, base_c] = out.loc[mask, fill_c]
        used = int(out[["补齐距离满意度", "补齐响应满意度", "补齐问题2综合满意度"]].notna().any(axis=1).sum())
        if used:
            print(f"INFO: 已从问题2其他sheet补齐 {used} 个小区的满意度信息。")
        out = out.drop(columns=[c for c in ["补齐距离满意度", "补齐响应满意度", "补齐问题2综合满意度", "来源sheet"] if c in out.columns])

    # 如果仍然没有距离/响应分项，但有问题2综合满意度，则按“距离+响应合成分”兜底：
    # 综合满意度 = 0.5*价格满意度 + 0.5*问题2服务可达满意度。
    # 等价实现为把距离满意度和响应满意度都设为问题2综合满意度。
    for c in ["距离满意度", "响应满意度", "问题2综合满意度"]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    has_p2s = out["问题2综合满意度"].notna() & (out["问题2综合满意度"] > 1e-12)
    for c in ["距离满意度", "响应满意度"]:
        mask = (out[c].fillna(0) <= 1e-12) & has_p2s
        out.loc[mask, c] = out.loc[mask, "问题2综合满意度"]
    out["距离满意度"] = out["距离满意度"].fillna(0.0).clip(lower=0, upper=1)
    out["响应满意度"] = out["响应满意度"].fillna(0.0).clip(lower=0, upper=1)
    if (out["距离满意度"].mean() <= 1e-12 and out["响应满意度"].mean() <= 1e-12):
        print("WARNING: 当前问题2结果未能提供距离/响应满意度，问题3可及性会退化为价格满意度贡献，图中约0.5不宜作为最终结果。建议在问题2输出距离满意度和响应满意度。")
    return out

def choose_best_allocation_from_all_sheets(
    wb2: dict[str, pd.DataFrame],
    stations_raw: pd.DataFrame,
    uncovered_communities: set[str] | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """从问题2所有可能的小区表中选择最可靠的小区-服务站分配。

    v8 修正点：不要固定使用名称上最像的 06_小区需求满足汇总。
    对每个包含“小区/社区”的 sheet 都尝试解析，按
    1) 覆盖小区数；2) 实际分配到的最优站点数；3) sheet 名是否含“分配”
    进行评分，自动选择最优表。
    """
    uncovered_communities = uncovered_communities or set()
    candidates: list[tuple[tuple[int, int, int, int], str, pd.DataFrame, pd.DataFrame]] = []
    selected_count = len(stations_raw) if stations_raw is not None else 0

    for name, df in wb2.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols_txt = "|".join(clean_text(c) for c in df.columns)
        name_txt = clean_text(name)
        if "小区" not in (name_txt + "|" + cols_txt) and "社区" not in (name_txt + "|" + cols_txt):
            continue
        try:
            a = load_allocation(df, stations_raw=stations_raw, wb2=wb2)
            if uncovered_communities:
                a = a[~a["小区"].map(clean_text).isin(uncovered_communities)].copy().reset_index(drop=True)
            used = a["服务站"].nunique() if not a.empty else 0
            n_comm = len(a)
            # sheet 名奖励：真正的分配明细优先；“需求满足汇总”只作为兜底。
            name_bonus = 0
            if "分配" in name_txt or "指派" in name_txt or "对应" in name_txt:
                name_bonus += 3
            if "需求满足" in name_txt or "满足汇总" in name_txt:
                name_bonus -= 1
            # used 不强行等于 selected_count，但越接近越好；超过选址数一定扣分。
            station_score = used if selected_count <= 0 else -abs(selected_count - used)
            score = (n_comm, station_score, name_bonus, used)
            candidates.append((score, name, df, a))
        except Exception as e:
            print(f"INFO: 跳过候选分配sheet {name}: {e}")
            continue

    if not candidates:
        raise ValueError("问题2结果中没有可解析的小区-服务站分配表。")

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_name, best_df, best_alloc = candidates[0]
    print("\n===== 问题2分配表候选评分 =====")
    for score, name, _df, a in candidates[:8]:
        print(f"候选sheet: {name} | 解析覆盖小区={len(a)} | 实际分配站点={a['服务站'].nunique() if not a.empty else 0} | score={score}")
    print(f"INFO: 最终采用分配sheet: {best_name}")
    best_alloc = enrich_allocation_satisfaction(best_alloc, wb2)
    return best_name, best_df, best_alloc


def split_community_values(x: Any) -> list[str]:
    """把“C、D”或“C;D”等单元格拆成小区名列表。"""
    s = clean_text(x)
    if not s or s.lower() in INVALID_STATION_TEXT:
        return []
    parts = re.split(r"[,，;；、/\\]+", s)
    return [clean_text(p) for p in parts if clean_text(p)]


def extract_uncovered_communities(df_cov: pd.DataFrame | None) -> set[str]:
    """从问题2覆盖统计/覆盖情况 sheet 中提取未覆盖小区。

    兼容三类常见写法：
    1) 列名为“未覆盖小区/未被覆盖小区”，单元格中直接列出小区；
    2) 存在“小区 + 是否覆盖/覆盖状态”列，值为否/未覆盖/0；
    3) 任意文本中包含“未覆盖小区: C、D”。
    """
    if df_cov is None or df_cov.empty:
        return set()
    df = normalize_columns(df_cov)
    out: set[str] = set()
    # 1) 专门的未覆盖小区列
    for c in df.columns:
        cn = clean_text(c)
        if "未覆盖" in cn or "未被覆盖" in cn:
            for v in df[c].dropna().tolist():
                for item in split_community_values(v):
                    if item and not any(tok in item for tok in ["未覆盖", "小区", "社区", "数量", "个"]):
                        out.add(item)
    # 2) 小区 + 覆盖状态列
    try:
        col_comm = pick_col_any(df, [["小区"], ["社区"]], required=False)
        col_cover = pick_col_any(df, [["是否覆盖"], ["覆盖状态"], ["覆盖情况"]], required=False)
        if col_comm and col_cover:
            for _, r in df[[col_comm, col_cover]].iterrows():
                comm = clean_text(r[col_comm])
                if comm and not is_covered_value(r[col_cover]):
                    out.add(comm)
    except Exception:
        pass
    # 3) 文本兜底：识别“未覆盖小区:C、D”后面的内容
    txt = "|".join(clean_text(v) for v in df.astype(str).values.ravel())
    for m in re.finditer(r"未覆盖(?:小区|社区)?[:：]?([^|]+)", txt):
        for item in split_community_values(m.group(1)):
            if item and not any(tok in item for tok in ["未覆盖", "小区", "社区", "数量", "个"]):
                out.add(item)
    return out

def harmonize_stations_with_allocation(stations: pd.DataFrame, alloc: pd.DataFrame, cost_ref: pd.DataFrame) -> pd.DataFrame:
    """
    以问题2“小区分配表”中实际被分配的站点为准，修正“最优选址表”和“分配表”站点名称不一致的问题。
    这是本版专门为避免“没有可优化的服务站”而增加的容错层。
    """
    stations = stations.copy()
    if "站点键" not in stations.columns:
        stations["站点键"] = stations["服务站"].map(station_key)
    if "站点键" not in alloc.columns:
        alloc = alloc.copy()
        alloc["站点键"] = alloc["服务站"].map(station_key)

    used = alloc[["服务站", "站点键"]].drop_duplicates()
    used = used[(used["服务站"].map(clean_text) != "") & (used["站点键"].map(clean_text) != "")]

    if used.empty:
        return stations

    out_rows = []
    station_by_key = {clean_text(r["站点键"]): r for _, r in stations.iterrows() if clean_text(r.get("站点键", ""))}
    default_scale = canonical_scale(stations["规模"].dropna().iloc[0]) if not stations.empty and stations["规模"].notna().any() else "中型"

    for _, u in used.iterrows():
        alloc_name = clean_text(u["服务站"])
        key = clean_text(u["站点键"])
        if key in station_by_key:
            row = station_by_key[key].copy()
            # 用分配表里的站点名称作为主名称，保证后续能和需求合并表匹配。
            row["原问题2选址表服务站"] = clean_text(row.get("服务站", ""))
            row["服务站"] = alloc_name
            row["站点键"] = key
        else:
            # 如果最优选址表没有匹配到该站点，仍按分配表实际站点继续计算，规模默认取已有站点规模；成本从附件3回填。
            row = pd.Series({
                "服务站": alloc_name,
                "原问题2选址表服务站": "未匹配，来自小区分配表",
                "规模": default_scale,
                "建设成本": np.nan,
                "日固定管理成本": np.nan,
                "日服务能力": np.nan,
                "站点键": key,
            })

        if cost_ref is not None and not cost_ref.empty:
            sc = canonical_scale(row.get("规模", default_scale))
            cr = cost_ref[cost_ref["规模"].map(canonical_scale) == sc]
            if not cr.empty:
                for c in ["建设成本", "日固定管理成本", "日服务能力"]:
                    if pd.isna(row.get(c, np.nan)):
                        row[c] = cr.iloc[0].get(c, row.get(c, np.nan))
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    for c in ["建设成本", "日固定管理成本", "日服务能力"]:
        if c not in out.columns:
            out[c] = np.nan
    if out["建设成本"].notna().any() and out["建设成本"].max() < 10000:
        out["建设成本"] = out["建设成本"] * 10000

    missing_cols = [c for c in ["建设成本", "日固定管理成本"] if out[c].isna().any()]
    if missing_cols:
        print("WARNING: 部分服务站成本字段缺失，将用同规模/中型默认值兜底：", missing_cols)
        for idx, row in out.iterrows():
            sc = canonical_scale(row.get("规模", default_scale))
            cr = cost_ref[cost_ref["规模"].map(canonical_scale) == sc] if cost_ref is not None and not cost_ref.empty else pd.DataFrame()
            if cr.empty and cost_ref is not None and not cost_ref.empty:
                cr = cost_ref[cost_ref["规模"].map(canonical_scale) == "中型"]
            if not cr.empty:
                for c in ["建设成本", "日固定管理成本", "日服务能力"]:
                    if pd.isna(out.at[idx, c]):
                        out.at[idx, c] = cr.iloc[0].get(c, out.at[idx, c])
        still_missing = [c for c in ["建设成本", "日固定管理成本"] if out[c].isna().any()]
        if still_missing:
            raise KeyError(f"服务站成本字段仍缺失: {still_missing}。请检查问题2最优选址表或附件3。\n{out}")

    return out.drop_duplicates(subset=["站点键"], keep="first").reset_index(drop=True)


# =========================
# 优化模型
# =========================
def station_operating_cost(station_row: pd.Series) -> tuple[float, float, float]:
    fixed = parse_number(station_row["日固定管理成本"], 0.0) * DAYS_PER_YEAR
    depr = parse_number(station_row["建设成本"], 0.0) / DEPR_YEARS_DEFAULT
    op_cost = fixed + depr
    return fixed, depr, op_cost


def evaluate_station_policy(
    sdem: pd.DataFrame,
    station_row: pd.Series,
    prices: dict[str, float],
    pmap: dict[str, dict[str, float]],
    allow_subsidy: bool = True,
    target_profit_rate: float = PROFIT_RATE_TARGET,
) -> dict[str, Any]:
    """评价某服务站在一组价格下的财务与满意度结果。

    allow_subsidy=True：问题3优化情景，政府补贴用于达到“保本微利”目标，但不超过补贴上限。
    allow_subsidy=False：无补贴基准情景，用于和优化方案比较可及性。
    """
    scale = canonical_scale(station_row["规模"])
    tmp = sdem.copy()
    tmp["价格"] = tmp["服务项目"].map(lambda x: float(prices.get(x, 0.0 if x == EMERGENCY else pmap[x]["base"])))
    tmp["基准价格"] = tmp["服务项目"].map(lambda x: float(pmap.get(x, {}).get("base", 0.0)))
    tmp["直接支出"] = tmp["服务项目"].map(lambda x: float(pmap.get(x, {}).get("cost", 0.0)))
    tmp["价格满意度"] = [s_price(p, b) for p, b in zip(tmp["价格"], tmp["基准价格"])]
    tmp["距离满意度"] = pd.to_numeric(tmp["距离满意度"], errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    tmp["响应满意度"] = pd.to_numeric(tmp["响应满意度"], errors="coerce").fillna(0.0).clip(lower=0, upper=1)
    tmp["实际需求"] = pd.to_numeric(tmp["实际需求"], errors="coerce").fillna(0.0).clip(lower=0)
    tmp["综合满意度"] = (0.2 * tmp["距离满意度"] + 0.3 * tmp["响应满意度"] + 0.5 * tmp["价格满意度"]).clip(lower=0, upper=1)
    tmp["有效服务人次"] = tmp["实际需求"] * tmp["综合满意度"]  # 月有效服务人次

    revenue = float((tmp["价格"] * tmp["有效服务人次"]).sum()) * MONTHS_PER_YEAR
    direct_cost = float((tmp["直接支出"] * tmp["有效服务人次"]).sum()) * MONTHS_PER_YEAR
    fixed, depr, op_cost = station_operating_cost(station_row)

    non_emg_monthly_eff = float(tmp[tmp["服务项目"] != EMERGENCY]["有效服务人次"].sum())
    subsidy_unclipped_daily = non_emg_monthly_eff / DAYS_PER_MONTH * SUBSIDY_PER_VISIT_CAP
    subsidy_daily_cap = SUBSIDY_DAILY_CAP.get(scale, max(SUBSIDY_DAILY_CAP.values()))
    subsidy_cap = min(subsidy_unclipped_daily, subsidy_daily_cap) * DAYS_PER_YEAR

    profit_before_subsidy = revenue - direct_cost - op_cost
    target_profit = max(0.0, target_profit_rate) * op_cost
    # 优化情景：补贴补到“保本微利目标”，但不超过人次与规模共同限定的上限。
    # 基准情景：不允许补贴，靠价格本身实现财务平衡。
    required_subsidy = max(0.0, target_profit - profit_before_subsidy) if allow_subsidy else 0.0
    subsidy = min(required_subsidy, subsidy_cap) if allow_subsidy else 0.0
    profit = profit_before_subsidy + subsidy
    profit_rate = profit / op_cost if op_cost > 0 else np.nan

    demand_total = float(tmp["实际需求"].sum())
    effective_total = float(tmp["有效服务人次"].sum())
    avg_s = weighted_mean(tmp["综合满意度"], tmp["实际需求"])
    price_s = weighted_mean(tmp["价格满意度"], tmp["实际需求"])

    type_acc_df = tmp.groupby("老人类型", as_index=False).agg(实际需求=("实际需求", "sum"), 有效服务人次=("有效服务人次", "sum"))
    type_acc_df["可及性"] = np.where(type_acc_df["实际需求"] > 0, type_acc_df["有效服务人次"] / type_acc_df["实际需求"], 0.0)
    min_type_acc = float(type_acc_df["可及性"].min()) if len(type_acc_df) else 0.0

    feasible = True
    reason = "可行"
    if allow_subsidy and required_subsidy > subsidy_cap + 1e-9:
        feasible = False
        reason = "补贴上限不足"
    elif profit_rate < PROFIT_RATE_LOW - 1e-9:
        feasible = False
        reason = "亏损"
    elif profit_rate > PROFIT_RATE_HIGH + 1e-9:
        feasible = False
        reason = "利润率超过8%"

    violation = 0.0
    if allow_subsidy and required_subsidy > subsidy_cap:
        violation += required_subsidy - subsidy_cap
    if not pd.isna(profit_rate) and profit_rate > PROFIT_RATE_HIGH:
        violation += (profit_rate - PROFIT_RATE_HIGH) * max(op_cost, 1.0)
    if profit < 0:
        violation += -profit

    return {
        "detail": tmp,
        "prices": prices,
        "revenue": revenue,
        "direct_cost": direct_cost,
        "fixed": fixed,
        "depr": depr,
        "op_cost": op_cost,
        "profit_before_subsidy": profit_before_subsidy,
        "target_profit": target_profit,
        "required_subsidy": required_subsidy,
        "subsidy_cap": subsidy_cap,
        "subsidy": subsidy,
        "profit": profit,
        "profit_rate": float(profit_rate) if not pd.isna(profit_rate) else np.nan,
        "avg_s": avg_s,
        "price_s": price_s,
        "demand_total": demand_total,
        "effective_total": effective_total,
        "accessibility": effective_total / demand_total if demand_total > 0 else 0.0,
        "min_type_accessibility": min_type_acc,
        "type_accessibility": type_acc_df,
        "feasible": feasible,
        "reason": reason,
        "violation": violation,
    }

def optimize_station(
    sdem: pd.DataFrame,
    station_row: pd.Series,
    pmap: dict[str, dict[str, float]],
    allow_subsidy: bool = True,
    target_profit_rate: float = PROFIT_RATE_TARGET,
) -> tuple[dict[str, Any], dict[str, Any]]:
    service_candidates = {k: build_candidate_prices(pmap[k]["base"], pmap[k]["cost"]) for k in CHARGE_SERVICES}
    enum_count = int(np.prod([len(v) for v in service_candidates.values()]))

    best: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None
    feasible_count = 0
    reason_count: dict[str, int] = {}

    for combo in product(*[service_candidates[k] for k in CHARGE_SERVICES]):
        prices = {k: float(combo[i]) for i, k in enumerate(CHARGE_SERVICES)}
        prices[EMERGENCY] = 0.0
        ev = evaluate_station_policy(sdem, station_row, prices, pmap, allow_subsidy=allow_subsidy, target_profit_rate=target_profit_rate)
        reason_count[ev["reason"]] = reason_count.get(ev["reason"], 0) + 1

        if ev["feasible"]:
            feasible_count += 1
            # 字典序目标：公平性 > 总满意度 > 价格满意度 > 少补贴 > 少支付 > 利润率接近微利目标
            ev["score"] = (
                ev["min_type_accessibility"],
                ev["avg_s"],
                ev["price_s"],
                -ev["subsidy"],
                -ev["revenue"],
                -abs(ev["profit_rate"] - target_profit_rate),
            )
            if best is None or ev["score"] > best["score"]:
                best = ev
        else:
            ev["fallback_score"] = (-ev["violation"], ev["min_type_accessibility"], ev["avg_s"], -ev["revenue"])
            if fallback is None or ev["fallback_score"] > fallback["fallback_score"]:
                fallback = ev

    if best is None:
        assert fallback is not None
        best = fallback

    stat = {
        "服务站": clean_text(station_row["服务站"]),
        "枚举组合数": enum_count,
        "可行组合数": feasible_count,
        "最终是否可行": bool(best["feasible"]),
        "最终不可行原因": "" if best["feasible"] else best["reason"],
        "补贴上限不足数量": reason_count.get("补贴上限不足", 0),
        "利润率超过8%数量": reason_count.get("利润率超过8%", 0),
        "亏损数量": reason_count.get("亏损", 0),
        "候选价格集合": str(service_candidates),
    }
    return best, stat


def station_row_from_eval(station_row: pd.Series, ev: dict[str, Any]) -> dict[str, Any]:
    pr = ev["prices"]
    return {
        "服务站": clean_text(station_row["服务站"]),
        "规模": canonical_scale(station_row["规模"]),
        "助餐价格": pr.get("助餐", np.nan),
        "日间照料价格": pr.get("日间照料", np.nan),
        "上门护理价格": pr.get("上门护理", np.nan),
        "康复理疗价格": pr.get("康复理疗", np.nan),
        "助浴价格": pr.get("助浴", np.nan),
        "紧急救助价格": 0.0,
        "最低老人类型可及性": ev["min_type_accessibility"],
        "站点可及性": ev["accessibility"],
        "站点价格满意度": ev["price_s"],
        "站点综合满意度": ev["avg_s"],
        "月实际需求总量": ev["demand_total"],
        "月有效服务人次": ev["effective_total"],
        "年服务收入": ev["revenue"],
        "年直接支出": ev["direct_cost"],
        "年政府补贴": ev["subsidy"],
        "补贴上限": ev["subsidy_cap"],
        "保本所需补贴": ev["required_subsidy"],
        "补贴前利润": ev["profit_before_subsidy"],
        "年固定管理成本": ev["fixed"],
        "年建设折旧": ev["depr"],
        "年运营成本总额": ev["op_cost"],
        "年利润": ev["profit"],
        "利润率": ev["profit_rate"],
        "是否满足保本微利": bool(ev["feasible"]),
        "诊断": ev["reason"],
    }


def summarize_accessibility(detail: pd.DataFrame, label: str) -> pd.DataFrame:
    g = detail.groupby("老人类型", as_index=False).agg(
        实际需求=("实际需求", "sum"),
        有效服务人次=("有效服务人次", "sum"),
        平均价格满意度=("价格满意度", "mean"),
        平均综合满意度=("综合满意度", "mean"),
        年支付金额=("年支付金额", "sum"),
    )
    g[f"{label}可及性"] = np.where(g["实际需求"] > 0, g["有效服务人次"] / g["实际需求"], 0.0)
    return g.rename(columns={
        "有效服务人次": f"{label}有效服务人次",
        "平均价格满意度": f"{label}平均价格满意度",
        "平均综合满意度": f"{label}平均综合满意度",
        "年支付金额": f"{label}年支付金额",
    })


def build_policy_detail(merged: pd.DataFrame, stations: pd.DataFrame, policy_by_station: dict[str, dict[str, float]], pmap: dict[str, dict[str, float]]) -> pd.DataFrame:
    frames = []
    if "站点键" not in merged.columns:
        merged = merged.copy()
        merged["站点键"] = merged["服务站"].map(station_key)
    if "站点键" not in stations.columns:
        stations = stations.copy()
        stations["站点键"] = stations["服务站"].map(station_key)
    for _, strow in stations.iterrows():
        st = clean_text(strow["服务站"])
        key = clean_text(strow.get("站点键", station_key(st)))
        sdem = merged[merged["站点键"].map(clean_text) == key].copy()
        if sdem.empty:
            continue
        ev = evaluate_station_policy(sdem, strow, policy_by_station[st], pmap)
        d = ev["detail"].copy()
        d["分配站点"] = st
        d["年支付金额"] = d["价格"] * d["有效服务人次"] * MONTHS_PER_YEAR
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# =========================
# 输出工具
# =========================
def safe_write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    out = path.resolve()
    try:
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            for n, d in sheets.items():
                df = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
                df.to_excel(w, index=False, sheet_name=n[:31])
        return out
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out.with_name(f"{out.stem}_{ts}{out.suffix}")
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            for n, d in sheets.items():
                df = d if isinstance(d, pd.DataFrame) else pd.DataFrame(d)
                df.to_excel(w, index=False, sheet_name=n[:31])
        return out


def display_station_name(x: Any, idx: int | None = None) -> str:
    s = clean_text(x)
    # 若问题2把坐标当作站点名，直接显示会变成很长的小数串；图中统一用顺序编号。
    if idx is not None:
        return f"站点{idx}"
    if s.startswith("站点") or s.startswith("服务站"):
        return s
    if len(s) > 12 or re.search(r"\d+\.\d+", s):
        return "站点"
    return f"站点{s}" if s else "未知站点"


def make_charts(charts_dir: Path, station_df: pd.DataFrame, comm_df: pd.DataFrame, access_df: pd.DataFrame) -> None:
    if not HAS_MPL:
        return
    charts_dir.mkdir(exist_ok=True)
    try:
        plot_df = station_df.copy().reset_index(drop=True)
        if "服务站显示" not in plot_df.columns:
            plot_df["服务站显示"] = [display_station_name(v, i + 1) for i, v in enumerate(plot_df["服务站"])]
        pcols = ["助餐价格", "日间照料价格", "上门护理价格", "康复理疗价格", "助浴价格"]
        ax = plot_df.set_index("服务站显示")[pcols].plot(kind="bar", figsize=(12, 5), title="各服务站最优服务价格")
        ax.set_ylabel("价格")
        plt.tight_layout(); plt.savefig(charts_dir / "1_各服务站最优服务价格柱状图.png", dpi=180); plt.close()

        ax = plot_df.plot(x="服务站显示", y="利润率", kind="bar", legend=False, figsize=(10, 4), title="各服务站利润率")
        ax.axhline(PROFIT_RATE_HIGH, linestyle="--", linewidth=1)
        ax.set_ylabel("利润率")
        ax.set_xlabel("服务站")
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{v:.3f}" for v in plot_df["利润率"].fillna(0)], fontsize=8, rotation=90, padding=2)
        plt.tight_layout(); plt.savefig(charts_dir / "2_各服务站利润率柱状图.png", dpi=180); plt.close()

        ax = plot_df.plot(x="服务站显示", y="年政府补贴", kind="bar", legend=False, figsize=(10, 4), title="各服务站年度政府补贴")
        ax.set_ylabel("元/年")
        ax.set_xlabel("服务站")
        plt.tight_layout(); plt.savefig(charts_dir / "3_各服务站政府补贴柱状图.png", dpi=180); plt.close()

        cplot = comm_df.sort_values("综合满意度", ascending=True).tail(30)
        ax = cplot.plot(x="小区", y="综合满意度", kind="bar", legend=False, figsize=(max(10, 0.32 * len(cplot)), 4), title="小区综合满意度（显示前30个）")
        ax.set_ylabel("综合满意度")
        plt.tight_layout(); plt.savefig(charts_dir / "4_小区综合满意度柱状图.png", dpi=180); plt.close()

        ycols = [c for c in ["基准可及性", "优化可及性"] if c in access_df.columns]
        if ycols:
            ax = access_df.plot(x="老人类型", y=ycols, kind="bar", figsize=(8, 4), title="不同老人类型可及性对比（基准=无补贴保本定价）")
            ax.set_ylabel("可及性")
            plt.tight_layout(); plt.savefig(charts_dir / "5_不同老人类型可及性对比图.png", dpi=180); plt.close()
    except Exception as e:
        print(f"WARNING: 图表生成失败，但结果表已输出。原因: {e}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="电工杯 B题问题3：基于最新 B_problem2_results.xlsx 的定价-补贴优化（站点列误识别修正版v7）")
    parser.add_argument("--root", default=".", help="项目根目录，默认当前目录")
    parser.add_argument("--problem1", default=None, help="B_problem1_results.xlsx 路径，可省略")
    parser.add_argument("--problem2", default=None, help="B_problem2_results.xlsx 路径；省略时自动选择修改时间最新的文件")
    parser.add_argument("--attachment2", default=None, help="附件2：服务需求数据.xlsx 路径，可省略")
    parser.add_argument("--attachment3", default=None, help="附件3：服务站建设与运营成本.xlsx 路径，可省略")
    parser.add_argument("--attachment5", default=None, help="附件5：满意度评分规则.xlsx 路径，可省略；本程序仅记录数据源")
    parser.add_argument("--output", default="B_problem3_results.xlsx", help="输出 Excel 文件名")
    parser.add_argument("--charts-dir", default="charts_problem3", help="输出图表目录")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    problem2_path = Path(args.problem2).resolve() if args.problem2 else find_latest_file(root, ["B_problem2_results.xlsx", "B_problem2_results*.xlsx"], label="最新 B_problem2_results.xlsx")
    problem1_path = Path(args.problem1).resolve() if args.problem1 else find_latest_file(root, ["B_problem1_results.xlsx", "B_problem1_results*.xlsx"], required=False, label="B_problem1_results.xlsx")
    attachment2_path = Path(args.attachment2).resolve() if args.attachment2 else find_latest_file(root, ["附件2*服务需求*.xlsx", "*服务需求数据*.xlsx"], label="附件2服务需求数据")
    attachment3_path = Path(args.attachment3).resolve() if args.attachment3 else find_latest_file(root, ["附件3*服务站建设*.xlsx", "*建设与运营成本*.xlsx", "附件3*.xlsx"], required=False, label="附件3建设与运营成本")
    attachment5_path = Path(args.attachment5).resolve() if args.attachment5 else find_latest_file(root, ["附件5*满意度*.xlsx", "*满意度评分规则*.xlsx", "附件5*.xlsx"], required=False, label="附件5满意度评分规则")

    print("\n===== 数据源选择 =====")
    print(f"问题2结果: {problem2_path}  | 修改时间: {datetime.fromtimestamp(problem2_path.stat().st_mtime)}")
    print(f"问题1结果: {problem1_path if problem1_path else '未找到，将尝试从附件2读取需求'}")
    print(f"附件2: {attachment2_path}")
    print(f"附件3: {attachment3_path if attachment3_path else '未找到，仅使用问题2中的成本字段'}")
    print(f"附件5: {attachment5_path if attachment5_path else '未找到'}")

    wb2 = read_workbook_sheets(problem2_path)
    (s_st, df_st), (s_alloc, df_alloc), (s_cov, df_cov), (s_sum, df_sum) = load_problem2_inputs(wb2)

    if problem1_path and problem1_path.exists():
        wb1 = read_workbook_sheets(problem1_path)
        s_demand, demand = extract_demand_from_sheets(wb1)
        demand_source = str(problem1_path)
    else:
        wb_a2 = read_workbook_sheets(attachment2_path)
        s_demand, demand = extract_demand_from_sheets(wb_a2)
        demand_source = str(attachment2_path)

    price_ref = load_price_ref_from_attachment2(attachment2_path)
    pmap = {r["服务项目"]: {"base": float(r["基准价格"]), "cost": float(r["直接支出"])} for _, r in price_ref.iterrows()}
    for srv in ALL_SERVICES:
        pmap.setdefault(srv, {"base": 0.0, "cost": 0.0})

    cost_ref = load_station_cost_ref_from_attachment3(attachment3_path)
    stations_raw = load_stations(df_st, cost_ref)

    uncovered_communities = extract_uncovered_communities(df_cov)
    if uncovered_communities:
        print(f"INFO: 问题2覆盖统计识别到未覆盖小区: {sorted(uncovered_communities)}")

    # v8：在所有可能的小区表里自动选择最可靠的小区-服务站分配，避免固定读 06_小区需求满足汇总导致 5 个选址站点只剩 4 个。
    s_alloc, df_alloc, alloc = choose_best_allocation_from_all_sheets(wb2, stations_raw, uncovered_communities)

    stations = harmonize_stations_with_allocation(stations_raw, alloc, cost_ref)

    print("\n===== 问题2解析检查 =====")
    print(f"选址规模sheet: {s_st} | 行数: {len(df_st)} | 解析服务站数: {len(stations_raw)}")
    actual_station_count = alloc['服务站'].nunique() if not alloc.empty else 0
    print(f"小区分配sheet: {s_alloc} | 行数: {len(df_alloc)} | 已分配小区数: {len(alloc)} | 实际分配站点数: {actual_station_count}")
    if not stations_raw.empty:
        print("选址表服务站示例:", stations_raw[["服务站", "规模"]].head(10).to_dict("records"))
        selected_keys_dbg = set(stations_raw["服务站"].map(station_key))
        used_keys_dbg = set(alloc["站点键"].map(clean_text)) if not alloc.empty and "站点键" in alloc.columns else set()
        missing_keys_dbg = sorted(selected_keys_dbg - used_keys_dbg)
        if missing_keys_dbg:
            print("WARNING: 以下最优选址站点没有在当前小区分配表中分到有效覆盖小区:", missing_keys_dbg)
    if not alloc.empty:
        print("分配表站点示例:", alloc[["小区", "服务站", "站点键"]].head(10).to_dict("records"))

    merged = demand.merge(alloc[["小区", "服务站", "站点键", "距离", "距离满意度", "响应满意度", "问题2综合满意度", "是否覆盖"]], on="小区", how="left")
    # 不能再用“服务站非空”兜底覆盖，否则 C 这类未覆盖小区若有最近站点字段，会被误当作已覆盖。
    cover_mask = merged["是否覆盖"].map(is_covered_value) & merged["服务站"].map(is_valid_station_name)
    merged = merged[cover_mask].copy()
    if merged.empty:
        raise ValueError("需求表与最新问题2分配结果合并后为空：请检查小区名称是否一致、是否覆盖列是否正确。")
    merged["服务项目"] = merged["服务项目"].map(canonical_service)
    merged = merged[merged["服务项目"].isin(ALL_SERVICES)].copy()
    merged["站点键"] = merged["服务站"].map(station_key)

    # 只保留最新问题2实际选出的站点；用站点键匹配，避免“服务站3/站点03/候选点3”等写法不一致。
    station_keys = set(stations["站点键"].map(clean_text)) if "站点键" in stations.columns else set(stations["服务站"].map(station_key))
    before_match_rows = len(merged)
    matched = merged[merged["站点键"].map(clean_text).isin(station_keys)].copy()
    if matched.empty:
        print("WARNING: 选址表站点和分配表站点没有匹配上，已改为以分配表中的实际站点继续计算。")
        print("选址表站点键:", sorted(station_keys)[:20])
        print("分配表站点键:", sorted(set(merged["站点键"].map(clean_text)))[:20])
    else:
        merged = matched

    # 只保留确实有有效需求的服务站，避免 0/未覆盖/无需求站点进入问题3图表。
    demand_by_key = merged.groupby("站点键")["实际需求"].sum()
    positive_demand_keys = set(demand_by_key[demand_by_key > 1e-9].index.map(clean_text))
    dropped_station_keys = sorted(set(stations["站点键"].map(clean_text)) - positive_demand_keys)
    if dropped_station_keys:
        print("INFO: 以下站点没有匹配到有效需求，已从问题3优化图表中剔除：", dropped_station_keys)
    stations = stations[stations["站点键"].map(clean_text).isin(positive_demand_keys)].copy().reset_index(drop=True)
    merged = merged[merged["站点键"].map(clean_text).isin(positive_demand_keys)].copy().reset_index(drop=True)

    print(f"需求-分配合并后记录数: {before_match_rows}，站点匹配后记录数: {len(merged)}，实际优化站点数: {len(stations)}")
    if stations.empty or merged.empty:
        raise ValueError("没有可优化的服务站：有效需求没有匹配到问题2分配站点。请检查问题2分配表中站点是否用0表示未覆盖。")

    station_rows = []
    infeasible_rows = []
    best_policy_by_station: dict[str, dict[str, float]] = {}
    detail_frames = []
    type_access_frames = []

    total_stations = len(stations)
    for si, (_, strow) in enumerate(stations.iterrows(), start=1):
        st = clean_text(strow["服务站"])
        key = clean_text(strow.get("站点键", station_key(st)))
        sdem = merged[merged["站点键"].map(clean_text) == key].copy()
        if sdem.empty:
            infeasible_rows.append({"服务站": st, "枚举组合数": 0, "可行组合数": 0, "最终是否可行": False, "最终不可行原因": "无分配需求"})
            continue
        print(f"正在优化服务站 {si}/{total_stations}: {st}，需求记录数={len(sdem)}")
        best, stat = optimize_station(sdem, strow, pmap)
        print(f"  完成：枚举组合数={stat['枚举组合数']}，可行组合数={stat['可行组合数']}，最终是否可行={stat['最终是否可行']}")
        best_policy_by_station[st] = best["prices"]
        d = best["detail"].copy()
        d["分配站点"] = st
        d["年支付金额"] = d["价格"] * d["有效服务人次"] * MONTHS_PER_YEAR
        detail_frames.append(d)
        station_rows.append(station_row_from_eval(strow, best))
        infeasible_rows.append(stat)
        ta = best["type_accessibility"].copy()
        ta.insert(0, "服务站", st)
        type_access_frames.append(ta)

    if not station_rows:
        raise ValueError("没有可优化的服务站：请检查问题2最优选址和小区分配表。")

    station_df = pd.DataFrame(station_rows).reset_index(drop=True)
    station_df.insert(1, "服务站显示", [display_station_name(v, i + 1) for i, v in enumerate(station_df["服务站"])])
    detail = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    station_type_access = pd.concat(type_access_frames, ignore_index=True) if type_access_frames else pd.DataFrame()

    # 基准情景改为“无补贴保本定价”：否则若直接用附件2基准价，价格满意度天然为1，
    # 优化前后可及性会完全一样，无法体现补贴与定价优化的作用。
    baseline_policy_by_station: dict[str, dict[str, float]] = {}
    baseline_eval_by_station: dict[str, dict[str, Any]] = {}
    for _, strow in stations.iterrows():
        st = clean_text(strow["服务站"])
        key = clean_text(strow.get("站点键", station_key(st)))
        sdem = merged[merged["站点键"].map(clean_text) == key].copy()
        if sdem.empty:
            continue
        base_best, _base_stat = optimize_station(sdem, strow, pmap, allow_subsidy=False, target_profit_rate=PROFIT_RATE_TARGET)
        baseline_policy_by_station[st] = base_best["prices"]
        baseline_eval_by_station[st] = base_best
    baseline_detail = build_policy_detail(merged, stations, baseline_policy_by_station, pmap)
    optimized_detail = detail.copy()

    base_acc = summarize_accessibility(baseline_detail, "基准") if not baseline_detail.empty else pd.DataFrame()
    opt_acc = summarize_accessibility(optimized_detail, "优化") if not optimized_detail.empty else pd.DataFrame()
    if not base_acc.empty and not opt_acc.empty:
        access = base_acc.merge(opt_acc, on=["老人类型", "实际需求"], how="outer")
        access["可及性变化"] = access["优化可及性"] - access["基准可及性"]
        access["年支付金额变化"] = access["优化年支付金额"] - access["基准年支付金额"]
        access["结论"] = np.where(access["可及性变化"] >= -1e-9, "优化后不低于基准", "优化后低于基准")
    else:
        access = pd.DataFrame()

    comm = optimized_detail.groupby(["小区", "分配站点"], as_index=False).agg(
        价格满意度=("价格满意度", "mean"),
        距离满意度=("距离满意度", "mean"),
        响应满意度=("响应满意度", "mean"),
        综合满意度=("综合满意度", "mean"),
        实际需求=("实际需求", "sum"),
        有效服务人次=("有效服务人次", "sum"),
        年支付金额=("年支付金额", "sum"),
    )
    comm["可及性"] = np.where(comm["实际需求"] > 0, comm["有效服务人次"] / comm["实际需求"], 0.0)

    baseline_station_rows = []
    for _, strow in stations.iterrows():
        st = clean_text(strow["服务站"])
        key = clean_text(strow.get("站点键", station_key(st)))
        sdem = merged[merged["站点键"].map(clean_text) == key].copy()
        if sdem.empty:
            continue
        ev = baseline_eval_by_station.get(st) or evaluate_station_policy(sdem, strow, baseline_policy_by_station[st], pmap, allow_subsidy=False)
        baseline_station_rows.append({
            "服务站": st,
            "基准综合满意度": ev["avg_s"],
            "基准价格满意度": ev["price_s"],
            "基准年服务收入": ev["revenue"],
            "基准年政府补贴": ev["subsidy"],
            "基准年利润": ev["profit"],
            "基准利润率": ev["profit_rate"],
            "基准是否可行": ev["feasible"],
            "基准诊断": ev["reason"],
        })
    baseline_station = pd.DataFrame(baseline_station_rows)
    compare_station = station_df.merge(baseline_station, on="服务站", how="left")
    compare_station["满意度变化"] = compare_station["站点综合满意度"] - compare_station["基准综合满意度"]
    compare_station["政府补贴变化"] = compare_station["年政府补贴"] - compare_station["基准年政府补贴"]
    compare_station["利润率变化"] = compare_station["利润率"] - compare_station["基准利润率"]

    p2_candidates = candidate_files(root, ["B_problem2_results.xlsx", "B_problem2_results*.xlsx"])
    data_check = pd.DataFrame([
        {
            "项目": "问题2结果文件",
            "实际使用路径": str(problem2_path),
            "修改时间": datetime.fromtimestamp(problem2_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "全部候选": " | ".join(str(p) for p in p2_candidates[:8]),
            "说明": "自动选择修改时间最新的 B_problem2_results*.xlsx，避免读取旧结果。",
        },
        {"项目": "问题1需求文件", "实际使用路径": demand_source, "修改时间": "", "全部候选": "", "说明": f"使用sheet={s_demand}"},
        {"项目": "附件2", "实际使用路径": str(attachment2_path), "修改时间": "", "全部候选": "", "说明": "读取服务基准价格和直接支出"},
        {"项目": "附件3", "实际使用路径": str(attachment3_path) if attachment3_path else "未使用", "修改时间": "", "全部候选": "", "说明": "当问题2结果缺少成本列时，用附件3按规模回填"},
        {"项目": "附件5", "实际使用路径": str(attachment5_path) if attachment5_path else "未使用", "修改时间": "", "全部候选": "", "说明": "满意度公式在代码中固定为0.2距离+0.3响应+0.5价格"},
        {"项目": "问题2未覆盖小区", "实际使用路径": str(problem2_path), "修改时间": "", "全部候选": ",".join(sorted(uncovered_communities)) if uncovered_communities else "", "说明": "这些小区不会进入问题3满意度图和优化明细"},
    ])

    candidate_price_df = pd.DataFrame([
        {
            "服务项目": k,
            "直接支出": pmap[k]["cost"],
            "基准价格": pmap[k]["base"],
            "候选价格列表": str(build_candidate_prices(pmap[k]["base"], pmap[k]["cost"])),
        }
        for k in CHARGE_SERVICES
    ])

    summary = pd.DataFrame([{
        "服务站数量": len(station_df),
        "覆盖小区数量": optimized_detail["小区"].nunique(),
        "总实际需求(月人次)": optimized_detail["实际需求"].sum(),
        "总有效服务人次(月)": optimized_detail["有效服务人次"].sum(),
        "总体可及性": optimized_detail["有效服务人次"].sum() / optimized_detail["实际需求"].sum() if optimized_detail["实际需求"].sum() > 0 else 0.0,
        "平均价格满意度": weighted_mean(optimized_detail["价格满意度"], optimized_detail["实际需求"]),
        "平均综合满意度": weighted_mean(optimized_detail["综合满意度"], optimized_detail["实际需求"]),
        "总政府补贴": station_df["年政府补贴"].sum(),
        "总服务收入": station_df["年服务收入"].sum(),
        "总直接支出": station_df["年直接支出"].sum(),
        "总运营成本": station_df["年运营成本总额"].sum(),
        "总利润": station_df["年利润"].sum(),
        "平均利润率": station_df["利润率"].mean(),
        "是否全部满足保本微利": bool(station_df["是否满足保本微利"].all()),
        "折旧年限(年)": DEPR_YEARS_DEFAULT,
        "算法说明": "以最新问题2选址和小区分配为约束，对各站点服务价格和达到3%微利目标的最低必要政府补贴联合优化；对比基准为无补贴保本定价方案。",
    }])

    model_note = pd.DataFrame([
        {"条目": "输入修正", "说明": "自动搜索并选择修改时间最新的 B_problem2_results*.xlsx；也可用 --problem2 显式指定。"},
        {"条目": "决策变量", "说明": "各服务站五类收费服务价格；紧急救助固定公益免费；政府补贴取达到3%保本微利目标所需的最低值，但不超过按人次和站点规模确定的上限。"},
        {"条目": "约束", "说明": "价格不低于直接支出；站点利润率在[0, 8%]；补贴不超过 min(有效服务人次/30*2元, 站点规模日补贴上限)*365。"},
        {"条目": "目标函数", "说明": "字典序：最大化最低老人类型可及性，其次最大化加权综合满意度和价格满意度，再最小化政府补贴、老人支付，并使利润率靠近3%微利目标；8%仍是上限。"},
        {"条目": "数据清洗", "说明": "分配表中服务站为0/无/未覆盖/未分配的记录会被剔除；覆盖统计sheet列出的未覆盖小区也会剔除；没有匹配到有效需求的站点不参与问题3图表。"},
        {"条目": "满意度", "说明": "综合满意度 = 0.2*距离满意度 + 0.3*响应满意度 + 0.5*价格满意度；价格满意度按基准价、1.1倍、1.2倍分段。"},
    ])

    outputs = {
        "00_模型说明": model_note,
        "01_数据源检查": data_check,
        "02_问题2最优站点输入": stations,
        "03_问题2小区分配输入": alloc,
        "04_服务需求输入": demand,
        "05_服务价格成本": price_ref,
        "06_候选价格集合": candidate_price_df,
        "07_最优定价补贴方案": station_df[[
            "服务站显示", "服务站", "规模", "助餐价格", "日间照料价格", "上门护理价格", "康复理疗价格", "助浴价格", "紧急救助价格",
            "最低老人类型可及性", "站点可及性", "站点价格满意度", "站点综合满意度", "是否满足保本微利", "诊断",
        ]],
        "08_服务站财务结果": station_df[[
            "服务站显示", "服务站", "年服务收入", "年直接支出", "年政府补贴", "补贴上限", "保本所需补贴", "补贴前利润",
            "月实际需求总量", "月有效服务人次", "年固定管理成本", "年建设折旧", "年运营成本总额", "年利润", "利润率", "是否满足保本微利", "诊断",
        ]],
        "09_小区满意度与可及性": comm,
        "10_老人类型可及性对比": access,
        "11_服务站老人类型可及性": station_type_access,
        "12_基准价格方案对比": compare_station,
        "13_明细结果": optimized_detail[[
            "小区", "分配站点", "老人类型", "服务项目", "实际需求", "价格", "基准价格", "直接支出",
            "距离满意度", "响应满意度", "价格满意度", "综合满意度", "有效服务人次", "年支付金额",
        ]],
        "14_不可行组合统计": pd.DataFrame(infeasible_rows),
        "15_总体指标": summary,
    }

    out_path = safe_write_excel(root / args.output, outputs)
    make_charts(root / args.charts_dir, station_df, comm, access)

    print("\n===== 问题3优化摘要 =====")
    print(station_df[["服务站", "规模", "站点综合满意度", "最低老人类型可及性", "年政府补贴", "利润率", "是否满足保本微利"]].to_string(index=False))
    print(f"\n结果文件: {out_path}")
    print(f"图表目录: {(root / args.charts_dir).resolve()}")


if __name__ == "__main__":
    main()
