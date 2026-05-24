from __future__ import annotations

import os
import shutil
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import problem1
import problem2
import B_problem3


# ============================================================
# 电工杯B题 问题4：灵敏度分析与方案比较
# ------------------------------------------------------------
# 情景设置按题目黑体分类，不再拆成 5 个单独小情景：
# S0：基准方案，全部使用附件默认参数，预算 120 万元；
# S1：老人增长与状态转移参数变化：年增长率=8%，p12=5.5%，p23=9.5%，
#     其余成本、预算均回到附件默认值；
# S2：成本变化：仅日固定管理成本增加 20%，老人增长/转移概率和预算均回到默认值；
# S3：预算调整：仅建设预算调整为 140 万元，老人增长/转移概率和成本均回到默认值。
#
# 注意：每个情景只改变当前情景对应参数，其他参数不继承前一个情景。
#      S3 使用 140 万元并启用全覆盖硬约束；若 140 万元无法全覆盖，程序直接报错，
#      不会私自搜索或采用超过 140 万元的预算。
#      问题3阶段保留问题2的分流比例：小区需求 × 分流比例 = 服务站实际承担需求，
#      再按服务站-老人类型-服务项目汇总后定价和补贴优化。
# ============================================================

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "problem4_outputs"
CHART_ROOT = ROOT / "charts_problem4"

ORIGINAL_SPLIT_DEMAND_SOLVER = problem2.solve_split_demand_milp


@dataclass
class Scenario:
    code: str
    name: str
    out_dir: str
    new_rate: float = 0.07
    p12: float | None = None
    p23: float | None = None
    fixed_cost_mult: float = 1.0
    budget: int = 1_200_000
    full_cover: bool = False


SCENARIOS = [
    Scenario("S0", "baseline", "S0_baseline"),
    # 情景1：老人增长与状态转移参数同时按题目调整；成本和预算保持默认。
    Scenario("S1", "elderly_growth_transition", "S1_elderly_growth_transition", new_rate=0.08, p12=0.055, p23=0.095),
    # 情景2：仅日固定管理成本增加20%；人口参数和预算保持默认。
    Scenario("S2", "fixed_cost_plus20", "S2_fixed_cost_plus20", fixed_cost_mult=1.2),
    # 情景3：仅建设预算调整为140万元；人口参数和成本保持默认，并要求全覆盖。
    Scenario("S3", "budget_140_full_cover", "S3_budget_140", budget=1_400_000, full_cover=True),
]


# ============================================================
# 运行时替换 problem2 的 MILP：加入全覆盖硬约束
# ============================================================

def _make_sparse_constraint(coeff: np.ndarray, lb: float = -np.inf, ub: float = np.inf):
    _, LinearConstraint, _, _, csr_matrix = problem2.require_scipy_milp()
    return LinearConstraint(csr_matrix(coeff.reshape(1, -1)), np.array([lb], dtype=float), np.array([ub], dtype=float))


def solve_split_demand_milp_fullcover(
    communities: list[str],
    P: dict[str, float],
    Qd: dict[str, float],
    dmat: pd.DataFrame,
    params: dict[str, dict[str, float]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """替换 problem2.solve_split_demand_milp 的全覆盖版。

    关键硬约束：
        对每个小区 i，sum_{j,s} x[i,j,s] == Qd[i]
    因此只要 MILP 求解成功，结果就不允许出现部分覆盖和未覆盖。
    """
    milp, LinearConstraint, Bounds, lil_matrix, _ = problem2.require_scipy_milp()

    scale_names = [s for s in ["小型", "中型", "大型"] if s in params]
    if not scale_names:
        raise ValueError("没有识别到小型/中型/大型服务站参数，请检查附件3。")

    feasible_pairs = [
        (i, j)
        for i in communities
        for j in communities
        if float(dmat.loc[i, j]) <= problem2.SERVICE_RADIUS_M and float(Qd.get(i, 0.0)) > 0
    ]
    if not feasible_pairs:
        raise RuntimeError("没有任何小区-服务站候选对满足服务半径约束。")

    # 每个有需求的小区至少要有一个半径内候选站，否则全覆盖从结构上不可行。
    no_candidate = []
    for i in communities:
        if float(Qd.get(i, 0.0)) <= 0:
            continue
        if not any(ii == i for ii, _ in feasible_pairs):
            no_candidate.append(i)
    if no_candidate:
        raise RuntimeError(f"以下小区在服务半径内没有候选站点，无法全覆盖：{no_candidate}")

    v_idx: dict[tuple[str, str, int], int] = {}
    x_idx: dict[tuple[str, str, str, int], int] = {}
    idx = 0

    for j in communities:
        for s in scale_names:
            v_idx[(j, s, 0)] = idx
            idx += 1

    for i, j in feasible_pairs:
        for s in scale_names:
            x_idx[(i, j, s, 0)] = idx
            idx += 1

    umax_idx = idx
    idx += 1
    n_var = idx

    lb = np.zeros(n_var, dtype=float)
    ub = np.full(n_var, np.inf, dtype=float)
    integrality = np.zeros(n_var, dtype=int)

    for k in v_idx.values():
        ub[k] = 1.0
        integrality[k] = 1
    ub[umax_idx] = 1.0

    rows: list[dict[int, float]] = []
    lows: list[float] = []
    ups: list[float] = []

    def add_row(coefs: dict[int, float], low: float = -np.inf, up: float = np.inf) -> None:
        rows.append(coefs)
        lows.append(low)
        ups.append(up)

    # 每个候选小区最多建一个服务站。
    for j in communities:
        add_row({v_idx[(j, s, 0)]: 1.0 for s in scale_names}, -np.inf, 1.0)

    # 总建设预算。
    budget_coefs = {}
    for (j, s, _), k in v_idx.items():
        budget_coefs[k] = float(params[s]["build"])
    add_row(budget_coefs, -np.inf, float(problem2.BUDGET_YUAN))

    # 全覆盖硬约束：每个小区的日均需求必须全部分配完。
    for i in communities:
        coefs = {}
        for (ii, j, s, _), k in x_idx.items():
            if ii == i:
                coefs[k] = 1.0
        qi = float(Qd.get(i, 0.0))
        add_row(coefs, qi, qi)

    # 站点容量和最大利用率。
    for j in communities:
        for s in scale_names:
            cap = float(params[s]["cap"])
            if cap <= 0:
                raise ValueError(f"服务站规模 {s} 的日服务能力无效：{cap}")
            y = v_idx[(j, s, 0)]
            load_terms = {}
            for (i2, j2, s2, _), k in x_idx.items():
                if j2 == j and s2 == s:
                    load_terms[k] = 1.0

            # sum_i x[i,j,s] <= cap_s * y[j,s]
            cap_row = dict(load_terms)
            cap_row[y] = cap_row.get(y, 0.0) - cap
            add_row(cap_row, -np.inf, 0.0)

            # sum_i x[i,j,s] / cap_s <= Umax
            umax_row = {k: val / cap for k, val in load_terms.items()}
            umax_row[umax_idx] = umax_row.get(umax_idx, 0.0) - 1.0
            add_row(umax_row, -np.inf, 0.0)

    A = lil_matrix((len(rows), n_var), dtype=float)
    for r, coefs in enumerate(rows):
        for c, val in coefs.items():
            A[r, c] = val
    base_constraint = LinearConstraint(A.tocsr(), np.array(lows, dtype=float), np.array(ups, dtype=float))

    coverage_coeff = np.zeros(n_var, dtype=float)
    sat_coeff = np.zeros(n_var, dtype=float)
    demand_coeff = np.zeros(n_var, dtype=float)
    build_coeff = np.zeros(n_var, dtype=float)
    capacity_coeff = np.zeros(n_var, dtype=float)
    umax_coeff = np.zeros(n_var, dtype=float)
    umax_coeff[umax_idx] = 1.0

    for (j, s, _), k in v_idx.items():
        build_coeff[k] = float(params[s]["build"])
        capacity_coeff[k] = float(params[s]["cap"])

    for (i, j, s, _), k in x_idx.items():
        q = float(Qd[i])
        pop_per_daily = float(P[i]) / q if q > 0 else 0.0
        s1 = problem2.distance_satisfaction(float(dmat.loc[i, j]))
        s3 = 1.0
        sij_proxy = weights["distance"] * s1 + weights["price"] * s3
        coverage_coeff[k] = pop_per_daily
        sat_coeff[k] = pop_per_daily * sij_proxy
        demand_coeff[k] = 30.0

    bounds = Bounds(lb, ub)

    def run_stage(coeff: np.ndarray, maximize: bool, extra_constraints: list[Any], stage_name: str):
        c = -coeff if maximize else coeff.copy()
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=[base_constraint] + extra_constraints,
            options={
                "time_limit": getattr(problem2, "MILP_TIME_LIMIT_SEC", 600),
                "mip_rel_gap": getattr(problem2, "MILP_REL_GAP", 1e-6),
                "disp": False,
            },
        )
        if not res.success:
            raise RuntimeError(f"FULL_COVER MILP阶段「{stage_name}」求解失败：status={res.status}, message={res.message}")
        val = float(coeff @ res.x)
        print(f"INFO: FULL_COVER MILP阶段「{stage_name}」完成，目标值={val:.6g}")
        return res, val

    constraints_extra: list[Any] = []

    # 覆盖量在全覆盖约束下已经固定，因此优化顺序改为：满意度 -> 建设成本 -> 总能力 -> 最大利用率。
    res1, opt_sat = run_stage(sat_coeff, True, constraints_extra, "1-全覆盖下最大距离价格满意度")
    tol_sat = max(1e-4, abs(opt_sat) * 1e-6)
    constraints_extra.append(_make_sparse_constraint(sat_coeff, lb=opt_sat - tol_sat, ub=np.inf))

    res2, opt_build = run_stage(build_coeff, False, constraints_extra, "2-全覆盖下最小建设成本")
    tol_build = max(1e-2, abs(opt_build) * 1e-6)
    constraints_extra.append(_make_sparse_constraint(build_coeff, lb=-np.inf, ub=opt_build + tol_build))

    res3, opt_cap = run_stage(capacity_coeff, True, constraints_extra, "3-全覆盖下最大总安装能力")
    tol_cap = max(1e-4, abs(opt_cap) * 1e-6)
    constraints_extra.append(_make_sparse_constraint(capacity_coeff, lb=opt_cap - tol_cap, ub=np.inf))

    res4, opt_umax = run_stage(umax_coeff, False, constraints_extra, "4-全覆盖下最小最大利用率")
    x = res4.x

    return {
        "x": x,
        "v_idx": v_idx,
        "x_idx": x_idx,
        "umax_idx": umax_idx,
        "coverage_coeff": coverage_coeff,
        "sat_coeff": sat_coeff,
        "demand_coeff": demand_coeff,
        "build_coeff": build_coeff,
        "capacity_coeff": capacity_coeff,
        "umax_coeff": umax_coeff,
        "opt_cov": float(coverage_coeff @ x),
        "opt_sat_num_proxy": float(sat_coeff @ x),
        "opt_dem_month": float(demand_coeff @ x),
        "opt_build": float(build_coeff @ x),
        "opt_capacity": float(capacity_coeff @ x),
        "opt_umax": float(umax_coeff @ x),
        "scale_names": scale_names,
    }


def install_full_cover_solver() -> None:
    problem2.solve_split_demand_milp = solve_split_demand_milp_fullcover


def restore_original_solver() -> None:
    problem2.solve_split_demand_milp = ORIGINAL_SPLIT_DEMAND_SOLVER


# ============================================================
# 数据与运行辅助函数
# ============================================================

def _ignore_excel_lock_files(dir_path: str, names: list[str]) -> set[str]:
    """复制 data 目录时忽略 Excel/WPS 生成的临时锁文件。

    Windows 下打开 xlsx 后会生成类似 ~$附件1：小区基础数据.xlsx 的隐藏临时文件，
    这些文件经常处于拒绝访问状态，不能复制，也不属于正式数据。
    """
    ignored: set[str] = set()
    for name in names:
        if name.startswith("~$") or name.endswith(".tmp"):
            ignored.add(name)
    return ignored


def _safe_rmtree(path: Path) -> None:
    """尽量删除旧情景数据目录。

    如果 Excel/WPS 正打开了 problem4_outputs 里的结果表，Windows 可能拒绝删除。
    这种情况下直接报出更明确的提示，避免后续读到旧结果。
    """
    if not path.exists():
        return
    def onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o666)
            func(p)
        except Exception as e:
            raise PermissionError(
                f"无法删除旧目录或文件：{p}。请先关闭 Excel/WPS/PyCharm 中打开的 problem4_outputs 结果文件后重试。"
            ) from e
    shutil.rmtree(path, onerror=onerror)


def _prepare_data_dir(work: Path, sc: Scenario) -> None:
    src = ROOT / "data"
    dst = work / "data"
    if dst.exists():
        _safe_rmtree(dst)
    shutil.copytree(src, dst, ignore=_ignore_excel_lock_files)

    # 固定管理成本扰动：直接修改情景目录中的附件3副本，不影响原始 data。
    # 注意：Windows 下 pd.ExcelFile 如果不关闭，会导致后续覆盖同一个 xlsx 时 WinError 5。
    # 因此这里先用 with 读完所有 sheet，关闭句柄后再写临时文件并替换。
    if abs(sc.fixed_cost_mult - 1.0) > 1e-12:
        a3 = dst / "附件3：服务站建设与运营成本.xlsx"
        sheets: dict[str, pd.DataFrame] = {}
        with pd.ExcelFile(a3) as xls:
            for sn in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sn)
                # 先提升表头，避免真实表头在第二行。
                try:
                    df2 = problem2.promote_header_if_needed(df, sheet_name=sn)
                except Exception:
                    df2 = df.copy()
                fixed_col = None
                for c in df2.columns:
                    cc = problem2.clean_text(c)
                    if ("固定" in cc and "成本" in cc) or ("管理" in cc and "成本" in cc):
                        fixed_col = c
                        break
                if fixed_col is not None:
                    df2[fixed_col] = pd.to_numeric(df2[fixed_col], errors="coerce") * sc.fixed_cost_mult
                sheets[sn] = df2

        tmp = a3.with_name(a3.stem + "_scaled_tmp.xlsx")
        if tmp.exists():
            tmp.unlink()
        with pd.ExcelWriter(tmp, engine="openpyxl") as w:
            for sn, df in sheets.items():
                df.to_excel(w, sheet_name=sn, index=False)
        os.replace(tmp, a3)
        print(f"INFO: {a3.name} / 服务站建设与运营成本 / 日固定管理成本 已乘以 {sc.fixed_cost_mult}")


def _run_problem1(work: Path, sc: Scenario) -> Path:
    old = os.getcwd()
    try:
        os.chdir(work)
        problem1.NEW_RATE = sc.new_rate
        base_df, _ = problem1.load_attachment1(work / "data/附件1：小区基础数据.xlsx")
        if sc.p12 is not None:
            if "p12" in base_df.columns:
                base_df["p12"] = sc.p12
            else:
                # 尽量兼容中文列名
                for c in base_df.columns:
                    cc = str(c)
                    if "自理" in cc and "半" in cc:
                        base_df[c] = sc.p12
        if sc.p23 is not None:
            if "p23" in base_df.columns:
                base_df["p23"] = sc.p23
            else:
                for c in base_df.columns:
                    cc = str(c)
                    if "半" in cc and "失能" in cc:
                        base_df[c] = sc.p23
        demand_df, _ = problem1.load_attachment2(work / "data/附件2：服务需求数据.xlsx")

        pred_df, area_sum = problem1.forecast_population(base_df)
        pop_yr5 = pred_df[pred_df["年份"] == problem1.YEARS][["小区", "自理", "半失能", "失能"]].merge(
            base_df[["小区", "人均月收入"]], on="小区", how="left"
        )
        demand_detail, theta_df = problem1.compute_demand(pop_yr5, demand_df)

        theory_detail = demand_detail[["小区", "老人类型", "服务项目", "第5年人数", "需求次数", "理论需求"]].copy()
        theory_sum = theory_detail.groupby("小区", as_index=False)["理论需求"].sum().rename(columns={"理论需求": "理论需求合计"})
        actual_detail = demand_detail[["小区", "老人类型", "服务项目", "第5年人数", "需求次数", "theta", "是否紧急救助", "实际需求"]].copy()
        actual_sum = actual_detail.groupby("小区", as_index=False)["实际需求"].sum().rename(columns={"实际需求": "实际需求合计"})
        cmp_df = theory_sum.merge(actual_sum, on="小区", how="outer")
        cmp_df["差值(理论-实际)"] = cmp_df["理论需求合计"] - cmp_df["实际需求合计"]
        cmp_df["实际/理论"] = cmp_df["实际需求合计"] / cmp_df["理论需求合计"]

        out_file = work / "B_problem1_results.xlsx"
        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            problem1.round_for_output(pred_df, ["自理", "半失能", "失能", "总人数"]).to_excel(writer, sheet_name="02_老人数量预测_逐小区", index=False)
            problem1.round_for_output(area_sum, ["自理", "半失能", "失能", "总人数"]).to_excel(writer, sheet_name="03_老人数量预测_区域汇总", index=False)
            problem1.round_for_output(theory_detail, ["第5年人数", "理论需求"]).to_excel(writer, sheet_name="04_理论需求_分小区分类型", index=False)
            problem1.round_for_output(theory_sum, ["理论需求合计"]).to_excel(writer, sheet_name="05_理论需求_小区汇总", index=False)
            theta_df.to_excel(writer, sheet_name="06_消费约束系数", index=False)
            problem1.round_for_output(actual_detail, ["第5年人数", "实际需求"]).to_excel(writer, sheet_name="07_实际需求_分小区分类型", index=False)
            problem1.round_for_output(actual_sum, ["实际需求合计"]).to_excel(writer, sheet_name="08_实际需求_小区汇总", index=False)
            problem1.round_for_output(cmp_df, ["理论需求合计", "实际需求合计", "差值(理论-实际)"]).to_excel(writer, sheet_name="09_理论与实际需求对比", index=False)
        return out_file
    finally:
        os.chdir(old)


def _coverage_info(p2: Path) -> dict[str, Any]:
    alloc = pd.read_excel(p2, sheet_name="06_小区需求满足汇总")
    total = len(alloc)

    def col_any(names: list[str]) -> str | None:
        for n in names:
            if n in alloc.columns:
                return n
        return None

    c_full = col_any(["是否完全覆盖"])
    c_part = col_any(["是否部分覆盖"])
    c_un = col_any(["是否未覆盖"])
    c_ratio = col_any(["需求满足比例", "需求满足率"])
    c_unmet = col_any(["日未满足需求", "未满足需求"])

    if c_ratio is not None:
        ratio = pd.to_numeric(alloc[c_ratio], errors="coerce").fillna(0.0)
    elif c_full is not None:
        ratio = pd.to_numeric(alloc[c_full], errors="coerce").fillna(0.0)
    else:
        ratio = pd.Series(np.zeros(total), index=alloc.index)

    unmet = pd.to_numeric(alloc[c_unmet], errors="coerce").fillna(0.0) if c_unmet is not None else pd.Series(np.zeros(total), index=alloc.index)

    full_mask = (ratio >= 0.999) & (unmet <= 1e-4)
    full = int(full_mask.sum())
    part = int(((ratio > 1e-6) & (~full_mask)).sum())
    un = int((ratio <= 1e-6).sum())

    # 若原表提供三类标识，则优先用于诊断，但全覆盖判断仍以 ratio/unmet 为准。
    if c_full is not None and c_part is not None and c_un is not None:
        raw_full = int(pd.to_numeric(alloc[c_full], errors="coerce").fillna(0).sum())
        raw_part = int(pd.to_numeric(alloc[c_part], errors="coerce").fillna(0).sum())
        raw_un = int(pd.to_numeric(alloc[c_un], errors="coerce").fillna(0).sum())
        full = min(full, raw_full) if raw_full != full else full
        part = max(part, raw_part)
        un = max(un, raw_un)

    return {
        "total": total,
        "full": full,
        "part": part,
        "uncovered": un,
        "coverage": full / total if total else 0.0,
        "is_full_cover": bool(total > 0 and full == total and part == 0 and un == 0),
    }


def _run_problem2_once(work: Path, budget: int, full_cover: bool = True) -> Path:
    old = os.getcwd()
    try:
        os.chdir(work)
        problem2.BUDGET_YUAN = int(budget)
        if full_cover:
            install_full_cover_solver()
        else:
            restore_original_solver()
        problem2.main()
        src = work / "B_problem2_split_results.xlsx"
        if not src.exists():
            raise RuntimeError("problem2.main() 未生成 B_problem2_split_results.xlsx")
        dst = work / "B_problem2_results.xlsx"
        shutil.copy2(src, dst)
        cov = _coverage_info(dst)
        if full_cover and not cov["is_full_cover"]:
            raise RuntimeError(
                f"FULL_COVER求解后仍未全覆盖：完全={cov['full']}，部分={cov['part']}，未覆盖={cov['uncovered']}"
            )
        return dst
    finally:
        os.chdir(old)


def _run_problem2_for_scenario(work: Path, sc: Scenario) -> tuple[Path, dict[str, Any]]:
    """按题目三类情景运行问题2。

    重要：每次只改变当前情景参数，其他参数均回到附件默认值。
    - S0：默认参数、120万元预算；
    - S1：仅老人增长与状态转移参数变化，120万元预算；
    - S2：仅固定管理成本增加20%，120万元预算；
    - S3：仅预算调整为140万元，并启用全覆盖硬约束；若不可行则直接报错。
    """
    budget = int(sc.budget)
    diag = {
        "原预算": budget,
        "原预算是否可行": True,
        "全覆盖最低预算": budget if sc.full_cover else np.nan,
        "实际采用预算": budget,
        "诊断": "预算调整情景：140万元预算下全覆盖可行。" if sc.full_cover else "按题目三类情景重跑；本情景未额外强制全覆盖。",
    }

    if sc.full_cover:
        print(f"INFO: 情景 {sc.code} 为预算调整情景：仅将建设预算调整为 140 万元，并启用 FULL_COVER 硬约束。")
        try:
            p2 = _run_problem2_once(work, budget, full_cover=True)
            return p2, diag
        except Exception as e:
            diag.update({
                "原预算是否可行": False,
                "诊断": f"140万元预算下无法实现全覆盖：{str(e)[:200]}",
            })
            raise RuntimeError(diag["诊断"]) from e

    p2 = _run_problem2_once(work, budget, full_cover=False)
    return p2, diag


def _pick_col_contains(df: pd.DataFrame, keywords: list[str], excludes: list[str] | None = None) -> str | None:
    """按列名关键词识别列，供问题4内部构造分流版问题3输入。"""
    excludes = excludes or []
    for c in df.columns:
        cn = str(c).replace(" ", "")
        if all(k in cn for k in keywords) and not any(e in cn for e in excludes):
            return c
    return None


def _parse_ratio_series(flow_df: pd.DataFrame, col_comm: str) -> pd.Series:
    """从问题2分流表中获得每条“小区-服务站”分流比例。

    优先使用“占本小区需求比例”；若不存在，则用日分配需求或月分配需求
    在同一小区内归一化。这样能保留 G->E/G->I/G->J 等次要分流，
    而不是压缩成一小区一主站。
    """
    col_ratio = _pick_col_contains(flow_df, ["占本小区需求比例"])
    if col_ratio is None:
        col_ratio = _pick_col_contains(flow_df, ["分配比例"])
    if col_ratio is not None:
        ratio = pd.to_numeric(flow_df[col_ratio], errors="coerce").fillna(0.0)
        # 兼容 15.09 这类百分数写法。
        if ratio.max() > 1.5:
            ratio = ratio / 100.0
        return ratio.clip(lower=0.0)

    for keys in [["日分配需求"], ["月分配需求"], ["分配需求"]]:
        c = _pick_col_contains(flow_df, keys)
        if c is not None:
            val = pd.to_numeric(flow_df[c], errors="coerce").fillna(0.0).clip(lower=0.0)
            denom = val.groupby(flow_df[col_comm].map(lambda x: B_problem3.clean_text(x))).transform("sum")
            return np.where(denom > 1e-12, val / denom, 0.0)

    raise KeyError("问题2分流明细中未找到分配比例、日分配需求或月分配需求列，无法保留分流比例。")


def _build_flow_inputs_for_problem3(p1: Path, p2: Path) -> tuple[Path, Path, pd.DataFrame]:
    """为 B_problem3 生成“保留分流比例”的专用输入文件。

    B_problem3.py 的原始逻辑是一小区对应一个服务站，并会按“小区”去重。
    为了不修改 B_problem3.py，这里采用等价变换：

    1. 读取问题2的 07_小区到站点分流明细；
    2. 将每条流构造为一个虚拟需求点：原小区__to__服务站；
    3. 将问题1需求按分流比例拆分到这些虚拟需求点；
    4. 让 B_problem3 按虚拟需求点合并，从而完整保留所有次要分流。

    例如：I 小区月需求 1000，I->D=15.09%，I->I=84.90%，则生成：
    I__to__D 需求 150.9；I__to__I 需求 849.0。
    """
    wb2 = pd.read_excel(p2, sheet_name=None)
    if "07_小区到站点分流明细" not in wb2:
        raise KeyError("问题2结果缺少 07_小区到站点分流明细，无法按分流比例拆分问题3需求。")

    flow = wb2["07_小区到站点分流明细"].copy()
    flow.columns = [B_problem3.clean_text(c) for c in flow.columns]

    col_comm = _pick_col_contains(flow, ["小区"])
    col_station = _pick_col_contains(flow, ["服务站"], excludes=["规模"])
    if col_station is None:
        # 兜底：排除“服务站规模”，选第一个包含服务站/站点的列。
        for c in flow.columns:
            cn = B_problem3.clean_text(c)
            if ("服务站" in cn or "站点" in cn) and "规模" not in cn:
                col_station = c
                break
    if col_comm is None or col_station is None:
        raise KeyError(f"无法从 07_小区到站点分流明细 中识别小区列或服务站列，当前列={list(flow.columns)}")

    col_scale = _pick_col_contains(flow, ["规模"])
    col_day = _pick_col_contains(flow, ["日分配需求"])
    col_month = _pick_col_contains(flow, ["月分配需求"])
    col_pop = _pick_col_contains(flow, ["等价覆盖老人数量"])
    col_dist = _pick_col_contains(flow, ["距离"], excludes=["满意"])
    col_s1 = _pick_col_contains(flow, ["距离满意度"])
    col_s2 = _pick_col_contains(flow, ["响应满意度"])
    col_s = _pick_col_contains(flow, ["综合满意度"])

    flow["原小区"] = flow[col_comm].map(B_problem3.clean_text)
    flow["服务站"] = flow[col_station].map(B_problem3.clean_text)
    flow = flow[(flow["原小区"] != "") & flow["服务站"].map(B_problem3.is_valid_station_name)].copy()
    flow["分流比例"] = _parse_ratio_series(flow, col_comm)
    flow = flow[flow["分流比例"] > 1e-10].copy().reset_index(drop=True)
    if flow.empty:
        raise ValueError("问题2分流表没有有效分流比例，问题3无法继续。")

    # 虚拟小区名必须唯一，否则 B_problem3 会按“小区”去重。
    flow["虚拟小区"] = [
        f"{comm}__to__{st}__{idx+1}"
        for idx, (comm, st) in enumerate(zip(flow["原小区"], flow["服务站"]))
    ]

    # 读取问题1需求明细，并按分流比例拆给服务站。
    wb1 = B_problem3.read_workbook_sheets(p1)
    _demand_sheet, demand = B_problem3.extract_demand_from_sheets(wb1)
    demand = demand.copy()
    demand["小区"] = demand["小区"].map(B_problem3.clean_text)
    demand["服务项目"] = demand["服务项目"].map(B_problem3.canonical_service)
    demand["实际需求"] = pd.to_numeric(demand["实际需求"], errors="coerce").fillna(0.0).clip(lower=0.0)

    split = demand.merge(
        flow[["原小区", "虚拟小区", "服务站", "分流比例"]],
        left_on="小区",
        right_on="原小区",
        how="inner",
    )
    if split.empty:
        raise ValueError("问题1需求表与问题2分流表按小区合并后为空，请检查小区名称。")
    split["原始实际需求"] = split["实际需求"]
    split["实际需求"] = split["实际需求"] * split["分流比例"]
    split["原小区"] = split["小区"]
    split["小区"] = split["虚拟小区"]
    split_demand = split[[
        "小区", "原小区", "服务站", "分流比例", "老人类型", "服务项目", "原始实际需求", "实际需求"
    ]].copy()

    # 构造给 B_problem3 读取的小区-服务站分配表。每条虚拟小区只有一个服务站，
    # 但这些虚拟小区合起来完整保留了原始可拆分流。
    alloc = pd.DataFrame()
    alloc["小区"] = flow["虚拟小区"]
    alloc["原小区"] = flow["原小区"]
    alloc["服务站"] = flow["服务站"]
    if col_scale is not None:
        alloc["服务站规模"] = flow[col_scale]
    alloc["分流比例"] = flow["分流比例"]
    if col_day is not None:
        alloc["日分配需求"] = pd.to_numeric(flow[col_day], errors="coerce")
    if col_month is not None:
        alloc["月分配需求"] = pd.to_numeric(flow[col_month], errors="coerce")
    if col_pop is not None:
        alloc["等价覆盖老人数量"] = pd.to_numeric(flow[col_pop], errors="coerce")
    if col_dist is not None:
        alloc["距离"] = pd.to_numeric(flow[col_dist], errors="coerce")
    if col_s1 is not None:
        alloc["距离满意度"] = pd.to_numeric(flow[col_s1], errors="coerce")
    if col_s2 is not None:
        alloc["响应满意度"] = pd.to_numeric(flow[col_s2], errors="coerce")
    if col_s is not None:
        alloc["综合满意度"] = pd.to_numeric(flow[col_s], errors="coerce")
    alloc["是否覆盖"] = True

    # 输出专用问题1文件。
    p1_out = p1.with_name("B_problem1_results_for_problem3_flow_split.xlsx")
    with pd.ExcelWriter(p1_out, engine="openpyxl") as w:
        split_demand.to_excel(w, sheet_name="07_实际需求_分小区分类型", index=False)
        # 辅助核对：按真实服务站汇总各服务项目需求。
        station_service = split_demand.groupby(["服务站", "老人类型", "服务项目"], as_index=False)["实际需求"].sum()
        station_service.to_excel(w, sheet_name="10_分流后站点服务需求", index=False)
        flow[["原小区", "虚拟小区", "服务站", "分流比例"]].to_excel(w, sheet_name="11_分流比例映射", index=False)

    # 输出专用问题2文件。
    p2_out = p2.with_name("B_problem2_results_for_problem3_flow_split.xlsx")
    with pd.ExcelWriter(p2_out, engine="openpyxl") as w:
        if "05_最优选址规模方案" in wb2:
            wb2["05_最优选址规模方案"].to_excel(w, sheet_name="05_最优选址规模方案", index=False)
        alloc.to_excel(w, sheet_name="06_小区分配结果", index=False)
        flow.to_excel(w, sheet_name="07_原始分流明细", index=False)
        for sn in ["06_小区需求满足汇总", "08_服务站覆盖明细", "11_总体指标"]:
            if sn in wb2:
                wb2[sn].to_excel(w, sheet_name=sn[:31], index=False)

    check = split_demand.groupby(["服务站", "服务项目"], as_index=False)["实际需求"].sum()
    print(
        f"INFO: 已为问题3构造分流比例版输入：虚拟小区数={alloc['小区'].nunique()}，"
        f"有效服务站数={alloc['服务站'].nunique()}，分流后需求记录={len(split_demand)}。"
    )
    return p1_out, p2_out, check

def _run_problem3(work: Path, p1: Path, p2: Path) -> Path:
    # 问题3必须保留问题2的“小区-服务站”分流比例。
    # 这里生成分流比例版 problem1/problem2 输入后，再调用原 B_problem3 主程序。
    p1_for_p3, p2_for_p3, _split_check = _build_flow_inputs_for_problem3(p1, p2)
    args = [
        "--root", str(work),
        "--problem1", str(p1_for_p3),
        "--problem2", str(p2_for_p3),
        "--attachment2", str(work / "data/附件2：服务需求数据.xlsx"),
        "--attachment3", str(work / "data/附件3：服务站建设与运营成本.xlsx"),
        "--attachment5", str(work / "data/附件5：满意度评分规则.xlsx"),
        "--output", "B_problem3_results.xlsx",
        "--charts-dir", "charts_problem3",
    ]
    B_problem3.main(args)
    return work / "B_problem3_results.xlsx"


def _read_overall_metric(path: Path, sheet: str, key_col: str = "指标", val_col: str = "数值") -> dict[str, Any]:
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return {}
    if key_col in df.columns and val_col in df.columns:
        return dict(zip(df[key_col].astype(str), df[val_col]))
    if len(df) >= 1:
        return df.iloc[0].to_dict()
    return {}


def _extract_p3_metrics(p3: Path) -> dict[str, Any]:
    try:
        df = pd.read_excel(p3, sheet_name="15_总体指标")
        if "指标" in df.columns and "数值" in df.columns:
            return dict(zip(df["指标"].astype(str), df["数值"]))
        if len(df) > 0:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


def _make_charts(summary_df: pd.DataFrame) -> None:
    CHART_ROOT.mkdir(exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        print("WARNING: matplotlib不可用，跳过问题4图表。")
        return

    def bar(col: str, title: str, ylabel: str, fname: str):
        if col not in summary_df.columns:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(summary_df))
        vals = pd.to_numeric(summary_df[col], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(summary_df["情景"].tolist())
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ymax = max(vals.max() if len(vals) else 0, 1e-9)
        ax.set_ylim(0, ymax * 1.15)
        for i, v in enumerate(vals):
            text = f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
            ax.text(i, v + ymax * 0.02, text, ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(CHART_ROOT / fname, dpi=180)
        plt.close(fig)

    bar("覆盖率", "各情景覆盖率对比", "覆盖率", "01_覆盖率对比.png")
    bar("总政府补贴", "各情景总政府补贴对比", "元", "02_总政府补贴对比.png")
    bar("平均综合满意度", "各情景平均综合满意度对比", "满意度", "03_平均综合满意度对比.png")
    bar("站点数量", "各情景站点数量对比", "站点数量", "04_站点数量对比.png")
    bar("实际采用预算", "各情景实际采用预算对比", "元", "05_实际采用预算对比.png")


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    CHART_ROOT.mkdir(exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    price_rows: list[pd.DataFrame] = []
    access_rows: list[pd.DataFrame] = []

    for sc in SCENARIOS:
        print("\n" + "=" * 80)
        print(f"开始运行情景 {sc.code}: {sc.name}")
        print("=" * 80)

        work = OUT_ROOT / sc.out_dir
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)

        _prepare_data_dir(work, sc)
        p1 = _run_problem1(work, sc)
        p2, diag = _run_problem2_for_scenario(work, sc)
        cov = _coverage_info(p2)

        # 二次防线：只有预算调整情景 S3（140万元）必须全覆盖。
        if sc.full_cover and not cov["is_full_cover"]:
            raise RuntimeError(
                f"情景 {sc.code} 使用140万元预算后仍未全覆盖，已停止。"
                f"完全={cov['full']}，部分={cov['part']}，未覆盖={cov['uncovered']}。"
            )

        p3 = _run_problem3(work, p1, p2)

        p2_metrics = _read_overall_metric(p2, "11_总体指标")
        p3_metrics = _extract_p3_metrics(p3)

        # 读取站点数量：problem2的总体指标里可能叫“站点数量”。
        station_count = p2_metrics.get("站点数量", np.nan)
        build_cost = p2_metrics.get("总建设成本", np.nan)
        avg_sat_p2 = p2_metrics.get("人口加权平均满意度", np.nan)
        demand_cov = p2_metrics.get("需求覆盖率", cov["coverage"])

        row = {
            "情景": sc.code,
            "情景名称": sc.name,
            "输出目录": str(work.relative_to(ROOT)),
            "老人增长率": sc.new_rate,
            "p12_自理转半失能": sc.p12 if sc.p12 is not None else 0.05,
            "p23_半失能转失能": sc.p23 if sc.p23 is not None else 0.09,
            "固定管理成本倍率": sc.fixed_cost_mult,
            "原预算": diag["原预算"],
            "原预算是否可行": diag["原预算是否可行"],
            "全覆盖最低预算": diag["全覆盖最低预算"],
            "实际采用预算": diag["实际采用预算"],
            "覆盖率": 1.0 if cov["is_full_cover"] else cov["coverage"],
            "是否全覆盖": cov["is_full_cover"],
            "完全覆盖小区数": cov["full"],
            "部分覆盖小区数": cov["part"],
            "未覆盖小区数": cov["uncovered"],
            "需求覆盖率": demand_cov,
            "站点数量": station_count,
            "总建设成本": build_cost,
            "问题2平均满意度": avg_sat_p2,
            "平均综合满意度": p3_metrics.get("平均综合满意度", p3_metrics.get("总体平均综合满意度", np.nan)),
            "平均价格满意度": p3_metrics.get("平均价格满意度", np.nan),
            "总政府补贴": p3_metrics.get("总政府补贴", np.nan),
            "总服务收入": p3_metrics.get("总服务收入", np.nan),
            "总运营成本": p3_metrics.get("总运营成本", np.nan),
            "总利润": p3_metrics.get("总利润", np.nan),
            "平均利润率": p3_metrics.get("平均利润率", np.nan),
            "诊断": diag["诊断"],
        }
        summary_rows.append(row)

        # 附加价格表和可及性表，若存在则汇总。
        try:
            price_df = pd.read_excel(p3, sheet_name="07_最优定价补贴方案")
            price_df.insert(0, "情景", sc.code)
            price_rows.append(price_df)
        except Exception:
            pass
        try:
            acc_df = pd.read_excel(p3, sheet_name="10_老人类型可及性对比")
            acc_df.insert(0, "情景", sc.code)
            access_rows.append(acc_df)
        except Exception:
            pass

        print(f"\n情景 {sc.code} 完成：完全覆盖={cov['full']}，部分覆盖={cov['part']}，未覆盖={cov['uncovered']}，采用预算={diag['实际采用预算']}")
        if sc.full_cover:
            print(f"{sc.code}为题目要求的140万元预算调整方案，应达到全覆盖。")

    summary_df = pd.DataFrame(summary_rows)

    # 灵敏度：相对 S0 的变化率。
    sens_rows = []
    base = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    metrics_for_sens = ["实际采用预算", "站点数量", "总建设成本", "平均综合满意度", "总政府补贴", "平均利润率"]
    for _, r in summary_df.iterrows():
        out = {"情景": r["情景"]}
        for m in metrics_for_sens:
            b = pd.to_numeric(pd.Series([base.get(m)]), errors="coerce").iloc[0]
            v = pd.to_numeric(pd.Series([r.get(m)]), errors="coerce").iloc[0]
            if pd.notna(b) and abs(b) > 1e-9 and pd.notna(v):
                out[f"{m}_相对S0变化率"] = (v - b) / b
            else:
                out[f"{m}_相对S0变化率"] = np.nan
        sens_rows.append(out)
    sens_df = pd.DataFrame(sens_rows)

    robust_df = summary_df[["情景", "是否全覆盖", "原预算是否可行", "全覆盖最低预算", "实际采用预算", "完全覆盖小区数", "部分覆盖小区数", "未覆盖小区数", "诊断"]].copy()

    other_uncertainty = pd.DataFrame([
        {"不确定因素": "实际老人需求偏离预测", "影响": "需求量和服务类型结构变化，导致站点能力不足或资源闲置", "应对策略": "滚动更新需求预测，按年度复算选址和补贴方案，预留可扩展服务能力"},
        {"不确定因素": "人工与运营成本继续上涨", "影响": "利润率下降、政府补贴压力增加", "应对策略": "建立成本指数联动补贴机制，并对固定成本高的站点设置专项运营补助"},
        {"不确定因素": "老人支付意愿和价格敏感性变化", "影响": "价格满意度和实际服务使用量偏离模型", "应对策略": "按老人类型和收入水平设置差异化服务券或阶梯补贴"},
        {"不确定因素": "道路交通和服务半径变化", "影响": "实际可达性低于距离矩阵评估结果", "应对策略": "使用实时交通时间替代静态距离，定期更新小区-站点可达关系"},
    ])

    out_file = ROOT / "B_problem4_results.xlsx"
    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        summary_df[["情景", "情景名称", "老人增长率", "p12_自理转半失能", "p23_半失能转失能", "固定管理成本倍率", "原预算", "实际采用预算"]].to_excel(writer, sheet_name="01_情景参数设置", index=False)
        summary_df[["情景", "覆盖率", "是否全覆盖", "完全覆盖小区数", "部分覆盖小区数", "未覆盖小区数", "站点数量", "总建设成本", "实际采用预算"]].to_excel(writer, sheet_name="02_问题2选址覆盖对比", index=False)
        summary_df[["情景", "平均综合满意度", "平均价格满意度", "总政府补贴", "总服务收入", "总运营成本", "总利润", "平均利润率"]].to_excel(writer, sheet_name="03_问题3财务满意度对比", index=False)
        (pd.concat(price_rows, ignore_index=True) if price_rows else pd.DataFrame()).to_excel(writer, sheet_name="04_服务定价变化", index=False)
        (pd.concat(access_rows, ignore_index=True) if access_rows else pd.DataFrame()).to_excel(writer, sheet_name="05_老人类型可及性变化", index=False)
        sens_df.to_excel(writer, sheet_name="06_灵敏度指标", index=False)
        robust_df.to_excel(writer, sheet_name="07_鲁棒性评价", index=False)
        other_uncertainty.to_excel(writer, sheet_name="08_其他不确定性与策略", index=False)

    _make_charts(summary_df)
    print("\n" + "=" * 80)
    print("问题4运行完成")
    print("结果文件:", out_file)
    print("情景目录:", OUT_ROOT)
    print("图表目录:", CHART_ROOT)
    print("=" * 80)


if __name__ == "__main__":
    main()
