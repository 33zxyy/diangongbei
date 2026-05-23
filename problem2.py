from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
import math
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    HAS_MPL = False

BUDGET_YUAN = 1_200_000
SERVICE_RADIUS_M = 1000
DEFAULT_WEIGHTS = {'distance': 0.2, 'response': 0.3, 'price': 0.5}


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ''
    s = str(x).strip()
    for a, b in [('\n', ''), ('\r', ''), (' ', ''), ('\u3000', ''), ('：', ':'), ('（', '('), ('）', ')'), ('－', '-'), ('—', '-')]:
        s = s.replace(a, b)
    return s


def parse_number(x: Any, default=np.nan) -> float:
    if pd.isna(x):
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return default
    val = float(m.group())
    if '%' in s or '％' in s:
        val /= 100
    return val


def read_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if path.name.startswith('~$'):
        return {}
    sheets = pd.read_excel(path, sheet_name=None)
    cleaned = {}
    for name, df in sheets.items():
        cdf = df.copy()
        cdf.columns = [clean_text(c) for c in cdf.columns]
        cleaned[name] = cdf
    return cleaned


def choose_sheet_by_keywords(sheets: dict[str, pd.DataFrame], keywords: list[str]) -> tuple[str, pd.DataFrame]:
    kws = [clean_text(k).lower() for k in keywords]
    for n, df in sheets.items():
        nn = clean_text(n).lower()
        if all(k in nn for k in kws):
            return n, df
    for n, df in sheets.items():
        headtxt = '|'.join(clean_text(v).lower() for v in df.head(10).astype(str).values.ravel())
        if all(k in headtxt for k in kws):
            return n, df
    print('无法匹配sheet，关键词:', keywords)
    for n, df in sheets.items():
        print(f'- {n}: cols={list(df.columns)}')
        print(df.head(3))
    raise KeyError(f'无法找到包含关键词 {keywords} 的sheet')


def pick_col(df: pd.DataFrame, includes: list[str], excludes: list[str] | None = None) -> str:
    excludes = excludes or []
    cols = list(df.columns)
    for c in cols:
        cc = clean_text(c)
        if all(clean_text(i) in cc for i in includes) and not any(clean_text(e) in cc for e in excludes):
            return c
    raise KeyError(f'列识别失败 includes={includes} excludes={excludes} 可选={cols}')


def parse_scale_name(s: str) -> str:
    t = clean_text(s)
    if '小' in t:
        return '小型'
    if '中' in t:
        return '中型'
    if '大' in t:
        return '大型'
    return t


def distance_satisfaction(d: float) -> float:
    if d <= 300:
        return 1.0
    if d <= 500:
        return 0.9
    if d <= 650:
        return 0.75
    if d <= 1000:
        return 0.6
    return 0.0


def response_satisfaction(u: float) -> float:
    if u <= 0.60:
        return 1.0
    if u <= 0.75:
        return 0.93
    if u <= 0.85:
        return 0.85
    if u <= 0.95:
        return 0.72
    if u <= 1.0:
        return 0.60
    return -1


@dataclass
class BestResult:
    key: tuple
    detail: dict


def safe_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    try:
        with open(path, 'a'):
            return path
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return path.with_name(f'{path.stem}_{ts}{path.suffix}')


def main() -> None:
    root = Path('.')
    p1 = root / 'B_problem1_results.xlsx'
    a2 = root / 'data/附件2：服务需求数据.xlsx'
    a3 = root / 'data/附件3：服务站建设与运营成本.xlsx'
    a4 = root / 'data/附件4：小区间距离矩阵.xlsx'
    a5 = root / 'data/附件5：满意度评分规则.xlsx'

    workbook_log = []

    p1s = read_workbook_sheets(p1)
    s02n, s02 = choose_sheet_by_keywords(p1s, ['老人', '逐小区'])
    s08n, s08 = choose_sheet_by_keywords(p1s, ['实际需求', '汇总'])
    s07n, s07 = choose_sheet_by_keywords(p1s, ['实际需求', '类型'])

    comm_col = pick_col(s02, ['小区'])
    p_col = pick_col(s02, ['总'])
    df_pop = s02[[comm_col, p_col]].copy()
    df_pop.columns = ['小区', '第5年末老人总数']
    df_pop['小区'] = df_pop['小区'].map(clean_text)
    df_pop['第5年末老人总数'] = df_pop['第5年末老人总数'].apply(parse_number)
    df_pop = df_pop.dropna().query("小区!=''")

    comm2 = pick_col(s08, ['小区'])
    q_col = pick_col(s08, ['实际', '合计'])
    df_dem = s08[[comm2, q_col]].copy()
    df_dem.columns = ['小区', '月实际需求']
    df_dem['小区'] = df_dem['小区'].map(clean_text)
    df_dem['月实际需求'] = df_dem['月实际需求'].apply(parse_number)
    df_dem = df_dem.dropna().query("小区!=''")

    base = df_pop.merge(df_dem, on='小区', how='inner')
    base['日均需求q'] = base['月实际需求'] / 30.0
    if len(base) != 10:
        raise ValueError(f'小区数量应为10，当前={len(base)}')

    a3s = read_workbook_sheets(a3)
    s3n, s3 = choose_sheet_by_keywords(a3s, ['规模'])
    size_col = pick_col(s3, ['规模'])
    cost_col = pick_col(s3, ['建设'])
    fixed_col = pick_col(s3, ['日'])
    cap_col = pick_col(s3, ['能力'])
    scale_df = s3[[size_col, cost_col, fixed_col, cap_col]].copy()
    scale_df.columns = ['规模', '建设成本原', '日固定成本', '日服务能力']
    scale_df['规模'] = scale_df['规模'].map(parse_scale_name)
    scale_df = scale_df[scale_df['规模'].isin(['小型', '中型', '大型'])].copy()
    scale_df['建设成本原'] = scale_df['建设成本原'].apply(parse_number)
    scale_df['日固定成本'] = scale_df['日固定成本'].apply(parse_number)
    scale_df['日服务能力'] = scale_df['日服务能力'].apply(parse_number)
    max_build = scale_df['建设成本原'].max()
    # 若看起来是万元，换算成元
    scale_df['建设成本'] = scale_df['建设成本原'] * (10000 if max_build < 1000 else 1)
    scale_df['年固定成本'] = scale_df['日固定成本'] * 365

    a4s = read_workbook_sheets(a4)
    s4n, s4 = choose_sheet_by_keywords(a4s, ['距离'])
    dmat = s4.copy()
    first = dmat.columns[0]
    dmat[first] = dmat[first].map(clean_text)
    dmat = dmat.set_index(first)
    dmat.index = dmat.index.map(clean_text)
    dmat.columns = [clean_text(c) for c in dmat.columns]
    communities = base['小区'].tolist()
    dmat = dmat.reindex(index=communities, columns=communities)
    for i in communities:
        for j in communities:
            dmat.loc[i, j] = parse_number(dmat.loc[i, j])
    for i in communities:
        dmat.loc[i, i] = 0.0
    if dmat.isna().sum().sum() > 0:
        raise ValueError('距离矩阵存在非对角缺失值')

    weights = DEFAULT_WEIGHTS.copy()
    a5s = read_workbook_sheets(a5)
    try:
        _, wdf = choose_sheet_by_keywords(a5s, ['权重'])
        txt = '|'.join(clean_text(v) for v in wdf.astype(str).values.ravel())
        nums = [float(x) for x in re.findall(r'\d+\.\d+|\d+', txt)]
        if len(nums) >= 3:
            ws = np.array(nums[:3], dtype=float)
            if ws.max() > 1:
                ws = ws / ws.sum()
            weights = {'distance': ws[0], 'response': ws[1], 'price': ws[2]}
    except Exception:
        print('WARNING: 无法解析附件5权重，使用默认权重', DEFAULT_WEIGHTS)

    # 收入/支出参数
    p1_comm = pick_col(s07, ['小区'])
    p1_type = pick_col(s07, ['服务'])
    p1_qty = pick_col(s07, ['实际'])
    demand_type = s07[[p1_comm, p1_type, p1_qty]].copy()
    demand_type.columns = ['小区', '服务项目', '月需求']
    demand_type['小区'] = demand_type['小区'].map(clean_text)
    demand_type['服务项目'] = demand_type['服务项目'].map(clean_text)
    demand_type['月需求'] = demand_type['月需求'].apply(parse_number)

    a2s = read_workbook_sheets(a2)
    s2pn, s2p = choose_sheet_by_keywords(a2s, ['营收'])
    c_service = pick_col(s2p, ['服务项目'])
    c_price = pick_col(s2p, ['营收'])
    c_direct = pick_col(s2p, ['支出'])
    price_df = s2p[[c_service, c_price, c_direct]].copy()
    price_df.columns = ['服务项目', '单价', '直接支出']
    price_df['服务项目'] = price_df['服务项目'].map(clean_text)
    price_df['单价'] = price_df['单价'].apply(parse_number)
    price_df['直接支出'] = price_df['直接支出'].apply(parse_number)

    dm = demand_type.merge(price_df, on='服务项目', how='left')
    dm['收入'] = dm['月需求'] * dm['单价']
    dm['直接成本'] = dm['月需求'] * dm['直接支出']

    unit_profit = (dm['收入'].sum() - dm['直接成本'].sum()) / max(dm['月需求'].sum(), 1)
    unit_rev = dm['收入'].sum() / max(dm['月需求'].sum(), 1)
    unit_dc = dm['直接成本'].sum() / max(dm['月需求'].sum(), 1)

    params = {r['规模']: {'build': r['建设成本'], 'fixed_day': r['日固定成本'], 'cap': r['日服务能力']} for _, r in scale_df.iterrows()}
    states = [None, '小型', '中型', '大型']
    total_enum = 4 ** len(communities)
    print('枚举建站方案总数:', total_enum)

    top = []
    best: BestResult | None = None
    feasible_budget = 0
    entered_search = 0

    P = dict(zip(base['小区'], base['第5年末老人总数']))
    Qd = dict(zip(base['小区'], base['日均需求q']))
    Qm = dict(zip(base['小区'], base['月实际需求']))
    total_pop = sum(P.values())
    total_month = sum(Qm.values())

    for combo in product(states, repeat=len(communities)):
        built = {communities[i]: combo[i] for i in range(len(communities)) if combo[i] is not None}
        if not built:
            continue
        build_cost = sum(params[s]['build'] for s in built.values())
        if build_cost > BUDGET_YUAN:
            continue
        feasible_budget += 1
        caps = {j: params[s]['cap'] for j, s in built.items()}
        rem = caps.copy()
        feasible_sites = {i: [j for j in built if dmat.loc[i, j] <= SERVICE_RADIUS_M] for i in communities}
        if all(len(v) == 0 for v in feasible_sites.values()):
            continue
        entered_search += 1
        order = sorted(communities, key=lambda x: Qd[x], reverse=True)
        best_local = {'key': (-1, -1, -1, math.inf, math.inf), 'assign': None, 'station_load': None}

        def eval_assign(assign: dict[str, str | None]):
            station_load = {j: 0.0 for j in built}
            covered_pop = covered_dem = 0.0
            for i, j in assign.items():
                if j:
                    station_load[j] += Qd[i]
                    covered_pop += P[i]
                    covered_dem += Qm[i]
            util = {j: station_load[j] / caps[j] for j in built}
            if any(u > 1 for u in util.values()):
                return
            s2 = {j: response_satisfaction(u) for j, u in util.items()}
            sat_num = 0.0
            for i, j in assign.items():
                if j:
                    s1 = distance_satisfaction(float(dmat.loc[i, j]))
                    sij = weights['distance'] * s1 + weights['response'] * s2[j] + weights['price'] * 1.0
                    sat_num += P[i] * sij
            avg_sat = sat_num / covered_pop if covered_pop > 0 else 0.0
            util_var = float(np.var(list(util.values()))) if util else 0.0
            key = (covered_pop, avg_sat, covered_dem, -build_cost, -util_var)
            if key > best_local['key']:
                best_local.update({'key': key, 'assign': assign.copy(), 'station_load': station_load.copy(), 'util': util.copy()})

        def dfs(idx: int, assign: dict[str, str | None], covered_pop_now: float):
            if idx == len(order):
                eval_assign(assign)
                return
            remain_upper = covered_pop_now + sum(P[x] for x in order[idx:])
            if remain_upper < best_local['key'][0]:
                return
            i = order[idx]
            for j in feasible_sites[i]:
                if rem[j] >= Qd[i]:
                    rem[j] -= Qd[i]
                    assign[i] = j
                    dfs(idx + 1, assign, covered_pop_now + P[i])
                    rem[j] += Qd[i]
            assign[i] = None
            dfs(idx + 1, assign, covered_pop_now)
            assign.pop(i, None)

        dfs(0, {}, 0.0)
        if best_local['assign'] is None:
            continue
        detail = {'built': built, 'build_cost': build_cost, **best_local}
        top.append(detail)
        top = sorted(top, key=lambda x: x['key'], reverse=True)[:20]
        if (best is None) or (detail['key'] > best.key):
            best = BestResult(key=detail['key'], detail=detail)
            print('更新最优:', detail['key'], '站点=', built)

    if best is None:
        raise RuntimeError('没有找到可行方案')

    print('预算可行方案数:', feasible_budget)
    print('进入分配搜索方案数:', entered_search)

    d = best.detail
    assign = d['assign']
    built = d['built']
    util = d['util']

    station_rows, cover_rows, alloc_rows, profit_rows = [], [], [], []
    for j, s in built.items():
        cap = params[s]['cap']
        ld = d['station_load'][j]
        station_rows.append({'站点': j, '规模': s, '建设成本': params[s]['build'], '日服务能力': cap, '日固定管理成本': params[s]['fixed_day'], '年固定管理成本': params[s]['fixed_day'] * 365, '利用率': ld / cap})
        served = [i for i, jj in assign.items() if jj == j]
        cpop = sum(P[i] for i in served)
        cm = sum(Qm[i] for i in served)
        cover_rows.append({'站点': j, '覆盖小区': '、'.join(served), '覆盖老人总数': cpop, '覆盖月需求': cm, '覆盖日需求': cm / 30, '利用率': ld / cap})
        rev = cm * unit_rev * 12
        dc = cm * unit_dc * 12
        fixed = params[s]['fixed_day'] * 365
        build = params[s]['build']
        dep = build / 20
        profit_rows.append({'站点': j, '年度收入': rev, '年直接支出': dc, '年固定管理成本': fixed, '建设成本': build, '年折旧': dep, '运营利润': rev - dc - fixed, '含折旧利润': rev - dc - fixed - dep})

    for i in communities:
        j = assign.get(i)
        covered = int(j is not None)
        dist = float(dmat.loc[i, j]) if j else np.nan
        s1 = distance_satisfaction(dist) if j else 0.0
        s2 = response_satisfaction(util[j]) if j else 0.0
        s3 = 1.0 if j else 0.0
        sat = weights['distance'] * s1 + weights['response'] * s2 + weights['price'] * s3 if j else 0.0
        alloc_rows.append({'小区': i, '第5年末老人总数': P[i], '月实际需求': Qm[i], '日均需求': Qd[i], '分配服务站': j or '未覆盖', '距离': dist, '距离满意度': s1, '响应满意度': s2, '价格满意度': s3, '综合满意度': sat, '是否被覆盖': covered})

    alloc_df = pd.DataFrame(alloc_rows)
    covered_pop = alloc_df.loc[alloc_df['是否被覆盖'] == 1, '第5年末老人总数'].sum()
    covered_month = alloc_df.loc[alloc_df['是否被覆盖'] == 1, '月实际需求'].sum()
    avg_sat = (alloc_df['第5年末老人总数'] * alloc_df['综合满意度']).sum() / max(covered_pop, 1)

    overall = pd.DataFrame([
        ('服务覆盖率', covered_pop / total_pop), ('人口加权平均满意度', avg_sat), ('被覆盖老人数量', covered_pop), ('总老人数量', total_pop),
        ('覆盖月需求', covered_month), ('总月需求', total_month), ('需求覆盖率', covered_month / total_month), ('总建设成本', d['build_cost']),
        ('总日服务能力', sum(params[s]['cap'] for s in built.values())), ('总日需求', total_month / 30), ('站点数量', len(built)),
        ('小型数量', sum(1 for x in built.values() if x == '小型')), ('中型数量', sum(1 for x in built.values() if x == '中型')), ('大型数量', sum(1 for x in built.values() if x == '大型'))
    ], columns=['指标', '数值'])

    top20 = pd.DataFrame([{'排名': i + 1, '覆盖老人数': t['key'][0], '平均满意度': t['key'][1], '覆盖月需求': t['key'][2], '建设成本': t['build_cost'], '利用率方差': -t['key'][4], '站点方案': ';'.join([f'{k}:{v}' for k, v in t['built'].items()])} for i, t in enumerate(sorted(top, key=lambda x: x['key'], reverse=True))])

    charts = Path('charts_problem2')
    charts.mkdir(exist_ok=True)
    if HAS_MPL:
        pd.DataFrame(alloc_rows).plot(x='小区', y='综合满意度', kind='bar', legend=False, title='各小区满意度')
        plt.tight_layout(); plt.savefig(charts / '02_各小区满意度柱状图.png', dpi=180); plt.close()
        pd.DataFrame(station_rows).plot(x='站点', y='利用率', kind='bar', legend=False, title='各服务站利用率')
        plt.tight_layout(); plt.savefig(charts / '03_各服务站利用率柱状图.png', dpi=180); plt.close()
        plt.figure(); plt.scatter(top20['建设成本'], top20['覆盖老人数'] / total_pop); plt.xlabel('建设成本'); plt.ylabel('覆盖率'); plt.title('覆盖率与建设成本'); plt.tight_layout(); plt.savefig(charts / '04_覆盖率与建设成本对比图.png', dpi=180); plt.close()
        plt.figure(); plt.scatter(top20['覆盖老人数'] / total_pop, top20['平均满意度']); plt.xlabel('覆盖率'); plt.ylabel('满意度'); plt.title('Top20 覆盖率-满意度散点'); plt.tight_layout(); plt.savefig(charts / '05_Top20覆盖率满意度散点图.png', dpi=180); plt.close()
        plt.figure(figsize=(8, 5))
        xs = np.arange(len(communities))
        pos = {c: (xs[i], 0) for i, c in enumerate(communities)}
        for c, (x, y) in pos.items():
            plt.scatter(x, y, c='tab:blue' if c in built else 'gray')
            plt.text(x, y + 0.03, c, ha='center', fontsize=8)
        for i, j in assign.items():
            if j:
                xi, yi = pos[i]; xj, yj = pos[j]
                plt.plot([xi, xj], [yi, yj], 'k-', alpha=0.4)
        plt.axis('off'); plt.title('最优站点覆盖关系图'); plt.tight_layout(); plt.savefig(charts / '01_最优站点覆盖关系图.png', dpi=180); plt.close()
    else:
        print('WARNING: matplotlib 不可用，已跳过图表生成。')

    output = safe_output_path(Path('B_problem2_results.xlsx'))
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame([
            {'文件': 'B_problem1_results.xlsx', 'sheet总数': len(p1s), '使用sheet': f'{s02n},{s07n},{s08n}'},
            {'文件': '附件2', 'sheet总数': len(a2s), '使用sheet': s2pn},
            {'文件': '附件3', 'sheet总数': len(a3s), '使用sheet': s3n},
            {'文件': '附件4', 'sheet总数': len(a4s), '使用sheet': s4n},
            {'文件': '附件5', 'sheet总数': len(a5s), '使用sheet': '权重sheet(若解析成功)'}
        ]).to_excel(writer, sheet_name='01_数据读取检查', index=False)
        base.to_excel(writer, sheet_name='02_问题1输入数据', index=False)
        scale_df[['规模', '建设成本', '日固定成本', '日服务能力']].to_excel(writer, sheet_name='03_服务站规模参数', index=False)
        dmat.to_excel(writer, sheet_name='04_距离矩阵')
        pd.DataFrame(station_rows).to_excel(writer, sheet_name='05_最优选址规模方案', index=False)
        alloc_df.to_excel(writer, sheet_name='06_小区分配结果', index=False)
        pd.DataFrame(cover_rows).to_excel(writer, sheet_name='07_服务站覆盖明细', index=False)
        pd.DataFrame(profit_rows).to_excel(writer, sheet_name='08_年度利润测算', index=False)
        overall.to_excel(writer, sheet_name='09_总体指标', index=False)
        top20.to_excel(writer, sheet_name='10_候选方案Top20', index=False)

    print('=== 最终最优方案摘要 ===')
    print('站点数量:', len(built), '方案:', built)
    print('服务覆盖率:', covered_pop / total_pop)
    print('平均满意度:', avg_sat)
    print('总建设成本:', d['build_cost'])
    print('总日服务能力:', sum(params[s]['cap'] for s in built.values()))
    print('被覆盖老人数:', covered_pop)
    print('需求覆盖率:', covered_month / total_month)
    for r in cover_rows:
        print(f"站点 {r['站点']} 覆盖: {r['覆盖小区']}")
    for r in profit_rows:
        print(f"站点 {r['站点']} 年利润(运营/含折旧): {r['运营利润']:.2f}/{r['含折旧利润']:.2f}")
    print('结果文件:', output)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
