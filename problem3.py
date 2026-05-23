from __future__ import annotations

from datetime import datetime
from itertools import product
from pathlib import Path
import re
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

DAYS_PER_YEAR = 365
DEPR_YEARS_DEFAULT = 20
SUBSIDY_PER_VISIT = 2.0
SUBSIDY_DAILY_CAP = {'小型': 1000.0, '中型': 1800.0, '大型': 2600.0}
CHARGE_SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴']
EMERGENCY = '紧急救助'


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ''
    s = str(x).strip()
    for a, b in [('\n', ''), ('\r', ''), ('\u3000', ''), (' ', ''), ('：', ':'), ('（', '('), ('）', ')')]:
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
    v = float(m.group())
    if '%' in s or '％' in s:
        v /= 100
    return v


def read_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if path.name.startswith('~$'):
        print(f'INFO: skip temp workbook {path}')
        return {}
    raw = pd.read_excel(path, sheet_name=None)
    out = {}
    for sn, df in raw.items():
        cdf = df.copy()
        cdf.columns = [clean_text(c) for c in cdf.columns]
        out[clean_text(sn)] = cdf
    return out


def choose_sheet_by_keywords(sheets: dict[str, pd.DataFrame], keywords: list[str]) -> tuple[str, pd.DataFrame]:
    kws = [clean_text(k).lower() for k in keywords]
    for name, df in sheets.items():
        txt = clean_text(name).lower()
        if all(k in txt for k in kws):
            return name, df
    for name, df in sheets.items():
        ctxt = '|'.join(clean_text(c).lower() for c in df.columns)
        dtxt = '|'.join(clean_text(v).lower() for v in df.head(5).astype(str).values.ravel())
        txt = f'{ctxt}|{dtxt}'
        if all(k in txt for k in kws):
            return name, df
    print(f'ERROR: 未找到关键词 {keywords} 对应 sheet，当前工作簿包括：')
    for n, d in sheets.items():
        print(f'  - {n}, 列: {list(d.columns)}')
        print(d.head(3))
    raise KeyError(f'cannot find sheet by {keywords}')


def pick_col(df: pd.DataFrame, keys: list[str]) -> str:
    cc = [(c, clean_text(c).lower()) for c in df.columns]
    for c, norm in cc:
        if all(k in norm for k in [clean_text(k).lower() for k in keys]):
            return c
    for c, norm in cc:
        if any(clean_text(k).lower() in norm for k in keys):
            return c
    raise KeyError(f'列未找到: {keys}, available={list(df.columns)}')


def s_price(price: float, p0: float) -> float:
    if price <= p0:
        return 1.0
    if price <= 1.1 * p0:
        return 0.9
    if price <= 1.2 * p0:
        return 0.75
    return 0.6


def build_candidate_prices(base: float, cost: float) -> list[float]:
    vals = sorted(set(round(v, 4) for v in [cost, base, 1.1 * base, 1.2 * base, 1.5 * base]))
    return [v for v in vals if v >= cost - 1e-9 and v <= 1.5 * base + 1e-9]


def safe_write_excel(path: Path, sheets: dict[str, pd.DataFrame]) -> Path:
    out = path
    try:
        with pd.ExcelWriter(out, engine='openpyxl') as w:
            for n, d in sheets.items():
                d.to_excel(w, index=False, sheet_name=n[:31])
        return out
    except PermissionError:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = path.with_name(f'{path.stem}_{ts}{path.suffix}')
        with pd.ExcelWriter(out, engine='openpyxl') as w:
            for n, d in sheets.items():
                d.to_excel(w, index=False, sheet_name=n[:31])
        return out


def main() -> None:
    root = Path('.')
    charts_dir = root / 'charts_problem3'
    charts_dir.mkdir(exist_ok=True)

    wb1 = read_workbook_sheets(root / 'B_problem1_results.xlsx')
    wb2 = read_workbook_sheets(root / 'B_problem2_results.xlsx')
    a2 = read_workbook_sheets(root / 'data/附件2：服务需求数据.xlsx')
    a3 = read_workbook_sheets(root / 'data/附件3：服务站建设与运营成本.xlsx')
    _a5 = read_workbook_sheets(root / 'data/附件5：满意度评分规则.xlsx')

    s_st, df_st = choose_sheet_by_keywords(wb2, ['最优', '选址', '规模'])
    s_alloc, df_alloc = choose_sheet_by_keywords(wb2, ['小区', '分配'])
    s_cov, df_cov = choose_sheet_by_keywords(wb2, ['覆盖'])
    s_sum, df_sum = choose_sheet_by_keywords(wb2, ['总体'])

    s_demand, df_demand = choose_sheet_by_keywords(wb1, ['实际', '需求'])
    s_srv, df_srv = choose_sheet_by_keywords(a2, ['营收', '支出'])

    col_station = pick_col(df_st, ['服务站'])
    col_scale = pick_col(df_st, ['规模'])
    col_build = pick_col(df_st, ['建设', '成本'])
    col_daily_fix = pick_col(df_st, ['日', '固定'])
    col_cap = pick_col(df_st, ['能力'])

    stations = df_st[[col_station, col_scale, col_build, col_daily_fix, col_cap]].copy()
    stations.columns = ['服务站', '规模', '建设成本', '日固定管理成本', '日服务能力']
    for c in ['建设成本', '日固定管理成本', '日服务能力']:
        stations[c] = stations[c].map(parse_number)

    col_comm = pick_col(df_alloc, ['小区'])
    col_alloc_st = pick_col(df_alloc, ['分配'])
    col_dist = pick_col(df_alloc, ['距离'])
    col_sd = pick_col(df_alloc, ['距离', '满意'])
    col_sr = pick_col(df_alloc, ['响应', '满意'])
    col_ss = pick_col(df_alloc, ['综合', '满意'])
    col_cover = pick_col(df_alloc, ['覆盖'])

    alloc = df_alloc[[col_comm, col_alloc_st, col_dist, col_sd, col_sr, col_ss, col_cover]].copy()
    alloc.columns = ['小区', '服务站', '距离', '距离满意度', '响应满意度', '问题2综合满意度', '是否覆盖']
    for c in ['距离', '距离满意度', '响应满意度', '问题2综合满意度']:
        alloc[c] = alloc[c].map(parse_number)

    col_dc = pick_col(df_demand, ['小区'])
    col_dt = pick_col(df_demand, ['老人类型'])
    col_ds = pick_col(df_demand, ['服务项目'])
    col_dn = pick_col(df_demand, ['实际', '需求'])
    demand = df_demand[[col_dc, col_dt, col_ds, col_dn]].copy()
    demand.columns = ['小区', '老人类型', '服务项目', '实际需求']
    demand['实际需求'] = demand['实际需求'].map(parse_number).fillna(0.0)

    c_serv = pick_col(df_srv, ['服务项目'])
    c_base = pick_col(df_srv, ['营收'])
    c_cost = pick_col(df_srv, ['支出'])
    price_ref = df_srv[[c_serv, c_base, c_cost]].copy()
    price_ref.columns = ['服务项目', '基准价格', '直接支出']
    price_ref['基准价格'] = price_ref['基准价格'].map(parse_number)
    price_ref['直接支出'] = price_ref['直接支出'].map(parse_number)
    pmap = {r['服务项目']: {'base': r['基准价格'], 'cost': r['直接支出']} for _, r in price_ref.iterrows()}

    merged = demand.merge(alloc[['小区', '服务站', '距离满意度', '响应满意度', '是否覆盖']], on='小区', how='left')

    station_results = []
    infeasible_stats = []
    best_prices = {}

    for _, srow in stations.iterrows():
        st = srow['服务站']
        st_scale = clean_text(srow['规模'])
        sdem = merged[merged['服务站'] == st].copy()
        service_candidates = {k: build_candidate_prices(pmap[k]['base'], pmap[k]['cost']) for k in CHARGE_SERVICES if k in pmap}
        enum_count = int(np.prod([len(v) for v in service_candidates.values()])) if service_candidates else 0

        best = None
        feasible = 0
        bad_loss, bad_high = 0, 0
        for combo in product(*[service_candidates[k] for k in CHARGE_SERVICES]):
            prices = {k: combo[i] for i, k in enumerate(CHARGE_SERVICES)}
            prices[EMERGENCY] = 0.0
            tmp = sdem.copy()
            tmp['price'] = tmp['服务项目'].map(prices).fillna(tmp['服务项目'].map(lambda x: pmap.get(x, {}).get('base', 0.0)))
            tmp['base'] = tmp['服务项目'].map(lambda x: pmap.get(x, {}).get('base', 0.0))
            tmp['cost'] = tmp['服务项目'].map(lambda x: pmap.get(x, {}).get('cost', 0.0))
            tmp['价格满意度'] = [s_price(p, b) if b > 0 else 1.0 for p, b in zip(tmp['price'], tmp['base'])]
            tmp['综合满意度'] = 0.2 * tmp['距离满意度'].fillna(0) + 0.3 * tmp['响应满意度'].fillna(0) + 0.5 * tmp['价格满意度']
            tmp['有效服务人次'] = tmp['实际需求'] * tmp['综合满意度']

            revenue = float((tmp['price'] * tmp['有效服务人次']).sum()) * DAYS_PER_YEAR
            dcost = float((tmp['cost'] * tmp['有效服务人次']).sum()) * DAYS_PER_YEAR
            sub_base_daily = float(tmp[tmp['服务项目'] != EMERGENCY]['有效服务人次'].sum()) * SUBSIDY_PER_VISIT
            subsidy = min(sub_base_daily, SUBSIDY_DAILY_CAP.get(st_scale, 1000.0)) * DAYS_PER_YEAR
            fixed = srow['日固定管理成本'] * DAYS_PER_YEAR
            depr = srow['建设成本'] / DEPR_YEARS_DEFAULT
            op_cost = fixed + depr
            profit = (revenue - dcost) + subsidy - op_cost
            rho = profit / op_cost if op_cost > 0 else -1
            avg_s = float(np.average(tmp['综合满意度'], weights=tmp['实际需求'].clip(lower=0) + 1e-9))
            pay = revenue
            if rho < 0:
                bad_loss += 1
                continue
            if rho > 0.08 + 1e-12:
                bad_high += 1
                continue
            feasible += 1
            score = (avg_s, -pay, -abs(0.08 - rho))
            if best is None or score > best['score']:
                best = dict(score=score, prices=prices, tmp=tmp, revenue=revenue, dcost=dcost, subsidy=subsidy,
                            fixed=fixed, depr=depr, op_cost=op_cost, profit=profit, rho=rho, avg_s=avg_s)

        if best is None:
            base_prices = {k: pmap[k]['base'] for k in CHARGE_SERVICES}
            base_prices[EMERGENCY] = 0.0
            best = dict(prices=base_prices, tmp=sdem.copy(), revenue=0.0, dcost=0.0, subsidy=0.0, fixed=0.0, depr=0.0,
                        op_cost=1.0, profit=-1.0, rho=-1.0, avg_s=0.0)
        best_prices[st] = best['prices']
        infeasible_stats.append({'服务站': st, '枚举组合数': enum_count, '可行组合数': feasible, '因亏损剔除数量': bad_loss,
                                 '因利润率超过8%剔除数量': bad_high, '最终选择方案': str(best['prices'])})

        pr = best['prices']
        station_results.append({'服务站': st, '规模': st_scale, '助餐价格': pr.get('助餐', np.nan), '日间照料价格': pr.get('日间照料', np.nan),
                                '上门护理价格': pr.get('上门护理', np.nan), '康复理疗价格': pr.get('康复理疗', np.nan),
                                '助浴价格': pr.get('助浴', np.nan), '紧急救助价格=0': 0.0,
                                '站点价格满意度': float(np.average(best['tmp']['价格满意度'], weights=best['tmp']['实际需求'] + 1e-9)) if len(best['tmp']) else 0.0,
                                '站点综合满意度': best['avg_s'], '年服务收入': best['revenue'], '年直接支出': best['dcost'],
                                '年政府补贴': best['subsidy'], '年固定管理成本': best['fixed'], '年建设折旧': best['depr'],
                                '年运营成本总额': best['op_cost'], '年利润': best['profit'], '利润率': best['rho']})

    st_df = pd.DataFrame(station_results)

    all_rows = []
    for _, r in merged.iterrows():
        st = r['服务站']
        srv = r['服务项目']
        prices = best_prices.get(st, {})
        price = prices.get(srv, 0.0 if srv == EMERGENCY else pmap.get(srv, {}).get('base', 0.0))
        base = pmap.get(srv, {}).get('base', 0.0)
        ps = s_price(price, base) if base > 0 else 1.0
        s = 0.2 * parse_number(r['距离满意度'], 0) + 0.3 * parse_number(r['响应满意度'], 0) + 0.5 * ps
        eff = parse_number(r['实际需求'], 0) * s
        all_rows.append({'小区': r['小区'], '分配站点': st, '老人类型': r['老人类型'], '服务项目': srv, '价格满意度': ps,
                         '距离满意度': parse_number(r['距离满意度'], 0), '响应满意度': parse_number(r['响应满意度'], 0),
                         '综合满意度': s, '有效服务人次': eff, '是否覆盖': r['是否覆盖'], '价格': price,
                         '基准价格': base, '直接支出': pmap.get(srv, {}).get('cost', 0.0), '实际需求': parse_number(r['实际需求'], 0)})
    detail = pd.DataFrame(all_rows)

    comm = detail.groupby(['小区', '分配站点', '是否覆盖'], as_index=False).agg({
        '价格满意度': 'mean', '距离满意度': 'mean', '响应满意度': 'mean', '综合满意度': 'mean', '有效服务人次': 'sum'
    })

    # baseline
    base_detail = detail.copy()
    base_detail['价格'] = base_detail['基准价格']
    base_detail['价格满意度'] = 1.0
    base_detail['综合满意度'] = 0.2 * base_detail['距离满意度'] + 0.3 * base_detail['响应满意度'] + 0.5 * base_detail['价格满意度']
    base_detail['有效服务人次'] = base_detail['实际需求'] * base_detail['综合满意度']

    acc_rows = []
    for elder_type, g in detail.groupby('老人类型'):
        gb = base_detail[base_detail['老人类型'] == elder_type]
        dem = g['实际需求'].sum()
        eff_b = gb['有效服务人次'].sum()
        eff_o = g['有效服务人次'].sum()
        pay_b = (gb['价格'] * gb['有效服务人次']).sum() * DAYS_PER_YEAR
        pay_o = (g['价格'] * g['有效服务人次']).sum() * DAYS_PER_YEAR
        sub_b = (gb[gb['服务项目'] != EMERGENCY]['有效服务人次'].sum() * SUBSIDY_PER_VISIT) * DAYS_PER_YEAR
        sub_o = (g[g['服务项目'] != EMERGENCY]['有效服务人次'].sum() * SUBSIDY_PER_VISIT) * DAYS_PER_YEAR
        acc_rows.append({'老人类型': elder_type, '消费约束后实际需求人次': dem, '基准有效服务人次': eff_b, '优化后有效服务人次': eff_o,
                         '可及性变化': (eff_o / dem if dem > 0 else 0) - (eff_b / dem if dem > 0 else 0), '基准价格满意度': gb['价格满意度'].mean(),
                         '优化后价格满意度': g['价格满意度'].mean(), '补贴金额': sub_o, '支付金额变化': pay_o - pay_b,
                         '平均综合满意度(优化)': g['综合满意度'].mean(), '结论': '优化后可及性提升' if eff_o >= eff_b else '优化后可及性下降'})
    acc = pd.DataFrame(acc_rows)

    summary = pd.DataFrame([{
        '平均价格满意度': detail['价格满意度'].mean(), '平均综合满意度': detail['综合满意度'].mean(),
        '总政府补贴': st_df['年政府补贴'].sum(), '总服务收入': st_df['年服务收入'].sum(), '总直接支出': st_df['年直接支出'].sum(),
        '总运营成本': st_df['年运营成本总额'].sum(), '总利润': st_df['年利润'].sum(), '平均利润率': st_df['利润率'].mean(),
        '是否所有服务站满足保本微利': bool(((st_df['利润率'] >= 0) & (st_df['利润率'] <= 0.08)).all()), '折旧年限(年)': DEPR_YEARS_DEFAULT
    }])

    data_check = pd.DataFrame([
        {'文件': 'B_problem2_results.xlsx', '全部sheet': ','.join(wb2.keys()), '使用sheet': ','.join([s_st, s_alloc, s_cov, s_sum]), '行数': len(df_st)+len(df_alloc), '关键列': '服务站/规模/小区分配'},
        {'文件': 'B_problem1_results.xlsx', '全部sheet': ','.join(wb1.keys()), '使用sheet': s_demand, '行数': len(df_demand), '关键列': '小区/老人类型/服务项目/实际需求'},
        {'文件': '附件2：服务需求数据.xlsx', '全部sheet': ','.join(a2.keys()), '使用sheet': s_srv, '行数': len(df_srv), '关键列': '服务项目/营收/支出'},
        {'文件': '附件3：服务站建设与运营成本.xlsx', '全部sheet': ','.join(a3.keys()), '使用sheet': '用于一致性检查', '行数': 0, '关键列': '规模/成本/能力'},
        {'文件': '附件5：满意度评分规则.xlsx', '全部sheet': ','.join(_a5.keys()), '使用sheet': '默认价格评分规则', '行数': 0, '关键列': 'S=0.2+0.3+0.5'},
    ])

    cand = pd.DataFrame([
        {'服务项目': k, '直接支出': pmap.get(k, {}).get('cost', np.nan), '基准价格': pmap.get(k, {}).get('base', np.nan),
         '候选价格列表': str(build_candidate_prices(pmap[k]['base'], pmap[k]['cost'])) if k in pmap else '[]'}
        for k in CHARGE_SERVICES
    ])

    baseline_station = pd.DataFrame({'服务站': st_df['服务站'], '优化利润率': st_df['利润率'], '优化满意度': st_df['站点综合满意度']})

    outputs = {
        '01_数据读取检查': data_check,
        '02_问题2最优方案输入': stations,
        '03_小区分配与基础满意度': alloc,
        '04_服务需求输入': demand,
        '05_候选价格集合': cand,
        '06_最优定价方案': st_df[['服务站', '规模', '助餐价格', '日间照料价格', '上门护理价格', '康复理疗价格', '助浴价格', '紧急救助价格=0', '站点价格满意度', '站点综合满意度']],
        '07_服务站利润与补贴': st_df[['服务站', '年服务收入', '年直接支出', '年政府补贴', '年固定管理成本', '年建设折旧', '年运营成本总额', '年利润', '利润率']].assign(**{'是否满足0到8%': lambda x: (x['利润率']>=0)&(x['利润率']<=0.08)}),
        '08_小区满意度结果': comm,
        '09_基准价格方案对比': baseline_station,
        '10_不同老人类型可及性分析': acc,
        '11_总体指标': summary,
        '12_不可行价格组合统计': pd.DataFrame(infeasible_stats),
    }
    out = safe_write_excel(root / 'B_problem3_results.xlsx', outputs)

    if HAS_MPL:
        pcols = ['助餐价格', '日间照料价格', '上门护理价格', '康复理疗价格', '助浴价格']
        st_df.set_index('服务站')[pcols].plot(kind='bar', figsize=(10, 5), title='各服务站最优服务价格')
        plt.tight_layout(); plt.savefig(charts_dir / '1_各服务站最优服务价格柱状图.png'); plt.close()
        st_df.plot(x='服务站', y='利润率', kind='bar', legend=False, title='各服务站利润率'); plt.tight_layout(); plt.savefig(charts_dir / '2_各服务站利润率柱状图.png'); plt.close()
        comm.plot(x='小区', y='综合满意度', kind='bar', legend=False, title='各小区综合满意度'); plt.tight_layout(); plt.savefig(charts_dir / '3_各小区综合满意度柱状图.png'); plt.close()
        acc.plot(x='老人类型', y=['基准有效服务人次', '优化后有效服务人次'], kind='bar', title='三类老人可及性对比'); plt.tight_layout(); plt.savefig(charts_dir / '4_三类老人可及性对比图.png'); plt.close()
        acc.plot(x='老人类型', y=['基准价格满意度', '优化后价格满意度'], kind='bar', title='基准与优化价格满意度对比'); plt.tight_layout(); plt.savefig(charts_dir / '5_基准价格与优化价格满意度对比图.png'); plt.close()

    print('\n===== 问题3求解摘要 =====')
    for _, r in st_df.iterrows():
        print(f"站点 {r['服务站']} 最优定价: 助餐{r['助餐价格']}, 日间照料{r['日间照料价格']}, 上门护理{r['上门护理价格']}, 康复理疗{r['康复理疗价格']}, 助浴{r['助浴价格']}; 利润率={r['利润率']:.4f}")
    print('总政府补贴:', st_df['年政府补贴'].sum())
    print('平均满意度:', detail['综合满意度'].mean())
    print('三类老人可及性变化:')
    print(acc[['老人类型', '可及性变化']])
    print(f'结果文件: {out}')


if __name__ == '__main__':
    main()
