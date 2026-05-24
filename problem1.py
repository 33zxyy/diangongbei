from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEATH_RATE = 0.05
NEW_RATE = 0.07
YEARS = 5
EMERGENCY_KEYWORDS = ['紧急救助', '应急救助', '紧急', '急救']


def clean_text(x):
    """统一清洗中文列名、单元格文本中的空格、换行、全角箭头等。"""
    if pd.isna(x):
        return ''
    s = str(x).strip()
    s = s.replace('\n', '').replace('\r', '')
    s = s.replace(' ', '').replace('\u3000', '')
    s = s.replace('→', '->').replace('—', '-').replace('－', '-')
    return s


def parse_number(x, default=np.nan):
    """从 0（公益免费）、≤ 20%、18万元 等文本里提取数字。百分数自动转为小数。"""
    if pd.isna(x):
        return default
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == '':
        return default
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m:
        return default
    val = float(m.group())
    if '%' in s or '％' in s:
        val /= 100.0
    return val


def normalize_columns(df: pd.DataFrame):
    out = df.copy()
    out.columns = [clean_text(c) for c in out.columns]
    return out


def validate_excel_file(path: Path):
    """校验 xlsx 文件是否有效；同时排除 Excel 临时文件 ~$xxx.xlsx。"""
    if path.name.startswith('~$'):
        raise ValueError(f'这是 Excel 临时锁定文件，不应读取: {path}')
    if path.suffix.lower() != '.xlsx':
        raise ValueError(f'文件不是 .xlsx: {path}')
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {path}')
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f'文件不是有效的 xlsx 压缩包，可能已损坏或后缀名错误: {path}\n'
            '请用 Excel/WPS 打开并另存为 .xlsx 后重试。'
        )


def list_xlsx_files(data_dir=Path('data')):
    files = [p for p in data_dir.glob('*.xlsx') if not p.name.startswith('~$')]
    if not files:
        raise FileNotFoundError(f'{data_dir.resolve()} 下没有找到正式 .xlsx 文件')
    return files


def pick_file_by_sheet(sheet_keywords, name_keywords=None, data_dir=Path('data')):
    """按文件名和 sheet 名自动识别附件。"""
    files = list_xlsx_files(data_dir)
    name_keywords = name_keywords or []

    # 先按文件名匹配
    for p in files:
        n = clean_text(p.name)
        if any(clean_text(k) in n for k in name_keywords):
            return p

    # 再按 sheet 名匹配
    for p in files:
        try:
            xls = pd.ExcelFile(p)
            sheet_text = '|'.join(clean_text(s) for s in xls.sheet_names)
            if all(clean_text(k) in sheet_text for k in sheet_keywords):
                return p
        except Exception:
            continue
    raise FileNotFoundError(
        f'未找到匹配文件。需要 sheet 关键词={sheet_keywords}，文件名关键词={name_keywords}；当前文件={[f.name for f in files]}'
    )


def choose_sheet(path: Path, keywords):
    xls = pd.ExcelFile(path)
    for s in xls.sheet_names:
        st = clean_text(s)
        if all(clean_text(k) in st for k in keywords):
            return s
    for s in xls.sheet_names:
        raw = pd.read_excel(path, sheet_name=s, header=None, nrows=8)
        txt = '|'.join(clean_text(v) for v in raw.astype(str).values.ravel())
        if all(clean_text(k) in txt for k in keywords):
            return s
    raise KeyError(f'{path.name} 中未找到包含关键词 {keywords} 的工作表；已有 sheet={xls.sheet_names}')


def read_table_auto(path: Path, sheet_name: str, header_keywords):
    """读取带标题行的 Excel sheet：自动定位真正表头行，适合第一行是大标题的情况。"""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    raw = raw.dropna(how='all').dropna(axis=1, how='all')
    if raw.empty:
        raise ValueError(f'{path.name} / {sheet_name} 是空表')

    best_idx, best_score = 0, -1
    for idx in range(min(len(raw), 15)):
        row_text = '|'.join(clean_text(v) for v in raw.iloc[idx].tolist())
        score = sum(1 for kw in header_keywords if clean_text(kw) in row_text)
        if score > best_score:
            best_idx, best_score = idx, score

    cols = [clean_text(c) if clean_text(c) else f'col_{i}' for i, c in enumerate(raw.iloc[best_idx].tolist())]
    df = raw.iloc[best_idx + 1:].copy().reset_index(drop=True)
    df.columns = cols
    df = normalize_columns(df)
    df = df.dropna(how='all').reset_index(drop=True)
    return df


def find_col(cols, include, exclude=None, required=True):
    exclude = exclude or []
    cols = [str(c) for c in cols]
    include = [clean_text(k) for k in include]
    exclude = [clean_text(k) for k in exclude]
    for c in cols:
        cc = clean_text(c)
        if all(k in cc for k in include) and not any(k in cc for k in exclude):
            return c
    if required:
        raise KeyError(f'未找到列，包含关键词={include}，排除关键词={exclude}，可选列={cols}')
    return None


def map_state_name(s):
    s = clean_text(s)
    if '半失能' in s or '半自理' in s:
        return '半失能'
    if '失能' in s:
        return '失能'
    if '自理' in s:
        return '自理'
    return s


def load_attachment1(path: Path):
    """读取附件1：人口与老人结构、转移概率。注意：这两个是不同 sheet，不能直接 concat 后找 p12/p23。"""
    validate_excel_file(path)
    pop_sheet = choose_sheet(path, ['人口', '老人'])
    trans_sheet = choose_sheet(path, ['转移'])

    pop = read_table_auto(path, pop_sheet, ['小区', '自理', '半失能', '失能', '收入'])
    trans = read_table_auto(path, trans_sheet, ['转移类型', '概率'])

    col_comm = find_col(pop.columns, ['小区'], required=False) or find_col(pop.columns, ['社区'], required=False) or find_col(pop.columns, ['编号'])
    col_n1 = find_col(pop.columns, ['自理'], exclude=['半'])
    col_n2 = find_col(pop.columns, ['半失能'])
    col_n3 = find_col(pop.columns, ['失能'], exclude=['半'])
    col_income = find_col(pop.columns, ['收入'])

    base = pop[[col_comm, col_n1, col_n2, col_n3, col_income]].copy()
    base.columns = ['小区', '自理', '半失能', '失能', '人均月收入']
    base['小区'] = base['小区'].map(clean_text)
    for c in ['自理', '半失能', '失能', '人均月收入']:
        base[c] = base[c].apply(parse_number)
    base = base.dropna(subset=['小区', '自理', '半失能', '失能', '人均月收入'])
    base = base[base['小区'] != '']

    col_type = find_col(trans.columns, ['转移类型'], required=False) or find_col(trans.columns, ['类型'])
    col_prob = find_col(trans.columns, ['概率'], required=False) or find_col(trans.columns, ['参考区间'])
    trans = trans[[col_type, col_prob]].copy()
    trans.columns = ['转移类型', '概率']
    trans['转移类型_clean'] = trans['转移类型'].map(clean_text)
    trans['概率'] = trans['概率'].apply(parse_number)

    p12_rows = trans[trans['转移类型_clean'].str.contains('自理') & trans['转移类型_clean'].str.contains('半失能')]
    p23_rows = trans[trans['转移类型_clean'].str.contains('半失能') & trans['转移类型_clean'].str.contains('失能')]
    if p12_rows.empty or p23_rows.empty:
        raise KeyError(f'转移概率表无法识别 p12/p23，请检查列：\n{trans}')

    base['p12'] = float(p12_rows['概率'].iloc[0])
    base['p23'] = float(p23_rows['概率'].iloc[0])

    raw_sheets = {pop_sheet: pop, trans_sheet: trans}
    return base, raw_sheets


def load_attachment2(path: Path):
    """读取附件2：三个 sheet 分别是需求次数、营收支出、消费上限。"""
    validate_excel_file(path)
    demand_sheet = choose_sheet(path, ['需求'])
    price_sheet = choose_sheet(path, ['营收'])
    cap_sheet = choose_sheet(path, ['消费', '上限'])

    demand_wide = read_table_auto(path, demand_sheet, ['服务项目', '自理', '失能'])
    price_df = read_table_auto(path, price_sheet, ['服务项目', '营收', '支出'])
    cap_df = read_table_auto(path, cap_sheet, ['老人类型', '消费上限'])

    # 1) 服务需求次数：宽表 -> 长表
    col_service_d = find_col(demand_wide.columns, ['服务项目'], required=False) or find_col(demand_wide.columns, ['项目'])
    state_cols = []
    for c in demand_wide.columns:
        mc = map_state_name(c)
        if mc in ['自理', '半失能', '失能'] and c != col_service_d:
            state_cols.append(c)
    if len(state_cols) < 3:
        raise KeyError(f'附件2需求次数表未识别到三类老人列，可选列={list(demand_wide.columns)}')

    demand_long = demand_wide[[col_service_d] + state_cols].copy()
    demand_long = demand_long.rename(columns={col_service_d: '服务项目'})
    demand_long['服务项目'] = demand_long['服务项目'].map(clean_text)
    demand_long = demand_long[demand_long['服务项目'] != '']
    demand_long = demand_long.melt(id_vars='服务项目', value_vars=state_cols, var_name='老人类型', value_name='需求次数')
    demand_long['老人类型'] = demand_long['老人类型'].map(map_state_name)
    demand_long['需求次数'] = demand_long['需求次数'].apply(parse_number)
    demand_long = demand_long.dropna(subset=['服务项目', '老人类型', '需求次数'])

    # 2) 服务价格与直接支出
    col_service_p = find_col(price_df.columns, ['服务项目'], required=False) or find_col(price_df.columns, ['项目'])
    col_price = find_col(price_df.columns, ['营收'], required=False) or find_col(price_df.columns, ['价格'], required=False) or find_col(price_df.columns, ['单次'])
    col_direct = find_col(price_df.columns, ['直接支出'], required=False) or find_col(price_df.columns, ['支出'])
    price2 = price_df[[col_service_p, col_price, col_direct]].copy()
    price2.columns = ['服务项目', '服务价格', '直接支出']
    price2['服务项目'] = price2['服务项目'].map(clean_text)
    price2['服务价格'] = price2['服务价格'].apply(parse_number)
    price2['直接支出'] = price2['直接支出'].apply(parse_number)
    price2 = price2.dropna(subset=['服务项目', '服务价格'])

    # 3) 消费上限比例
    col_state_cap = find_col(cap_df.columns, ['老人类型'], required=False) or find_col(cap_df.columns, ['类型'])
    col_cap = find_col(cap_df.columns, ['消费上限'], required=False) or find_col(cap_df.columns, ['上限'])
    cap2 = cap_df[[col_state_cap, col_cap]].copy()
    cap2.columns = ['老人类型', '消费上限']
    cap2['老人类型'] = cap2['老人类型'].map(map_state_name)
    cap2['消费上限'] = cap2['消费上限'].apply(parse_number)
    cap2 = cap2.dropna(subset=['老人类型', '消费上限'])

    out = demand_long.merge(price2, on='服务项目', how='left')
    out = out.merge(cap2, on='老人类型', how='left')
    if out[['服务价格', '直接支出', '消费上限']].isna().any().any():
        bad = out[out[['服务价格', '直接支出', '消费上限']].isna().any(axis=1)]
        raise ValueError(f'附件2合并后存在缺失，请检查服务名称或老人类型是否一致：\n{bad}')

    raw_sheets = {demand_sheet: demand_wide, price_sheet: price2, cap_sheet: cap2}
    return out, raw_sheets


def forecast_population(base_df: pd.DataFrame):
    records = []
    for _, row in base_df.iterrows():
        c = row['小区']
        n1, n2, n3 = float(row['自理']), float(row['半失能']), float(row['失能'])
        p12, p23 = float(row['p12']), float(row['p23'])
        for t in range(YEARS + 1):
            T = n1 + n2 + n3
            records.append({'小区': c, '年份': t, '自理': n1, '半失能': n2, '失能': n3, '总人数': T})
            if t < YEARS:
                n1_next = (1 - DEATH_RATE) * (1 - p12) * n1 + NEW_RATE * T
                n2_next = (1 - DEATH_RATE) * (p12 * n1 + (1 - p23) * n2)
                n3_next = (1 - DEATH_RATE) * (p23 * n2 + n3)
                n1, n2, n3 = n1_next, n2_next, n3_next
    pred = pd.DataFrame(records)
    area_sum = pred.groupby('年份', as_index=False)[['自理', '半失能', '失能', '总人数']].sum()
    return pred, area_sum


def compute_demand(pop_yr5, demand_df):
    long_pop = pop_yr5.melt(
        id_vars=['小区', '人均月收入'],
        value_vars=['自理', '半失能', '失能'],
        var_name='老人类型',
        value_name='第5年人数'
    )

    merged = long_pop.merge(demand_df, on='老人类型', how='left')
    if merged['服务项目'].isna().any():
        raise ValueError('人口类型与服务需求表未成功匹配，请检查老人类型字段')

    merged['理论需求'] = merged['第5年人数'] * merged['需求次数']
    merged['是否紧急救助'] = merged['服务项目'].apply(lambda x: any(k in str(x) for k in EMERGENCY_KEYWORDS))

    # E_m 是每个老人类型每人每月的理论消费。紧急救助公益免费，不计入消费上限。
    non_emg = merged[~merged['是否紧急救助']].copy()
    non_emg['单人理论消费'] = non_emg['需求次数'] * non_emg['服务价格']
    em = non_emg.groupby(['小区', '老人类型'], as_index=False)['单人理论消费'].sum()
    em = em.rename(columns={'单人理论消费': '理论月消费E_m'})

    lam = demand_df.groupby('老人类型', as_index=False)['消费上限'].median().rename(columns={'消费上限': 'lambda_m'})
    theta_df = long_pop[['小区', '老人类型', '人均月收入']].merge(lam, on='老人类型', how='left').merge(em, on=['小区', '老人类型'], how='left')
    theta_df['理论月消费E_m'] = theta_df['理论月消费E_m'].fillna(0.0)
    theta_df['消费上限B_i,m'] = theta_df['lambda_m'] * theta_df['人均月收入']
    theta_df['theta'] = np.where(
        theta_df['理论月消费E_m'] > 0,
        np.minimum(1.0, theta_df['消费上限B_i,m'] / theta_df['理论月消费E_m']),
        1.0
    )

    merged = merged.merge(theta_df[['小区', '老人类型', 'theta']], on=['小区', '老人类型'], how='left')
    merged['实际需求'] = np.where(merged['是否紧急救助'], merged['理论需求'], merged['理论需求'] * merged['theta'])
    return merged, theta_df


def save_charts(pred_df, cmp_df):
    out_dir = Path('charts')
    out_dir.mkdir(exist_ok=True)

    area = pred_df.groupby('年份')[['自理', '半失能', '失能']].sum()
    plt.figure(figsize=(9, 5))
    for c in area.columns:
        plt.plot(area.index, area[c], marker='o', label=c)
    plt.title('未来5年老人数量预测（区域汇总）')
    plt.xlabel('年份')
    plt.ylabel('人数')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / '01_老人数量趋势.png', dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(cmp_df))
    w = 0.36
    plt.bar(x - w / 2, cmp_df['理论需求合计'], width=w, label='理论需求')
    plt.bar(x + w / 2, cmp_df['实际需求合计'], width=w, label='实际需求')
    plt.xticks(x, cmp_df['小区'], rotation=45, ha='right')
    plt.ylabel('需求次数')
    plt.title('第5年各小区理论与实际需求对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / '02_理论与实际需求对比.png', dpi=160)
    plt.close()


def round_for_output(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = np.round(pd.to_numeric(out[c], errors='coerce')).astype('Int64')
    return out


def main():
    data_dir = Path('data')
    print('当前工作目录:', Path.cwd())
    print('data目录下的xlsx文件:', [p.name for p in list_xlsx_files(data_dir)])

    file1 = pick_file_by_sheet(['人口', '转移'], ['附件1'], data_dir)
    file2 = pick_file_by_sheet(['需求', '营收', '消费'], ['附件2'], data_dir)

    print('正在读取附件1:', file1)
    print('正在读取附件2:', file2)

    base_df, raw1 = load_attachment1(file1)
    demand_df, raw2 = load_attachment2(file2)

    pred_df, area_sum = forecast_population(base_df)
    pop_yr5 = pred_df[pred_df['年份'] == YEARS][['小区', '自理', '半失能', '失能']].merge(
        base_df[['小区', '人均月收入']], on='小区', how='left'
    )

    demand_detail, theta_df = compute_demand(pop_yr5, demand_df)

    theory_detail = demand_detail[['小区', '老人类型', '服务项目', '第5年人数', '需求次数', '理论需求']].copy()
    theory_sum = theory_detail.groupby('小区', as_index=False)['理论需求'].sum().rename(columns={'理论需求': '理论需求合计'})

    actual_detail = demand_detail[['小区', '老人类型', '服务项目', '第5年人数', '需求次数', 'theta', '是否紧急救助', '实际需求']].copy()
    actual_sum = actual_detail.groupby('小区', as_index=False)['实际需求'].sum().rename(columns={'实际需求': '实际需求合计'})

    cmp_df = theory_sum.merge(actual_sum, on='小区', how='outer')
    cmp_df['差值(理论-实际)'] = cmp_df['理论需求合计'] - cmp_df['实际需求合计']
    cmp_df['实际/理论'] = np.where(cmp_df['理论需求合计'] > 0, cmp_df['实际需求合计'] / cmp_df['理论需求合计'], np.nan)

    raw_check = pd.DataFrame([
        {'附件': '附件1', '文件': str(file1), 'sheet': ', '.join(raw1.keys()), '处理后行数': len(base_df)},
        {'附件': '附件2', '文件': str(file2), 'sheet': ', '.join(raw2.keys()), '处理后行数': len(demand_df)},
    ])

    # 控制台摘要
    yr5_area = area_sum[area_sum['年份'] == YEARS].iloc[0]
    print('\n===== 问题1计算摘要 =====')
    print(f"第5年末全区域老人总数: {yr5_area['总人数']:.0f}")
    print(f"自理: {yr5_area['自理']:.0f}, 半失能: {yr5_area['半失能']:.0f}, 失能: {yr5_area['失能']:.0f}")
    print(f"全区域理论月需求总量: {theory_detail['理论需求'].sum():.0f}")
    print(f"全区域消费约束后实际月需求总量: {actual_detail['实际需求'].sum():.0f}")
    print('\n消费约束影响最大的记录:')
    print(theta_df.sort_values('theta').head(8).to_string(index=False))

    out_file = Path('B_problem1_results.xlsx')
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        raw_check.to_excel(writer, sheet_name='01_原始数据检查', index=False)
        round_for_output(pred_df, ['自理', '半失能', '失能', '总人数']).to_excel(writer, sheet_name='02_老人数量预测_逐小区', index=False)
        round_for_output(area_sum, ['自理', '半失能', '失能', '总人数']).to_excel(writer, sheet_name='03_老人数量预测_区域汇总', index=False)
        round_for_output(theory_detail, ['第5年人数', '理论需求']).to_excel(writer, sheet_name='04_理论需求_分小区分类型', index=False)
        round_for_output(theory_sum, ['理论需求合计']).to_excel(writer, sheet_name='05_理论需求_小区汇总', index=False)
        theta_df.to_excel(writer, sheet_name='06_消费约束系数', index=False)
        round_for_output(actual_detail, ['第5年人数', '实际需求']).to_excel(writer, sheet_name='07_实际需求_分小区分类型', index=False)
        round_for_output(actual_sum, ['实际需求合计']).to_excel(writer, sheet_name='08_实际需求_小区汇总', index=False)
        round_for_output(cmp_df, ['理论需求合计', '实际需求合计', '差值(理论-实际)']).to_excel(writer, sheet_name='09_理论与实际需求对比', index=False)

    save_charts(pred_df, cmp_df)
    print(f'\n结果已输出: {out_file.resolve()}')
    print('图表目录: charts/')


if __name__ == '__main__':
    main()
