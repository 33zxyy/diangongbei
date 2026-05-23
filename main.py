from pathlib import Path
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


def pick_file(candidates):
    for p in candidates:
        fp = Path(p)
        if fp.exists():
            return fp
    raise FileNotFoundError(f'未找到数据文件: {candidates}')




def validate_excel_file(path: Path):
    """校验xlsx文件是否为有效Office压缩包，避免把损坏文件当作编码问题。"""
    if path.suffix.lower() != '.xlsx':
        raise ValueError(f'文件不是 .xlsx: {path}')
    if not path.exists():
        raise FileNotFoundError(f'文件不存在: {path}')
    if not zipfile.is_zipfile(path):
        raise ValueError(
            f'文件不是有效的xlsx压缩包，可能已损坏或后缀名错误: {path}\n'
            '请用Excel/WPS打开并另存为 .xlsx 后重试。'
        )
    with zipfile.ZipFile(path, 'r') as zf:
        names = set(zf.namelist())
        if '[Content_Types].xml' not in names:
            raise ValueError(f'文件缺少 [Content_Types].xml，疑似损坏: {path}')

def normalize_columns(df: pd.DataFrame):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def is_empty(v):
    return pd.isna(v) or str(v).strip() == ''


def build_columns(header_row, next_row=None):
    cols = []
    for i, cur in enumerate(header_row):
        cur_txt = '' if is_empty(cur) else str(cur).strip()
        nxt_txt = ''
        if next_row is not None and i < len(next_row) and not is_empty(next_row[i]):
            nxt_txt = str(next_row[i]).strip()

        if nxt_txt and cur_txt and cur_txt != nxt_txt:
            cols.append(f'{cur_txt}_{nxt_txt}')
        elif nxt_txt:
            cols.append(nxt_txt)
        else:
            cols.append(cur_txt if cur_txt else f'col_{i}')
    return cols


def choose_header_row(df_raw: pd.DataFrame, required_keywords):
    best_idx = 0
    best_score = -1
    n = len(df_raw)
    for i in range(min(n, 12)):
        row = df_raw.iloc[i].tolist()
        row_next = df_raw.iloc[i + 1].tolist() if i + 1 < n else None
        col_candidates = build_columns(row, row_next)
        txt = ' | '.join(col_candidates)
        score = sum(1 for kws in required_keywords for kw in kws if kw in txt)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def read_sheet_auto(file_path: Path, sheet_name: str, required_keywords):
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    header_idx = choose_header_row(df_raw, required_keywords)

    # 优先尝试双行表头，再回退单行表头
    row = df_raw.iloc[header_idx].tolist()
    row_next = df_raw.iloc[header_idx + 1].tolist() if header_idx + 1 < len(df_raw) else None

    cols_two = build_columns(row, row_next)
    txt_two = ' | '.join(cols_two)
    score_two = sum(1 for kws in required_keywords for kw in kws if kw in txt_two)

    cols_one = build_columns(row, None)
    txt_one = ' | '.join(cols_one)
    score_one = sum(1 for kws in required_keywords for kw in kws if kw in txt_one)

    if score_two >= score_one and row_next is not None:
        cols = cols_two
        data_start = header_idx + 2
    else:
        cols = cols_one
        data_start = header_idx + 1

    data = df_raw.iloc[data_start:].copy().reset_index(drop=True)
    data.columns = cols
    data = normalize_columns(data)
    return data


def find_col(cols, keywords, required=True):
    cols = [str(c).strip() for c in cols]
    for kw in keywords:
        for c in cols:
            if kw in c:
                return c
    if required:
        raise KeyError(f'未找到列，关键词={keywords}，可选列={cols}')
    return None


def to_numeric(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out


def load_attachment1(path: Path):
    xls = pd.ExcelFile(path)
    required = [
        ['小区', '社区'], ['自理'], ['半失能'], ['失能'], ['人均月收入', '月收入'], ['p12', '自理转半失能'], ['p23', '半失能转失能']
    ]
    raw_sheets = {s: read_sheet_auto(path, s, required) for s in xls.sheet_names}
    merged = pd.concat(raw_sheets.values(), ignore_index=True)

    col_comm = find_col(merged.columns, ['小区', '社区'])
    col_n1 = find_col(merged.columns, ['自理'])
    col_n2 = find_col(merged.columns, ['半失能'])
    col_n3 = find_col(merged.columns, ['失能'])
    col_income = find_col(merged.columns, ['人均月收入', '月收入', '收入'])
    col_p12 = find_col(merged.columns, ['p12', '自理转半失能', '转半失能'])
    col_p23 = find_col(merged.columns, ['p23', '半失能转失能', '转失能'])

    base = merged[[col_comm, col_n1, col_n2, col_n3, col_income, col_p12, col_p23]].copy()
    base.columns = ['小区', '自理', '半失能', '失能', '人均月收入', 'p12', 'p23']
    base['小区'] = base['小区'].astype(str).str.strip()
    base = to_numeric(base, ['自理', '半失能', '失能', '人均月收入', 'p12', 'p23'])
    base = base.dropna(subset=['小区', '自理', '半失能', '失能', '人均月收入', 'p12', 'p23'])
    base = base[base['小区'] != '']
    base = base.groupby('小区', as_index=False).first()

    return base, raw_sheets


def load_attachment2(path: Path):
    xls = pd.ExcelFile(path)
    required = [
        ['老人类型', '状态', '群体'], ['服务', '项目'], ['需求次数', '频次'], ['服务价格', '单价', '价格'], ['直接支出'], ['消费上限', '上限系数', 'λ']
    ]
    raw_sheets = {s: read_sheet_auto(path, s, required) for s in xls.sheet_names}
    merged = pd.concat(raw_sheets.values(), ignore_index=True)

    col_state = find_col(merged.columns, ['老人类型', '状态', '群体'])
    col_service = find_col(merged.columns, ['服务', '项目'])
    col_r = find_col(merged.columns, ['需求次数', '需求频次', '频次', '次数'])
    col_price = find_col(merged.columns, ['服务价格', '单价', '价格'])
    col_direct = find_col(merged.columns, ['直接支出'])
    col_lambda = find_col(merged.columns, ['消费上限', '上限系数', 'λ'])

    df = merged[[col_state, col_service, col_r, col_price, col_direct, col_lambda]].copy()
    df.columns = ['老人类型', '服务项目', '需求次数', '服务价格', '直接支出', '消费上限']
    df['老人类型'] = df['老人类型'].astype(str).str.strip()
    df['服务项目'] = df['服务项目'].astype(str).str.strip()
    df = to_numeric(df, ['需求次数', '服务价格', '直接支出', '消费上限'])
    df = df.dropna(subset=['老人类型', '服务项目', '需求次数', '服务价格'])

    lam = df.groupby('老人类型')['消费上限'].median()
    df['消费上限'] = df['老人类型'].map(lam)
    return df, raw_sheets


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


def map_state_name(s):
    s = str(s)
    if '半失能' in s:
        return '半失能'
    if '失能' in s:
        return '失能'
    if '自理' in s:
        return '自理'
    return s


def compute_demand(pop_yr5, demand_df):
    long_pop = pop_yr5.melt(id_vars=['小区', '人均月收入'], value_vars=['自理', '半失能', '失能'], var_name='老人类型', value_name='第5年人数')
    d2 = demand_df.copy()
    d2['老人类型'] = d2['老人类型'].map(map_state_name)

    merged = long_pop.merge(d2, on='老人类型', how='left')
    merged['理论需求'] = merged['第5年人数'] * merged['需求次数']
    merged['是否紧急救助'] = merged['服务项目'].apply(lambda x: any(k in str(x) for k in EMERGENCY_KEYWORDS))

    non_emg = merged[~merged['是否紧急救助']]
    em = non_emg.groupby(['小区', '老人类型'], as_index=False).apply(
        lambda g: pd.Series({'E_m': (g['需求次数'] * g['服务价格']).sum()})
    ).reset_index(drop=True)

    lam = merged.groupby('老人类型', as_index=False)['消费上限'].median().rename(columns={'消费上限': 'lambda_m'})
    theta_df = long_pop[['小区', '老人类型', '人均月收入']].merge(lam, on='老人类型', how='left').merge(em, on=['小区', '老人类型'], how='left')
    theta_df['E_m'] = theta_df['E_m'].fillna(0.0)
    theta_df['B_i,m'] = theta_df['lambda_m'] * theta_df['人均月收入']
    theta_df['theta'] = np.where(theta_df['E_m'] > 0, np.minimum(1.0, theta_df['B_i,m'] / theta_df['E_m']), 1.0)

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


def main():
    file1 = pick_file(['data/附件1.xlsx', 'data/附件1：小区基础数据.xlsx'])
    file2 = pick_file(['data/附件2.xlsx', 'data/附件2：服务需求数据.xlsx'])

    validate_excel_file(file1)
    validate_excel_file(file2)

    base_df, raw1 = load_attachment1(file1)
    demand_df, raw2 = load_attachment2(file2)

    pred_df, area_sum = forecast_population(base_df)
    pop_yr5 = pred_df[pred_df['年份'] == YEARS][['小区', '自理', '半失能', '失能']].merge(base_df[['小区', '人均月收入']], on='小区', how='left')

    demand_detail, theta_df = compute_demand(pop_yr5, demand_df)

    theory_detail = demand_detail[['小区', '老人类型', '服务项目', '第5年人数', '需求次数', '理论需求']].copy()
    theory_sum = theory_detail.groupby('小区', as_index=False)['理论需求'].sum().rename(columns={'理论需求': '理论需求合计'})

    actual_detail = demand_detail[['小区', '老人类型', '服务项目', '第5年人数', '需求次数', 'theta', '是否紧急救助', '实际需求']].copy()
    actual_sum = actual_detail.groupby('小区', as_index=False)['实际需求'].sum().rename(columns={'实际需求': '实际需求合计'})

    cmp_df = theory_sum.merge(actual_sum, on='小区', how='outer')
    cmp_df['差值(理论-实际)'] = cmp_df['理论需求合计'] - cmp_df['实际需求合计']
    cmp_df['实际/理论'] = np.where(cmp_df['理论需求合计'] > 0, cmp_df['实际需求合计'] / cmp_df['理论需求合计'], np.nan)

    for df, cols in [
        (pred_df, ['自理', '半失能', '失能', '总人数']),
        (area_sum, ['自理', '半失能', '失能', '总人数']),
        (theory_detail, ['第5年人数', '理论需求']),
        (theory_sum, ['理论需求合计']),
        (actual_detail, ['第5年人数', '实际需求']),
        (actual_sum, ['实际需求合计']),
        (cmp_df, ['理论需求合计', '实际需求合计', '差值(理论-实际)'])
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = np.round(df[c]).astype('Int64')

    raw_check = pd.DataFrame([
        {'附件': '附件1', '文件': str(file1), 'sheet数': len(raw1), '读取行数合计': sum(len(v) for v in raw1.values())},
        {'附件': '附件2', '文件': str(file2), 'sheet数': len(raw2), '读取行数合计': sum(len(v) for v in raw2.values())},
        {'附件': '处理后小区数', '文件': '-', 'sheet数': np.nan, '读取行数合计': len(base_df)},
        {'附件': '服务需求记录数', '文件': '-', 'sheet数': np.nan, '读取行数合计': len(demand_df)}
    ])

    out_file = Path('B_problem1_results.xlsx')
    with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
        raw_check.to_excel(writer, sheet_name='01_原始数据检查', index=False)
        pred_df.to_excel(writer, sheet_name='02_老人数量预测_逐小区', index=False)
        area_sum.to_excel(writer, sheet_name='03_老人数量预测_区域汇总', index=False)
        theory_detail.to_excel(writer, sheet_name='04_理论需求_分小区分类型', index=False)
        theory_sum.to_excel(writer, sheet_name='05_理论需求_小区汇总', index=False)
        theta_df.to_excel(writer, sheet_name='06_消费约束系数', index=False)
        actual_detail.to_excel(writer, sheet_name='07_实际需求_分小区分类型', index=False)
        actual_sum.to_excel(writer, sheet_name='08_实际需求_小区汇总', index=False)
        cmp_df.to_excel(writer, sheet_name='09_理论与实际需求对比', index=False)

    save_charts(pred_df, cmp_df)
    print(f'结果已输出: {out_file.resolve()}')
    print('图表目录: charts/')


if __name__ == '__main__':
    main()
