from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
import math
import re
import sys
import zipfile
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


HEADER_KEYWORDS = [
    '小区', '社区', '年份', '年度', '老人', '总数', '需求', '实际', '合计',
    '服务', '项目', '规模', '建设', '成本', '固定', '运营', '管理', '能力',
    '距离', '矩阵', '营收', '收入', '支出', '单价', '满意', '权重'
]


def promote_header_if_needed(df: pd.DataFrame, sheet_name: str = '') -> pd.DataFrame:
    """处理 Excel 顶部有标题行、导致列名变成 Unnamed:1/2/3 的情况。

    例如附件3被读成：
    ['服务站建设与运营成本', 'Unnamed:1', 'Unnamed:2', 'Unnamed:3']，
    真正的表头其实在下一行：['规模', '建设成本', '日固定成本', '日服务能力']。
    """
    cdf = df.copy()
    cdf.columns = [clean_text(c) for c in cdf.columns]

    cols_txt = '|'.join(cdf.columns)
    unnamed_cnt = sum((c == '') or c.lower().startswith('unnamed') for c in cdf.columns)
    col_hits = sum(1 for k in HEADER_KEYWORDS if k in cols_txt)

    # 如果当前列名已经像正常表头，就不动。
    if unnamed_cnt == 0 and col_hits >= 2:
        return cdf

    best_idx = None
    best_score = -1
    # 只在前若干行找真正表头，避免把数据行误当成表头。
    for idx in range(min(12, len(cdf))):
        vals = [clean_text(v) for v in cdf.iloc[idx].tolist()]
        nonempty = sum(v != '' and v.lower() != 'nan' for v in vals)
        row_txt = '|'.join(vals)
        hits = sum(1 for k in HEADER_KEYWORDS if k in row_txt)
        # 表头通常包含多个关键词，且非空单元格不止一个。
        score = hits * 10 + nonempty
        if hits >= 2 and nonempty >= 2 and score > best_score:
            best_idx = idx
            best_score = score

    if best_idx is None:
        return cdf

    raw_header = [clean_text(v) for v in cdf.iloc[best_idx].tolist()]
    new_header: list[str] = []
    seen: dict[str, int] = {}
    for j, h in enumerate(raw_header):
        if h == '' or h.lower() == 'nan':
            h = f'列{j+1}'
        if h in seen:
            seen[h] += 1
            h = f'{h}_{seen[h]}'
        else:
            seen[h] = 1
        new_header.append(h)

    out = cdf.iloc[best_idx + 1:].copy()
    out.columns = new_header
    out = out.dropna(how='all').reset_index(drop=True)
    print(f'INFO: sheet「{sheet_name}」检测到标题行/空列名，已将第 {best_idx + 2} 行提升为表头：{new_header}')
    return out



def parse_satisfaction_weights(a5s: dict[str, pd.DataFrame]) -> dict[str, float]:
    """从附件5“满意度评分规则”中解析综合满意度权重。

    只解析综合公式中的权重，例如：
        S = 0.2*S1 + 0.3*S2 + 0.5*S3

    注意：附件5后续评分规则中还会出现 S1=1.00、S2=1.00、S3=1.00，
    这些是单项满意度取值，不是综合权重，不能拿来归一化。
    """
    def norm_formula_text(x: Any) -> str:
        t = clean_text(x)
        t = (t.replace('＝', '=')
               .replace('＊', '*')
               .replace('×', '*')
               .replace('x', '*')
               .replace('X', '*')
               .replace('Ｓ', 'S')
               .replace('s', 'S'))
        # 兼容“0. 2”或“0。2”这类从 Excel 读出来的异常写法。
        t = t.replace('。', '.')
        t = re.sub(r'(\d)\.\s+(\d)', r'\1.\2', t)
        return t

    candidates: list[str] = []
    for _, df in a5s.items():
        for c in df.columns:
            candidates.append(norm_formula_text(c))
        for v in df.astype(str).values.ravel():
            candidates.append(norm_formula_text(v))

    # 优先只看包含完整综合公式的单元格/列名，避免误读后面的 S1=1.00 等评分规则。
    formula_candidates = [
        c for c in candidates
        if all(term in c.upper() for term in ['S1', 'S2', 'S3'])
        and ('S=' in c.upper() or '记为S' in c or '共同决定' in c or '核心因素' in c)
    ]
    # 如果公式被拆散，退一步把所有文本合在一起，但仍然只解析 S1/S2/S3 前面的系数。
    if not formula_candidates:
        formula_candidates = [''.join(candidates)]

    def coef_before(term: str, text: str) -> float | None:
        # 只接受 0.2*S1、0.2S1、20*S1 这种“系数在 S1/S2/S3 前”的综合公式写法。
        # 不接受 S1=1.00，因为那是单项评分规则，不是权重。
        m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*\*?\s*' + term + r'(?!\d)', text, flags=re.IGNORECASE)
        return float(m.group(1)) if m else None

    for cand in formula_candidates:
        text = cand.upper()
        w1 = coef_before('S1', text)
        w2 = coef_before('S2', text)
        w3 = coef_before('S3', text)
        if w1 is None or w2 is None or w3 is None:
            continue
        ws = np.array([w1, w2, w3], dtype=float)
        if not np.all(np.isfinite(ws)) or ws.sum() <= 0:
            continue
        # 如果表里写 20、30、50，则归一化；若写 0.2、0.3、0.5，保持原值。
        if ws.max() > 1:
            ws = ws / ws.sum()
        # 如果解析结果不是 1 附近，但又明确是公式，仍保留原系数；通常应为 0.2+0.3+0.5=1。
        weights = {'distance': float(ws[0]), 'response': float(ws[1]), 'price': float(ws[2])}
        print('INFO: 已从附件5综合公式解析满意度权重:', weights)
        return weights

    print('WARNING: 未从附件5综合公式解析到 0.2*S1+0.3*S2+0.5*S3，使用默认权重', DEFAULT_WEIGHTS)
    return DEFAULT_WEIGHTS.copy()

def read_workbook_sheets(path: Path) -> dict[str, pd.DataFrame]:
    if path.name.startswith('~$'):
        return {}
    sheets = pd.read_excel(path, sheet_name=None)
    cleaned = {}
    for name, df in sheets.items():
        cdf = promote_header_if_needed(df, sheet_name=name)
        cdf.columns = [clean_text(c) for c in cdf.columns]
        cleaned[name] = cdf
    return cleaned


def choose_sheet_by_keywords(sheets: dict[str, pd.DataFrame], keywords: list[str]) -> tuple[str, pd.DataFrame]:
    """按关键词选择工作表。

    v2 的问题是只在 sheet 名称和前几行数据里找关键词，
    没有搜索列名。附件3的 sheet 名是“服务站建设与运营成本”，
    真正的关键词“规模”在列名“站点规模”里，因此会误报找不到。
    """
    kws = [clean_text(k).lower() for k in keywords]

    # 1) 先匹配 sheet 名称
    for n, df in sheets.items():
        nn = clean_text(n).lower()
        if all(k in nn for k in kws):
            return n, df

    # 2) 再匹配“列名 + 前10行数据”
    for n, df in sheets.items():
        coltxt = '|'.join(clean_text(c).lower() for c in df.columns)
        datatxt = '|'.join(clean_text(v).lower() for v in df.head(10).astype(str).values.ravel())
        fulltxt = coltxt + '|' + datatxt
        if all(k in fulltxt for k in kws):
            return n, df

    # 3) 如果整个文件只有一个 sheet，直接使用它，避免附件3这类单表文件误停。
    if len(sheets) == 1:
        n, df = next(iter(sheets.items()))
        print(f'WARNING: 未严格匹配关键词 {keywords}，但文件只有一个sheet，已使用：{n}')
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


def pick_col_any(df: pd.DataFrame, candidates: list[tuple[list[str], list[str]]]) -> str:
    """按优先级识别列。candidates=[(必须包含关键词, 排除关键词), ...]"""
    last_error: Exception | None = None
    for includes, excludes in candidates:
        try:
            return pick_col(df, includes, excludes)
        except Exception as e:
            last_error = e
    raise KeyError(f'列识别失败，候选规则均未命中：{candidates}；最后错误：{last_error}')


def find_period_col(df: pd.DataFrame) -> str | None:
    """识别年份/期数列，用于从多期预测结果中筛出最后一年。"""
    strong_names = {'年份', '年度', '预测年份', '预测年度', '年序号', '第几年', '期数', '时间', '阶段'}
    for c in df.columns:
        cc = clean_text(c)
        if cc in strong_names:
            return c
    for c in df.columns:
        cc = clean_text(c)
        # 避免把“第5年末老人总数”“月实际需求”等指标列误判成年份列
        looks_like_period = re.search(r'(年份|年度|预测年|年序号|第几年|期数|阶段)', cc)
        looks_like_metric = any(k in cc for k in ['老人', '需求', '总数', '人数', '成本', '收入', '支出', '满意', '能力'])
        if looks_like_period and not looks_like_metric:
            return c
    return None


def filter_latest_period(df: pd.DataFrame, tag: str = '') -> pd.DataFrame:
    """若表中含年份/期数列，则只保留最后一期；否则原样返回。"""
    period_col = find_period_col(df)
    if period_col is None:
        return df.copy()
    tmp = df.copy()
    tmp['_period_num'] = tmp[period_col].apply(parse_number)
    valid = tmp.dropna(subset=['_period_num']).copy()
    if valid['_period_num'].nunique() <= 1:
        return tmp.drop(columns=['_period_num'])
    latest = valid['_period_num'].max()
    out = valid[valid['_period_num'] == latest].drop(columns=['_period_num'])
    print(f'INFO: {tag} 检测到多期数据，已按 {period_col}={latest:g} 筛选最后一期，行数 {len(df)} -> {len(out)}')
    return out


def collapse_metric_by_community(
    df: pd.DataFrame,
    community_col: str,
    value_col: str,
    value_name: str,
    agg: str = 'last'
) -> pd.DataFrame:
    """把重复小区记录压缩为一行，避免 10 个小区 × 多期 = 60 行。"""
    tmp = df[[community_col, value_col]].copy()
    tmp.columns = ['小区', value_name]
    tmp['小区'] = tmp['小区'].map(clean_text)
    tmp[value_name] = tmp[value_name].apply(parse_number)
    tmp = tmp.dropna(subset=[value_name])
    tmp = tmp[tmp['小区'] != ''].copy()

    if tmp['小区'].duplicated().any():
        before = len(tmp)
        if agg == 'sum':
            tmp = tmp.groupby('小区', as_index=False)[value_name].sum()
        elif agg == 'mean':
            tmp = tmp.groupby('小区', as_index=False)[value_name].mean()
        else:
            # 对“逐年预测表”最安全：取每个小区最后一条记录；若存在年份列，前面已筛最后一期
            tmp = tmp.groupby('小区', as_index=False)[value_name].last()
        print(f'INFO: {value_name} 存在重复小区记录，已按小区合并：{before} -> {len(tmp)}')

    return tmp


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



def clean_excel_cell(x: Any) -> Any:
    """去除 Excel XML 不允许的控制字符，避免生成的 xlsx 被 Excel 判定损坏。"""
    if isinstance(x, str):
        return re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', x)
    return x


def clean_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """写入 Excel 前做轻量清洗：去掉 inf 和非法控制字符。"""
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(clean_excel_cell)
    out.columns = [clean_excel_cell(c) for c in out.columns]
    return out


def write_df(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str, index: bool = False) -> None:
    """统一写表，确保 sheet 名合法且表格内容不会破坏 xlsx。"""
    invalid = r'[]:*?/\\'
    safe_name = ''.join('_' if ch in invalid else ch for ch in sheet_name)[:31]
    clean_excel_df(df).to_excel(writer, sheet_name=safe_name, index=index)


def write_validated_excel(output: Path, tables: list[tuple[str, pd.DataFrame, bool]]) -> Path:
    """先写临时 xlsx，验证可被 openpyxl 读取后再替换正式文件。

    这样可以避免运行中断、Excel 正打开文件、或写入一半时用户打开导致的损坏文件。
    """
    output = safe_output_path(output)
    tmp = output.with_name(f'{output.stem}__tmp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')

    with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
        for sheet_name, df, index in tables:
            write_df(writer, df, sheet_name, index=index)

    if not zipfile.is_zipfile(tmp):
        raise RuntimeError(f'Excel临时文件不是有效xlsx压缩包: {tmp}')

    try:
        import openpyxl
        wb = openpyxl.load_workbook(tmp, read_only=True, data_only=False)
        _ = wb.sheetnames
        wb.close()
    except Exception as e:
        raise RuntimeError(f'Excel临时文件生成后校验失败: {tmp}, {e}') from e

    # 如果目标文件被 Excel 打开，Windows 下可能无法删除/覆盖；自动改用带时间戳的新文件名。
    try:
        if output.exists():
            output.unlink()
    except PermissionError:
        output = output.with_name(f'{output.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}{output.suffix}')

    tmp.replace(output)
    print('INFO: Excel结果文件已通过有效性校验:', output)
    return output



def community_key(x: Any) -> str:
    """用于匹配“小区1 / 1小区 / 小区01 / A社区”等命名差异。"""
    s = clean_text(x)
    if s == '':
        return ''
    trans = str.maketrans('０１２３４５６７８９abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                          '0123456789abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz')
    s = s.translate(trans)
    s = s.replace('社区', '小区')
    s = re.sub(r'[\s_\-—－、,，:：;；（）()\[\]【】]', '', s)
    # 只要有“小区”和数字，就统一成“小区N”，避免“小区01”和“小区1”匹配不上。
    m = re.search(r'(\d+)', s)
    if '小区' in s and m:
        return f'小区{int(m.group(1))}'
    # 纯数字也按“小区N”处理，兼容距离矩阵行列名写成 1,2,...,10 的情况。
    if re.fullmatch(r'\d+(?:\.0)?', s):
        return f'小区{int(float(s))}'
    return s


def match_positions_to_communities(labels: list[Any], communities: list[str]) -> dict[str, int]:
    """把一组行/列标签匹配到题目中的10个小区，返回 {小区名: 位置}。"""
    key_to_comm: dict[str, str] = {}
    for c in communities:
        k = community_key(c)
        if k and k not in key_to_comm:
            key_to_comm[k] = c

    matched: dict[str, int] = {}
    used_pos: set[int] = set()

    # 1) 规范化完全匹配
    for pos, lab in enumerate(labels):
        k = community_key(lab)
        if k in key_to_comm and key_to_comm[k] not in matched:
            matched[key_to_comm[k]] = pos
            used_pos.add(pos)

    # 2) 唯一包含匹配，兼容“到小区1距离”“小区1(m)”这类标签
    for c in communities:
        if c in matched:
            continue
        ck = community_key(c)
        if not ck:
            continue
        cand = []
        for pos, lab in enumerate(labels):
            if pos in used_pos:
                continue
            lk = community_key(lab)
            if lk and (ck in lk or lk in ck):
                cand.append(pos)
        if len(cand) == 1:
            matched[c] = cand[0]
            used_pos.add(cand[0])

    return matched


def fill_symmetric_distance_matrix(dmat: pd.DataFrame) -> pd.DataFrame:
    """距离矩阵若只填了上三角/下三角，则用对称位置补齐。"""
    out = dmat.copy()
    for i in out.index:
        for j in out.columns:
            if i == j:
                out.loc[i, j] = 0.0
                continue
            a = out.loc[i, j]
            b = out.loc[j, i] if (j in out.index and i in out.columns) else np.nan
            if pd.isna(a) and not pd.isna(b):
                out.loc[i, j] = b
            elif not pd.isna(a) and pd.isna(b):
                out.loc[j, i] = a
    return out


def build_distance_matrix_from_labeled_df(df: pd.DataFrame, communities: list[str], sheet_name: str = '') -> pd.DataFrame | None:
    """处理已经有表头的距离矩阵：第一列为行小区，列名为列小区。"""
    if df.empty or len(df.columns) < 2:
        return None

    best: tuple[int, pd.DataFrame] | None = None
    for first_col in list(df.columns)[:min(5, len(df.columns))]:
        tmp = df.copy()
        row_labels = tmp[first_col].tolist()
        col_labels = list(tmp.columns)
        row_map = match_positions_to_communities(row_labels, communities)
        col_map = match_positions_to_communities(col_labels, communities)
        score = len(row_map) + len(col_map)
        if len(row_map) < max(3, len(communities) // 2) or len(col_map) < max(3, len(communities) // 2):
            continue

        mat = pd.DataFrame(np.nan, index=communities, columns=communities, dtype=float)
        for i in communities:
            if i not in row_map:
                continue
            rpos = row_map[i]
            for j in communities:
                if j not in col_map:
                    continue
                cpos = col_map[j]
                mat.loc[i, j] = parse_number(tmp.iloc[rpos, cpos])
        mat = fill_symmetric_distance_matrix(mat)
        if best is None or score > best[0]:
            best = (score, mat)

    return best[1] if best else None


def build_distance_matrix_from_raw_df(raw: pd.DataFrame, communities: list[str], sheet_name: str = '') -> pd.DataFrame | None:
    """处理带合并标题行/空白行的原始距离矩阵。"""
    raw = raw.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
    if raw.empty:
        return None

    best: tuple[int, pd.DataFrame, int, int] | None = None
    nrow, ncol = raw.shape
    # 在前15行、前6列内寻找“列小区标签行”和“行小区标签列”。
    for header_row in range(min(15, nrow)):
        header_vals = raw.iloc[header_row, :].tolist()
        col_map = match_positions_to_communities(header_vals, communities)
        if len(col_map) < max(3, len(communities) // 2):
            continue
        for label_col in range(min(6, ncol)):
            row_vals = raw.iloc[header_row + 1:, label_col].tolist()
            row_map_rel = match_positions_to_communities(row_vals, communities)
            if len(row_map_rel) < max(3, len(communities) // 2):
                continue
            row_map = {c: header_row + 1 + r for c, r in row_map_rel.items()}
            score = len(row_map) + len(col_map)
            mat = pd.DataFrame(np.nan, index=communities, columns=communities, dtype=float)
            for i in communities:
                if i not in row_map:
                    continue
                rpos = row_map[i]
                for j in communities:
                    if j not in col_map:
                        continue
                    cpos = col_map[j]
                    mat.loc[i, j] = parse_number(raw.iloc[rpos, cpos])
            mat = fill_symmetric_distance_matrix(mat)
            if best is None or score > best[0]:
                best = (score, mat, header_row, label_col)

    if best:
        _, mat, header_row, label_col = best
        print(f'INFO: 距离矩阵 sheet「{sheet_name}」已按原始表解析：列标签行={header_row + 1}，行标签列={label_col + 1}')
        return mat
    return None


def load_distance_matrix(path: Path, communities: list[str]) -> tuple[str, pd.DataFrame]:
    """稳健读取附件4距离矩阵，并输出缺失位置诊断。"""
    # 先用已有的自动表头逻辑读；失败再用 header=None 原始读取。
    parsed_sheets = read_workbook_sheets(path)
    candidates: list[tuple[str, pd.DataFrame]] = []
    for name, df in parsed_sheets.items():
        if '距离' in clean_text(name) or len(parsed_sheets) == 1:
            candidates.append((name, df))
    candidates += [(name, df) for name, df in parsed_sheets.items() if (name, df) not in candidates]

    best_name = ''
    best_mat: pd.DataFrame | None = None
    best_missing = 10**9
    for name, df in candidates:
        mat = build_distance_matrix_from_labeled_df(df, communities, sheet_name=name)
        if mat is None:
            continue
        missing = int(mat.isna().sum().sum())
        if missing < best_missing:
            best_name, best_mat, best_missing = name, mat, missing

    if best_mat is None or best_missing > 0:
        raw_sheets = pd.read_excel(path, sheet_name=None, header=None)
        for name, raw in raw_sheets.items():
            mat = build_distance_matrix_from_raw_df(raw, communities, sheet_name=name)
            if mat is None:
                continue
            missing = int(mat.isna().sum().sum())
            if missing < best_missing:
                best_name, best_mat, best_missing = name, mat, missing

    if best_mat is None:
        raise ValueError('无法解析附件4距离矩阵：没有找到同时包含小区行标签和列标签的矩阵。请检查附件4第一列和首行是否包含10个小区名称。')

    best_mat = fill_symmetric_distance_matrix(best_mat)
    for c in communities:
        best_mat.loc[c, c] = 0.0

    missing_pairs = []
    for i in communities:
        for j in communities:
            if i != j and pd.isna(best_mat.loc[i, j]):
                missing_pairs.append((i, j))

    if missing_pairs:
        print('ERROR: 距离矩阵仍有非对角缺失。前20个缺失位置如下：')
        for i, j in missing_pairs[:20]:
            print(f'  {i} -> {j}')
        print('INFO: 问题1结果中的小区名称:', communities)
        # 打印附件4可见行列标签，方便人工对照。
        for name, df in parsed_sheets.items():
            print(f'INFO: 附件4 sheet「{name}」列名预览:', list(df.columns)[:15])
            if len(df.columns) > 0:
                print(f'INFO: 附件4 sheet「{name}」第一列预览:', df.iloc[:15, 0].map(clean_text).tolist())
        raise ValueError(f'距离矩阵存在非对角缺失值：共 {len(missing_pairs)} 处。通常是附件4的小区名称与问题1输出的小区名称不一致，或附件4只有部分矩阵。')

    print(f'INFO: 距离矩阵读取成功，使用sheet「{best_name}」，规模={best_mat.shape[0]}×{best_mat.shape[1]}')
    return best_name, best_mat


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

    # 问题1结果中可能是“10个小区 × 6个年份/期数 = 60行”。
    # 问题2只需要第5年末/最后一期数据，因此必须先筛最后一期，再按小区压缩成10行。
    s02_latest = filter_latest_period(s02, tag='老人逐小区表')
    s08_latest = filter_latest_period(s08, tag='实际需求汇总表')
    s07_latest = filter_latest_period(s07, tag='实际需求类型表')

    comm_col = pick_col(s02_latest, ['小区'])
    p_col = pick_col_any(s02_latest, [
        (['第5', '老人', '总'], []),
        (['5', '老人', '总'], []),
        (['年末', '老人', '总'], []),
        (['老人', '总'], []),
        (['总'], [])
    ])
    df_pop = collapse_metric_by_community(
        s02_latest,
        community_col=comm_col,
        value_col=p_col,
        value_name='第5年末老人总数',
        agg='last'
    )

    comm2 = pick_col(s08_latest, ['小区'])
    q_col = pick_col_any(s08_latest, [
        (['第5', '实际', '合计'], []),
        (['实际', '合计'], []),
        (['月', '实际'], []),
        (['实际'], ['率'])
    ])
    df_dem = collapse_metric_by_community(
        s08_latest,
        community_col=comm2,
        value_col=q_col,
        value_name='月实际需求',
        agg='last'
    )

    base = df_pop.merge(df_dem, on='小区', how='inner')
    base['日均需求q'] = base['月实际需求'] / 30.0
    if len(base) != 10:
        raise ValueError(
            f'小区数量应为10，当前={len(base)}，唯一小区数={base["小区"].nunique()}。'
            f'老人表小区数={df_pop["小区"].nunique()}，需求表小区数={df_dem["小区"].nunique()}。'
            '请检查 B_problem1_results.xlsx 中小区名称是否一致，或是否混入了非小区汇总行。'
        )

    a3s = read_workbook_sheets(a3)
    s3n, s3 = choose_sheet_by_keywords(a3s, ['规模'])
    size_col = pick_col(s3, ['规模'])
    cost_col = pick_col_any(s3, [
        (['建设', '成本'], []),
        (['建设'], []),
        (['成本'], ['固定', '运营', '管理'])
    ])
    fixed_col = pick_col_any(s3, [
        (['日', '固定'], []),
        (['固定', '成本'], []),
        (['运营', '成本'], []),
        (['管理', '成本'], []),
        (['日'], ['能力', '服务能力'])
    ])
    cap_col = pick_col_any(s3, [
        (['日', '服务', '能力'], []),
        (['服务', '能力'], []),
        (['最大', '服务', '人次'], []),
        (['服务', '人次'], []),
        (['人次'], ['成本', '固定', '运营', '管理']),
        (['能力'], ['成本', '固定', '运营', '管理'])
    ])
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

    communities = base['小区'].tolist()

    # 附件4经常存在三类格式差异：
    # 1) 顶部有合并标题行，真正表头不在第1行；
    # 2) 小区名写法与问题1结果不完全一致，如“小区01/小区1/1小区”；
    # 3) 只给出上三角或下三角距离。
    # 因此这里不再简单 set_index + reindex，而是用稳健解析函数，并自动做名称归一化和对称补齐。
    s4n, dmat = load_distance_matrix(a4, communities)

    a5s = read_workbook_sheets(a5)
    weights = parse_satisfaction_weights(a5s)

    # 收入/支出参数
    p1_comm = pick_col(s07_latest, ['小区'])
    p1_type = pick_col(s07_latest, ['服务'])
    p1_qty = pick_col_any(s07_latest, [
        (['第5', '实际'], []),
        (['月', '实际'], []),
        (['实际'], ['率'])
    ])
    demand_type = s07_latest[[p1_comm, p1_type, p1_qty]].copy()
    demand_type.columns = ['小区', '服务项目', '月需求']
    demand_type['小区'] = demand_type['小区'].map(clean_text)
    demand_type['服务项目'] = demand_type['服务项目'].map(clean_text)
    demand_type['月需求'] = demand_type['月需求'].apply(parse_number)
    demand_type = demand_type.dropna(subset=['月需求'])
    demand_type = demand_type[(demand_type['小区'] != '') & (demand_type['服务项目'] != '')]
    # 若同一小区-服务项目仍有重复记录，则合并，避免收入成本重复计算
    demand_type = demand_type.groupby(['小区', '服务项目'], as_index=False)['月需求'].sum()

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
        # 用附件4距离矩阵做二维近似布局。
        # 旧版代码把所有小区的 y 坐标都写成 0，所以图会被画成一条直线；
        # 这里使用经典 MDS 将 10×10 距离矩阵投影到二维平面，便于展示覆盖关系。
        def classical_mds_positions(distance_df: pd.DataFrame, labels: list[str]) -> dict[str, tuple[float, float]]:
            D = distance_df.reindex(index=labels, columns=labels).astype(float).to_numpy()
            D = np.nan_to_num(D, nan=0.0)
            D = (D + D.T) / 2.0
            np.fill_diagonal(D, 0.0)

            n = D.shape[0]
            J = np.eye(n) - np.ones((n, n)) / n
            B = -0.5 * J @ (D ** 2) @ J
            eigvals, eigvecs = np.linalg.eigh(B)
            idx = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            coords = np.zeros((n, 2), dtype=float)
            positive_dims = [k for k, v in enumerate(eigvals) if v > 1e-9][:2]
            for out_dim, eig_idx in enumerate(positive_dims):
                coords[:, out_dim] = eigvecs[:, eig_idx] * np.sqrt(eigvals[eig_idx])

            # 极端情况下若距离矩阵只能形成一维结构，则给第二维一个很小的错位，避免标签完全压在一起。
            if len(positive_dims) < 2 or np.ptp(coords[:, 1]) < 1e-9:
                coords[:, 1] = np.linspace(-0.3, 0.3, n)

            # 归一化到便于绘图的尺度。
            for k in range(2):
                span = np.ptp(coords[:, k])
                if span > 1e-9:
                    coords[:, k] = (coords[:, k] - coords[:, k].mean()) / span

            return {lab: (float(coords[i, 0]), float(coords[i, 1])) for i, lab in enumerate(labels)}

        pos = classical_mds_positions(dmat, communities)
        plt.figure(figsize=(8, 6))
        for i, j in assign.items():
            if j:
                xi, yi = pos[i]
                xj, yj = pos[j]
                plt.plot([xi, xj], [yi, yj], color='0.65', linewidth=1.2, alpha=0.75, zorder=1)

        for c, (x, y) in pos.items():
            if c in built:
                plt.scatter(x, y, s=90, marker='*', color='tab:blue', label='建站小区' if '建站小区' not in plt.gca().get_legend_handles_labels()[1] else '', zorder=3)
                plt.text(x, y + 0.045, f'{c}\n{built[c]}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            elif assign.get(c) is not None:
                plt.scatter(x, y, s=45, marker='o', color='tab:green', label='被覆盖小区' if '被覆盖小区' not in plt.gca().get_legend_handles_labels()[1] else '', zorder=2)
                plt.text(x, y + 0.035, c, ha='center', va='bottom', fontsize=8)
            else:
                plt.scatter(x, y, s=45, marker='o', color='0.55', label='未覆盖小区' if '未覆盖小区' not in plt.gca().get_legend_handles_labels()[1] else '', zorder=2)
                plt.text(x, y + 0.035, c, ha='center', va='bottom', fontsize=8)

        plt.title('最优站点覆盖关系图（基于距离矩阵的MDS近似布局）')
        plt.legend(loc='best', fontsize=8, frameon=False)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(charts / '01_最优站点覆盖关系图.png', dpi=180)
        plt.close()
    else:
        print('WARNING: matplotlib 不可用，已跳过图表生成。')

    data_check_df = pd.DataFrame([
        {'文件': 'B_problem1_results.xlsx', 'sheet总数': len(p1s), '使用sheet': f'{s02n},{s07n},{s08n}'},
        {'文件': '附件2', 'sheet总数': len(a2s), '使用sheet': s2pn},
        {'文件': '附件3', 'sheet总数': len(a3s), '使用sheet': s3n},
        {'文件': '附件4', 'sheet总数': '自动解析', '使用sheet': s4n},
        {'文件': '附件5', 'sheet总数': len(a5s), '使用sheet': '满意度评分规则'}
    ])
    output = write_validated_excel(Path('B_problem2_results.xlsx'), [
        ('01_数据读取检查', data_check_df, False),
        ('02_问题1输入数据', base, False),
        ('03_服务站规模参数', scale_df[['规模', '建设成本', '日固定成本', '日服务能力']], False),
        ('04_距离矩阵', dmat.reset_index(), False),
        ('05_最优选址规模方案', pd.DataFrame(station_rows), False),
        ('06_小区分配结果', alloc_df, False),
        ('07_服务站覆盖明细', pd.DataFrame(cover_rows), False),
        ('08_年度利润测算', pd.DataFrame(profit_rows), False),
        ('09_总体指标', overall, False),
        ('10_候选方案Top20', top20, False),
    ])

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