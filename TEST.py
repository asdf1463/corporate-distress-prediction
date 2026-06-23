import os
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = 'dart_finance.db'
TABLE_NAME = 'FINANCE_FEATURES'

ID_COLS = ['corp_code', 'corp_name', 'bsns_year', 'target_label', 'target_name']

RAW_COLS = [
    'total_assets', 'total_liab', 'total_equity', 'current_assets', 'current_liab',
    'revenue', 'operating_profit', 'net_income', 'interest_expense',
    'operating_cash_flow', 'retained_earnings'
]

MODEL_FEATURES = [
    'roa',
    'signed_log_ocf',
    'asset_turnover',
    'debt_to_assets',
    'log_assets'
]

REFERENCE_RATIOS = [
    'debt_ratio', 'equity_ratio', 'current_ratio', 'op_margin', 'net_margin',
    'roe', 'interest_coverage', 'interest_coverage_raw', 'ocf_to_assets',
    'ocf_to_debt', 'retained_earnings_ratio', 'working_capital_ratio',
    'operating_profit_to_assets'
]

CHECK_COLS = MODEL_FEATURES + REFERENCE_RATIOS
DESCRIBE_PERCENTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
IQR_K = 3.0
IQR_MIN_COUNT = 20
INVALID_PREVIEW_ROWS = 50
OUTLIER_PREVIEW_ROWS = 10


# ==========================================
# 1. 공통 유틸
# ==========================================
def configure_display():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)


def existing(df, cols):
    return [col for col in cols if col in df.columns]


def print_section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


def load_table(db_path, table_name):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"'{db_path}' 파일이 없습니다.")

    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)['name'].tolist()
        if table_name not in tables:
            raise ValueError(f"'{table_name}' 테이블이 없습니다. 현재 테이블: {tables}")
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def coerce_numeric_columns(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ==========================================
# 2. 리포트 구성 함수
# ==========================================
def print_overview(df):
    print_section(f"📊 [{TABLE_NAME} 데이터 건강 검진 리포트]")
    print(f"전체 행 수: {len(df)}건")

    if 'target' in df.columns:
        print('\n[정상 vs 부실 분포]')
        print(df['target'].value_counts(dropna=False).rename(index={0: '정상', 1: '부실'}).to_string())
    elif 'target_label' in df.columns:
        print('\n[target_label 분포]')
        print(df['target_label'].value_counts(dropna=False).sort_index().to_string())

    if {'bsns_year', 'target'}.issubset(df.columns):
        print('\n[연도 x 정상/부실 분포]')
        print(pd.crosstab(df['bsns_year'], df['target']).rename(columns={0: '정상', 1: '부실'}).to_string())


def print_missing_report(df):
    print_section('✅ 1. 결측치 확인')
    missing = pd.DataFrame({
        'missing_count': df.isna().sum(),
        'missing_rate_pct': (df.isna().mean() * 100).round(2)
    }).query('missing_count > 0').sort_values('missing_rate_pct', ascending=False)

    print(missing.to_string() if not missing.empty else '결측치 없음')


def build_invalid_records(df, id_cols, raw_cols):
    rules = {
        '자산총계 0 이하': df['total_assets'] <= 0 if 'total_assets' in df else False,
        '매출액 0 이하': df['revenue'] <= 0 if 'revenue' in df else False,
        '부채총계 음수': df['total_liab'] < 0 if 'total_liab' in df else False,
        '유동자산 음수': df['current_assets'] < 0 if 'current_assets' in df else False,
        '유동부채 음수': df['current_liab'] < 0 if 'current_liab' in df else False,
    }

    invalid_list = []
    report_cols = existing(df, id_cols + raw_cols)

    for issue, mask in rules.items():
        if isinstance(mask, bool):
            continue

        tmp = df.loc[mask.fillna(False), report_cols].copy()
        if not tmp.empty:
            tmp['issue'] = issue
            invalid_list.append(tmp)

    return pd.concat(invalid_list, ignore_index=True) if invalid_list else pd.DataFrame()


def print_invalid_report(df, id_cols, raw_cols):
    print_section('✅ 2. 기본 이상 데이터 스캔')
    invalid = build_invalid_records(df, id_cols, raw_cols)

    print(f"기본 이상 데이터: {len(invalid)}건")
    if not invalid.empty:
        print(invalid.head(INVALID_PREVIEW_ROWS).to_string(index=False))


def print_describe_report(df, cols, title, empty_message):
    print_section(title)
    if cols:
        print(df[cols].describe(percentiles=DESCRIBE_PERCENTILES).T.round(4).to_string())
    else:
        print(empty_message)


def detect_iqr_outliers(df, cols, id_cols, k=IQR_K, min_count=IQR_MIN_COUNT):
    rows = []

    for col in cols:
        s = df[col].dropna()
        if len(s) < min_count or s.nunique() <= 1:
            continue

        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue

        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask = (df[col] < lo) | (df[col] > hi)

        if mask.any():
            tmp = df.loc[mask, id_cols + [col]].copy()
            tmp['variable'] = col
            tmp['value'] = tmp[col]
            tmp['lower_bound'] = lo
            tmp['upper_bound'] = hi
            tmp['direction'] = np.where(tmp[col] > hi, '상단 이상치', '하단 이상치')
            tmp.drop(columns=[col], inplace=True)
            rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def print_outlier_report(df, check_cols, id_cols):
    print_section('✅ 5. IQR 기준 이상치 탐지')
    outliers = detect_iqr_outliers(df, check_cols, id_cols)

    if outliers.empty:
        print('IQR 기준 극단 이상치 없음')
        return

    print(f"IQR 기준 극단 이상치: {len(outliers)}건")
    print('\n[변수별 이상치 개수]')
    print(outliers['variable'].value_counts().to_string())
    print('\n[이상치 목록 상위 10건]')
    print(outliers.head(OUTLIER_PREVIEW_ROWS).to_string(index=False))


# ==========================================
# 3. 실행
# ==========================================
def main():
    configure_display()

    df = load_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise SystemExit(f"🚨 {TABLE_NAME} 테이블에 데이터가 없습니다.")

    id_cols = existing(df, ID_COLS)
    raw_cols = existing(df, RAW_COLS)
    model_features = existing(df, MODEL_FEATURES)
    reference_ratios = existing(df, REFERENCE_RATIOS)
    check_cols = existing(df, CHECK_COLS)

    numeric_cols = list(dict.fromkeys(raw_cols + check_cols + ['bsns_year', 'target_label', 'target']))
    df = coerce_numeric_columns(df, numeric_cols)

    print_overview(df)
    print_missing_report(df)
    print_invalid_report(df, id_cols, raw_cols)
    print_describe_report(df, model_features, '✅ 3. 최종 모델 Feature 요약', '모델 feature 컬럼 없음')
    print_describe_report(df, reference_ratios, '✅ 4. 참고 재무비율 요약', '참고 재무비율 컬럼 없음')
    print_outlier_report(df, check_cols, id_cols)

    print('\n✅ 데이터 건강 검진 완료')


if __name__ == '__main__':
    main()
