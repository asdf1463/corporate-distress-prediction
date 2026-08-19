import os
import sqlite3

import numpy as np
import pandas as pd

# ==========================================
# 0. 설정
# ==========================================
DB_PATH = 'dart_finance.db'
SOURCE_TABLE = 'FINANCE'
FEATURE_TABLE = 'FINANCE_FEATURES'

MODEL_FEATURES = [
    'roa',             # 수익성: 순이익 / 총자산
    'signed_log_ocf',  # 현금흐름: 부호 유지 로그 변환 영업현금흐름
    'asset_turnover',  # 활동성: 매출액 / 총자산
    'equity_ratio',    # 안정성: 자기자본 / 총자산
    'log_assets'       # 규모: ln(총자산)
]

REFERENCE_RATIOS = [
    'debt_ratio',
    'debt_to_assets',
    'current_ratio',
    'op_margin',
    'net_margin',
    'roe',
    'interest_coverage',
    'interest_coverage_raw',
    'ocf_to_assets',
    'ocf_to_debt',
    'retained_earnings_ratio',
    'working_capital_ratio',
    'operating_profit_to_assets'
]

RAW_COLS = [
    'corp_code',
    'corp_name',
    'bsns_year',
    'target_label',
    'total_assets',
    'total_liab',
    'total_equity',
    'current_assets',
    'current_liab',
    'revenue',
    'operating_profit',
    'net_income',
    'interest_expense',
    'operating_cash_flow',
    'retained_earnings'
]

NUMERIC_COLS = [
    'bsns_year',
    'target_label',
    'total_assets',
    'total_liab',
    'total_equity',
    'current_assets',
    'current_liab',
    'revenue',
    'operating_profit',
    'net_income',
    'interest_expense',
    'operating_cash_flow',
    'retained_earnings'
]

BASE_OUTPUT_COLS = [
    'corp_code',
    'corp_name',
    'bsns_year',
    'target_label',
    'target',
    'target_name'
]


# ==========================================
# 1. 유틸 함수
# ==========================================
def safe_div(num, den, scale=1.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = (num / den) * scale

    invalid = num.isna() | den.isna() | (den == 0)
    return result.mask(invalid, np.nan).replace([np.inf, -np.inf], np.nan)


def validate_source_columns(df):
    missing = [col for col in RAW_COLS if col not in df.columns]

    if missing:
        raise ValueError(f"{SOURCE_TABLE} 테이블에 필수 원천 컬럼이 없습니다: {missing}")


def coerce_numeric_columns(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def unique_existing_columns(df, cols):
    return list(dict.fromkeys([col for col in cols if col in df.columns]))


# ==========================================
# 2. 원천 데이터 정제
# ==========================================
def clean_source_dataframe(df_raw):
    validate_source_columns(df_raw)

    df = coerce_numeric_columns(df_raw, NUMERIC_COLS)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 기존 설계 유지: 자산총계와 매출액이 양수인 표본만 사용
    df = df[
        (df['total_assets'] > 0) &
        (df['revenue'] > 0)
    ].copy()

    df = df[df['target_label'].isin([0, 1, 2])].copy()
    df = df.dropna(subset=['bsns_year']).copy()
    df['bsns_year'] = df['bsns_year'].astype(int)

    df['target'] = df['target_label'].isin([1, 2]).astype(int)
    df['target_name'] = np.where(df['target'] == 1, '부실', '정상')

    return df


# ==========================================
# 3. 최종 모델용 핵심 변수
# ==========================================
def add_model_features(df):
    df = df.copy()

    df['roa'] = safe_div(df['net_income'], df['total_assets'], scale=100)
    df['signed_log_ocf'] = (
        np.sign(df['operating_cash_flow']) *
        np.log1p(np.abs(df['operating_cash_flow']))
    )
    df['asset_turnover'] = safe_div(df['revenue'], df['total_assets'])
    df['equity_ratio'] = safe_div(df['total_equity'], df['total_assets'], scale=100)
    df['log_assets'] = np.log(df['total_assets'])

    return df


# ==========================================
# 4. 참고/확장 분석용 일반 재무비율
# ==========================================
def add_reference_ratios(df):
    df = df.copy()

    df['debt_ratio'] = safe_div(df['total_liab'], df['total_equity'], scale=100)
    df['debt_to_assets'] = safe_div(df['total_liab'], df['total_assets'])
    df['current_ratio'] = safe_div(df['current_assets'], df['current_liab'], scale=100)
    df['op_margin'] = safe_div(df['operating_profit'], df['revenue'], scale=100)
    df['net_margin'] = safe_div(df['net_income'], df['revenue'], scale=100)
    df['roe'] = safe_div(df['net_income'], df['total_equity'], scale=100)

    df['interest_coverage_raw'] = safe_div(
        df['operating_profit'],
        df['interest_expense']
    )
    df['interest_coverage'] = df['interest_coverage_raw'].clip(lower=-20, upper=20)

    zero_interest = (df['interest_expense'] == 0) & df['operating_profit'].notna()
    df.loc[zero_interest & (df['operating_profit'] > 0), 'interest_coverage'] = 20.0
    df.loc[zero_interest & (df['operating_profit'] <= 0), 'interest_coverage'] = -20.0

    df['ocf_to_assets'] = safe_div(df['operating_cash_flow'], df['total_assets'])
    df['ocf_to_debt'] = safe_div(df['operating_cash_flow'], df['total_liab'], scale=100)
    df['retained_earnings_ratio'] = safe_div(
        df['retained_earnings'],
        df['total_assets'],
        scale=100
    )

    working_capital = df['current_assets'] - df['current_liab']
    df['working_capital_ratio'] = safe_div(working_capital, df['total_assets'], scale=100)

    df['operating_profit_to_assets'] = safe_div(
        df['operating_profit'],
        df['total_assets'],
        scale=100
    )

    return df


# ==========================================
# 5. Feature 데이터프레임 생성
# ==========================================
def build_feature_dataframe(df_raw):
    df = clean_source_dataframe(df_raw)
    df = add_model_features(df)
    df = add_reference_ratios(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    save_cols = unique_existing_columns(
        df,
        BASE_OUTPUT_COLS + RAW_COLS + MODEL_FEATURES + REFERENCE_RATIOS
    )

    return df[save_cols].copy()


# ==========================================
# 6. 실행
# ==========================================
def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"'{DB_PATH}' 파일이 없습니다. DART_API.py를 먼저 실행하세요.")

    with sqlite3.connect(DB_PATH) as conn:
        df_raw = pd.read_sql(f"SELECT * FROM {SOURCE_TABLE}", conn)

        if df_raw.empty:
            raise ValueError(f"{SOURCE_TABLE} 테이블에 데이터가 없습니다. DART_API.py를 먼저 완료하세요.")

        df_features = build_feature_dataframe(df_raw)
        df_features.to_sql(
            FEATURE_TABLE,
            con=conn,
            if_exists='replace',
            index=False
        )
        conn.commit()

    print(f"✅ {FEATURE_TABLE} 테이블 생성 완료: {len(df_features)}건 저장")


if __name__ == '__main__':
    main()
