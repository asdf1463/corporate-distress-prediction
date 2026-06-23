import sqlite3

import pandas as pd

# =====================================================================
# 0. 설정
# =====================================================================
FILE_PATH = 'KIND DATA.xlsx'
DB_PATH = 'dart_finance.db'
TABLE_NAME = 'KRX_TARGET_COMPANY'

COL_MAPPING = {
    '번호': 'id',
    '종목코드': 'stock_code',
    '회사명': 'corp_name',
    '시간': 'event_date',
    '공시제목': 'report_title'
}

SHEET_RULES = [
    {
        'sheet_name': '관리종목',
        'pattern': '자본잠식|영업손실|계속사업손실|매출액|회생|파산|부적정|의견거절|범위제한|상장폐지사유',
        'target_label': 1,
        'exclude': False
    },
    {
        'sheet_name': '환기종목',
        'pattern': '손실|잠식|계속기업',
        'target_label': 1,
        'exclude': False
    },
    {
        'sheet_name': '상장폐지',
        'pattern': '합병|스팩|피흡수|자회사|유가증권|주식교환|우선주',
        'target_label': 2,
        'exclude': True
    }
]

REQUIRED_COLS = ['stock_code', 'corp_name', 'event_date', 'report_title']


# =====================================================================
# 1. 로드 및 검증
# =====================================================================
def load_kind_sheet(file_path, sheet_name):
    df = pd.read_excel(file_path, sheet_name=sheet_name, dtype={'종목코드': str})
    df = df.rename(columns=COL_MAPPING)
    validate_columns(df, REQUIRED_COLS, sheet_name)
    return df


def validate_columns(df, required_cols, sheet_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"'{sheet_name}' 시트에 필수 컬럼이 없습니다: {missing}")


# =====================================================================
# 2. 시트별 필터링
# =====================================================================
def filter_target_events(df, pattern, target_label, exclude=False):
    mask = df['report_title'].str.contains(pattern, na=False)
    if exclude:
        mask = ~mask

    filtered = df.loc[mask].copy()
    filtered['target_label'] = target_label
    return filtered


def build_master_dataframe(file_path):
    cleaned_frames = []

    for rule in SHEET_RULES:
        df_sheet = load_kind_sheet(file_path, rule['sheet_name'])
        df_clean = filter_target_events(
            df_sheet,
            pattern=rule['pattern'],
            target_label=rule['target_label'],
            exclude=rule['exclude']
        )
        cleaned_frames.append(df_clean)

    df_master = pd.concat(cleaned_frames, ignore_index=True)
    print(f"시트 1차 정제 및 통합 완료 (미정렬 건수: {len(df_master)}건)")

    return preprocess_master_dataframe(df_master)


# =====================================================================
# 3. 날짜, 사업연도, 중복 처리
# =====================================================================
def preprocess_master_dataframe(df_master):
    df_master = df_master.copy()

    df_master['event_date'] = pd.to_datetime(
        df_master['event_date'].astype(str).str[:10],
        errors='coerce'
    )

    df_master['target_bsns_year'] = (
        (df_master['event_date'] - pd.DateOffset(months=3)).dt.year - 1
    ).astype(str)

    df_master = df_master.sort_values(by='event_date', ascending=True)
    df_master['stock_code'] = df_master['stock_code'].astype(str).str.zfill(6)

    return df_master.drop_duplicates(subset=['stock_code'], keep='first')


# =====================================================================
# 4. DB 저장
# =====================================================================
def save_to_db(df, db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
        conn.commit()


# =====================================================================
# 5. 실행
# =====================================================================
def main():
    print("데이터 로딩을 시작합니다...")
    df_final_target = build_master_dataframe(FILE_PATH)
    save_to_db(df_final_target, DB_PATH, TABLE_NAME)

    print("\n==================================================")
    print("✅ 전처리 완료! (한글 변수명 -> 영어 변수명 매핑 적용)")
    print(f"✅ 순수 부실기업(Target) 수: {len(df_final_target)}개")
    print(f"✅ DB 저장 완료: '{DB_PATH}' ➡️ '{TABLE_NAME}' 테이블")
    print("==================================================")


if __name__ == '__main__':
    main()
