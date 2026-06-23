import io
import os
import random
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

# ==========================================
# 0. 설정 및 상수
# ==========================================
CONFIG = {
    'API_KEY': os.getenv('DART_API_KEY'),
    'REPRT_CODE': '11011',
    'DB_NAME': 'dart_finance.db',
    'TARGET_NORMAL_COUNT': 600,
    'RANDOM_SEED': 42,

    'USE_NORMAL_SNAPSHOT': True,
    'NORMAL_SNAPSHOT_PATH': 'normal_sample_snapshot.csv',
    'OVERWRITE_NORMAL_SNAPSHOT': False
}

if not CONFIG['API_KEY']:
    raise ValueError(
        'DART_API_KEY 환경변수가 설정되어 있지 않습니다. '
        'PowerShell에서 $env:DART_API_KEY="본인_API_KEY" 설정 후 실행하세요.'
    )

DART_CORP_CODE_URL = 'https://opendart.fss.or.kr/api/corpCode.xml'
DART_DISCLOSURE_LIST_URL = 'https://opendart.fss.or.kr/api/list.json'
DART_FINANCIAL_URL = 'https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json'
FINANCE_TABLE = 'FINANCE'
NORMAL_SAMPLE_SIZE = 2000
NORMAL_YEAR_RANGE = list(range(2015, 2026))
REQUEST_SLEEP_SECONDS = 0.1
NORMAL_SNAPSHOT_COLUMNS = ['corp_code', 'corp_name', 'bsns_year', 'target_label']

rng = random.Random(CONFIG['RANDOM_SEED'])

ACCOUNT_ALIASES = {
    'total_assets': ['자산총계'],
    'total_liab': ['부채총계'],
    'total_equity': ['자본총계'],
    'current_assets': ['유동자산'],
    'current_liab': ['유동부채'],
    'revenue': ['매출액', '수익', '영업수익', '매출', '순영업수익'],
    'operating_profit': ['영업이익', '영업손익'],
    'net_income': ['당기순이익', '당기순손익', '연결당기순이익', '연결당기순손익', '반기순이익', '분기순이익'],
    'interest_expense': ['이자비용', '금융원가', '금융비용', '이자의지급', '이자지급', '지급이자', '이자지급액'],
    'operating_cash_flow': ['영업활동현금흐름', '영업활동으로인한현금흐름', '영업활동순현금흐름'],
    'retained_earnings': ['이익잉여금', '미처분이익잉여금', '결손금', '미처리결손금']
}

ACCOUNT_NAME_TO_FIELD = {
    alias: field
    for field, aliases in ACCOUNT_ALIASES.items()
    for alias in aliases
}

FIELD_SJ_FILTER = {
    'total_assets': ['BS'],
    'total_liab': ['BS'],
    'total_equity': ['BS'],
    'current_assets': ['BS'],
    'current_liab': ['BS'],
    'retained_earnings': ['BS', 'SCE'],
    'revenue': ['IS', 'CIS'],
    'operating_profit': ['IS', 'CIS'],
    'net_income': ['IS', 'CIS'],
    'interest_expense': ['IS', 'CIS', 'CF'],
    'operating_cash_flow': ['CF']
}

SAFE_KEYWORDS = [
    '자진상장폐지', '자진 상장폐지', '피흡수합병',
    '주식의포괄적교환', '주식의 포괄적 교환', '주식교환ㆍ이전', '유가증권시장 상장'
]

REQUIRED_FINANCE_FIELDS = [
    'total_assets', 'total_liab', 'total_equity',
    'revenue', 'operating_profit', 'net_income'
]

FINANCE_SCHEMA_SQL = f'''
CREATE TABLE IF NOT EXISTS {FINANCE_TABLE} (
    fin_id INTEGER PRIMARY KEY AUTOINCREMENT,
    corp_code VARCHAR(8),
    corp_name VARCHAR(100),
    bsns_year VARCHAR(4),
    total_assets BIGINT,
    total_liab BIGINT,
    total_equity BIGINT,
    current_assets BIGINT,
    current_liab BIGINT,
    revenue BIGINT,
    operating_profit BIGINT,
    net_income BIGINT,
    interest_expense BIGINT,
    operating_cash_flow BIGINT,
    retained_earnings BIGINT,
    target_label INTEGER
)
'''


# ==========================================
# 1. 유틸리티 함수
# ==========================================
def norm_stock_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s[:-2].zfill(6) if s.endswith('.0') else s.zfill(6)


def parse_amount(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(',', '')
    if s in ['', '-', 'None', 'nan']:
        return None
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return int(s)
    except ValueError:
        return None


def get_xml_text(parent, tag_name, default=''):
    child = parent.find(tag_name)
    if child is None or not child.text:
        return default
    return child.text.strip()


def clean_account_name(account_name):
    clean_nm = re.sub(r'\(.*?\)', '', str(account_name or ''))
    clean_nm = re.sub(r'^[^가-힣]+', '', clean_nm)
    return clean_nm.replace(' ', '')


def table_exists(conn, table_name):
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?;"
    return conn.execute(query, (table_name,)).fetchone() is not None


def normalize_corp_code(x):
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s.zfill(8)


def load_normal_snapshot(snapshot_path):
    df_snapshot = pd.read_csv(
        snapshot_path,
        dtype={
            'corp_code': str,
            'corp_name': str,
            'bsns_year': str
        }
    )

    missing = [col for col in NORMAL_SNAPSHOT_COLUMNS if col not in df_snapshot.columns]
    if missing:
        raise ValueError(f"정상군 snapshot 파일에 필수 컬럼이 없습니다: {missing}")

    df_snapshot = df_snapshot[NORMAL_SNAPSHOT_COLUMNS].copy()
    df_snapshot['corp_code'] = df_snapshot['corp_code'].apply(normalize_corp_code)
    df_snapshot['corp_name'] = df_snapshot['corp_name'].fillna('').astype(str).str.strip()
    df_snapshot['bsns_year'] = (
        pd.to_numeric(df_snapshot['bsns_year'], errors='coerce')
        .astype('Int64')
        .astype(str)
    )
    df_snapshot['target_label'] = pd.to_numeric(
        df_snapshot['target_label'],
        errors='coerce'
    ).astype('Int64')

    invalid_year = df_snapshot['bsns_year'].eq('<NA>')
    if invalid_year.any():
        raise ValueError(f"정상군 snapshot 파일에 사업연도 오류가 있습니다: {invalid_year.sum()}건")

    invalid_label = df_snapshot['target_label'].ne(0) | df_snapshot['target_label'].isna()
    if invalid_label.any():
        raise ValueError(
            f"정상군 snapshot의 target_label은 모두 0이어야 합니다: 오류 {invalid_label.sum()}건"
        )

    df_snapshot['target_label'] = df_snapshot['target_label'].astype(int)

    normal_list = df_snapshot.to_dict('records')
    print(f"   📌 저장된 정상군 snapshot 사용: {snapshot_path} ({len(normal_list)}건)")
    return normal_list


def save_normal_snapshot(normal_list, snapshot_path):
    snapshot_dir = os.path.dirname(snapshot_path)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    df_snapshot = pd.DataFrame(normal_list, columns=NORMAL_SNAPSHOT_COLUMNS)
    df_snapshot.to_csv(snapshot_path, index=False, encoding='utf-8-sig')
    print(f"   💾 정상군 snapshot 저장 완료: {snapshot_path} ({len(df_snapshot)}건)")


# ==========================================
# 2. DART 마스터 및 표본 준비
# ==========================================
def fetch_dart_master():
    print('1. DART 상장사 고유번호 마스터 데이터 다운로드 중...')

    res = requests.get(
        DART_CORP_CODE_URL,
        params={'crtfc_key': CONFIG['API_KEY']},
        timeout=20
    )

    if res.status_code != 200:
        print(f"\n🚨 DART 마스터 다운로드 실패: HTTP {res.status_code}")
        sys.exit()

    zip_buffer = io.BytesIO(res.content)
    if not zipfile.is_zipfile(zip_buffer):
        print('\n🚨 정상적인 ZIP 파일이 아닙니다. API 키를 확인하거나 일일 트래픽 한도를 초과했는지 확인하세요.')
        print(f"👉 DART 서버 응답: {res.text[:300]}")
        sys.exit()

    corps = []
    try:
        zip_buffer.seek(0)
        with zipfile.ZipFile(zip_buffer) as z:
            with z.open('CORPCODE.xml') as f:
                root = ET.parse(f).getroot()

        for list_tag in root.findall('list'):
            stock_code = get_xml_text(list_tag, 'stock_code')
            if not stock_code:
                continue

            corp_code = get_xml_text(list_tag, 'corp_code')
            corps.append({
                'corp_code': corp_code.zfill(8) if corp_code else '',
                'corp_name': get_xml_text(list_tag, 'corp_name'),
                'stock_code': stock_code.zfill(6)
            })

        df_dart = pd.DataFrame(corps)
        print(f"✅ 상장사 총 {len(df_dart)}개 확인 완료.")
        return df_dart

    except zipfile.BadZipFile:
        print('\n🚨 정상적인 ZIP 파일을 받지 못했습니다. API 키를 확인하세요.')
        sys.exit()


def check_voluntary_delisting(api_key, corp_code, corp_name, event_year):
    if '스팩' in corp_name or 'SPAC' in corp_name.upper():
        return True, '스팩(SPAC) 정상 해산'

    params = {
        'crtfc_key': api_key,
        'corp_code': corp_code,
        'bgn_de': f'{event_year - 1}0101',
        'end_de': f'{event_year + 1}1231',
        'page_no': 1,
        'page_count': 100
    }

    try:
        res = requests.get(DART_DISCLOSURE_LIST_URL, params=params, timeout=10)
        data = res.json()
        if data.get('status') == '000':
            for report in data.get('list', []):
                title = report.get('report_nm', '')
                if any(keyword in title for keyword in SAFE_KEYWORDS):
                    return True, title
        return False, ''
    except Exception:
        return False, ''


def add_event_year(df_target):
    df_target = df_target.copy()

    if 'event_date' in df_target.columns:
        df_target['event_date'] = pd.to_datetime(
            df_target['event_date'].astype(str).str[:10],
            errors='coerce'
        )
        fallback_year = pd.to_numeric(df_target['target_bsns_year']) + 1
        df_target['event_year'] = df_target['event_date'].dt.year.fillna(fallback_year).astype(int)
    else:
        df_target['event_year'] = pd.to_numeric(df_target['target_bsns_year']).astype(int) + 1

    return df_target


def prepare_target_data(conn, df_dart):
    print('\n2. 부실기업 정답군 추출 및 API 기반 노이즈 제거(스마트 필터링) 중...')

    query = 'SELECT * FROM KRX_TARGET_COMPANY WHERE CAST(target_bsns_year AS INTEGER) >= 2015'
    df_target = pd.read_sql(query, conn)
    df_target = add_event_year(df_target)

    df_target['target_bsns_year'] = (
        pd.to_numeric(df_target['target_bsns_year'], errors='coerce')
        .astype('Int64')
        .astype(str)
    )
    df_target = df_target[df_target['target_bsns_year'] != '<NA>']
    df_target['stock_code'] = df_target['stock_code'].apply(norm_stock_code)

    df_matched = pd.merge(
        df_target,
        df_dart[['corp_code', 'stock_code']],
        on='stock_code',
        how='inner'
    )

    suspect_corps = df_matched['corp_code'].unique()
    print(f"   🔍 원본에서 부실 처리된 {len(suspect_corps)}개 기업의 공시를 실시간 검사합니다... (잠시 대기)")

    fake_fails = []
    for corp_code in suspect_corps:
        time.sleep(REQUEST_SLEEP_SECONDS)
        corp_info = df_matched[df_matched['corp_code'] == corp_code].iloc[0]
        corp_name = corp_info['corp_name']
        event_year = corp_info['event_year']

        is_safe, _ = check_voluntary_delisting(CONFIG['API_KEY'], corp_code, corp_name, event_year)
        if is_safe:
            fake_fails.append(corp_code)

    df_matched = df_matched[~df_matched['corp_code'].isin(fake_fails)].copy()

    target_list = [
        {
            'corp_code': row['corp_code'],
            'corp_name': row['corp_name'],
            'bsns_year': row['target_bsns_year'],
            'target_label': row['target_label']
        }
        for _, row in df_matched.iterrows()
    ]

    print(f"✅ 정제 결과: 우량기업/스팩 {len(fake_fails)}개를 무사히 걸러냈습니다.")
    return target_list


def prepare_normal_data(conn, df_dart):
    print('\n3. 정상 상장사 예비 추출 중...')

    snapshot_path = CONFIG.get('NORMAL_SNAPSHOT_PATH')
    use_snapshot = CONFIG.get('USE_NORMAL_SNAPSHOT', True)
    overwrite_snapshot = CONFIG.get('OVERWRITE_NORMAL_SNAPSHOT', False)

    if use_snapshot and snapshot_path and os.path.exists(snapshot_path) and not overwrite_snapshot:
        return load_normal_snapshot(snapshot_path)

    query = 'SELECT DISTINCT stock_code FROM KRX_TARGET_COMPANY'
    df_all_targets = pd.read_sql(query, conn)
    target_codes = df_all_targets['stock_code'].apply(norm_stock_code).dropna().tolist()

    df_pool = df_dart[~df_dart['stock_code'].isin(target_codes)]
    df_sampled = df_pool.sample(n=NORMAL_SAMPLE_SIZE, random_state=CONFIG['RANDOM_SEED'])

    normal_list = [
        {
            'corp_code': row['corp_code'],
            'corp_name': row['corp_name'],
            'bsns_year': str(rng.choice(NORMAL_YEAR_RANGE)),
            'target_label': 0
        }
        for _, row in df_sampled.iterrows()
    ]

    if use_snapshot and snapshot_path:
        save_normal_snapshot(normal_list, snapshot_path)

    return normal_list


# ==========================================
# 3. DB 처리
# ==========================================
def setup_database(conn):
    conn.execute(FINANCE_SCHEMA_SQL)
    conn.commit()


def save_records_to_db(conn, records):
    if not records:
        print('\n⚠️ 수집된 데이터가 없습니다.')
        return

    df_final = pd.DataFrame(records)
    print(f"\n📊 데이터 수집 성공 (총 {len(df_final)}건). DB를 업데이트합니다...")

    conn.execute(f'DROP TABLE IF EXISTS {FINANCE_TABLE}')
    conn.commit()
    setup_database(conn)

    df_final.to_sql(FINANCE_TABLE, con=conn, if_exists='append', index=False)
    conn.commit()
    print(f"🎉 완료! 총 {len(df_final)}개의 데이터가 DB에 저장되었습니다.")


# ==========================================
# 4. 단일 기업 재무제표 처리
# ==========================================
def fetch_financial_json(corp, fs_div):
    params = {
        'crtfc_key': CONFIG['API_KEY'],
        'corp_code': corp['corp_code'],
        'bsns_year': corp['bsns_year'],
        'reprt_code': CONFIG['REPRT_CODE'],
        'fs_div': fs_div
    }
    res = requests.get(DART_FINANCIAL_URL, params=params, timeout=10)
    return res.json()


def parse_financial_rows(df_finance):
    parsed = {key: None for key in ACCOUNT_ALIASES.keys()}

    for _, row in df_finance.iterrows():
        raw_nm = str(row.get('account_nm') or '')
        sj_div = str(row.get('sj_div') or '')

        clean_nm = clean_account_name(raw_nm)
        field = ACCOUNT_NAME_TO_FIELD.get(clean_nm)

        if field and parsed[field] is None and sj_div in FIELD_SJ_FILTER.get(field, []):
            val = parse_amount(row.get('thstrm_amount'))
            if val is not None:
                if '결손' in clean_nm:
                    val = -abs(val)
                elif field == 'interest_expense':
                    val = abs(val)

                parsed[field] = val

    return parsed


def has_required_finance(parsed):
    return all(parsed[field] is not None for field in REQUIRED_FINANCE_FIELDS)


def process_single_company(corp):
    try:
        data = fetch_financial_json(corp, fs_div='CFS')

        if data.get('status') != '000' or not data.get('list'):
            data = fetch_financial_json(corp, fs_div='OFS')

            if data.get('status') != '000' or not data.get('list'):
                return None, 'CFS/OFS 전체 재무제표 데이터가 모두 없습니다.'

        df_temp = pd.DataFrame(data.get('list', []))
        if df_temp.empty:
            return None, '응답 list 비어 있음'

        parsed = parse_financial_rows(df_temp)

        if not has_required_finance(parsed):
            return None, '필수 재무값 누락'

        return {**corp, **parsed}, '성공'

    except Exception as e:
        return None, f'예외 발생: {str(e)}'


# ==========================================
# 5. 수집 루프 및 메인 실행
# ==========================================
def collect_finance_records(final_list):
    print(f"\n🚀 총 {len(final_list)}개 기업 명단으로 재무 데이터 수집 시작...")

    finance_records = []
    normal_count = 0
    target_normal_count = CONFIG['TARGET_NORMAL_COUNT']

    for i, corp in enumerate(final_list):
        if corp['target_label'] == 0 and normal_count >= target_normal_count:
            continue

        time.sleep(REQUEST_SLEEP_SECONDS)
        result, msg = process_single_company(corp)

        if result:
            finance_records.append(result)
            if corp['target_label'] == 0:
                normal_count += 1
                print(
                    f"[{i + 1}/{len(final_list)}] 🏢 {corp['corp_name']} "
                    f"({corp['bsns_year']}) 완료! (정상 누적: {normal_count}/{target_normal_count})"
                )
            else:
                print(f"[{i + 1}/{len(final_list)}] 🚨 {corp['corp_name']} ({corp['bsns_year']}) 완료! (진짜 부실 정답군)")
        else:
            print(f"[{i + 1}/{len(final_list)}] ⚠️ {corp['corp_name']} 패스 사유: {msg}")

    return finance_records


def main():
    conn = sqlite3.connect(CONFIG['DB_NAME'])

    try:
        setup_database(conn)

        if not table_exists(conn, 'KRX_TARGET_COMPANY'):
            print("🚨 'KRX_TARGET_COMPANY' 테이블이 없습니다. DB 초기 세팅을 먼저 진행해주세요.")
            return

        df_dart = fetch_dart_master()
        target_list = prepare_target_data(conn, df_dart)
        normal_list = prepare_normal_data(conn, df_dart)
        final_list = target_list + normal_list

        finance_records = collect_finance_records(final_list)
        save_records_to_db(conn, finance_records)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
