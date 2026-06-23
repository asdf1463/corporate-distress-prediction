import os
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================================
# 0. 설정
# ==========================================
DB_PATH = 'dart_finance.db'
TABLE_NAME = 'FINANCE_FEATURES'

FEATURES = [
    'roa',
    'signed_log_ocf',
    'asset_turnover',
    'debt_to_assets',
    'log_assets'
]

USE_OUTLIER_CLIP = True
CLIP_LOWER_Q = 0.01
CLIP_UPPER_Q = 0.99
TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024


# ==========================================
# 1. 유틸 함수
# ==========================================
def load_data(db_path, table_name):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"'{db_path}' 파일이 없습니다.")

    with sqlite3.connect(db_path) as conn:
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'",
            conn
        )['name'].tolist()

        if table_name not in tables:
            raise ValueError(f"'{table_name}' 테이블이 없습니다. 현재 테이블: {tables}")

        return pd.read_sql(f"SELECT * FROM {table_name}", conn)


def validate_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"{TABLE_NAME} 테이블에 필수 컬럼이 없습니다: {missing}")


def coerce_numeric_columns(df, cols):
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def add_significance_label(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


def calc_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (pred == 1))
    tn = np.sum((y_true == 0) & (pred == 0))
    fp = np.sum((y_true == 0) & (pred == 1))
    fn = np.sum((y_true == 1) & (pred == 0))

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    return acc, prec, rec


# ==========================================
# 2. 출력 함수
# ==========================================
def print_regression_summary(summary_df):
    print('\n' + '=' * 73)
    print('[GLM Regression Summary]')
    print('=' * 73)
    print(f"{'Variable':<20} {'Coef':>10} {'Std.Err':>10} {'z':>10} {'p-value':>10} {'Sig':>8}")
    print('-' * 73)

    for variable, row in summary_df.iterrows():
        print(
            f"{variable:<20} "
            f"{row['계수(Coef)']:>10.4f} "
            f"{row['표준오차(Std.Err)']:>10.4f} "
            f"{row['z-값(z)']:>10.4f} "
            f"{row['p-value(P>|z|)']:>10.4f} "
            f"{row['유의성 평가']:>8}"
        )

    print('-' * 73)
    print('Sig: *** p<0.001, ** p<0.01, * p<0.05, n.s. not significant')
    print('=' * 73)


def print_threshold_simulation(y_test, y_test_prob):
    print('\n' + '=' * 70)
    print('🎛️ [임계값(Threshold) 변경 시뮬레이션]')
    print('=' * 70)
    print(f"{'임계값':^6} | {'정확도':^8} | {'정밀도(진짜부실 비율)':^18} | {'재현율(폭탄 방어율)':^18}")
    print('-' * 70)

    for th in np.arange(0.1, 1.0, 0.1):
        acc, prec, rec = calc_metrics(y_test, y_test_prob, th)
        print(f"{th * 100:4.0f}%  | {acc * 100:7.1f}% | {prec * 100:18.1f}% | {rec * 100:17.1f}%")

    print('=' * 70)


def print_auc(y_test, y_test_prob):
    try:
        auc_score = roc_auc_score(y_test, y_test_prob)
        print(f"🏆 모델의 최종 종합 예측 성능 (ROC-AUC): {auc_score:.4f}")
    except ValueError:
        print('⚠️ Test 데이터에 한쪽 클래스만 있어 ROC-AUC를 계산할 수 없습니다.')


def print_vif(X_train_scaled):
    vif_df = pd.DataFrame({
        'feature': FEATURES,
        'VIF': [
            variance_inflation_factor(X_train_scaled, i)
            for i in range(len(FEATURES))
        ]
    })

    print('\n📌 [다중공선성 VIF]')
    print(vif_df.round(4).to_string(index=False))


# ==========================================
# 3. 데이터 정제 및 분할
# ==========================================
def prepare_analysis_dataframe(df):
    required_cols = ['corp_name', 'bsns_year', 'target_label'] + FEATURES
    validate_columns(df, required_cols)

    numeric_cols = ['bsns_year', 'target_label', 'target'] + FEATURES
    df = coerce_numeric_columns(df, numeric_cols)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if 'target' not in df.columns:
        df = df[df['target_label'].isin([0, 1, 2])].copy()
        df['target'] = df['target_label'].isin([1, 2]).astype(int)

    df = df.dropna(subset=['bsns_year', 'target']).copy()
    df['bsns_year'] = df['bsns_year'].astype(int)
    df['target'] = df['target'].astype(int)

    return df


def split_train_test(df):
    train_data = df[df['bsns_year'] <= TRAIN_END_YEAR].copy()
    test_data = df[df['bsns_year'] >= TEST_START_YEAR].copy()

    if train_data.empty or test_data.empty:
        raise ValueError('Train 또는 Test 데이터가 비어 있습니다. bsns_year 분포를 확인하세요.')

    if train_data['target'].nunique() < 2:
        raise ValueError('Train 데이터에 정상/부실 중 한쪽 클래스만 있습니다.')

    return train_data, test_data


def preprocess_features(train_data, test_data):
    X_train = train_data[FEATURES].copy()
    y_train = train_data['target'].values
    X_test = test_data[FEATURES].copy()
    y_test = test_data['target'].values

    print('\n🔧 [결측치 처리: Train 중앙값 기준]')
    train_medians = X_train.median().fillna(0)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    if USE_OUTLIER_CLIP:
        print('\n✂️ [이상치 완화: Train 기준 1%~99% Clip]')
        lower_bounds = X_train[FEATURES].quantile(CLIP_LOWER_Q)
        upper_bounds = X_train[FEATURES].quantile(CLIP_UPPER_Q)

        X_train[FEATURES] = X_train[FEATURES].clip(
            lower=lower_bounds,
            upper=upper_bounds,
            axis=1
        )
        X_test[FEATURES] = X_test[FEATURES].clip(
            lower=lower_bounds,
            upper=upper_bounds,
            axis=1
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# ==========================================
# 4. 모델 학습
# ==========================================
def build_sample_weights(y_train):
    num_zeros = np.sum(y_train == 0)
    num_ones = np.sum(y_train == 1)

    return np.where(
        y_train == 0,
        (1 / num_zeros) * (len(y_train) / 2.0),
        (1 / num_ones) * (len(y_train) / 2.0)
    )


def fit_glm_model(X_train_scaled, y_train):
    print('\n🔬 [통계학적 로지스틱 회귀 모델(GLM) 분석 시작]')

    X_train_stat = sm.add_constant(X_train_scaled, has_constant='add')
    model = sm.GLM(
        y_train,
        X_train_stat,
        family=sm.families.Binomial(),
        freq_weights=build_sample_weights(y_train)
    )

    return model.fit()


def build_summary_dataframe(result):
    summary_df = pd.DataFrame({
        '계수(Coef)': result.params,
        '표준오차(Std.Err)': result.bse,
        'z-값(z)': result.tvalues,
        'p-value(P>|z|)': result.pvalues
    })

    summary_df.index = ['Intercept'] + FEATURES
    summary_df['유의성 평가'] = summary_df['p-value(P>|z|)'].apply(add_significance_label)
    return summary_df

# ==========================================
# 5. 실행
# ==========================================
def main():
    df = load_data(DB_PATH, TABLE_NAME)

    if df.empty:
        raise SystemExit(f"🚨 {TABLE_NAME} 테이블에 데이터가 없습니다.")

    df = prepare_analysis_dataframe(df)

    print(f"✅ {TABLE_NAME}에서 불러온 데이터: {len(df)}건")
    print('\n[전체 라벨 분포]')
    print(df['target'].value_counts().rename(index={0: '정상', 1: '부실'}).to_string())

    train_data, test_data = split_train_test(df)
    print(f"\n📊 학습용 데이터(<=2023년): {len(train_data)}건")
    print(f"📊 테스트용 데이터(>=2024년): {len(test_data)}건")

    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_features(train_data, test_data)
    result = fit_glm_model(X_train_scaled, y_train)

    summary_df = build_summary_dataframe(result)
    print_regression_summary(summary_df)

    X_test_stat = sm.add_constant(X_test_scaled, has_constant='add')
    y_test_prob = result.predict(X_test_stat)
    print_threshold_simulation(y_test, y_test_prob)
    print_auc(y_test, y_test_prob)
    print_vif(X_train_scaled)


if __name__ == '__main__':
    main()
