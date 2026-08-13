import os
import sqlite3

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
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

TRAIN_END_YEAR = 2023
TEST_START_YEAR = 2024
CLIP_LOWER_Q = 0.01
CLIP_UPPER_Q = 0.99
THRESHOLDS = np.arange(0.1, 1.0, 0.1)
CONFUSION_THRESHOLDS = [0.5, 0.6]

RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 5,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}


# ==========================================
# 1. 데이터 불러오기 / 분석 데이터 준비
# ==========================================
def load_and_prepare_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"'{DB_PATH}' 파일이 없습니다.")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(f'SELECT * FROM {TABLE_NAME}', conn)

    required = ['corp_name', 'bsns_year', 'target_label'] + FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f'필수 컬럼이 없습니다: {missing}')

    # 숫자로 변환할 수 없는 값은 NaN으로 처리한다.
    numeric_cols = ['bsns_year', 'target_label', 'target'] + FEATURES
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 기존 DB에 target이 없을 때만 target_label로 생성한다.
    if 'target' not in df.columns:
        df = df[df['target_label'].isin([0, 1, 2])].copy()
        df['target'] = df['target_label'].isin([1, 2]).astype(int)

    df = df.dropna(subset=['bsns_year', 'target']).copy()
    df['bsns_year'] = df['bsns_year'].astype(int)
    df['target'] = df['target'].astype(int)

    return df


# ==========================================
# 2. Train / Test 분리 + 공통 전처리
# ==========================================
def prepare_train_test(df):
    # 미래 데이터가 학습에 섞이지 않도록 연도 기준으로 나눈다.
    train = df[df['bsns_year'] <= TRAIN_END_YEAR].copy()
    test = df[df['bsns_year'] >= TEST_START_YEAR].copy()

    if train.empty or test.empty:
        raise ValueError('Train 또는 Test 데이터가 비어 있습니다.')

    X_train = train[FEATURES].copy()
    X_test = test[FEATURES].copy()
    y_train = train['target'].to_numpy()
    y_test = test['target'].to_numpy()

    # Test 정보를 사용하지 않기 위해 중앙값과 clipping 경계를 Train에서만 계산한다.
    train_medians = X_train.median().fillna(0)
    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    lower = X_train.quantile(CLIP_LOWER_Q)
    upper = X_train.quantile(CLIP_UPPER_Q)
    X_train = X_train.clip(lower=lower, upper=upper, axis=1)
    X_test = X_test.clip(lower=lower, upper=upper, axis=1)

    return train, test, X_train, X_test, y_train, y_test


# ==========================================
# 3. Logistic Regression
# ==========================================
def run_logistic(X_train, X_test, y_train):
    # Logistic은 변수 스케일 차이를 줄이기 위해 표준화한다.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 정상/부실 클래스가 학습에서 비슷한 영향력을 갖도록 가중치를 준다.
    n0 = np.sum(y_train == 0)
    n1 = np.sum(y_train == 1)
    sample_weights = np.where(
        y_train == 0,
        len(y_train) / (2 * n0),
        len(y_train) / (2 * n1)
    )

    X_train_glm = sm.add_constant(X_train_scaled, has_constant='add')
    model = sm.GLM(
        y_train,
        X_train_glm,
        family=sm.families.Binomial(),
        freq_weights=sample_weights
    )
    result = model.fit()

    X_test_glm = sm.add_constant(X_test_scaled, has_constant='add')
    y_prob = np.asarray(result.predict(X_test_glm))

    # VIF는 Logistic의 변수 간 다중공선성 점검용이다.
    vif = pd.DataFrame({
        'feature': FEATURES,
        'VIF': [
            variance_inflation_factor(X_train_scaled, i)
            for i in range(len(FEATURES))
        ]
    })

    return result, y_prob, vif


# ==========================================
# 4. Random Forest
# ==========================================
def run_random_forest(X_train, X_test, y_train):
    # Tree 모델은 변수 크기보다 분할 기준이 중요하므로 StandardScaler를 쓰지 않는다.
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, y_prob, importance


# ==========================================
# 5. 공통 평가 함수
# ==========================================
def get_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    accuracy = (tn + tp) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return accuracy, precision, recall, tn, fp, fn, tp


def significance(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'


# ==========================================
# 6. 결과 출력
# ==========================================
def print_results(
    df,
    train,
    test,
    y_test,
    logistic_result,
    logistic_prob,
    vif,
    rf_prob,
    rf_importance
):
    print('\n' + '=' * 78)
    print('📊 [분석 데이터]')
    print('=' * 78)
    print(f'전체: {len(df)}건 (정상 {(df.target == 0).sum()} / 부실 {(df.target == 1).sum()})')
    print(f'Train: {len(train)}건 (<= {TRAIN_END_YEAR})')
    print(f'Test : {len(test)}건 (>= {TEST_START_YEAR})')
    print(f'Test 라벨: 정상 {(y_test == 0).sum()} / 부실 {(y_test == 1).sum()}')

    # Logistic 해석
    print('\n' + '=' * 78)
    print('🔬 [1. GLM Logistic Regression]')
    print('=' * 78)
    print(f"{'Variable':<20} {'Coef':>10} {'p-value':>10} {'Sig':>8}")
    print('-' * 52)

    names = ['Intercept'] + FEATURES
    for name, coef, p in zip(names, logistic_result.params, logistic_result.pvalues):
        print(f'{name:<20} {coef:>10.4f} {p:>10.4f} {significance(p):>8}')

    print('\n[VIF]')
    print(vif.round(4).to_string(index=False))

    # Random Forest 해석
    print('\n' + '=' * 78)
    print('🌲 [2. Random Forest]')
    print('=' * 78)
    for key, value in RF_PARAMS.items():
        print(f' - {key}: {value}')

    print('\n[Feature Importance]')
    print(rf_importance.to_string(index=False, formatters={
        'importance': lambda x: f'{x:.4f}'
    }))

    # 종합 성능
    logit_roc = roc_auc_score(y_test, logistic_prob)
    rf_roc = roc_auc_score(y_test, rf_prob)
    logit_ap = average_precision_score(y_test, logistic_prob)
    rf_ap = average_precision_score(y_test, rf_prob)

    print('\n' + '=' * 78)
    print('🏆 [3. 모델 종합 성능 비교]')
    print('=' * 78)
    print(f"{'Model':<24} {'ROC-AUC':>10} {'AP':>10}")
    print('-' * 46)
    print(f"{'Logistic Regression':<24} {logit_roc:>10.4f} {logit_ap:>10.4f}")
    print(f"{'Random Forest':<24} {rf_roc:>10.4f} {rf_ap:>10.4f}")
    print(f'참고: Test 부실 비중 = {np.mean(y_test):.4f} ({np.mean(y_test) * 100:.1f}%)')

    # 같은 threshold에서 두 모델을 직접 비교한다.
    print('\n' + '=' * 100)
    print('🎛️ [4. Threshold별 성능 비교]')
    print('=' * 100)
    print(
        f"{'Th':>5} | {'Logit Acc':>9} {'RF Acc':>9} | "
        f"{'Logit Prec':>10} {'RF Prec':>9} | "
        f"{'Logit Rec':>9} {'RF Rec':>9}"
    )
    print('-' * 100)

    for th in THRESHOLDS:
        l_acc, l_prec, l_rec, *_ = get_metrics(y_test, logistic_prob, th)
        r_acc, r_prec, r_rec, *_ = get_metrics(y_test, rf_prob, th)

        print(
            f'{th:>4.0%} | '
            f'{l_acc:>8.1%} {r_acc:>8.1%} | '
            f'{l_prec:>9.1%} {r_prec:>8.1%} | '
            f'{l_rec:>8.1%} {r_rec:>8.1%}'
        )

    # 50%, 60%에서는 실제 오분류 건수도 확인한다.
    print('\n' + '=' * 78)
    print('🧩 [5. Confusion Matrix 비교]')
    print('=' * 78)
    print('TN=정상→정상 / FP=정상→부실 / FN=부실→정상 / TP=부실→부실')

    for th in CONFUSION_THRESHOLDS:
        print(f'\n[Threshold = {th:.0%}]')
        print(f"{'Model':<24} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}")

        for name, prob in [
            ('Logistic Regression', logistic_prob),
            ('Random Forest', rf_prob)
        ]:
            _, _, _, tn, fp, fn, tp = get_metrics(y_test, prob, th)
            print(f'{name:<24} {tn:>6} {fp:>6} {fn:>6} {tp:>6}')


# ==========================================
# 7. 실행
# ==========================================
def main():
    df = load_and_prepare_data()
    train, test, X_train, X_test, y_train, y_test = prepare_train_test(df)

    logistic_result, logistic_prob, vif = run_logistic(
        X_train, X_test, y_train
    )

    _, rf_prob, rf_importance = run_random_forest(
        X_train, X_test, y_train
    )

    print_results(
        df,
        train,
        test,
        y_test,
        logistic_result,
        logistic_prob,
        vif,
        rf_prob,
        rf_importance
    )


if __name__ == '__main__':
    main()
