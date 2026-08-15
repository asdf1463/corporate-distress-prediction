# KRX KIND & OpenDART 부실기업 예측 프로젝트

KRX KIND 공시와 OpenDART 사업보고서 재무제표를 연결해 상장기업의 부실 가능성을 예측한 프로젝트입니다.

이 프로젝트는 단순히 모델 성능을 높이는 것보다 **부실 라벨 설계 → 정상 대조군 구성 → 재무제표 수집 → Feature 생성 → Out-of-Time 검증 → 모델 비교**의 전체 분석 파이프라인을 직접 구성하는 데 초점을 뒀습니다.

GLM Logistic Regression을 해석 가능한 기준 모델로 사용하고, 동일한 5개 재무변수와 동일한 테스트셋에 Random Forest를 추가해 **동일한 데이터 조건에서 선형 모델과 비선형 모델의 예측 성능을 비교**했습니다.

## 프로젝트 흐름

```text
KIND DATA.xlsx
  ↓
KIND_EXCEL.py
  - 관리종목, 투자주의환기종목, 상장폐지 공시에서 부실 후보 추출
  - 동일 종목은 최초 이벤트만 사용
  ↓
DART_API.py
  - OpenDART 기업코드 매칭
  - 스팩·합병 등 일부 비부실 상장폐지 케이스 제거
  - 정상 대조군 구성 및 사업보고서 재무제표 수집
  ↓
FEATURES.py
  - 재무 원천값 정제
  - 모델 변수 및 참고 재무비율 생성
  ↓
TEST.py
  - 라벨/연도 분포, 결측치, 이상치 점검
  ↓
ANALYZE.py
  - GLM Logistic Regression
  - Random Forest
  - ROC-AUC / Average Precision / Threshold / Confusion Matrix 비교
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| `KIND_EXCEL.py` | KIND 엑셀에서 부실 후보군 생성 |
| `DART_API.py` | DART 기업 매칭, 정상군 구성, 재무제표 수집 |
| `normal_sample_snapshot.csv` | 정상 대조군 후보와 사업연도 고정용 snapshot |
| `FEATURES.py` | 재무제표 원천값을 분석 변수로 변환 |
| `TEST.py` | 데이터 분포, 결측치, 이상치 점검 |
| `ANALYZE.py` | Logistic Regression과 Random Forest 학습·평가·비교 |
| `dart_finance.db` | 수집 및 Feature 생성 결과가 저장된 SQLite DB |
| `REPORT.md` | 데이터 설계, 모델링, 결과 해석, 한계를 정리한 상세 분석 보고서 |

## 데이터와 Feature

최종 분석 데이터는 **903건(정상 599 / 부실 304)**입니다.

두 모델에는 동일한 5개 재무변수를 사용했습니다.

| 변수 | 의미 | 관점 |
|---|---|---|
| `roa` | 순이익 / 총자산 | 수익성 |
| `signed_log_ocf` | 영업현금흐름의 부호 유지 로그값 | 현금창출력 |
| `asset_turnover` | 매출액 / 총자산 | 자산 효율성 |
| `debt_to_assets` | 총부채 / 총자산 | 부채 부담 |
| `log_assets` | 총자산 로그값 | 기업 규모 |

Train/Test는 사업연도를 기준으로 나눴습니다.

- Train: **2023년 이하 733건**
- Test: **2024년 이상 170건**
- Test 라벨: 정상 123 / 부실 47
- 결측치: Train 중앙값으로 Train/Test 모두 대체
- 이상치: Train 1%~99% quantile 기준 clipping
- Logistic Regression에만 `StandardScaler` 적용
- 클래스 불균형 보정
  - Logistic Regression: class-balanced sample weight
  - Random Forest: `class_weight='balanced'`

연도 기준으로 분리한 이유는 실제 사용 상황처럼 **과거 데이터로 학습하고 이후 시점의 기업을 평가하는 Out-of-Time 검증**을 하기 위해서입니다.

## 모델

### GLM Logistic Regression

해석 가능한 기준 모델입니다.

- 회귀계수와 p-value로 변수 방향 및 통계적 유의성 확인
- VIF로 다중공선성 점검
- StandardScaler 적용

### Random Forest

비선형 관계와 변수 간 상호작용을 확인하기 위한 비교 모델입니다.

```text
n_estimators = 300
max_depth = 5
min_samples_leaf = 5
class_weight = balanced
random_state = 42
```

Tree 기반 모델이므로 별도의 StandardScaler는 적용하지 않았습니다.

## 실행 방법

패키지 설치:

```bash
pip install -r requirements.txt
```

전체 파이프라인 실행:

```bash
python KIND_EXCEL.py
python DART_API.py
python FEATURES.py
python TEST.py
python ANALYZE.py
```

OpenDART API를 다시 호출하려면 API 키가 필요합니다.

```powershell
$env:DART_API_KEY="본인_API_KEY"
python DART_API.py
```

이미 생성된 `dart_finance.db`가 있다면 API 호출 없이 분석 단계만 다시 실행할 수 있습니다.

```bash
python TEST.py
python ANALYZE.py
```

## 핵심 결과

| Model | ROC-AUC | Average Precision |
|---|---:|---:|
| **GLM Logistic Regression** | **0.8838** | 0.7516 |
| **Random Forest** | 0.8829 | **0.7597** |

두 모델의 ROC-AUC는 사실상 유사했습니다. Random Forest는 Average Precision과 일부 Threshold의 Recall이 소폭 높았지만, Logistic Regression은 같은 구간에서 Precision과 Accuracy가 더 높았습니다.

예를 들어 60% Threshold에서는 Random Forest가 Logistic Regression보다 부실기업을 3건 더 탐지해 FN을 10건에서 7건으로 줄였지만, 정상기업 오탐 FP는 19건에서 25건으로 증가했습니다.

또한 `roa`는 Logistic Regression에서 가장 큰 절댓값의 계수를 보였고, Random Forest에서도 가장 높은 Feature Importance를 보여 두 모델에서 공통적으로 강한 부실 구분 신호로 나타났습니다.

## 정리

현재 5개 Feature 구성에서는 Random Forest의 비선형 구조가 ROC-AUC 개선으로 이어지지 않았습니다. 따라서 Logistic Regression을 **주요 해석 모델**, Random Forest를 **비선형 비교 모델**로 두었습니다.

다음 개선에서는 모델 수를 늘리기보다 산업·규모·연도를 고려한 정상군 매칭, 다년도 재무 변화율, 산업 상대지표 등 **데이터와 Feature 정보량을 확장하는 방향**을 우선 검토할 수 있습니다.

데이터 구성 과정, 회귀계수·VIF·Feature Importance, Threshold별 성능과 Confusion Matrix, 한계 및 개선 방향은 **[REPORT.md](REPORT.md)**에 상세히 정리했습니다.
