# KRX KIND & OpenDART 부실기업 예측 프로젝트

KRX KIND 공시와 OpenDART 사업보고서 재무제표를 연결해 상장기업의 부실 가능성을 예측한 프로젝트입니다.

단순히 모델 성능만 비교하기보다 **부실 라벨을 어떻게 만들고**, **정상 대조군을 어떻게 구성하고**, **재무제표 원천값을 어떤 분석 변수로 바꾸며**, **미래 시점 데이터에서 모델을 어떻게 검증할지**를 하나의 분석 파이프라인으로 정리하는 데 초점을 뒀습니다.

초기에는 해석 가능한 GLM Logistic Regression을 기준 모델로 사용했고, 이후 동일한 5개 재무변수와 동일한 Out-of-Time 테스트셋에 Random Forest를 추가해 **동일한 데이터 조건에서 선형 기준 모델과 비선형 Bagging 모델의 예측 성능을 비교**했습니다.

## 프로젝트 흐름

```text
KIND DATA.xlsx
  ↓
KIND_EXCEL.py
  - 관리종목, 환기종목, 상장폐지 공시에서 부실 후보 추출
  - 동일 종목은 최초 이벤트만 사용
  ↓
DART_API.py
  - OpenDART 기업코드 매칭
  - 스팩, 합병 등 정상적 상장폐지 케이스 일부 제거
  - 정상 대조군 추출 또는 snapshot 파일 로드
  - 사업보고서 재무제표 수집
  ↓
FEATURES.py
  - 재무 원천값 정제
  - 모델 변수와 참고 재무비율 생성
  ↓
TEST.py
  - 라벨/연도 분포, 결측치, 이상치 점검
  ↓
ANALYZE.py
  - GLM Logistic Regression 학습 및 통계적 해석
  - Random Forest 학습 및 Feature Importance 확인
  - 동일 테스트셋에서 ROC-AUC, Average Precision, Threshold 성능 비교
  - 50%/60% Threshold Confusion Matrix 비교
```

## 사용 데이터

| 구분 | 내용 |
|---|---|
| KRX KIND | 관리종목, 투자주의환기종목, 상장폐지 관련 공시 엑셀 |
| OpenDART | 기업 고유번호, 공시 목록, 사업보고서 재무제표 API |
| 저장소 | SQLite `dart_finance.db` |
| 재현성 보완 | `normal_sample_snapshot.csv` |

정상 대조군은 DART 상장사 중 부실 후보를 제외하고 무작위로 추출했습니다. DART 기업목록은 실행 시점에 따라 달라질 수 있기 때문에 정상군 후보와 사업연도를 `normal_sample_snapshot.csv`로 고정해 재현성을 보완했습니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `KIND_EXCEL.py` | KIND 엑셀에서 부실 후보군 생성 |
| `DART_API.py` | DART 매칭, 정상군 구성, 재무제표 수집 |
| `normal_sample_snapshot.csv` | 정상 대조군 후보 및 사업연도 고정용 snapshot |
| `FEATURES.py` | 재무제표 원천값을 분석 변수로 변환 |
| `TEST.py` | 데이터 분포, 결측치, 이상치 점검 |
| `ANALYZE.py` | Logistic Regression과 Random Forest 학습·평가·비교 |
| `dart_finance.db` | 실행 결과가 저장된 SQLite DB |
| `REPORT.md` | 데이터 구성, 모델링, 결과 해석 상세 보고서 |

## 모델 변수

두 모델 모두 동일한 5개 재무변수를 사용했습니다. 모델만 바꾸고 데이터와 Feature 조건을 동일하게 유지해 모델 구조에 따른 차이를 비교하기 위한 설계입니다.

| 변수 | 의미 | 신용분석 관점 |
|---|---|---|
| `roa` | 순이익 / 총자산 | 수익성 |
| `signed_log_ocf` | 영업현금흐름의 부호 유지 로그값 | 현금창출력 |
| `asset_turnover` | 매출액 / 총자산 | 자산 효율성 |
| `debt_to_assets` | 총부채 / 총자산 | 부채 부담 |
| `log_assets` | 총자산 로그값 | 기업 규모 |

## 실험 설계

모델 간 비교가 전처리 차이 때문에 왜곡되지 않도록 다음 조건을 공통으로 적용했습니다.

- 전체 분석 데이터: **903건** (정상 599 / 부실 304)
- Train: **2023년 이하 733건**
- Test: **2024년 이상 170건**
- Test 라벨: 정상 123 / 부실 47
- 결측치: **Train 중앙값**으로 Train/Test 모두 대체
- 이상치: **Train 1%~99% quantile 기준 clipping**을 Train/Test에 동일 적용
- Logistic Regression에만 `StandardScaler` 적용
- 클래스 불균형 보정
  - Logistic Regression: class-balanced sample weight
  - Random Forest: `class_weight='balanced'`

연도 기준으로 Train/Test를 나눈 이유는 실제 사용 상황처럼 **과거 정보로 학습하고 이후 시점의 기업을 평가하는 Out-of-Time 검증**을 하기 위해서입니다.

### 비교 모델

**GLM Logistic Regression**

- 해석 가능한 기준 모델
- 회귀계수, p-value, VIF를 이용해 변수 방향과 통계적 유의성 확인

**Random Forest**

- 비선형 관계와 변수 간 상호작용을 포착할 수 있는 Bagging 기반 비교 모델
- `n_estimators=300`
- `max_depth=5`
- `min_samples_leaf=5`
- `class_weight='balanced'`
- `random_state=42`

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
# Windows PowerShell
$env:DART_API_KEY="본인_API_KEY"
python DART_API.py
```

이미 생성된 `dart_finance.db`가 있으면 API 키 없이 분석 단계만 다시 실행할 수 있습니다.

```bash
python TEST.py
python ANALYZE.py
```

## 결과 요약

### 1. 모델 종합 성능

| Model | ROC-AUC | Average Precision (AP) |
|---|---:|---:|
| **GLM Logistic Regression** | **0.8838** | 0.7516 |
| **Random Forest** | 0.8829 | **0.7597** |

두 모델의 ROC-AUC는 거의 동일했습니다. Random Forest가 더 복잡한 비선형 구조를 사용했지만 현재 5개 변수 구성에서는 추가적인 ROC-AUC 개선으로 이어지지 않았습니다.

반면 부실 클래스를 중심으로 Precision-Recall 성능을 요약하는 **Average Precision(AP)**은 Random Forest가 0.7597로 Logistic Regression의 0.7516보다 소폭 높았습니다. Test 데이터의 실제 부실 비중은 약 27.6%였습니다.

### 2. Threshold별 성능 비교

| Threshold | Model | Accuracy | Precision | Recall |
|---:|---|---:|---:|---:|
| 50% | Logistic | **78.2%** | **57.4%** | 83.0% |
|  | Random Forest | 76.5% | 54.8% | **85.1%** |
| 60% | Logistic | **82.9%** | **66.1%** | 78.7% |
|  | Random Forest | 81.2% | 61.5% | **85.1%** |
| 70% | Logistic | **85.3%** | **72.9%** | 74.5% |
|  | Random Forest | 82.4% | 66.0% | 74.5% |

Random Forest는 50~60% Threshold에서 더 높은 Recall을 보여 실제 부실기업을 조금 더 많이 탐지했습니다. 대신 정상기업을 부실로 경고하는 오탐이 늘어나 Precision과 Accuracy는 Logistic Regression보다 낮았습니다.

예를 들어 60% Threshold에서:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 104 | **19** | 10 | 37 |
| Random Forest | 98 | 25 | **7** | **40** |

Random Forest는 Logistic Regression보다 부실기업을 3건 더 탐지해 FN을 10건에서 7건으로 줄였지만, 정상기업 오탐 FP는 19건에서 25건으로 증가했습니다. 따라서 실제 적용에서는 **부실 누락 비용과 정상기업 오탐 비용의 균형**을 고려해 Threshold를 결정해야 합니다.

### 3. 모델별 변수 해석

Logistic Regression에서는 5개 변수 모두 예상한 방향으로 나타났습니다.

- `roa`: -1.4092 (p<0.001)
- `signed_log_ocf`: -0.3393 (p=0.0016)
- `asset_turnover`: -0.2771 (p=0.0174)
- `debt_to_assets`: +0.4323 (p=0.0005)
- `log_assets`: -1.0294 (p<0.001)

즉 수익성, 현금흐름, 자산 효율성, 기업 규모가 낮을수록 부실 가능성이 커지고, 자산 대비 부채 부담이 높을수록 부실 가능성이 커지는 방향이었습니다.

Random Forest의 impurity-based Feature Importance는 다음과 같습니다.

| Feature | Importance |
|---|---:|
| `roa` | **0.5114** |
| `signed_log_ocf` | 0.2019 |
| `log_assets` | 0.1662 |
| `debt_to_assets` | 0.0615 |
| `asset_turnover` | 0.0590 |

특히 `roa`는 Random Forest에서 가장 높은 중요도를 보였고, Logistic Regression에서도 가장 큰 절댓값의 계수를 보여 **수익성 악화가 두 모델에서 공통적으로 강한 부실 구분 신호**로 나타났습니다.

단, Random Forest의 Feature Importance는 변수의 인과효과나 위험 방향을 의미하지 않으며, Tree 분할 과정에서 해당 변수가 상대적으로 얼마나 활용되었는지를 나타냅니다.

## 결론

동일한 5개 재무변수와 동일한 Out-of-Time 테스트셋에서 Logistic Regression과 Random Forest를 비교한 결과, **모델 복잡도를 높이는 것만으로는 예측 성능이 유의미하게 개선되지 않았습니다.**

ROC-AUC는 Logistic Regression 0.8838, Random Forest 0.8829로 사실상 유사했습니다. Random Forest는 일부 Threshold에서 Recall과 AP가 소폭 높았지만, Logistic Regression은 Precision·Accuracy와 설명 가능성 측면에서 장점이 있었습니다.

따라서 현재 변수 구성에서는 Logistic Regression을 **주요 해석 모델**, Random Forest를 **비선형 비교 모델**로 두는 것이 적절하다고 판단했습니다. 다음 개선에서는 모델 수를 늘리기보다 산업·규모·연도를 고려한 대조군 구성이나 다년도 재무 변화율 등 **데이터와 Feature 정보량을 확장하는 방향**을 우선 검토할 수 있습니다.

자세한 데이터 구성과 모델별 결과는 [REPORT.md](REPORT.md)에 정리했습니다.
