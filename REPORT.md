# 부실기업 예측 분석 보고서

## 1. 시작점

이 프로젝트는 **“부실기업을 재무제표만 보고 어느 정도 구분할 수 있을까?”**라는 질문에서 시작했습니다.

부실 라벨은 KRX KIND의 관리종목, 투자주의환기종목, 상장폐지 관련 공시에서 만들었습니다. 이후 OpenDART API로 사업보고서 재무제표를 수집해 분석용 데이터셋을 구성했습니다.

초기에는 해석 가능한 **GLM Logistic Regression**을 기준 모델로 사용했습니다. 이후 동일한 데이터, 동일한 5개 재무변수, 동일한 Out-of-Time 테스트셋에 **Random Forest**를 추가해 선형 기준 모델과 비선형 Bagging 모델의 예측 성능을 비교했습니다.

포트폴리오 목적상 단순히 더 복잡한 모델을 적용하는 것보다, **데이터 수집 → 라벨 설계 → 변수 생성 → 시점 분할 → 모델 비교 → 결과 해석**의 흐름을 일관되게 구성하는 데 초점을 뒀습니다.

---

## 2. 데이터셋 구성

### 2.1 부실 후보군

KIND 엑셀의 세 시트를 사용했습니다.

| 시트 | 처리 방식 |
|---|---|
| 관리종목 | 자본잠식, 영업손실, 감사의견 등 부실 관련 키워드 추출 |
| 환기종목 | 손실, 잠식, 계속기업 관련 키워드 추출 |
| 상장폐지 | 합병, 스팩, 주식교환 등 비부실성 이벤트 제외 |

같은 종목에서 여러 이벤트가 있으면 최초 이벤트만 남겼습니다.

또한 이벤트 직전의 확정 재무제표를 보기 위해, 이벤트일에서 3개월을 뺀 뒤 그 전년도 사업연도를 기준으로 잡았습니다. 예를 들어 2024년 3월 이벤트라면 2023년 재무제표가 아직 확정되지 않았을 수 있으므로 더 보수적인 연도를 사용합니다.

### 2.2 DART 처리

DART에서는 기업 고유번호를 종목코드와 매칭한 뒤, 사업보고서 재무제표를 가져왔습니다. 상장폐지 후보 중에서도 합병, 스팩 등 정상적 이벤트로 보이는 경우는 공시 목록 API를 이용해 한 번 더 걸렀습니다.

정상 대조군은 DART 상장사 모집단에서 부실 후보를 제외하고 뽑았습니다. 이때 DART 기업목록은 실행 시점에 따라 바뀔 수 있어서, 정상군 후보와 사업연도를 `normal_sample_snapshot.csv`로 저장해 재현성을 보완했습니다.

---

## 3. 최종 데이터

최종 분석 테이블 `FINANCE_FEATURES`에는 903건이 남았습니다.

| 구분 | 건수 |
|---|---:|
| 정상 | 599 |
| 부실 | 304 |
| 합계 | 903 |

연도별 라벨 분포는 아래와 같습니다.

| 사업연도 | 정상 | 부실 |
|---:|---:|---:|
| 2015 | 30 | 23 |
| 2016 | 52 | 22 |
| 2017 | 43 | 47 |
| 2018 | 62 | 30 |
| 2019 | 53 | 30 |
| 2020 | 46 | 24 |
| 2021 | 58 | 15 |
| 2022 | 70 | 21 |
| 2023 | 62 | 45 |
| 2024 | 55 | 40 |
| 2025 | 68 | 7 |

Train/Test는 사업연도 기준으로 나눴습니다.

| 구분 | 조건 | 건수 |
|---|---|---:|
| Train | 2023년 이하 | 733 |
| Test | 2024년 이상 | 170 |

Test 데이터는 정상 123건, 부실 47건으로 구성됐습니다.

결측률은 전반적으로 낮았습니다. 가장 높은 결측도 영업현금흐름 관련 변수 7건, 약 0.78% 수준이었습니다. 기본 점검에서 자산총계 0 이하, 매출액 0 이하, 음수 부채 같은 명확한 오류는 발견되지 않았습니다.

변수별 IQR 기준 이상치 관측은 총 595건으로 확인됐습니다. 부실기업 데이터는 손실률, 이자보상배율, 현금흐름 비율에서 극단값이 많이 나타나는 편이라, 모델링 단계에서는 **Train 데이터 기준 1%~99% clipping**을 적용했습니다.

---

## 4. 변수 설계

두 모델에는 동일한 5개 변수를 사용했습니다.

| 변수 | 의미 | 기대 방향 |
|---|---|---|
| `roa` | 순이익 / 총자산 | 낮을수록 위험 |
| `signed_log_ocf` | 영업현금흐름 부호 유지 로그값 | 낮을수록 위험 |
| `asset_turnover` | 매출액 / 총자산 | 낮을수록 위험 |
| `equity_ratio` | 자기자본 / 총자산 | 낮을수록 위험 |
| `log_assets` | 기업 규모 | 작을수록 위험 |

동일한 Feature를 두 모델에 사용한 이유는 **Feature 차이가 아니라 모델 구조 자체의 차이**를 비교하기 위해서입니다.

결측치는 Train 데이터의 중앙값으로 Train/Test 모두 대체했습니다. 이상치는 Train 데이터에서 계산한 1%~99% 분위수 범위를 Train/Test에 동일하게 적용했습니다. 이를 통해 Test 데이터의 정보를 전처리 단계에서 미리 사용하는 것을 피했습니다.

---

## 5. 모델링 및 실험 설계

### 5.1 공통 비교 조건

| 항목 | 방식 |
|---|---|
| Feature | 동일 5개 재무변수 |
| Train | 2023년 이하, 733건 |
| Test | 2024년 이상, 170건 |
| 결측치 | Train 중앙값으로 대체 |
| 이상치 | Train 1%~99% 기준 clipping |
| 평가 | ROC-AUC, Average Precision, Threshold별 Accuracy/Precision/Recall, Confusion Matrix |

연도 기준으로 Train/Test를 나눈 이유는 실제 사용 상황처럼 **과거 정보로 학습하고 이후 시점의 기업을 평가하는 Out-of-Time 검증**을 하기 위해서입니다.

### 5.2 GLM Logistic Regression

Logistic Regression은 해석 가능한 기준 모델로 사용했습니다.

| 항목 | 방식 |
|---|---|
| 모델 | GLM Binomial Logistic Regression |
| 스케일링 | StandardScaler |
| 불균형 처리 | class-balanced sample weight |
| 주요 해석 | 계수, p-value, VIF |

Logistic Regression은 변수별 계수의 방향과 통계적 유의성을 확인할 수 있다는 장점이 있습니다.

### 5.3 Random Forest

Random Forest는 비선형 관계와 변수 간 상호작용을 포착할 수 있는 Bagging 기반 비교 모델로 사용했습니다.

| Parameter | 값 |
|---|---:|
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `min_samples_leaf` | 5 |
| `class_weight` | balanced |
| `random_state` | 42 |

Random Forest에는 별도의 StandardScaler를 적용하지 않았습니다. Tree 기반 모델은 변수값의 임계점을 기준으로 분할하므로 스케일링이 필수적이지 않기 때문입니다.

---

## 6. 결과

### 6.1 Logistic Regression 회귀계수

| 변수 | 계수 | p-value | 유의성 | 해석 |
|---|---:|---:|---:|---|
| Intercept | -0.2227 | 0.0494 | * | 기준 절편 |
| `roa` | -1.4092 | 0.0000 | *** | 수익성이 높을수록 부실 가능성 감소 |
| `signed_log_ocf` | -0.3393 | 0.0016 | ** | 현금흐름이 좋을수록 부실 가능성 감소 |
| `asset_turnover` | -0.2771 | 0.0174 | * | 자산 효율성이 높을수록 부실 가능성 감소 |
| `equity_ratio` | -0.4323 | 0.0005 | *** | 자기자본비율이 높을수록 부실 가능성 감소 |
| `log_assets` | -1.0294 | 0.0000 | *** | 규모가 클수록 부실 가능성 감소 |

계수 방향은 예상한 재무 해석과 대체로 일치했습니다. 특히 `roa`, `equity_ratio`, `log_assets`가 뚜렷하게 나타났습니다.

### 6.2 Logistic Regression 다중공선성

| 변수 | VIF |
|---|---:|
| `roa` | 1.6353 |
| `signed_log_ocf` | 1.3574 |
| `asset_turnover` | 1.1699 |
| `equity_ratio` | 1.3550 |
| `log_assets` | 1.3579 |

모든 변수의 VIF가 낮아, 현재 5개 변수 사이의 다중공선성 문제는 크지 않은 것으로 봤습니다.

### 6.3 Random Forest Feature Importance

| Feature | Importance |
|---|---:|
| `roa` | **0.5077** |
| `signed_log_ocf` | 0.2016 |
| `log_assets` | 0.1699 |
| `equity_ratio` | 0.0612 |
| `asset_turnover` | 0.0596 |

Random Forest에서는 `roa`의 Feature Importance가 0.5077로 가장 높았고, `signed_log_ocf`, `log_assets`가 뒤를 이었습니다.

특히 `roa`는 Logistic Regression에서도 가장 큰 절댓값의 계수(-1.4092, p<0.001)를 보여, 서로 다른 모델에서도 **수익성 악화가 강한 부실 구분 신호**로 나타났습니다.

다만 Random Forest의 Feature Importance는 변수의 인과효과나 위험 방향을 의미하지 않습니다. 이 값은 Tree가 정상/부실을 구분하기 위해 분할하는 과정에서 해당 변수가 상대적으로 얼마나 활용되었는지를 나타냅니다.

### 6.4 모델 종합 성능 비교

| Model | ROC-AUC | Average Precision (AP) |
|---|---:|---:|
| **GLM Logistic Regression** | **0.8838** | 0.7516 |
| **Random Forest** | 0.8831 | **0.7632** |

Test 데이터의 부실 비중은 47/170으로 약 27.6%입니다.

두 모델의 ROC-AUC 차이는 0.0007로 매우 작았습니다. 즉 현재 5개 Feature와 Out-of-Time 테스트셋에서는 **Random Forest의 비선형 구조가 추가적인 전체 구분 성능 개선으로 이어지지 않았습니다.**

반면 부실 클래스의 Precision-Recall 성능을 요약하는 Average Precision에서는 Random Forest가 0.7632로 Logistic Regression의 0.7516보다 소폭 높았습니다. 다만 차이가 크지는 않아 두 모델의 성능을 단순히 한 지표만으로 우열화하기보다는 Threshold별 결과와 함께 해석했습니다.

### 6.5 Threshold별 성능 비교

| Threshold | Model | Accuracy | Precision | Recall |
|---:|---|---:|---:|---:|
| 50% | Logistic | **78.2%** | **57.4%** | 83.0% |
|  | Random Forest | 77.1% | 55.6% | **85.1%** |
| 60% | Logistic | **82.9%** | **66.1%** | 78.7% |
|  | Random Forest | 81.2% | 61.5% | **85.1%** |
| 70% | Logistic | **85.3%** | **72.9%** | 74.5% |
|  | Random Forest | 82.4% | 66.0% | 74.5% |

50~60% Threshold에서는 Random Forest의 Recall이 더 높았습니다. 즉 실제 부실기업을 조금 더 많이 탐지했습니다. 대신 Precision과 Accuracy는 Logistic Regression보다 낮았습니다.

70% Threshold에서는 두 모델의 Recall이 74.5%로 동일했지만, Logistic Regression의 Precision은 72.9%로 Random Forest의 66.0%보다 높았습니다.

### 6.6 Confusion Matrix 비교

#### Threshold = 50%

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 94 | 29 | 8 | 39 |
| Random Forest | 91 | 32 | 7 | 40 |

Random Forest는 부실기업을 1건 더 탐지해 FN을 8건에서 7건으로 줄였지만, 정상기업 오탐 FP는 29건에서 32건으로 증가했습니다.

#### Threshold = 60%

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic Regression | 104 | 19 | 10 | 37 |
| Random Forest | 98 | 25 | 7 | 40 |

60% Threshold에서는 Random Forest가 Logistic Regression보다 실제 부실기업을 3건 더 탐지해 FN을 10건에서 7건으로 줄였습니다. 반면 정상기업 오탐 FP는 19건에서 25건으로 증가했습니다.

이 결과는 실제 적용에서 **부실기업을 놓치는 비용(FN)**과 **정상기업을 과도하게 경고하는 비용(FP)** 사이의 trade-off를 고려해야 함을 보여줍니다.

---

## 7. 모델 비교 해석 및 선택

이번 비교의 목적은 Random Forest가 Logistic Regression보다 더 복잡하다는 이유만으로 우수하다고 가정하는 것이 아니라, **동일한 데이터 조건에서 복잡도 증가가 실제 예측력 개선으로 이어지는지 확인하는 것**이었습니다.

결과는 다음과 같습니다.

1. ROC-AUC는 Logistic Regression 0.8838, Random Forest 0.8831로 사실상 유사했습니다.
2. Average Precision에서는 Random Forest가 0.7632로 소폭 높았습니다.
3. Random Forest는 50~60% Threshold에서 Recall이 더 높아 부실기업 누락을 줄였습니다.
4. Logistic Regression은 동일 구간에서 Precision과 Accuracy가 더 높아 정상기업 오탐이 상대적으로 적었습니다.
5. Logistic Regression은 계수 방향과 p-value를 통해 변수의 관계를 직접 해석할 수 있습니다.
6. Random Forest에서는 `roa`가 가장 높은 Feature Importance를 보여 Logistic Regression과 공통적으로 수익성의 중요성이 확인됐습니다.

따라서 현재 데이터와 Feature 구성에서는 **Logistic Regression을 주요 해석 모델**, **Random Forest를 비선형 비교 모델**로 두는 것이 적절하다고 판단했습니다.

복잡한 모델을 추가했다고 해서 자동으로 더 좋은 예측 결과가 나오는 것은 아니며, 현재 결과는 오히려 **모델 복잡도보다 데이터와 Feature가 담고 있는 정보량이 다음 개선에서 더 중요할 수 있음**을 보여줍니다.

---

## 8. 정리

이 프로젝트에서 가장 신경 쓴 부분은 모델 자체보다 분석 가능한 데이터셋을 만드는 과정이었습니다.

KRX KIND 공시에서 부실 후보를 만들고, OpenDART 공시 및 사업보고서 재무제표를 연결하고, SQLite에 저장한 뒤 신용분석 관점의 재무변수를 구성했습니다. 이후 과거 시점 데이터로 학습하고 이후 시점 데이터에서 검증하는 Out-of-Time 구조를 적용했습니다.

5개 재무변수만으로 Logistic Regression은 테스트 ROC-AUC 0.8838, Random Forest는 0.8831을 기록했습니다. Random Forest가 더 복잡한 모델임에도 ROC-AUC 개선은 거의 없었고, 두 모델 모두 `roa`를 주요 부실 구분 신호로 활용했습니다.

안정성 변수는 `equity_ratio`를 사용해 **자기자본비율이 낮을수록 부실 가능성이 높아지는 방향**으로 해석했습니다. 이를 통해 단순히 모델의 복잡도를 높이기보다 **해석 가능성, 부실 누락 비용, 정상기업 오탐 비용, 실제 업무 목적에 따른 Threshold 선택**을 함께 고려해야 한다는 점을 확인했습니다.

---

## 9. 한계와 개선 방향

- KIND 공시 기반 라벨은 실제 법적 부도와 완전히 동일하지 않을 수 있습니다.
- 정상군은 무작위 추출 방식이므로 산업, 규모, 연도를 고려한 매칭을 추가하면 더 엄밀한 대조군을 구성할 수 있습니다.
- DART API 응답 실패나 공시 데이터 상태에 따라 재수집 결과가 일부 달라질 수 있습니다.
- 현재 모델은 특정 사업연도의 재무 수준을 중심으로 구성되어 있어, 다년도 데이터를 활용한 매출·ROA·자기자본비율·현금흐름 변화율을 추가할 여지가 있습니다.
- 산업별 재무구조 차이를 반영하기 위해 산업 평균 대비 상대지표를 추가할 수 있습니다.
- 향후 XGBoost 등 다른 모델을 추가 비교할 수 있지만, 이번 Random Forest 비교 결과를 고려하면 모델 수를 늘리는 것보다 데이터 및 Feature 확장을 우선 검토하는 것이 더 자연스러운 개선 방향입니다.