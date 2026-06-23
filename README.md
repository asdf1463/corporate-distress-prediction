# KRX KIND & OpenDART 부실기업 예측 프로젝트

KRX KIND 공시와 OpenDART 재무제표 데이터를 연결해 상장기업의 부실 가능성을 예측해본 프로젝트입니다.

모델 자체보다도 **부실기업 라벨을 어떻게 만들고**, **정상 대조군을 어떻게 구성하고**, **재무제표 값을 분석 변수로 어떻게 바꾸는지**를 하나의 파이프라인으로 정리하는 데 초점을 뒀습니다.

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
  - 라벨 분포, 연도 분포, 결측치, 이상치 확인
  ↓
ANALYZE.py
  - GLM 로지스틱 회귀 학습 및 평가
```

## 사용 데이터

| 구분 | 내용 |
|---|---|
| KRX KIND | 관리종목, 투자주의환기종목, 상장폐지 관련 공시 엑셀 |
| OpenDART | 기업 고유번호, 공시 목록, 사업보고서 재무제표 API |
| 저장소 | SQLite `dart_finance.db` |
| 재현성 보완 | `normal_sample_snapshot.csv` |

정상 대조군은 DART 상장사 중 부실 후보를 제외하고 무작위로 추출했습니다. DART 기업목록은 실행 시점에 따라 바뀔 수 있어서, 정상군 후보와 사업연도는 `normal_sample_snapshot.csv`로 고정했습니다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `KIND_EXCEL.py` | KIND 엑셀에서 부실 후보군 생성 |
| `DART_API.py` | DART 매칭, 정상군 구성, 재무제표 수집 |
| `normal_sample_snapshot.csv` | 정상 대조군 후보 고정용 snapshot |
| `FEATURES.py` | 재무제표 원천값을 분석 변수로 변환 |
| `TEST.py` | 데이터 분포와 이상치 점검 |
| `ANALYZE.py` | 로지스틱 회귀 분석 |
| `dart_finance.db` | 실행 결과가 저장된 SQLite DB |

## 모델 변수

해석 가능성을 우선해서 5개 재무 변수를 사용했습니다.

| 변수 | 의미 |
|---|---|
| `roa` | 순이익 / 총자산 (수익성) |
| `signed_log_ocf` | 영업현금흐름의 부호 유지 로그값 (현금흐름) |
| `asset_turnover` | 매출액 / 총자산 (회전성) |
| `debt_to_assets` | 총부채 / 총자산 (재무 안정성) |
| `log_assets` | 총자산 로그값 (기업 규모) |

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

이미 생성된 `dart_finance.db`가 있으면 API 키 없이도 아래 분석 단계는 실행할 수 있습니다.

```bash
python TEST.py
python ANALYZE.py
```

## 결과 요약

| 항목 | 결과 |
|---|---:|
| 최종 분석 데이터 | 903건 |
| 정상 / 부실 | 599건 / 304건 |
| 학습 / 테스트 | 733건 / 170건 |
| ROC-AUC | 0.8838 |

Threshold별 테스트 성능은 아래와 같습니다.

| Threshold | 정확도 | 정밀도 | 재현율 |
|---:|---:|---:|---:|
| 50% | 78.2% | 57.4% | 83.0% |
| 60% | 82.9% | 66.1% | 78.7% |
| 70% | 85.3% | 72.9% | 74.5% |

threshold를 높이면 오탐은 줄어들지만 부실기업을 놓칠 가능성은 커집니다. 반대로 threshold를 낮추면 재현율은 올라가지만 정상기업을 부실로 경고하는 경우가 늘어납니다.

회귀계수의 방향은 일반적인 재무 해석과 대체로 일치했습니다. 수익성, 현금흐름, 자산 효율성, 기업 규모가 낮을(작을)수록 부실 가능성이 커지고, 부채 비중이 높을수록 부실 가능성이 커지는 방향이었습니다.

자세한 내용은 [REPORT.md](REPORT.md)에 정리했습니다.
