# 주식 분석 프로그램 분석 설명서

작성일: 2026-05-04  
프로젝트 위치: `c:\Users\JKKIM\OneDrive\Desktop\auto`

> 중요: 이 프로그램은 학습, 연구, 자동화 실험용입니다. 예측 결과와 목표주가는 투자 조언이 아니며 실제 매수/매도 판단의 근거로 단독 사용하면 안 됩니다.

## 1. 프로그램 개요

이 프로그램은 Python으로 만든 주식 분석 자동화 도구입니다. 처음에는 명령어로 실행하는 CLI 프로그램으로 만들었고, 이후 Streamlit 웹앱을 붙여 브라우저에서 1-5단계를 실행하고 결과를 확인할 수 있게 구성했습니다.

현재 구현된 전체 기능은 다음과 같습니다.

1. 특정 종목의 다음 거래일 상승/하락 예측
2. 관심종목 목록을 자동 실행하여 여러 종목 예측
3. 시가총액 상위 300개 종목 중 다음 거래일 상승 확률 Top 10 선정
4. 특정 종목의 재무 데이터 기반 목표주가 산출
5. 시가총액 상위 300개 종목 중 목표주가 대비 상승여력 Top 10 선정
6. Streamlit 웹앱에서 1-5단계 실행 및 CSV 결과 조회

## 2. 사용 기술과 라이브러리

주요 언어는 Python입니다.

사용 라이브러리:

- `pandas`: 주가/재무/결과 데이터 처리
- `numpy`: 수치 계산, 직접 구현한 로지스틱 회귀 학습
- `yfinance`: Yahoo Finance에서 주가와 재무 데이터 수집
- `requests`: 네이버 금융 HTML 페이지 요청
- `beautifulsoup4`: 네이버 금융 시가총액 표 파싱
- `pykrx`: KRX 데이터 수집용 선택지로 설치되어 있음
- `streamlit`: 로컬 웹앱 UI

의존성 파일:

```powershell
requirements.txt
```

설치:

```powershell
python -m pip install -r requirements.txt
```

## 3. 주요 파일 구조

```text
auto/
  app.py                         Streamlit 웹앱
  main.py                        1단계 단일 종목 예측 실행
  auto_run.py                    2단계 관심종목 자동 실행
  top300_run.py                  3단계 상승확률 Top 실행
  target_price.py                4단계 단일 종목 목표주가 실행
  target300_run.py               5단계 상승여력 Top 실행
  watchlist.csv                  관심종목 목록
  requirements.txt               Python 패키지 목록
  README.md                      간단 사용법
  PROGRAM_MANUAL.md              현재 문서

  stock_analysis/
    data.py                      주가 데이터 수집/정리
    features.py                  기술적 지표 생성
    model.py                     로지스틱 회귀 모델
    predictor.py                 단일 종목 예측 조립
    auto_runner.py               관심종목 일괄 실행
    market_cap.py                시가총액 상위 종목 수집
    top_candidates.py            상승확률 Top 후보 선정
    valuation.py                 재무 데이터 기반 목표주가 계산
    valuation_screen.py          목표주가 상승여력 Top 후보 선정
    *_cli.py                     각 기능별 CLI 인터페이스

  scripts/
    run_web.ps1                  웹앱 실행
    run_auto.ps1                 2단계 실행
    run_top300.ps1               3단계 실행
    run_target_price.ps1         4단계 실행
    run_target300.ps1            5단계 실행
    register_daily_task.ps1      Windows 작업 스케줄러 등록

  outputs/
    실행 결과 CSV/JSON 저장 폴더
```

## 4. 웹앱 실행 방법

가장 추천하는 사용 방식은 웹앱입니다.

```powershell
.\scripts\run_web.ps1
```

또는 직접 실행:

```powershell
python -m streamlit run app.py
```

브라우저 접속:

```text
http://localhost:8501
```

웹앱 탭 구성:

- `1. 단일 예측`: 특정 종목 다음 거래일 상승/하락 예측
- `2. 관심종목`: `watchlist.csv`에 등록된 종목을 일괄 예측
- `3. 상승확률 Top`: 시가총액 상위 종목 중 상승 확률 Top 선정
- `4. 목표주가`: 특정 종목의 목표주가 계산
- `5. 상승여력 Top`: 시가총액 상위 종목 중 목표주가 상승여력 Top 선정
- `결과`: 저장된 CSV 결과 조회 및 다운로드

## 5. 데이터 소스

### 5.1 가격 데이터

가격 데이터는 Yahoo Finance에서 가져옵니다.

사용 모듈:

```text
stock_analysis/data.py
```

수집 컬럼:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

한국 종목 처리:

- `005930`처럼 6자리 코드를 입력하면 기본적으로 `005930.KS`로 변환
- KOSPI: `.KS`
- KOSDAQ: `.KQ`

예:

```powershell
python main.py --ticker 005930 --exchange KS
python main.py --ticker 091990 --exchange KQ
```

### 5.2 시가총액 데이터

시가총액 상위 종목은 기본적으로 네이버 금융 시가총액 페이지에서 가져옵니다.

사용 모듈:

```text
stock_analysis/market_cap.py
```

기본 수집 대상:

- KOSPI
- KOSDAQ
- 두 시장을 합쳐 시가총액 기준 정렬
- 기본 상위 300개

`pykrx`도 설치되어 있지만, 현재 기본값은 네이버 금융입니다.

### 5.3 재무 데이터

재무 데이터는 Yahoo Finance의 `Ticker` 객체에서 가져옵니다.

사용 모듈:

```text
stock_analysis/valuation.py
```

사용 데이터:

- 현재가
- 시가총액
- 발행주식수
- 매출
- 순이익
- 잉여현금흐름, FCF
- 영업현금흐름
- 설비투자, CapEx
- 자기자본
- 총부채
- 현금
- EPS
- BPS
- ROE
- 매출 성장률
- 순이익 성장률
- FCF 성장률
- PER/PBR 관련 값

## 6. 1단계: 특정 종목 다음 거래일 상승/하락 예측

실행 파일:

```text
main.py
```

실행 예:

```powershell
python main.py --ticker AAPL --period 5y
python main.py --ticker 005930 --exchange KS --period 5y
```

### 6.1 분석에 사용하는 기술적 지표

사용 모듈:

```text
stock_analysis/features.py
```

현재 생성하는 특징값은 다음과 같습니다.

- `return_1d`: 1일 수익률
- `return_2d`: 2일 수익률
- `return_5d`: 5일 수익률
- `return_10d`: 10일 수익률
- `price_vs_sma_5`: 종가와 5일 이동평균 괴리율
- `price_vs_sma_20`: 종가와 20일 이동평균 괴리율
- `price_vs_sma_60`: 종가와 60일 이동평균 괴리율
- `sma_5_vs_20`: 5일 이동평균과 20일 이동평균 괴리율
- `sma_20_vs_60`: 20일 이동평균과 60일 이동평균 괴리율
- `volatility_5`: 5일 수익률 변동성
- `volatility_20`: 20일 수익률 변동성
- `volume_change_1d`: 전일 대비 거래량 변화율
- `volume_vs_sma_20`: 거래량과 20일 평균 거래량 괴리율
- `intraday_range`: 하루 고가-저가 범위
- `close_position`: 당일 저가~고가 범위에서 종가의 위치
- `gap_open`: 전일 종가 대비 당일 시가 갭
- `rsi_14`: 14일 RSI

### 6.2 예측 대상

예측 대상은 다음 거래일 종가가 오늘 종가보다 높으면 `1`, 아니면 `0`입니다.

```text
target_up = 다음 거래일 종가 > 현재 거래일 종가
```

즉, 이 프로그램은 다음 거래일의 절대 가격을 맞히는 것이 아니라 상승/하락 방향을 분류합니다.

### 6.3 예측 모델

사용 모듈:

```text
stock_analysis/model.py
```

모델은 `numpy`로 직접 구현한 로지스틱 회귀입니다.

현재 설정:

- 모델: Logistic Regression
- 학습률: `0.05`
- 학습 반복: `2500 epoch`
- L2 정규화: `0.001`
- 입력값 표준화: 평균 0, 표준편차 1
- 출력값: 상승 확률

외부 머신러닝 라이브러리인 `scikit-learn`은 사용하지 않았습니다. 가볍게 실행되도록 직접 구현했습니다.

### 6.4 검증 방식

데이터를 랜덤으로 섞지 않고 시간순으로 나눕니다.

기본값:

- 앞쪽 80%: 학습
- 뒤쪽 20%: 검증

이렇게 한 이유는 주식 데이터가 시간 순서를 가지기 때문입니다. 미래 데이터를 과거 학습에 섞으면 실제보다 좋은 성능처럼 보이는 문제가 생깁니다.

출력 지표:

- 정확도
- 단순 기준 정확도
- 상승 precision
- 상승 recall
- 학습 행 수
- 검증 행 수

`단순 기준 정확도`는 검증 구간에서 상승/하락 중 더 많이 나온 쪽만 계속 찍었을 때의 정확도입니다. 모델 정확도가 이 값보다 낮으면 그 기간에서는 모델 신호가 약하다고 봐야 합니다.

## 7. 2단계: 관심종목 자동 실행

실행 파일:

```text
auto_run.py
```

관심종목 파일:

```text
watchlist.csv
```

형식:

```csv
ticker,exchange,name,enabled
AAPL,,Apple,1
MSFT,,Microsoft,1
005930,KS,Samsung Electronics,1
000660,KS,SK Hynix,1
035420,KS,NAVER,1
```

실행:

```powershell
python auto_run.py
```

일부만 테스트:

```powershell
python auto_run.py --limit 2 --period 2y
```

저장 위치:

```text
outputs/
  실행시각/
    predictions.csv
    predictions.json
```

결과는 상승 확률 높은 순서로 정렬됩니다. 실패한 종목은 `status=failed`, `error` 컬럼에 원인이 기록됩니다.

## 8. 3단계: 시가총액 상위 300개 중 상승 확률 Top 10

실행 파일:

```text
top300_run.py
```

실행:

```powershell
python top300_run.py
```

안정적인 전체 실행:

```powershell
python top300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2
```

테스트용:

```powershell
python top300_run.py --rank-limit 30 --run-limit 5 --top 3 --period 2y
```

처리 순서:

1. 네이버 금융에서 KOSPI/KOSDAQ 시가총액 표 수집
2. 시가총액 기준 상위 N개 종목 선정
3. 각 종목의 Yahoo Finance 가격 데이터 수집
4. 1단계 로지스틱 회귀 모델을 종목별로 학습
5. 최신 거래일 기준 다음 거래일 상승 확률 계산
6. 상승 확률 높은 순서로 Top N 저장

저장 위치:

```text
outputs/
  top_market_cap/
    실행시각/
      market_cap_universe.csv
      market_cap_watchlist.csv
      predictions.csv
      predictions.json
      top10.csv
      top10.json
```

파일 설명:

- `market_cap_universe.csv`: 시가총액 상위 종목 원본 목록
- `market_cap_watchlist.csv`: 실제 학습 대상으로 변환된 목록
- `predictions.csv`: 전체 예측 결과
- `top10.csv`: 상승 확률 상위 10개

## 9. 4단계: 특정 종목 목표주가 산출

실행 파일:

```text
target_price.py
```

실행:

```powershell
python target_price.py --ticker AAPL
python target_price.py --ticker 005930 --exchange KS
```

가정을 직접 지정:

```powershell
python target_price.py --ticker 005930 --exchange KS --target-pe 12 --target-pbr 1.4 --growth 0.04 --discount-rate 0.10
```

### 9.1 목표주가 계산 방식

사용 모듈:

```text
stock_analysis/valuation.py
```

현재는 3가지 방식을 함께 사용합니다.

### 9.2 PER 기반 목표주가

```text
PER 목표가 = EPS x 목표 PER
```

EPS가 양수이고 목표 PER이 존재할 때 계산합니다.

기본 목표 PER은 자동 계산합니다.

대략적인 구조:

```text
목표 PER = 10 + ROE 보너스 + 성장률 보너스
```

범위는 8-30 사이로 제한합니다.

### 9.3 PBR 기반 목표주가

```text
PBR 목표가 = BPS x 목표 PBR
```

BPS가 양수이고 목표 PBR이 존재할 때 계산합니다.

기본 목표 PBR은 ROE를 기반으로 자동 계산합니다.

범위는 0.5-5.0 사이로 제한합니다.

### 9.4 DCF 기반 목표주가

DCF는 Free Cash Flow를 이용합니다.

기본 가정:

- 명시 예측 기간: 5년
- 할인율: 10%
- 영구성장률: 2%
- 성장률: 매출 성장률, 순이익 성장률, FCF 성장률에서 추정
- 성장률 제한: -5%에서 15%

계산 구조:

1. 현재 FCF를 성장률로 5년간 증가시킴
2. 각 연도 FCF를 할인율로 현재가치화
3. 마지막 해 이후는 영구성장률로 터미널 가치를 계산
4. 순현금, 즉 현금 - 부채를 반영
5. 발행주식수로 나누어 주당 가치 계산

### 9.5 최종 목표주가

사용 가능한 방식만 가중 평균합니다.

기본 가중치:

- PER: 45%
- PBR: 15%
- DCF: 40%

예를 들어 DCF 계산에 필요한 FCF가 없으면 PER/PBR만 남은 가중치로 재조정됩니다.

출력:

- 현재가
- 목표주가
- 상승여력
- EPS/BPS/FCF/share/ROE
- 매출/순이익/FCF 성장률
- 방법별 목표주가
- 적용 가정

## 10. 5단계: 시가총액 상위 300개 중 목표주가 상승여력 Top 10

실행 파일:

```text
target300_run.py
```

실행:

```powershell
python target300_run.py
```

안정적인 전체 실행:

```powershell
python target300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2
```

테스트용:

```powershell
python target300_run.py --rank-limit 30 --run-limit 5 --top 3
```

처리 순서:

1. 시가총액 상위 종목 수집
2. 각 종목의 Yahoo Finance 재무 데이터 수집
3. 4단계 방식으로 목표주가 계산
4. 현재가 대비 상승여력 계산
5. 상승여력이 높은 순서로 Top N 저장

저장 위치:

```text
outputs/
  target_market_cap/
    실행시각/
      market_cap_universe.csv
      market_cap_watchlist.csv
      valuations.csv
      valuations.json
      top10.csv
      top10.json
```

파일 설명:

- `market_cap_universe.csv`: 시가총액 상위 종목 목록
- `market_cap_watchlist.csv`: 계산 대상 목록
- `valuations.csv`: 전체 목표주가 계산 결과
- `valuations.json`: 전체 목표주가 결과와 상세 계산 데이터
- `top10.csv`: 상승여력 상위 10개
- `top10.json`: 상승여력 상위 10개 JSON

5단계는 기본적으로 우선주를 제외합니다. Yahoo Finance에서 우선주의 재무제표와 발행주식수 데이터가 보통주 기준으로 섞이는 경우가 있어 목표주가가 크게 왜곡될 수 있기 때문입니다.

우선주 포함:

```powershell
python target300_run.py --include-preferred
```

## 11. 웹앱 사용 설명

웹앱 실행:

```powershell
.\scripts\run_web.ps1
```

접속:

```text
http://localhost:8501
```

### 11.1 사이드바

사이드바에는 공통 실행값이 있습니다.

- 가격 데이터 기간
- 종목 간 대기 초
- 재시도 횟수
- 결과 저장 폴더

3단계와 5단계처럼 여러 종목을 처리할 때는 데이터 제공처 요청 제한을 피하기 위해 종목 간 대기 시간을 1초 정도 주는 것을 권장합니다.

### 11.2 1. 단일 예측

입력:

- 종목코드
- 거래소
- 조회기간
- 상승 판단 기준

버튼:

- `예측 실행`

출력:

- 최근 종가
- 상승 확률
- 하락 확률
- 상승/하락 예측
- 검증 정확도

### 11.3 2. 관심종목

입력:

- 관심종목 CSV 경로
- 조회기간
- 실행 종목 수 제한

출력:

- 전체 처리 수
- 성공/실패 수
- 예측 결과 테이블
- CSV 다운로드

### 11.4 3. 상승확률 Top

입력:

- 시총 수집 수
- 학습 종목 수 제한
- Top N
- 시장
- 우선주 제외 여부
- 조회기간

출력:

- 상승 확률 Top 종목
- 시가총액 순위
- 예측 확률
- 검증 지표

### 11.5 4. 목표주가

입력:

- 종목코드
- 거래소
- 목표 PER/PBR/성장률 직접 입력 여부
- 할인율

출력:

- 현재가
- 목표주가
- 상승여력
- ROE
- 방식별 목표가
- 핵심 재무 데이터

### 11.6 5. 상승여력 Top

입력:

- 시총 수집 수
- 계산 종목 수 제한
- Top N
- 시장
- 우선주 포함 여부
- 공통 가치평가 가정

출력:

- 목표주가 상승여력 Top 종목
- 현재가
- 목표주가
- 상승여력
- 계산 방식

### 11.7 결과 탭

`outputs/` 아래 저장된 CSV 파일을 찾아 보여줍니다.

기능:

- 최근 결과 조회
- 과거 결과 선택
- 테이블 확인
- CSV 다운로드

## 12. 자동 실행 스크립트

웹앱 실행:

```powershell
.\scripts\run_web.ps1
```

관심종목 자동 실행:

```powershell
.\scripts\run_auto.ps1
```

3단계 실행:

```powershell
.\scripts\run_top300.ps1
```

4단계 목표주가 실행:

```powershell
.\scripts\run_target_price.ps1 -Ticker 005930 -Exchange KS
```

5단계 실행:

```powershell
.\scripts\run_target300.ps1
```

Windows 작업 스케줄러 등록:

```powershell
.\scripts\register_daily_task.ps1 -Time "18:30"
```

## 13. 결과 파일 컬럼 설명

### 13.1 예측 결과 주요 컬럼

- `run_at`: 실행 시각
- `status`: 성공/실패
- `name`: 종목명
- `ticker_input`: 입력 종목코드
- `exchange`: 거래소
- `ticker`: Yahoo Finance용 종목코드
- `latest_date`: 최근 거래일
- `latest_close`: 최근 종가
- `signal`: 상승/하락 예측
- `probability_up`: 상승 확률
- `probability_down`: 하락 확률
- `accuracy`: 검증 정확도
- `baseline_accuracy`: 단순 기준 정확도
- `precision_up`: 상승 precision
- `recall_up`: 상승 recall
- `train_rows`: 학습 행 수
- `test_rows`: 검증 행 수
- `error`: 실패 원인

### 13.2 목표주가 결과 주요 컬럼

- `market_cap_rank`: 시가총액 순위
- `market`: KOSPI/KOSDAQ
- `market_cap`: 시가총액
- `current_price`: 현재가
- `target_price`: 목표주가
- `upside`: 상승여력
- `upside_percent`: 상승여력 퍼센트
- `methods_used`: 사용된 계산 방식
- `per_target`: PER 방식 목표가
- `pbr_target`: PBR 방식 목표가
- `dcf_target`: DCF 방식 목표가
- `eps`: 주당순이익
- `bps`: 주당순자산
- `fcf_per_share`: 주당 잉여현금흐름
- `roe`: 자기자본이익률
- `revenue_growth`: 매출 성장률
- `net_income_growth`: 순이익 성장률
- `free_cash_flow_growth`: FCF 성장률
- `target_pe`: 적용 목표 PER
- `target_pbr`: 적용 목표 PBR
- `dcf_growth`: DCF 성장률
- `discount_rate`: 할인율
- `terminal_growth`: 영구성장률

## 14. 현재 프로그램의 장점

- Python만으로 실행 가능
- 웹앱과 CLI 모두 지원
- 결과를 CSV/JSON으로 자동 저장
- 단일 종목과 대량 종목 분석 모두 지원
- 가격 기반 단기 예측과 재무 기반 목표주가 계산을 분리
- 한국 종목 코드 자동 변환 지원
- 시가총액 상위 종목 자동 수집 가능

## 15. 현재 한계점

### 15.1 예측 모델 한계

현재 상승/하락 예측 모델은 단순 로지스틱 회귀입니다. 복잡한 시장 구조, 뉴스, 수급, 금리, 환율, 섹터 흐름 등은 반영하지 않습니다.

또한 종목마다 개별 모델을 빠르게 학습하는 방식이라, 장기적으로 안정적인 알파 모델이라고 보기는 어렵습니다.

### 15.2 백테스트 부재

현재는 단순 시간순 검증만 있습니다. 실제 매매전략으로 평가하려면 다음이 필요합니다.

- 매수/매도 규칙
- 거래비용
- 슬리피지
- 보유기간
- 최대낙폭
- 승률
- 기대수익
- 벤치마크 대비 성과

### 15.3 데이터 품질 한계

Yahoo Finance와 네이버 금융은 무료 데이터 소스입니다. 일부 종목은 다음 문제가 생길 수 있습니다.

- 데이터 누락
- 재무제표 항목 누락
- 우선주 데이터 왜곡
- 분할/병합/상장폐지 반영 문제
- 한국 종목 재무 데이터 지연

### 15.4 목표주가 모델 한계

현재 목표주가는 PER, PBR, DCF를 단순 가중 평균합니다. 업종별 적정 멀티플, 경쟁사 비교, 이익 전망치, 사업부별 가치평가 등은 아직 반영하지 않습니다.

## 16. 추천 개선 방향

우선순위 높은 개선:

1. 백테스트 기능 추가
2. 업종별 PER/PBR 기준 적용
3. 결과 DB 저장
4. 매일 자동 실행 후 최신 결과만 웹앱에 표시
5. 차트 추가
6. 종목 상세 페이지 추가
7. 예측 모델 성능 비교
8. 에러 종목 자동 재시도/제외 목록 관리

모델 개선 후보:

- Random Forest
- Gradient Boosting
- XGBoost/LightGBM
- 시계열 교차검증
- 섹터/시장지수/거래대금 특징 추가
- 외국인/기관 수급 데이터 추가
- 뉴스/공시 감성 분석

목표주가 개선 후보:

- 업종별 멀티플
- 경쟁사 비교 valuation
- 애널리스트 컨센서스 반영
- ROE/PBR 회귀모델
- 배당할인모형
- 잔여이익모형

## 17. 빠른 실행 명령 모음

웹앱:

```powershell
.\scripts\run_web.ps1
```

1단계:

```powershell
python main.py --ticker 005930 --exchange KS --period 5y
```

2단계:

```powershell
python auto_run.py
```

3단계:

```powershell
python top300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2
```

4단계:

```powershell
python target_price.py --ticker 005930 --exchange KS
```

5단계:

```powershell
python target300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2
```

## 18. 결론

현재 프로그램은 1-5단계의 기본 골격이 완성된 상태입니다. 단기 방향 예측, 관심종목 자동 분석, 시가총액 상위 종목 스크리닝, 재무 기반 목표주가 계산, 목표주가 상승여력 랭킹, 웹앱 실행까지 모두 가능합니다.

다만 지금 단계는 “작동하는 분석 자동화 기본판”입니다. 실제 투자 판단에 가까운 도구로 발전시키려면 백테스트, 데이터 검증, 업종별 가치평가, 모델 성능 비교, 결과 저장 DB, 자동 스케줄링 고도화가 다음 핵심 작업입니다.
