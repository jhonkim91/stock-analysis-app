# 주식 분석 프로그램

단계별로 확장하는 주식 분석 실험 프로젝트입니다. 현재는 **5단계: 시가총액 상위 종목의 목표주가 대비 상승여력 Top 10 선별**까지 구현되어 있습니다.

> 주의: 이 프로그램은 학습/연구용입니다. 출력 결과는 투자 조언이 아니며, 실제 매수/매도 판단에는 사용할 수 없습니다.

## 현재 구현 상태

1. 완료: 특정 종목의 다음 거래일 상승/하락 예측
2. 완료: 관심종목 데이터를 자동 수집하고 1단계를 일괄 실행
3. 완료: 시가총액 상위 300개 종목 학습 후 상승 확률 상위 10개 선정
4. 완료: 특정 종목의 재무 데이터 기반 목표주가 산출
5. 완료: 시가총액 상위 300개 종목의 목표주가 대비 상승여력 상위 10개 선정

## 설치

```powershell
python -m pip install -r requirements.txt
```

## 웹앱 실행

로컬 웹앱:

```powershell
python -m streamlit run app.py
```

또는 스크립트:

```powershell
.\scripts\run_web.ps1
```

브라우저에서 접속:

```text
http://localhost:8501
```

웹앱에는 1-5단계 실행 탭과 저장된 CSV 결과 조회 탭이 포함되어 있습니다. 3단계와 5단계는 시간이 오래 걸릴 수 있으므로 처음에는 `시총 수집 수`와 `실행/계산 제한 수`를 작게 두고 테스트하는 구성이 기본입니다.

## 1단계: 단일 종목 예측

미국 종목:

```powershell
python main.py --ticker AAPL --period 5y
```

한국 KOSPI 종목:

```powershell
python main.py --ticker 005930.KS --period 5y
```

6자리 코드만 입력하면 기본적으로 `.KS`를 붙입니다.

```powershell
python main.py --ticker 005930 --period 5y
```

KOSDAQ 종목은 거래소를 지정합니다.

```powershell
python main.py --ticker 091990 --exchange KQ --period 5y
```

## 2단계: 관심종목 자동 실행

`watchlist.csv`에 실행할 종목을 등록합니다.

```csv
ticker,exchange,name,enabled
AAPL,,Apple,1
MSFT,,Microsoft,1
005930,KS,Samsung Electronics,1
000660,KS,SK Hynix,1
035420,KS,NAVER,1
```

자동 실행:

```powershell
python auto_run.py
```

## 3단계: 시가총액 상위 300개 중 상승 확률 Top 10

전체 실행:

```powershell
python top300_run.py
```

테스트용 일부 실행:

```powershell
python top300_run.py --rank-limit 30 --run-limit 5 --top 3 --period 2y
```

결과 파일:

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

## 4단계: 재무 데이터 기반 목표주가 산출

특정 종목의 재무 데이터를 가져와 목표주가를 산출합니다.

```powershell
python target_price.py --ticker AAPL
python target_price.py --ticker 005930 --exchange KS
python target_price.py --ticker 091990 --exchange KQ
```

직접 가정 지정:

```powershell
python target_price.py --ticker 005930 --exchange KS --target-pe 12 --target-pbr 1.4 --growth 0.04 --discount-rate 0.10
```

계산 방식:

- PER 기반 목표가: `EPS x 목표 PER`
- PBR 기반 목표가: `BPS x 목표 PBR`
- DCF 기반 목표가: FCF를 성장률/할인율/영구성장률로 할인
- 최종 목표가: 사용 가능한 방식만 가중 평균

## 5단계: 목표주가 상승여력 Top 10

시가총액 상위 300개 종목의 목표주가를 일괄 계산하고, 현재가 대비 상승여력이 가장 높은 10개 종목을 고릅니다.

```powershell
python target300_run.py
```

안정적으로 전체 실행:

```powershell
python target300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2
```

테스트용 일부 실행:

```powershell
python target300_run.py --rank-limit 30 --run-limit 5 --top 3
```

KOSPI만 실행:

```powershell
python target300_run.py --market KOSPI
```

직접 가정 지정:

```powershell
python target300_run.py --target-pe 12 --target-pbr 1.4 --growth 0.04 --discount-rate 0.10
```

5단계는 기본적으로 우선주로 보이는 종목을 제외합니다. Yahoo Finance의 우선주 재무제표와 주식수 데이터가 보통주 기준으로 섞이는 경우가 있어 랭킹이 왜곡될 수 있기 때문입니다. 우선주까지 포함하려면 다음 옵션을 사용합니다.

```powershell
python target300_run.py --include-preferred
```

결과 파일:

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

## 자동 스크립트

2단계 관심종목 자동 실행을 Windows 작업 스케줄러에 등록:

```powershell
.\scripts\register_daily_task.ps1 -Time "18:30"
```

3단계 Top 300 실행:

```powershell
.\scripts\run_top300.ps1
```

4단계 목표주가 실행:

```powershell
.\scripts\run_target_price.ps1 -Ticker 005930 -Exchange KS
```

5단계 목표주가 상승여력 Top 10 실행:

```powershell
.\scripts\run_target300.ps1
```

## 데이터 소스

- 시가총액 목록 기본값: 네이버 금융 시가총액 페이지
- 가격 데이터: Yahoo Finance
- 재무 데이터: Yahoo Finance
- `pykrx`도 설치되어 있으며, 필요하면 `python top300_run.py --source pykrx --date 20260430`처럼 사용할 수 있습니다.
