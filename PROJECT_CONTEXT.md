# Project Context

이 문서는 앞으로 이 프로젝트를 이어서 작업할 때 기준점으로 쓰는 메모입니다.

## Canonical References

- GitHub repository: `https://github.com/jhonkim91/stock-analysis-app`
- Streamlit deployment: `https://stockanalysis-5o8kxiqrrrrfwccgn7bvkg.streamlit.app/`
- Local workspace root: `C:\Users\JKKIM\OneDrive\Desktop\auto`

## Product Summary

이 프로젝트는 Python + Streamlit 기반 주식 분석 웹앱입니다.

주요 기능:

1. 단일 종목 다음 거래일 상승/하락 예측
2. 관심종목 일괄 실행
3. 시가총액 상위 종목 상승확률 Top 선별
4. 단일 종목 목표주가 계산
5. 시가총액 상위 종목 상승여력 Top 선별
6. 코스피 공포탐욕 지수 조회

## Current App Direction

- 종목 입력은 `코드/이름` 둘 다 지원
- 관심종목은 검색 후 바로 등록 가능
- 사용자별 관심종목과 결과를 분리 저장
- Supabase가 연결되면 실제 로그인/영구 저장 사용
- Supabase가 없으면 로컬 프로필 모드로 fallback

## Data Ownership Rules

이 프로젝트에서 GitHub에 올리는 것:

- 소스코드
- 문서
- 설정 예시 파일
- DB 스키마

GitHub에 올리지 않는 것:

- `.streamlit/secrets.toml`
- 실행 결과 `outputs/`
- 로컬 사용자 데이터 `data/`
- 실제 서비스 계정/토큰

실서비스 사용자 데이터의 기준 저장소:

- 관심종목: Supabase
- 로그인 계정: Supabase Auth
- 분석 결과 스냅샷: Supabase

## Important Files

- `app.py`: Streamlit 웹앱 진입점
- `stock_analysis/stock_search.py`: 종목명/티커 검색 및 해석
- `stock_analysis/user_store.py`: 로컬 사용자/관심종목 저장
- `stock_analysis/supabase_store.py`: Supabase 인증/저장 연동
- `stock_analysis/predictor.py`: 다음 거래일 상승/하락 예측
- `stock_analysis/valuation.py`: 목표주가 산출
- `stock_analysis/fear_greed.py`: 공포탐욕 지수 로딩
- `supabase/schema.sql`: Supabase 테이블/RLS 정책
- `SUPABASE_SETUP.md`: Supabase 연결 절차

## Working Preference

앞으로 작업은 가능하면 이 GitHub 저장소 기준으로 진행합니다.

- 변경사항은 로컬에서 수정
- 검증 후 Git 커밋
- 원격 `origin`에 푸시
- 배포 반영이 필요하면 Streamlit Cloud에서 확인

## Immediate Next Steps

1. 현재 미커밋 변경사항을 GitHub에 푸시
2. Supabase 프로젝트 생성 및 `schema.sql` 적용
3. Streamlit Cloud Secrets에 Supabase 값 등록
4. 배포 앱에서 회원가입/로그인/관심종목 영구 저장 검증
