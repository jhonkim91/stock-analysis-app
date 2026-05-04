# Supabase Setup

이 앱은 기본적으로 로컬 프로필 모드로 동작합니다.  
Supabase를 연결하면 `회원가입/로그인`, `관심종목 영구 저장`, `계정별 실행 결과 저장`이 활성화됩니다.

## 1. Supabase 프로젝트 생성

1. Supabase에서 새 프로젝트를 만듭니다.
2. `Authentication > Providers`에서 Email 로그인을 활성화합니다.
3. 필요하면 `Confirm email` 정책을 확인합니다.

## 2. DB 스키마 적용

Supabase SQL Editor에서 아래 파일 내용을 실행하세요.

- [supabase/schema.sql](C:/Users/JKKIM/OneDrive/Desktop/auto/supabase/schema.sql)

이 스키마는 아래 테이블을 만듭니다.

- `public.watchlist_items`
- `public.analysis_snapshots`

그리고 `auth.uid() = user_id` 기준 RLS 정책도 같이 설정합니다.

## 3. Streamlit secrets 설정

로컬 개발:

1. [.streamlit/secrets.example.toml](C:/Users/JKKIM/OneDrive/Desktop/auto/.streamlit/secrets.example.toml)을 참고해서
2. `.streamlit/secrets.toml` 파일을 만들고 값을 채웁니다.

예시:

```toml
[supabase]
url = "https://your-project-id.supabase.co"
key = "your-supabase-anon-or-publishable-key"
```

Streamlit Community Cloud:

1. 앱 설정의 `Secrets` 메뉴로 이동
2. 같은 내용을 붙여넣기

## 4. 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

## 5. 앱 실행

```powershell
python -m streamlit run app.py
```

## 동작 방식

- Supabase 미설정: 로컬 프로필 모드
- Supabase 설정 + 로그인 전: 임시 게스트 모드
- Supabase 설정 + 로그인 후: 계정 기반 영구 저장 모드

## 저장 범위

- 관심종목: Supabase `watchlist_items`
- 실행 결과 요약/행 데이터: Supabase `analysis_snapshots`
- 로컬 CSV 결과: `outputs/users/<user_id>/...`

## 참고

회원가입 후 바로 세션이 생기지 않으면, 이메일 인증이 필요한 설정입니다.  
이 경우 메일 인증을 완료한 뒤 로그인하면 됩니다.
