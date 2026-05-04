# GitHub 업로드 및 웹앱 배포 가이드

이 프로젝트는 Streamlit 웹앱입니다. GitHub는 코드를 저장하고 버전 관리하는 곳이고, 실제 웹앱 실행은 Streamlit Community Cloud에 GitHub 저장소를 연결해서 배포합니다.

## 1. 현재 준비 상태

현재 프로젝트는 Git 저장소로 초기화되어 있고 첫 커밋이 만들어진 상태입니다.

커밋:

```text
Initial stock analysis web app
```

GitHub에 올릴 파일:

- `app.py`
- `requirements.txt`
- `.streamlit/config.toml`
- `stock_analysis/`
- `scripts/`
- `README.md`
- `PROGRAM_MANUAL.md`
- 실행용 CLI 파일들

GitHub에 올리지 않는 파일:

- `outputs/`
- `__pycache__/`
- `.venv/`

제외 설정은 `.gitignore`에 들어 있습니다.

## 2. GitHub CLI 설치

PowerShell에서 실행:

```powershell
winget install --id GitHub.cli -e
```

설치 확인:

```powershell
gh --version
```

## 3. GitHub 로그인

```powershell
gh auth login
```

추천 선택:

```text
GitHub.com
HTTPS
Login with a web browser
```

로그인 확인:

```powershell
gh auth status
```

## 4. GitHub 저장소 생성 및 push

프로젝트 폴더에서 실행:

```powershell
cd "C:\Users\JKKIM\OneDrive\Desktop\auto"
```

비공개 저장소로 만들기:

```powershell
gh repo create stock-analysis-app --private --source=. --remote=origin --push
```

공개 저장소로 만들고 싶으면:

```powershell
gh repo create stock-analysis-app --public --source=. --remote=origin --push
```

업로드 후 확인:

```powershell
git remote -v
git status -sb
```

## 5. Streamlit Community Cloud 배포

1. https://share.streamlit.io 접속
2. GitHub 계정으로 로그인
3. `Create app` 클릭
4. `Yup, I have an app` 선택
5. GitHub 저장소 선택
6. Branch: `main`
7. Main file path: `app.py`
8. Advanced settings에서 Python version은 `3.12` 권장
9. Deploy 클릭

배포가 완료되면 다음과 같은 주소가 만들어집니다.

```text
https://원하는이름.streamlit.app
```

## 6. 이후 수정/업데이트 방법

파일 수정 후:

```powershell
git status -sb
git add .
git commit -m "Update stock analysis app"
git push
```

GitHub에 push하면 Streamlit Community Cloud가 변경사항을 감지해서 앱을 다시 배포합니다.

## 7. 주의사항

### 7.1 무료 배포 환경 속도

3단계와 5단계는 시가총액 상위 종목을 많이 처리하므로 시간이 오래 걸릴 수 있습니다. 무료 웹 환경에서는 처음부터 300개 전체를 돌리기보다 아래처럼 작은 값으로 테스트하는 것을 권장합니다.

```text
시총 수집 수: 30
실행/계산 제한 수: 5
Top N: 3
```

### 7.2 저장 결과

웹 배포 환경의 파일 시스템은 영구 저장소처럼 쓰기 어렵습니다. 장기적으로는 결과 저장을 DB나 Google Sheets, Supabase, S3 같은 외부 저장소로 옮기는 것이 좋습니다.

현재 기본 구조에서는 `outputs/`에 CSV/JSON을 저장합니다.

### 7.3 Python 버전

로컬에서는 Python 3.12 사용을 권장합니다. Streamlit Community Cloud도 배포 시 Advanced settings에서 Python 3.12를 선택하는 것을 권장합니다.

## 8. 공식 문서

- Streamlit Community Cloud 배포: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit 앱 파일 구조: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
- Streamlit 의존성 설정: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
