from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from stock_analysis.auto_runner import run_watchlist
from stock_analysis.predictor import Prediction, predict_next_day
from stock_analysis.top_candidates import run_top_market_cap_screen
from stock_analysis.valuation import ValuationResult, calculate_target_price
from stock_analysis.valuation_screen import run_valuation_screen


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"


st.set_page_config(
    page_title="Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    render_header()
    render_sidebar()

    tabs = st.tabs(
        [
            "1. 단일 예측",
            "2. 관심종목",
            "3. 상승확률 Top",
            "4. 목표주가",
            "5. 상승여력 Top",
            "결과",
        ]
    )

    with tabs[0]:
        render_single_prediction()
    with tabs[1]:
        render_watchlist_run()
    with tabs[2]:
        render_top_probability()
    with tabs[3]:
        render_single_valuation()
    with tabs[4]:
        render_top_valuation()
    with tabs[5]:
        render_results()


def render_header() -> None:
    st.title("주식 분석 프로그램")
    st.caption("1-5단계 실행 및 결과 조회")


def render_sidebar() -> None:
    st.sidebar.header("실행 기본값")
    st.session_state.default_period = st.sidebar.selectbox("가격 데이터 기간", ["2y", "5y", "10y"], index=1)
    st.session_state.sleep_seconds = st.sidebar.number_input("종목 간 대기(초)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    st.session_state.retries = st.sidebar.number_input("재시도", min_value=0, max_value=5, value=1, step=1)
    st.sidebar.divider()
    st.sidebar.caption("출력 폴더")
    st.sidebar.code(str(OUTPUTS), language=None)


def render_single_prediction() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        ticker = st.text_input("종목코드", value="005930", key="single_pred_ticker")
        exchange = st.selectbox("거래소", ["", "KS", "KQ", "KOSPI", "KOSDAQ"], index=1, key="single_pred_exchange")
        period = st.selectbox("조회기간", ["2y", "5y", "10y"], index=1, key="single_pred_period")
        threshold = st.slider("상승 판단 기준", 0.1, 0.9, 0.5, 0.05)
        run = st.button("예측 실행", type="primary", use_container_width=True)

    with right:
        if run:
            with st.spinner("예측 중"):
                prediction = predict_next_day(
                    ticker,
                    exchange=exchange or None,
                    period=period,
                    threshold=threshold,
                )
            render_prediction_card(prediction)
        else:
            render_latest_csv_preview("outputs", "predictions.csv", key_prefix="single_prediction_latest")


def render_watchlist_run() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        watchlist = st.text_input("관심종목 CSV", value="watchlist.csv")
        period = st.selectbox("조회기간", ["2y", "5y", "10y"], index=1, key="watchlist_period")
        limit_enabled = st.checkbox("실행 종목 수 제한", value=False)
        limit = st.number_input("제한 수", min_value=1, max_value=300, value=5, disabled=not limit_enabled)
        run = st.button("관심종목 실행", type="primary", use_container_width=True)

    with right:
        if run:
            progress = st.progress(0)
            status = st.empty()

            def on_progress(index: int, total: int, row: dict[str, Any]) -> None:
                progress.progress(index / total)
                status.write(f"{index}/{total} {row.get('ticker') or row.get('ticker_input')} {row.get('status')}")

            with st.spinner("관심종목 실행 중"):
                summary = run_watchlist(
                    watchlist_path=watchlist,
                    period=period,
                    limit=int(limit) if limit_enabled else None,
                    sleep_seconds=float(st.session_state.sleep_seconds),
                    retries=int(st.session_state.retries),
                    progress=on_progress,
                )
            render_summary_metrics(summary)
            render_csv(summary.csv_path, key_prefix="watchlist_summary")
        else:
            render_latest_csv_preview("outputs", "predictions.csv", key_prefix="watchlist_latest")


def render_top_probability() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        rank_limit = st.number_input("시총 수집 수", min_value=1, max_value=300, value=30, step=10, key="prob_rank")
        run_limit_enabled = st.checkbox("학습 종목 수 제한", value=True, key="prob_run_limit_enabled")
        run_limit = st.number_input("학습 제한 수", min_value=1, max_value=300, value=5, step=1, disabled=not run_limit_enabled)
        top = st.number_input("Top N", min_value=1, max_value=50, value=10, step=1, key="prob_top")
        market = st.selectbox("시장", ["ALL", "KOSPI", "KOSDAQ"], key="prob_market")
        exclude_preferred = st.checkbox("우선주 제외", value=True, key="prob_exclude_pref")
        period = st.selectbox("조회기간", ["2y", "5y", "10y"], index=1, key="prob_period")
        run = st.button("상승확률 Top 실행", type="primary", use_container_width=True)

    with right:
        if run:
            progress = st.progress(0)
            status = st.empty()

            def on_progress(index: int, total: int, row: dict[str, Any]) -> None:
                progress.progress(index / total)
                status.write(f"{index}/{total} {row.get('ticker') or row.get('ticker_input')} {row.get('status')}")

            with st.spinner("상승확률 Top 계산 중"):
                summary = run_top_market_cap_screen(
                    rank_limit=int(rank_limit),
                    run_limit=int(run_limit) if run_limit_enabled else None,
                    top=int(top),
                    market=market,
                    exclude_preferred=exclude_preferred,
                    period=period,
                    sleep_seconds=float(st.session_state.sleep_seconds),
                    retries=int(st.session_state.retries),
                    progress=on_progress,
                )
            render_summary_metrics(summary)
            render_csv(summary.top_csv_path, key_prefix="top_probability_summary")
        else:
            render_latest_csv_preview("outputs/top_market_cap", "top*.csv", key_prefix="top_probability_latest")


def render_single_valuation() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        ticker = st.text_input("종목코드", value="005930", key="valuation_ticker")
        exchange = st.selectbox("거래소", ["", "KS", "KQ", "KOSPI", "KOSDAQ"], index=1, key="valuation_exchange")
        use_custom = st.checkbox("가정 직접 입력", value=False)
        target_pe = st.number_input("목표 PER", min_value=0.1, max_value=100.0, value=12.0, disabled=not use_custom)
        target_pbr = st.number_input("목표 PBR", min_value=0.1, max_value=20.0, value=1.4, disabled=not use_custom)
        growth = st.number_input("DCF 성장률", min_value=-0.2, max_value=0.3, value=0.04, step=0.01, disabled=not use_custom)
        discount_rate = st.number_input("할인율", min_value=0.01, max_value=0.5, value=0.10, step=0.01)
        run = st.button("목표주가 계산", type="primary", use_container_width=True)

    with right:
        if run:
            with st.spinner("목표주가 계산 중"):
                result = calculate_target_price(
                    ticker,
                    exchange=exchange or None,
                    target_pe=float(target_pe) if use_custom else None,
                    target_pbr=float(target_pbr) if use_custom else None,
                    growth=float(growth) if use_custom else None,
                    discount_rate=float(discount_rate),
                )
            render_valuation_card(result)
        else:
            st.info("대기 중")


def render_top_valuation() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        rank_limit = st.number_input("시총 수집 수", min_value=1, max_value=300, value=30, step=10, key="val_rank")
        run_limit_enabled = st.checkbox("계산 종목 수 제한", value=True, key="val_run_limit_enabled")
        run_limit = st.number_input("계산 제한 수", min_value=1, max_value=300, value=5, step=1, disabled=not run_limit_enabled)
        top = st.number_input("Top N", min_value=1, max_value=50, value=10, step=1, key="val_top")
        market = st.selectbox("시장", ["ALL", "KOSPI", "KOSDAQ"], key="val_market")
        include_preferred = st.checkbox("우선주 포함", value=False)
        use_custom = st.checkbox("공통 가정 직접 입력", value=False, key="val_custom")
        target_pe = st.number_input("목표 PER", min_value=0.1, max_value=100.0, value=12.0, disabled=not use_custom, key="val_pe")
        target_pbr = st.number_input("목표 PBR", min_value=0.1, max_value=20.0, value=1.4, disabled=not use_custom, key="val_pbr")
        growth = st.number_input("DCF 성장률", min_value=-0.2, max_value=0.3, value=0.04, step=0.01, disabled=not use_custom, key="val_growth")
        run = st.button("상승여력 Top 실행", type="primary", use_container_width=True)

    with right:
        if run:
            progress = st.progress(0)
            status = st.empty()

            def on_progress(index: int, total: int, row: dict[str, Any]) -> None:
                progress.progress(index / total)
                ticker_value = row.get("ticker") or row.get("ticker_input")
                status.write(f"{index}/{total} {ticker_value} {row.get('status')}")

            with st.spinner("상승여력 Top 계산 중"):
                summary = run_valuation_screen(
                    rank_limit=int(rank_limit),
                    run_limit=int(run_limit) if run_limit_enabled else None,
                    top=int(top),
                    market=market,
                    exclude_preferred=not include_preferred,
                    target_pe=float(target_pe) if use_custom else None,
                    target_pbr=float(target_pbr) if use_custom else None,
                    growth=float(growth) if use_custom else None,
                    sleep_seconds=float(st.session_state.sleep_seconds),
                    retries=int(st.session_state.retries),
                    progress=on_progress,
                )
            render_summary_metrics(summary)
            render_csv(summary.top_csv_path, key_prefix="top_valuation_summary")
        else:
            render_latest_csv_preview("outputs/target_market_cap", "top*.csv", key_prefix="top_valuation_latest")


def render_results() -> None:
    result_files = list_result_files()
    if not result_files:
        st.info("저장된 결과가 없습니다.")
        return

    selected = st.selectbox("결과 파일", result_files, format_func=lambda path: str(path.relative_to(ROOT)))
    render_csv(selected, key_prefix="results")


def render_prediction_card(prediction: Prediction) -> None:
    metrics = prediction.metrics
    st.subheader(f"{prediction.ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최근 종가", f"{prediction.latest_close:,.2f}")
    c2.metric("상승 확률", f"{prediction.probability_up:.2%}")
    c3.metric("예측", "상승" if prediction.signal == "UP" else "하락")
    c4.metric("검증 정확도", f"{metrics.accuracy:.2%}")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "최근 거래일": prediction.latest_date,
                    "하락 확률": prediction.probability_down,
                    "단순 기준 정확도": metrics.baseline_accuracy,
                    "상승 precision": metrics.precision_up,
                    "상승 recall": metrics.recall_up,
                    "학습 행": metrics.train_rows,
                    "검증 행": metrics.test_rows,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_valuation_card(result: ValuationResult) -> None:
    snapshot = result.snapshot
    st.subheader(f"{result.ticker} · {snapshot.name}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"{result.current_price:,.2f} {snapshot.currency}")
    c2.metric("목표주가", f"{result.target_price:,.2f}")
    c3.metric("상승여력", f"{result.upside:.2%}")
    c4.metric("ROE", "N/A" if snapshot.roe is None else f"{snapshot.roe:.2%}")

    methods = pd.DataFrame([method.to_dict() for method in result.methods])
    if not methods.empty:
        st.dataframe(methods, use_container_width=True, hide_index=True)

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "기준일": snapshot.statement_date,
                    "EPS": snapshot.eps,
                    "BPS": snapshot.book_value_per_share,
                    "FCF/share": snapshot.free_cash_flow_per_share,
                    "매출 성장률": snapshot.revenue_growth,
                    "순이익 성장률": snapshot.net_income_growth,
                    "FCF 성장률": snapshot.free_cash_flow_growth,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_summary_metrics(summary: Any) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체", getattr(summary, "total", getattr(summary, "evaluated_count", 0)))
    c2.metric("성공", getattr(summary, "succeeded", 0))
    c3.metric("실패", getattr(summary, "failed", 0))
    c4.metric("저장", Path(getattr(summary, "output_dir", "")).name)


def render_latest_csv_preview(base_dir: str, pattern: str, *, key_prefix: str) -> None:
    latest = latest_file(OUTPUTS / base_dir if not base_dir.startswith("outputs") else ROOT / base_dir, pattern)
    if latest:
        st.caption(f"최근 결과: {latest.relative_to(ROOT)}")
        render_csv(latest, key_prefix=key_prefix)
    else:
        st.info("최근 결과가 없습니다.")


def render_csv(path: str | Path, *, key_prefix: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        st.warning(f"파일을 찾을 수 없습니다: {file_path}")
        return
    frame = pd.read_csv(file_path)
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.download_button(
        "CSV 다운로드",
        file_path.read_bytes(),
        file_name=file_path.name,
        mime="text/csv",
        key=f"download_{key_prefix}_{abs(hash(str(file_path.resolve())))}",
        use_container_width=True,
    )


def list_result_files() -> list[Path]:
    if not OUTPUTS.exists():
        return []
    files = [path for path in OUTPUTS.rglob("*.csv") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def latest_file(base_dir: Path, pattern: str) -> Path | None:
    if not base_dir.exists():
        return None
    files = [path for path in base_dir.rglob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
