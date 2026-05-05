from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

from stock_analysis.auto_runner import run_watchlist
from stock_analysis.fear_greed import fetch_fear_greed_data
from stock_analysis.predictor import Prediction, predict_next_day
from stock_analysis.stock_search import search_stock_candidates
from stock_analysis.supabase_store import (
    AuthenticatedUser,
    export_cloud_watchlist_csv,
    get_analysis_snapshot,
    list_analysis_snapshots,
    list_cloud_watchlist_items,
    load_supabase_config,
    remove_cloud_watchlist_item,
    restore_session,
    save_analysis_snapshot,
    save_cloud_watchlist_item,
    sign_in_with_password,
    sign_out as sign_out_supabase,
    sign_up_with_password,
)
from stock_analysis.top_candidates import run_top_market_cap_screen
from stock_analysis.user_store import (
    ensure_user,
    export_watchlist_csv,
    list_user_ids,
    list_watchlist_items,
    normalize_user_id,
    remove_watchlist_item,
    save_watchlist_item,
    user_data_dir,
    user_outputs_dir,
)
from stock_analysis.valuation import ValuationResult, calculate_target_price
from stock_analysis.valuation_screen import run_valuation_screen


ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
AUTH_STATE_KEYS = ("auth_access_token", "auth_refresh_token", "auth_user_id", "auth_email")


st.set_page_config(
    page_title="Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    inject_responsive_styles()
    render_header()
    render_sidebar()

    tabs = st.tabs(
        [
            "대시보드",
            "1. 단일 예측",
            "2. 관심종목",
            "3. 상승확률 Top",
            "4. 목표주가",
            "5. 상승여력 Top",
            "결과",
        ]
    )

    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_single_prediction()
    with tabs[2]:
        render_watchlist_run()
    with tabs[3]:
        render_top_probability()
    with tabs[4]:
        render_single_valuation()
    with tabs[5]:
        render_top_valuation()
    with tabs[6]:
        render_results()


def render_header() -> None:
    st.title("주식 분석 프로그램")
    st.caption("1-5단계 실행 및 결과 조회")


def inject_responsive_styles() -> None:
    st.markdown(
        """
        <style>
        @media (max-width: 768px) {
          .block-container {
            padding-top: 1rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
            padding-bottom: 2rem !important;
          }

          div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.75rem !important;
          }

          div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
          }

          button, div.stButton > button {
            min-height: 2.6rem !important;
          }

          div[data-baseweb="tab-list"] {
            gap: 0.25rem !important;
            overflow-x: auto !important;
            scrollbar-width: none;
          }

          div[data-baseweb="tab-list"]::-webkit-scrollbar {
            display: none;
          }

          div[data-baseweb="tab"] {
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
            white-space: nowrap !important;
            font-size: 0.88rem !important;
          }

          div[data-testid="stMetric"] {
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
          }

          div[data-testid="stDataFrame"] {
            overflow-x: auto !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    user_id = get_active_user_id()
    st.subheader("오늘의 요약")
    st.caption("자주 보는 핵심 정보만 먼저 모아봤습니다.")
    detail_open = str(st.query_params.get("fg_detail", "0")) == "1"

    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("**내일 상승확률 높은 종목**")
        top_frame = load_dashboard_top_probability(user_id)
        if top_frame.empty:
            st.info("아직 저장된 상승확률 결과가 없습니다. `3. 상승확률 Top`을 한 번 실행하면 여기서 바로 볼 수 있습니다.")
        else:
            render_dashboard_top_cards(top_frame)

        st.markdown("**등록된 관심종목**")
        watchlist_frame = load_dashboard_watchlist(user_id)
        if watchlist_frame.empty:
            st.info("등록된 관심종목이 없습니다. `2. 관심종목` 탭에서 검색해서 추가해보세요.")
        else:
            st.dataframe(watchlist_frame, use_container_width=True, hide_index=True)

    with right:
        st.markdown("**공포탐욕 지수**")
        render_dashboard_fear_greed()

    if detail_open:
        st.divider()
        st.markdown("<div id='fear-greed-detail'></div>", unsafe_allow_html=True)
        close_cols = st.columns([0.82, 0.18])
        with close_cols[1]:
            if st.button("상세 닫기", use_container_width=True, key="fear_greed_close_detail"):
                try:
                    del st.query_params["fg_detail"]
                except Exception:
                    st.query_params["fg_detail"] = "0"
                st.rerun()
        render_fear_greed_detail_section()


def load_dashboard_top_probability(user_id: str) -> pd.DataFrame:
    latest_top = latest_file(user_outputs_dir(user_id) / "top_market_cap", "top*.csv")
    if latest_top is None:
        latest_top = latest_file(user_outputs_dir(user_id) / "watchlist_runs", "predictions.csv")
    if latest_top is None or not latest_top.exists():
        return pd.DataFrame()

    frame = pd.read_csv(latest_top)
    normalized = frame.copy()
    rename_map = {
        "ticker": "종목코드",
        "name": "종목명",
        "probability_up": "상승확률",
        "signal": "판단",
        "latest_close": "종가",
        "up_probability": "상승확률",
    }
    normalized = normalized.rename(columns={key: value for key, value in rename_map.items() if key in normalized.columns})

    if "상승확률" not in normalized.columns:
        return normalized.head(5)

    normalized["상승확률"] = pd.to_numeric(normalized["상승확률"], errors="coerce")
    normalized = normalized.sort_values("상승확률", ascending=False)
    display_columns = [column for column in ["종목명", "종목코드", "상승확률", "판단", "종가"] if column in normalized.columns]
    if not display_columns:
        return normalized.head(5)

    return normalized.loc[:, display_columns].head(5).copy()


def load_dashboard_watchlist(user_id: str) -> pd.DataFrame:
    items = list_saved_watchlist(user_id)
    if not items:
        return pd.DataFrame()

    frame = pd.DataFrame(
        [
            {
                "종목명": item.name or item.symbol,
                "종목코드": item.symbol,
                "거래소": item.exchange or "-",
            }
            for item in items[:10]
        ]
    )
    return frame


def render_dashboard_top_cards(frame: pd.DataFrame) -> None:
    rows = frame.reset_index(drop=True).to_dict(orient="records")
    for start in range(0, len(rows), 2):
        columns = st.columns(2, gap="medium")
        for column, row in zip(columns, rows[start : start + 2]):
            with column:
                render_dashboard_top_card(row)


def render_dashboard_top_card(row: dict[str, Any]) -> None:
    name = str(row.get("종목명") or row.get("종목코드") or "-")
    ticker = str(row.get("종목코드") or "-")
    probability_raw = pd.to_numeric(row.get("상승확률"), errors="coerce")
    probability_label = f"{probability_raw:.1%}" if pd.notna(probability_raw) else "-"
    signal = str(row.get("판단") or "-")
    close_raw = pd.to_numeric(row.get("종가"), errors="coerce")
    close_label = f"{close_raw:,.0f}" if pd.notna(close_raw) else "-"
    signal_label, accent_color, badge_bg, badge_fg = dashboard_signal_style(signal)
    strength_label, _ = describe_prediction_strength(float(probability_raw) if pd.notna(probability_raw) else 0.5)

    with st.container(border=True):
        st.caption(ticker)
        st.markdown(f"**{name}**")
        st.markdown(
            (
                f"<div style='display:flex; gap:8px; align-items:center; margin:0.15rem 0 0.6rem 0;'>"
                f"<span style='display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; "
                f"background:{badge_bg}; color:{badge_fg}; font-size:0.82rem; font-weight:600;'>{signal_label}</span>"
                f"<span style='display:inline-block; padding:0.2rem 0.55rem; border-radius:999px; "
                f"background:#f3f4f6; color:#111827; font-size:0.82rem; font-weight:600;'>{strength_label}</span>"
                f"</div>"
            ),
            unsafe_allow_html=True,
        )
        metric_left, metric_right = st.columns(2)
        metric_left.metric("상승확률", probability_label)
        metric_right.metric("판단", signal)
        st.markdown(
            f"<div style='color:{accent_color}; font-size:0.95rem; font-weight:600;'>최근 종가 {close_label}</div>",
            unsafe_allow_html=True,
        )


def dashboard_signal_style(signal: str) -> tuple[str, str, str, str]:
    normalized = signal.strip().upper()
    if normalized == "UP":
        return ("상승 우세", "#15803d", "#dcfce7", "#166534")
    if normalized == "DOWN":
        return ("하락 우세", "#c2410c", "#ffedd5", "#9a3412")
    return ("중립", "#475569", "#e2e8f0", "#334155")


def render_dashboard_fear_greed() -> None:
    try:
        data = load_fear_greed_data()
    except Exception as exc:
        st.warning(f"공포탐욕 지수를 불러오지 못했습니다: {exc}")
        return

    current = data.summary.iloc[0]
    score = float(current["score"])
    status = str(current["status"])
    st.markdown(build_fear_greed_dashboard_card(data), unsafe_allow_html=True)

    compact_summary = data.summary.copy()
    compact_summary = compact_summary.rename(
        columns={
            "period": "기간",
            "score": "점수",
            "status": "상태",
            "change": "변화",
        }
    )
    keep_columns = [column for column in ["기간", "점수", "상태", "변화"] if column in compact_summary.columns]
    st.dataframe(compact_summary.loc[:, keep_columns], use_container_width=True, hide_index=True)

    if data.source_age_days > 7:
        st.caption(f"원본 데이터가 {data.source_age_days}일 전 기준이라 최신 반영이 늦을 수 있습니다.")


def fear_greed_palette(score: float) -> tuple[str, str, str]:
    if score <= 24:
        return ("#dc2626", "#fee2e2", "#991b1b")
    if score <= 44:
        return ("#ea580c", "#ffedd5", "#9a3412")
    if score <= 54:
        return ("#ca8a04", "#fef3c7", "#854d0e")
    if score <= 74:
        return ("#16a34a", "#dcfce7", "#166534")
    return ("#059669", "#d1fae5", "#065f46")


def build_fear_greed_dashboard_card(data) -> str:
    current = data.summary.iloc[0]
    score = float(current["score"])
    status = str(current["status"])
    accent_color, badge_bg, badge_fg = fear_greed_palette(score)
    gauge_html = build_fear_greed_gauge(score=score, status=status, accent_color=accent_color)
    factors_html = build_fear_greed_factor_rows(data.factors)

    return (
        "<a href='?fg_detail=1#fear-greed-detail' style='text-decoration:none; color:inherit; display:block;'>"
        "<div style='border:1px solid #e5e7eb; background:#f8fafc; border-radius:16px; padding:18px 18px 16px 18px; cursor:pointer;'>"
        "<div style='display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:10px;'>"
        "<div>"
        "<div style='font-size:0.78rem; letter-spacing:0.18em; text-transform:uppercase; color:#64748b; margin-bottom:0.35rem;'>한국 시장</div>"
        "<div style='font-size:1.55rem; font-weight:800; color:#0f172a;'>KOSPI · KOSDAQ</div>"
        "</div>"
        f"<div style='text-align:right; font-size:0.8rem; color:#64748b;'>기준일<br><strong style=\"font-size:0.96rem; color:#0f172a;\">{data.latest_date}</strong></div>"
        "</div>"
        f"{gauge_html}"
        "<div style='height:1px; background:#e2e8f0; margin:14px 0 12px 0;'></div>"
        f"{factors_html}"
        "<div style='margin-top:12px; display:flex; align-items:center; justify-content:space-between; gap:12px;'>"
        f"<span style='display:inline-block; padding:0.28rem 0.7rem; border-radius:999px; background:{badge_bg}; color:{badge_fg}; font-size:0.84rem; font-weight:700;'>{status}</span>"
        "<span style='font-size:0.76rem; color:#64748b;'>게이지를 누르면 상세 지표가 열립니다</span>"
        "</div>"
        "</div>"
        "</a>"
    )


def build_fear_greed_gauge(*, score: float, status: str, accent_color: str) -> str:
    score = max(0.0, min(100.0, float(score)))
    cx = 150.0
    cy = 150.0
    outer_radius = 110.0
    inner_radius = 82.0
    ring_radius = 64.0
    needle_radius = 88.0
    label_radius = 126.0
    number_radius = 80.0

    segments = [
        (0.0, 25.0, "#ef4444", "극단적\n공포"),
        (25.0, 45.0, "#f97316", "공포"),
        (45.0, 55.0, "#facc15", "중립"),
        (55.0, 75.0, "#22c55e", "탐욕"),
        (75.0, 100.0, "#06b6d4", "극단적\n탐욕"),
    ]

    active_index = 0
    for index, (start, end, _, _) in enumerate(segments):
        if score <= end or index == len(segments) - 1:
            active_index = index
            break

    sector_paths: list[str] = []
    label_nodes: list[str] = []
    for index, (start, end, color, label) in enumerate(segments):
        is_active = index == active_index
        fill = f"{color}22" if is_active else "#ececec"
        stroke = color if is_active else "#f8fafc"
        text_color = "#111827" if is_active else "#737373"
        sector_paths.append(
            build_fear_greed_sector(
                start=start,
                end=end,
                cx=cx,
                cy=cy,
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                fill=fill,
                stroke=stroke,
                stroke_width=2.0 if is_active else 1.5,
            )
        )
        label_nodes.append(
            build_fear_greed_sector_label(
                start=start,
                end=end,
                label=label,
                cx=cx,
                cy=cy,
                radius=label_radius,
                color=text_color,
            )
        )

    tick_nodes: list[str] = []
    for value in range(0, 101, 5):
        angle = fear_greed_angle(value)
        x1, y1 = polar_to_cartesian(cx, cy, ring_radius + 2, angle)
        x2, y2 = polar_to_cartesian(cx, cy, ring_radius + (7 if value % 25 == 0 else 4), angle)
        width = 2 if value % 25 == 0 else 1
        tick_nodes.append(
            f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' stroke='#9ca3af' stroke-width='{width}' stroke-linecap='round' />"
        )

    number_nodes: list[str] = []
    for value in [0, 25, 45, 55, 75, 100]:
        angle = fear_greed_angle(float(value))
        x, y = polar_to_cartesian(cx, cy, number_radius, angle)
        number_nodes.append(
            f"<text x='{x:.2f}' y='{y + 4:.2f}' text-anchor='middle' font-size='11' font-weight='700' fill='#6b7280'>{value}</text>"
        )

    angle = fear_greed_angle(score)
    nx, ny = polar_to_cartesian(cx, cy, needle_radius, angle)

    return (
        "<div style='display:flex; justify-content:center; padding:8px 0 4px 0;'>"
        "<svg width='100%' viewBox='0 0 300 206' style='max-width:430px;'>"
        f"{''.join(sector_paths)}"
        f"{''.join(label_nodes)}"
        f"{''.join(tick_nodes)}"
        f"{''.join(number_nodes)}"
        f"<line x1='{cx:.2f}' y1='{cy:.2f}' x2='{nx:.2f}' y2='{ny:.2f}' stroke='#1f2937' stroke-width='6' stroke-linecap='round' />"
        f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='8' fill='#1f2937' />"
        f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='35' fill='#ffffff' opacity='0.98' />"
        f"<text x='{cx:.2f}' y='{cy + 2:.2f}' text-anchor='middle' font-size='22' font-weight='900' fill='#111827'>{int(round(score))}</text>"
        "</svg>"
        "</div>"
    )


def build_fear_greed_sector(
    *,
    start: float,
    end: float,
    cx: float,
    cy: float,
    outer_radius: float,
    inner_radius: float,
    fill: str,
    stroke: str,
    stroke_width: float,
) -> str:
    start_angle = fear_greed_angle(start)
    end_angle = fear_greed_angle(end)
    x1o, y1o = polar_to_cartesian(cx, cy, outer_radius, start_angle)
    x2o, y2o = polar_to_cartesian(cx, cy, outer_radius, end_angle)
    x1i, y1i = polar_to_cartesian(cx, cy, inner_radius, start_angle)
    x2i, y2i = polar_to_cartesian(cx, cy, inner_radius, end_angle)
    large_arc = 1 if abs(end_angle - start_angle) > 180 else 0
    path = (
        f"M {x1o:.2f} {y1o:.2f} "
        f"A {outer_radius:.2f} {outer_radius:.2f} 0 {large_arc} 1 {x2o:.2f} {y2o:.2f} "
        f"L {x2i:.2f} {y2i:.2f} "
        f"A {inner_radius:.2f} {inner_radius:.2f} 0 {large_arc} 0 {x1i:.2f} {y1i:.2f} Z"
    )
    return f"<path d='{path}' fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width:.1f}' />"


def build_fear_greed_sector_label(
    *,
    start: float,
    end: float,
    label: str,
    cx: float,
    cy: float,
    radius: float,
    color: str,
) -> str:
    angle = (fear_greed_angle(start) + fear_greed_angle(end)) / 2
    x, y = polar_to_cartesian(cx, cy, radius, angle)
    rotate = angle - 90
    parts = label.split("\n")
    lines = []
    if len(parts) == 1:
        lines.append(f"<tspan x='{x:.2f}' dy='0'>{parts[0]}</tspan>")
    else:
        lines.append(f"<tspan x='{x:.2f}' dy='-6'>{parts[0]}</tspan>")
        lines.append(f"<tspan x='{x:.2f}' dy='14'>{parts[1]}</tspan>")
    return (
        f"<text x='{x:.2f}' y='{y:.2f}' text-anchor='middle' font-size='9.5' font-weight='800' "
        f"fill='{color}' transform='rotate({rotate:.2f} {x:.2f} {y:.2f})'>{''.join(lines)}</text>"
    )


def fear_greed_angle(score: float) -> float:
    clamped = max(0.0, min(100.0, float(score)))
    return 180.0 - (clamped / 100.0 * 180.0)


def polar_to_cartesian(cx: float, cy: float, radius: float, angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return (
        cx + radius * math.cos(radians),
        cy - radius * math.sin(radians),
    )


def build_fear_greed_factor_rows(factors: pd.DataFrame) -> str:
    if factors.empty:
        return "<div style='font-size:0.9rem; color:#64748b;'>구성 지표를 불러오지 못했습니다.</div>"

    bar_colors = ["#22c5f3", "#f97316", "#22c55e", "#f97316", "#ef4444", "#22c5f3", "#3b82f6"]
    rows: list[str] = []
    for index, (_, row) in enumerate(factors.head(7).iterrows()):
        color = bar_colors[index % len(bar_colors)]
        factor_name = str(row.get("factor") or row.get("factor_key") or "-")
        score = max(0.0, min(100.0, float(row.get("score_0_100") or 0.0)))
        raw_value = row.get("raw")
        unit = str(row.get("unit") or "")
        if pd.notna(raw_value):
            try:
                numeric_raw = float(raw_value)
                value_label = f"{numeric_raw:+.2f}{unit}" if unit else f"{numeric_raw:.0f}"
            except (TypeError, ValueError):
                value_label = str(raw_value)
        else:
            value_label = f"{score:.0f}"

        rows.append(
            "<div style='display:grid; grid-template-columns:96px 1fr auto; gap:10px; align-items:center; margin:0 0 10px 0;'>"
            f"<div style='font-size:0.86rem; color:#475569; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{factor_name}</div>"
            "<div style='height:8px; border-radius:999px; background:#dbe4f0; overflow:hidden;'>"
            f"<div style='width:{score:.0f}%; height:100%; border-radius:999px; background:{color};'></div>"
            "</div>"
            f"<div style='font-size:0.84rem; font-weight:700; color:{color}; min-width:58px; text-align:right;'>{value_label}</div>"
            "</div>"
        )

    return "".join(rows)


def render_sidebar() -> None:
    st.sidebar.header("사용자")
    config = get_supabase_config()
    auth_user = get_authenticated_user()

    if config:
        render_supabase_auth_section(config, auth_user)
    else:
        recent_users = list_user_ids()
        if recent_users:
            st.sidebar.caption("최근 사용자 ID")
            st.sidebar.code(", ".join(recent_users[:8]), language=None)
        st.sidebar.text_input("사용자 ID", value="guest", key="app_user_id")
        user_id = get_active_user_id()
        st.sidebar.caption(f"현재 저장 프로필: `{user_id}`")

    st.sidebar.divider()
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
        ticker = st.text_input(
            "종목코드/이름",
            value="005930",
            key="single_pred_ticker",
            help="한국 종목코드(예: 005930), 한국 종목명(예: 삼성전자), 미국 티커(예: AAPL), 미국 종목명(예: Apple)을 입력할 수 있습니다.",
        )
        st.caption("예: 005930, 삼성전자, AAPL, Apple")
        st.caption("거래소는 검색 결과를 바탕으로 자동 판별합니다.")
        render_stock_candidates(ticker, "", key_prefix="single_pred")
        period = st.selectbox(
            "조회기간",
            ["2y", "5y", "10y"],
            index=1,
            key="single_pred_period",
            help="모델이 학습할 과거 데이터 길이입니다. 보통 5년이 가장 무난하고, 2년은 최근 흐름에 더 민감합니다.",
        )
        threshold = st.slider(
            "상승 판단 기준",
            0.1,
            0.9,
            0.5,
            0.05,
            help="상승 확률이 이 값 이상이면 '상승 우세'로 표시합니다. 0.50은 기본값, 0.55~0.60은 더 보수적인 기준입니다.",
        )
        auto_threshold = st.checkbox(
            "종목별 기준 자동 추천",
            value=True,
            help="이 종목의 최근 검증 구간에서 더 안정적이었던 threshold 후보를 무료로 자동 탐색해 적용합니다.",
        )
        compare_tree_model = st.checkbox(
            "무료 트리 모델도 비교",
            value=False,
            help=(
                "로지스틱 회귀와 무료 트리 모델(HistGradientBoosting)을 함께 검증해서 "
                "최근 성능이 더 안정적인 쪽을 자동으로 선택합니다. 대신 실행 시간이 조금 더 걸립니다."
            ),
        )
        with st.expander("입력값 설명", expanded=False):
            st.markdown(
                "- `조회기간`: 모델이 참고하는 과거 데이터 길이입니다.\n"
                "- `상승 판단 기준`: 상승 확률을 어디서부터 상승으로 볼지 정합니다.\n"
                "- `종목별 기준 자동 추천`을 켜면 0.40~0.60 사이 후보 중 이 종목에 더 맞는 기준을 자동 적용합니다.\n"
                "- `무료 트리 모델도 비교`를 켜면 기본 로지스틱 회귀와 트리 모델을 둘 다 테스트한 뒤 더 나은 쪽을 사용합니다.\n"
                "- 처음엔 `5y`와 자동 추천 켜짐 상태로 보는 것이 가장 무난합니다."
            )
        apply_market_filter = st.checkbox(
            "시장 심리 필터 같이 보기",
            value=True,
            help="한국 공포탐욕 지수를 함께 읽어서, 과열/과매도 구간에서는 예측 해석을 조금 더 보수적이거나 역발상 관점으로 조정합니다.",
        )
        run = st.button("예측 실행", type="primary", use_container_width=True)

    with right:
        if run:
            with st.spinner("예측 중"):
                prediction = predict_next_day(
                    ticker,
                    period=period,
                    threshold=threshold,
                    compute_walk_forward_metrics=True,
                    optimize_threshold=auto_threshold,
                    compare_tree_model=compare_tree_model,
                )
            market_context = None
            if apply_market_filter:
                try:
                    fear_greed = load_fear_greed_data()
                    current = fear_greed.summary.iloc[0]
                    market_context = {
                        "score": float(current["score"]),
                        "status": str(current["status"]),
                        "date": str(fear_greed.latest_date),
                    }
                except Exception:
                    market_context = None
            render_prediction_card(prediction, market_context=market_context)
        else:
            render_latest_csv_preview("outputs", "predictions.csv", key_prefix="single_prediction_latest")


def render_watchlist_run() -> None:
    user_id = get_active_user_id()
    left, right = st.columns([0.4, 0.6], gap="large")
    with left:
        query = st.text_input("종목 검색", value="", key="watchlist_search_query")
        st.caption("이름이나 코드를 검색하면 거래소를 자동으로 찾아서 바로 등록할 수 있습니다.")
        render_watchlist_search(user_id, query, "")
        st.divider()
        period = st.selectbox("조회기간", ["2y", "5y", "10y"], index=1, key="watchlist_period")
        limit_enabled = st.checkbox("실행 종목 수 제한", value=False)
        limit = st.number_input("제한 수", min_value=1, max_value=300, value=5, disabled=not limit_enabled)
        run = st.button("관심종목 실행", type="primary", use_container_width=True)

    with right:
        render_saved_watchlist(user_id)
        if run:
            watchlist_items = list_saved_watchlist(user_id)
            if not watchlist_items:
                st.warning("먼저 관심종목을 등록하세요.")
                return

            progress = st.progress(0)
            status = st.empty()

            def on_progress(index: int, total: int, row: dict[str, Any]) -> None:
                progress.progress(index / total)
                status.write(f"{index}/{total} {row.get('ticker') or row.get('ticker_input')} {row.get('status')}")

            with st.spinner("관심종목 실행 중"):
                watchlist_path = export_active_watchlist(user_id)
                summary = run_watchlist(
                    watchlist_path=watchlist_path,
                    output_dir=user_outputs_dir(user_id) / "watchlist_runs",
                    period=period,
                    limit=int(limit) if limit_enabled else None,
                    sleep_seconds=float(st.session_state.sleep_seconds),
                    retries=int(st.session_state.retries),
                    progress=on_progress,
                )
            persist_run_snapshot_if_possible(
                run_type="watchlist",
                title="관심종목 예측",
                csv_path=summary.csv_path,
                summary={
                    "total": summary.total,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                    "output_dir": summary.output_dir,
                },
            )
            render_summary_metrics(summary)
            render_csv(summary.csv_path, key_prefix="watchlist_summary")
        else:
            render_latest_csv_preview(user_outputs_dir(user_id) / "watchlist_runs", "predictions.csv", key_prefix="watchlist_latest")


def render_top_probability() -> None:
    user_id = get_active_user_id()
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        rank_limit = st.number_input("시총 수집 수", min_value=1, max_value=300, value=30, step=10, key="prob_rank")
        run_limit_enabled = st.checkbox("학습 종목 수 제한", value=True, key="prob_run_limit_enabled")
        run_limit = st.number_input("학습 제한 수", min_value=1, max_value=300, value=5, step=1, disabled=not run_limit_enabled)
        top = st.number_input("Top N", min_value=1, max_value=50, value=10, step=1, key="prob_top")
        market = st.selectbox("시장", ["ALL", "KOSPI", "KOSDAQ"], key="prob_market")
        exclude_preferred = st.checkbox("우선주 제외", value=True, key="prob_exclude_pref")
        period = st.selectbox("조회기간", ["2y", "5y", "10y"], index=1, key="prob_period")
        with st.expander("추가 필터", expanded=True):
            min_price_enabled = st.checkbox("최소 주가 필터", value=True, key="prob_min_price_enabled")
            min_price = st.number_input(
                "최소 주가",
                min_value=0,
                max_value=500000,
                value=5000,
                step=1000,
                disabled=not min_price_enabled,
                key="prob_min_price",
            )
            min_edge_enabled = st.checkbox("최소 검증 개선폭 필터", value=True, key="prob_min_edge_enabled")
            min_accuracy_edge = st.number_input(
                "최소 개선폭",
                min_value=-0.20,
                max_value=0.20,
                value=0.00,
                step=0.01,
                format="%.2f",
                disabled=not min_edge_enabled,
                key="prob_min_edge",
            )
            market_cap_enabled = st.checkbox("시장별 최대 선택 수 제한", value=False, key="prob_market_cap_enabled")
            max_per_market = st.number_input(
                "시장별 최대 수",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                disabled=not market_cap_enabled,
                key="prob_max_per_market",
            )
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
                    output_dir=user_outputs_dir(user_id) / "top_market_cap",
                    rank_limit=int(rank_limit),
                    run_limit=int(run_limit) if run_limit_enabled else None,
                    top=int(top),
                    market=market,
                    exclude_preferred=exclude_preferred,
                    period=period,
                    sleep_seconds=float(st.session_state.sleep_seconds),
                    retries=int(st.session_state.retries),
                    min_price=float(min_price) if min_price_enabled else None,
                    min_accuracy_edge=float(min_accuracy_edge) if min_edge_enabled else None,
                    max_per_market=int(max_per_market) if market_cap_enabled else None,
                    progress=on_progress,
                )
            persist_run_snapshot_if_possible(
                run_type="top_probability",
                title=f"상승확률 Top {int(top)}",
                csv_path=summary.top_csv_path,
                summary={
                    "universe_count": summary.universe_count,
                    "evaluated_count": summary.evaluated_count,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                    "output_dir": summary.output_dir,
                },
            )
            render_summary_metrics(summary)
            render_csv(summary.top_csv_path, key_prefix="top_probability_summary")
        else:
            render_latest_csv_preview(user_outputs_dir(user_id) / "top_market_cap", "top*.csv", key_prefix="top_probability_latest")


def render_single_valuation() -> None:
    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        ticker = st.text_input("종목코드/이름", value="005930", key="valuation_ticker")
        st.caption("예: 005930, 삼성전자, AAPL, Apple")
        st.caption("거래소는 검색 결과를 바탕으로 자동 판별합니다.")
        render_stock_candidates(ticker, "", key_prefix="valuation")
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
                    target_pe=float(target_pe) if use_custom else None,
                    target_pbr=float(target_pbr) if use_custom else None,
                    growth=float(growth) if use_custom else None,
                    discount_rate=float(discount_rate),
                )
            render_valuation_card(result)
        else:
            st.info("대기 중")


def render_top_valuation() -> None:
    user_id = get_active_user_id()
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
                    output_dir=user_outputs_dir(user_id) / "target_market_cap",
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
            persist_run_snapshot_if_possible(
                run_type="top_valuation",
                title=f"상승여력 Top {int(top)}",
                csv_path=summary.top_csv_path,
                summary={
                    "universe_count": summary.universe_count,
                    "evaluated_count": summary.evaluated_count,
                    "succeeded": summary.succeeded,
                    "failed": summary.failed,
                    "output_dir": summary.output_dir,
                },
            )
            render_summary_metrics(summary)
            render_csv(summary.top_csv_path, key_prefix="top_valuation_summary")
        else:
            render_latest_csv_preview(user_outputs_dir(user_id) / "target_market_cap", "top*.csv", key_prefix="top_valuation_latest")


def render_fear_greed_detail_section() -> None:
    left, right = st.columns([0.36, 0.64], gap="large")
    with left:
        refresh = st.button("지표 새로고침", type="primary", use_container_width=True, key="fear_greed_detail_refresh")
        if refresh:
            st.rerun()

        with st.spinner("공포탐욕 지표 로딩 중"):
            data = load_fear_greed_data()

        current = data.summary.iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("현재 지수", f"{current['score']:.1f}")
        c2.metric("상태", str(current["status"]))
        st.caption(f"원본 사이트: {data.source_url}")
        st.caption(f"원본 데이터 기준일: {data.latest_date}")

        if data.source_age_days > 7:
            today_label = pd.Timestamp.today().date().isoformat()
            st.warning(
                f"원본 데이터 최신일이 {data.latest_date}입니다. "
                f"오늘({today_label}) 기준 {data.source_age_days}일 이전 데이터입니다."
            )

        st.subheader("비교 구간")
        st.dataframe(data.summary, use_container_width=True, hide_index=True)

        st.subheader("세부 팩터")
        st.dataframe(data.factors, use_container_width=True, hide_index=True)

    with right:
        st.subheader("최근 공포탐욕 타임라인")
        recent = data.timeline.tail(120).copy()
        recent_fg = recent.loc[:, ["date", "fear_greed"]].set_index("date")
        st.line_chart(recent_fg, use_container_width=True)

        if "us_fear_greed" in recent.columns and recent["us_fear_greed"].notna().any():
            st.subheader("한국/미국 심리 비교")
            compare_frame = recent.loc[:, ["date", "fear_greed", "us_fear_greed"]].set_index("date")
            st.line_chart(compare_frame, use_container_width=True)
        elif "kospi_close" in recent.columns and recent["kospi_close"].notna().any():
            st.subheader("최근 코스피 지수")
            recent_kospi = recent.loc[:, ["date", "kospi_close"]].set_index("date")
            st.line_chart(recent_kospi, use_container_width=True)

        st.subheader("해석 기준")
        guide = pd.DataFrame(
            [
                {"range": "0-20", "meaning": "극도의 공포"},
                {"range": "20-40", "meaning": "공포"},
                {"range": "40-60", "meaning": "중립"},
                {"range": "60-80", "meaning": "탐욕"},
                {"range": "80-100", "meaning": "극도의 탐욕"},
            ]
        )
        st.dataframe(guide, use_container_width=True, hide_index=True)


def render_results() -> None:
    auth_user = get_authenticated_user()
    config = get_supabase_config()
    rendered_anything = False

    if config and auth_user:
        rendered_anything = render_cloud_results(auth_user, config)
        st.divider()

    user_id = get_active_user_id()
    show_all = st.checkbox("전체 결과 보기", value=False, key="results_show_all")
    result_files = list_result_files(None if show_all else user_id)
    if result_files:
        if not show_all:
            st.caption(f"현재 사용자 `{user_id}` 의 로컬 결과만 표시합니다.")
        selected = st.selectbox("로컬 결과 파일", result_files, format_func=lambda path: str(path.relative_to(ROOT)))
        render_csv(selected, key_prefix="results")
        rendered_anything = True

    if not rendered_anything:
        if show_all:
            st.info("저장된 결과가 없습니다.")
        else:
            st.info("현재 사용자에 저장된 결과가 없습니다.")


def get_supabase_config():
    return load_supabase_config(st.secrets)


def persist_authenticated_user(user: AuthenticatedUser) -> None:
    st.session_state["auth_access_token"] = user.access_token
    st.session_state["auth_refresh_token"] = user.refresh_token
    st.session_state["auth_user_id"] = user.user_id
    st.session_state["auth_email"] = user.email
    st.session_state["_auth_resolved_signature"] = (user.access_token, user.refresh_token)
    st.session_state["_auth_resolved_user"] = user


def clear_authenticated_user() -> None:
    for key in AUTH_STATE_KEYS + ("_auth_resolved_signature", "_auth_resolved_user"):
        st.session_state.pop(key, None)


def get_authenticated_user() -> AuthenticatedUser | None:
    config = get_supabase_config()
    if not config:
        return None

    access_token = st.session_state.get("auth_access_token")
    refresh_token = st.session_state.get("auth_refresh_token")
    if not access_token or not refresh_token:
        return None

    signature = (access_token, refresh_token)
    if st.session_state.get("_auth_resolved_signature") == signature:
        cached = st.session_state.get("_auth_resolved_user")
        if isinstance(cached, AuthenticatedUser):
            return cached

    try:
        user = restore_session(config, access_token, refresh_token)
    except Exception:
        clear_authenticated_user()
        return None

    persist_authenticated_user(user)
    return user


def render_supabase_auth_section(config, auth_user: AuthenticatedUser | None) -> None:
    if auth_user:
        st.sidebar.success(f"로그인됨: {auth_user.email or auth_user.user_id}")
        st.sidebar.caption(f"계정 ID: `{auth_user.user_id}`")
        if st.sidebar.button("로그아웃", use_container_width=True):
            try:
                sign_out_supabase(config, auth_user.access_token, auth_user.refresh_token)
            except Exception:
                pass
            clear_authenticated_user()
            st.rerun()
        return

    mode = st.sidebar.radio("계정 작업", ["로그인", "회원가입"], horizontal=True, key="auth_mode")
    with st.sidebar.form("supabase_auth_form", clear_on_submit=False):
        email = st.text_input("이메일", key="auth_email_input")
        password = st.text_input("비밀번호", type="password", key="auth_password_input")
        submitted = st.form_submit_button(mode, use_container_width=True)

    if submitted:
        try:
            if mode == "로그인":
                user = sign_in_with_password(config, email, password)
                persist_authenticated_user(user)
                st.sidebar.success("로그인되었습니다.")
                st.rerun()
            else:
                result = sign_up_with_password(config, email, password)
                if result.user:
                    persist_authenticated_user(result.user)
                    st.sidebar.success(result.message)
                    st.rerun()
                else:
                    st.sidebar.success(result.message)
        except Exception as exc:
            st.sidebar.error(f"계정 처리 중 오류가 발생했습니다: {exc}")

    guest_id = get_active_user_id()
    st.sidebar.caption(f"임시 프로필: `{guest_id}`")


def get_active_user_id() -> str:
    auth_user = get_authenticated_user()
    if auth_user:
        return auth_user.user_id

    if get_supabase_config():
        guest_id = st.session_state.setdefault("guest_user_id", f"guest-{uuid4().hex[:8]}")
        return ensure_user(normalize_user_id(guest_id))

    raw_value = st.session_state.get("app_user_id", "guest")
    return ensure_user(normalize_user_id(raw_value))


def list_saved_watchlist(user_id: str):
    config = get_supabase_config()
    auth_user = get_authenticated_user()
    if config and auth_user:
        return list_cloud_watchlist_items(config, auth_user)
    return list_watchlist_items(user_id)


def save_watchlist_selection(user_id: str, item: dict[str, Any]) -> None:
    config = get_supabase_config()
    auth_user = get_authenticated_user()
    if config and auth_user:
        save_cloud_watchlist_item(
            config,
            auth_user,
            ticker=item["ticker"],
            exchange=item["exchange"],
            name=item["name"],
            symbol=item["symbol"],
            source=item["source"],
        )
        return

    save_watchlist_item(
        user_id,
        ticker=item["ticker"],
        exchange=item["exchange"],
        name=item["name"],
        symbol=item["symbol"],
        source=item["source"],
    )


def remove_watchlist_selection(user_id: str, symbol: str) -> None:
    config = get_supabase_config()
    auth_user = get_authenticated_user()
    if config and auth_user:
        remove_cloud_watchlist_item(config, auth_user, symbol)
        return
    remove_watchlist_item(user_id, symbol)


def export_active_watchlist(user_id: str) -> Path:
    config = get_supabase_config()
    auth_user = get_authenticated_user()
    if config and auth_user:
        return export_cloud_watchlist_csv(config, auth_user)
    return export_watchlist_csv(user_id)


def persist_run_snapshot_if_possible(
    *,
    run_type: str,
    title: str,
    csv_path: str | Path,
    summary: dict[str, Any],
) -> None:
    config = get_supabase_config()
    auth_user = get_authenticated_user()
    if not config or not auth_user:
        return

    path = Path(csv_path)
    if not path.exists():
        return

    try:
        frame = pd.read_csv(path)
        rows = frame.where(pd.notnull(frame), None).to_dict(orient="records")
        save_analysis_snapshot(config, auth_user, run_type=run_type, title=title, rows=rows, summary=summary)
    except Exception as exc:
        st.info(f"클라우드 결과 저장은 건너뛰었습니다: {exc}")


def render_cloud_results(auth_user: AuthenticatedUser, config) -> bool:
    st.subheader("클라우드 저장 결과")
    try:
        snapshots = list_analysis_snapshots(config, auth_user, limit=30)
    except Exception as exc:
        st.warning(f"Supabase 저장 결과를 불러오지 못했습니다: {exc}")
        return False

    if not snapshots:
        st.caption("아직 Supabase에 저장된 실행 결과가 없습니다.")
        return False

    selected = st.selectbox(
        "클라우드 결과",
        snapshots,
        format_func=lambda item: f"{item.created_at} · {item.title} ({item.row_count}건)",
        key="cloud_results_select",
    )
    snapshot = get_analysis_snapshot(config, auth_user, selected.snapshot_id)
    if snapshot is None:
        st.warning("선택한 클라우드 결과를 찾지 못했습니다.")
        return False

    st.caption(f"유형: `{snapshot.run_type}`")
    if snapshot.summary:
        st.json(snapshot.summary, expanded=False)
    if snapshot.rows:
        st.dataframe(pd.DataFrame(snapshot.rows), use_container_width=True, hide_index=True)
    else:
        st.info("저장된 행 데이터가 없습니다.")
    return True


def render_watchlist_search(user_id: str, query: str, exchange: str) -> None:
    value = query.strip()
    if not value:
        st.info("종목 이름이나 코드를 입력하면 바로 등록 후보가 나타납니다.")
        return
    if len(value) < 2:
        st.caption("검색어를 2글자 이상 입력하세요.")
        return

    try:
        candidates = load_stock_candidates(value, exchange or "")
    except Exception as exc:
        st.warning(f"검색 후보를 가져오지 못했습니다: {exc}")
        return

    if not candidates:
        st.caption("검색 후보가 없습니다.")
        return

    st.subheader("검색 결과")
    for item in candidates[:8]:
        cols = st.columns([0.62, 0.18, 0.20], gap="small")
        with cols[0]:
            st.markdown(f"**{item['name']}**")
            exchange_label = item["exchange"] or "US/Global"
            st.caption(f"`{item['symbol']}` · {exchange_label}")
        with cols[1]:
            st.caption(item["source"])
        with cols[2]:
            if st.button("등록", key=f"watch_add_{user_id}_{item['symbol']}", use_container_width=True):
                try:
                    save_watchlist_selection(user_id, item)
                except Exception as exc:
                    st.error(f"관심종목 저장 중 오류가 발생했습니다: {exc}")
                else:
                    st.rerun()


def render_saved_watchlist(user_id: str) -> None:
    items = list_saved_watchlist(user_id)
    st.subheader(f"{user_id} 관심종목")
    if not items:
        st.info("아직 등록된 관심종목이 없습니다. 왼쪽에서 검색해서 추가해보세요.")
        return

    frame = pd.DataFrame(
        [
            {
                "종목명": item.name,
                "심볼": item.symbol,
                "거래소": item.exchange or "-",
                "출처": item.source or "-",
            }
            for item in items
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption(f"총 {len(items)}개 종목이 저장되어 있습니다.")

    with st.expander("등록 목록 관리", expanded=False):
        for item in items:
            cols = st.columns([0.78, 0.22], gap="small")
            with cols[0]:
                title = item.name or item.symbol
                exchange_label = item.exchange or "US/Global"
                st.markdown(f"**{title}**")
                st.caption(f"`{item.symbol}` · {exchange_label}")
            with cols[1]:
                if st.button("삭제", key=f"watch_remove_{user_id}_{item.symbol}", use_container_width=True):
                    try:
                        remove_watchlist_selection(user_id, item.symbol)
                    except Exception as exc:
                        st.error(f"관심종목 삭제 중 오류가 발생했습니다: {exc}")
                    else:
                        st.rerun()


def render_stock_candidates(query: str, exchange: str, *, key_prefix: str) -> None:
    value = query.strip()
    if not value:
        return
    if len(value) < 2:
        st.caption("종목코드 또는 종목명을 입력하세요.")
        return

    try:
        candidates = load_stock_candidates(value, exchange or "")
    except Exception as exc:
        st.caption(f"검색 후보를 가져오지 못했습니다: {exc}")
        return

    if not candidates:
        st.caption("검색 후보가 없습니다.")
        return

    preview = pd.DataFrame(
        [
            {
                "종목명": item["name"],
                "심볼": item["symbol"],
                "거래소": item["exchange"],
                "출처": item["source"],
            }
            for item in candidates[:5]
        ]
    )
    st.dataframe(
        preview,
        use_container_width=True,
        hide_index=True,
    )


def describe_prediction_strength(probability_up: float) -> tuple[str, str]:
    edge = abs(probability_up - 0.5)
    if edge < 0.03:
        return ("매우 애매", "상승과 하락 확률이 거의 비슷합니다.")
    if edge < 0.07:
        return ("약함", "방향은 있지만 확신은 크지 않습니다.")
    if edge < 0.15:
        return ("보통", "한쪽 가능성을 조금 더 높게 보고 있습니다.")
    if edge < 0.25:
        return ("높음", "한쪽 가능성을 비교적 분명하게 보고 있습니다.")
    return ("매우 높음", "모델이 한쪽 방향을 강하게 보고 있습니다.")


def model_basis_label(basis: str) -> str:
    return "walk-forward" if basis == "walk_forward" else "holdout"


def model_primary_metrics(metrics: Any, basis: str) -> tuple[float, float, float]:
    if basis == "walk_forward" and metrics.walk_forward_accuracy is not None:
        return (
            float(metrics.walk_forward_accuracy),
            float(metrics.walk_forward_balanced_accuracy or 0.0),
            float(metrics.walk_forward_edge_vs_baseline or 0.0),
        )
    return (
        float(metrics.accuracy),
        float(metrics.balanced_accuracy),
        float(metrics.edge_vs_baseline),
    )


def build_model_comparison_frame(prediction: Prediction) -> pd.DataFrame:
    rows = []
    for compared in prediction.compared_models:
        accuracy, balanced_accuracy, edge = model_primary_metrics(
            compared.metrics,
            prediction.model_selection_basis,
        )
        rows.append(
            {
                "모델": f"{compared.label} (선택)" if compared.selected else compared.label,
                "평가 기준": model_basis_label(prediction.model_selection_basis),
                "정확도": round(accuracy, 4),
                "균형 정확도": round(balanced_accuracy, 4),
                "개선폭": round(edge, 4),
                "추천 기준": round(compared.metrics.recommended_threshold or 0.5, 4),
            }
        )
    return pd.DataFrame(rows)


def render_prediction_backtest(prediction: Prediction) -> None:
    if prediction.backtest is None:
        return

    summary = prediction.backtest.summary
    st.markdown("**최근 검증 백테스트**")
    st.caption("walk-forward 검증 구간에서 당시까지의 데이터만으로 하루씩 예측한 기록입니다.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("검증 일수", f"{summary.rows}")
    c2.metric("상승 신호 수", f"{summary.signal_count}")
    c3.metric("신호 적중률", f"{summary.hit_rate:.1%}")
    c4.metric("신호 평균 수익", f"{summary.average_return:.2%}")

    c5, c6 = st.columns(2)
    c5.metric("전략 누적 수익", f"{summary.cumulative_strategy_return:.2%}")
    c6.metric("단순 보유 수익", f"{summary.cumulative_buy_hold_return:.2%}")

    chart_frame = pd.DataFrame(prediction.backtest.chart_rows)
    if not chart_frame.empty:
        chart_frame["date"] = pd.to_datetime(chart_frame["date"])
        chart_frame = chart_frame.set_index("date").rename(
            columns={
                "strategy_return": "예측 신호 전략",
                "buy_hold_return": "단순 보유",
            }
        )
        st.line_chart(chart_frame, use_container_width=True)

    with st.expander("최근 검증 로그 보기", expanded=False):
        recent_frame = pd.DataFrame(prediction.backtest.recent_rows).rename(
            columns={
                "date": "날짜",
                "probability_up": "상승확률",
                "signal": "예측",
                "actual": "실제",
                "next_day_return": "다음날 수익률",
                "strategy_return": "전략 수익률",
            }
        )
        if not recent_frame.empty:
            recent_frame["상승확률"] = recent_frame["상승확률"].map(lambda value: f"{value:.1%}")
            recent_frame["다음날 수익률"] = recent_frame["다음날 수익률"].map(lambda value: f"{value:.2%}")
            recent_frame["전략 수익률"] = recent_frame["전략 수익률"].map(lambda value: f"{value:.2%}")
        st.dataframe(recent_frame, use_container_width=True, hide_index=True)


def interpret_prediction_action(
    prediction: Prediction,
    market_context: dict[str, Any] | None = None,
) -> tuple[str, str, str, list[str]]:
    metrics = prediction.metrics
    probability_up = float(prediction.probability_up)
    probability_down = float(prediction.probability_down)
    confidence_gap = abs(probability_up - probability_down)
    validation_edge = (
        float(metrics.walk_forward_edge_vs_baseline)
        if metrics.walk_forward_edge_vs_baseline is not None
        else float(metrics.edge_vs_baseline)
    )
    hit_rate = prediction.backtest.summary.hit_rate if prediction.backtest is not None else None

    reasons = [
        f"상승확률 {probability_up:.1%}, 하락확률 {probability_down:.1%}",
        f"판단 기준선 {prediction.threshold:.2f}, 방향 차이 {confidence_gap:.1%}p",
    ]
    if metrics.walk_forward_accuracy is not None:
        reasons.append(
            f"최근 walk-forward 개선폭 {float(metrics.walk_forward_edge_vs_baseline or 0.0):+.2%}p"
        )
    if hit_rate is not None:
        reasons.append(f"최근 상승 신호 적중률 {hit_rate:.1%}")
    market_score = None if market_context is None else float(market_context["score"])
    market_status = None if market_context is None else str(market_context["status"])
    if market_score is not None and market_status is not None:
        reasons.append(f"현재 시장 심리 {market_score:.1f}점 ({market_status})")

    if prediction.signal == "DOWN":
        if market_score is not None and market_score <= 25:
            return (
                "과매도 구간 관망",
                "info",
                "모델은 하락 우세로 보지만 시장 심리가 이미 극단적 공포 구간입니다. 추가 하락 추종보다는 반등 가능성까지 함께 보며 신중하게 해석하는 편이 좋습니다.",
                reasons,
            )
        if confidence_gap < 0.06:
            return (
                "관망",
                "info",
                "모델은 하락 쪽으로 보지만 확신은 크지 않습니다. 지금은 방향 확인용으로만 보는 편이 좋습니다.",
                reasons,
            )
        return (
            "보수적 관망",
            "warning",
            "하락 우세 신호가 더 강합니다. 신규 진입보다 추세 확인 쪽에 무게를 두는 해석이 더 자연스럽습니다.",
            reasons,
        )

    if validation_edge < 0:
        return (
            "참고용 신호",
            "warning",
            "상승 신호는 나왔지만 최근 검증 성적이 단순 기준보다 약합니다. 바로 의사결정에 쓰기보다 참고용으로 보는 편이 안전합니다.",
            reasons,
        )

    if market_score is not None and market_score >= 75 and probability_up < 0.62:
        return (
            "신중한 관심",
            "warning",
            "상승 신호는 있지만 시장 심리가 이미 탐욕 구간입니다. 추격 매수보다는 눌림이나 추가 근거를 함께 확인하는 해석이 더 좋습니다.",
            reasons,
        )

    if market_score is not None and market_score <= 25 and probability_up >= 0.57:
        return (
            "역발상 관심 후보",
            "success",
            "상승 신호에 더해 시장 심리가 극단적 공포 구간입니다. 반등 후보 관점에서 우선 체크해볼 만합니다.",
            reasons,
        )

    if probability_up >= 0.65 and confidence_gap >= 0.15 and (hit_rate is None or hit_rate >= 0.55):
        return (
            "강한 관심 후보",
            "success",
            "상승 우세가 비교적 뚜렷하고 최근 검증 흐름도 나쁘지 않습니다. 우선 점검할 후보로 보기 좋습니다.",
            reasons,
        )

    if probability_up >= 0.57 and confidence_gap >= 0.08:
        return (
            "관심 후보",
            "success",
            "상승 쪽으로 기울어 있지만 아주 강한 확신 단계는 아닙니다. 거래량, 뉴스, 시장 분위기를 함께 보는 해석이 좋습니다.",
            reasons,
        )

    return (
        "약한 관심",
        "info",
        "상승 쪽으로 조금 기울었지만 아직은 애매합니다. 섣불리 확신하기보다 다른 근거와 함께 보는 편이 좋습니다.",
        reasons,
    )


def render_prediction_card(
    prediction: Prediction,
    market_context: dict[str, Any] | None = None,
) -> None:
    metrics = prediction.metrics
    signal_is_up = prediction.signal == "UP"
    signal_label = "상승 우세" if signal_is_up else "하락 우세"
    strength_label, strength_detail = describe_prediction_strength(prediction.probability_up)
    leading_probability = prediction.probability_up if signal_is_up else prediction.probability_down
    confidence_gap = abs(prediction.probability_up - prediction.probability_down)

    st.subheader(f"{prediction.ticker} 예측 결과")
    st.caption(f"기준 거래일: {prediction.latest_date}")
    if prediction.threshold_source == "recommended" and metrics.recommended_threshold is not None:
        basis_label = model_basis_label(metrics.recommended_threshold_basis or "holdout")
        st.caption(
            f"자동 추천 기준 적용: `{prediction.threshold:.2f}` "
            f"(평가 기준: {basis_label}, 개선폭: {metrics.recommended_threshold_edge or 0.0:+.2%}p)"
        )
    st.caption(f"사용 모델: `{prediction.model_label}`")
    if len(prediction.compared_models) > 1:
        _, _, selected_edge = model_primary_metrics(metrics, prediction.model_selection_basis)
        st.caption(
            f"무료 모델 비교 결과: `{prediction.model_label}` 선택 "
            f"(비교 기준: {model_basis_label(prediction.model_selection_basis)}, 개선폭: {selected_edge:+.2%}p)"
        )
    if market_context is not None:
        st.caption(
            f"시장 심리 필터: `{market_context['score']:.1f}`점 / `{market_context['status']}` "
            f"(기준일: {market_context['date']})"
        )

    action_label, action_tone, action_message, action_reasons = interpret_prediction_action(
        prediction,
        market_context=market_context,
    )
    if action_tone == "success":
        st.success(f"행동 해석: **{action_label}**\n\n{action_message}")
    elif action_tone == "warning":
        st.warning(f"행동 해석: **{action_label}**\n\n{action_message}")
    else:
        st.info(f"행동 해석: **{action_label}**\n\n{action_message}")
    with st.expander("왜 이렇게 해석했나요?", expanded=False):
        st.markdown("\n".join(f"- {reason}" for reason in action_reasons))

    if strength_label == "매우 애매":
        st.info(
            f"현재 모델은 `{signal_label}`로 보고 있지만, 확률 차이가 작아서 방향성이 뚜렷하지 않습니다."
        )
    elif signal_is_up:
        st.success(
            f"현재 모델은 다음 거래일에 `{signal_label}`로 보고 있습니다. "
            f"가장 높은 쪽 확률은 {leading_probability:.1%}입니다."
        )
    else:
        st.warning(
            f"현재 모델은 다음 거래일에 `{signal_label}`로 보고 있습니다. "
            f"가장 높은 쪽 확률은 {leading_probability:.1%}입니다."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "현재 판단",
        signal_label,
        help=f"상승 확률이 사용자가 정한 판단 기준 {prediction.threshold:.2f} 이상이면 상승 우세, 아니면 하락 우세로 표시합니다.",
    )
    c2.metric(
        "상승 가능성",
        f"{prediction.probability_up:.1%}",
        help="다음 거래일 종가가 오늘 종가보다 높을 가능성을 모델이 추정한 값입니다.",
    )
    c3.metric(
        "하락 가능성",
        f"{prediction.probability_down:.1%}",
        help="다음 거래일 종가가 오늘 종가보다 낮거나 비슷한 쪽으로 갈 가능성을 단순 보완적으로 보여줍니다.",
    )
    c4.metric(
        "판단 강도",
        strength_label,
        help="상승/하락 확률이 50%에서 얼마나 멀리 떨어져 있는지를 쉽게 해석한 값입니다.",
    )

    st.progress(int(round(leading_probability * 100)))
    st.caption(
        f"{strength_detail} 최근 종가는 {prediction.latest_close:,.2f}이고, "
        f"상승/하락 확률 차이는 {confidence_gap:.1%}p 입니다."
    )

    with st.expander("이 결과를 어떻게 읽나요?", expanded=False):
        st.markdown(
            "- `상승 가능성`이 50%에 가까우면 방향성이 약합니다.\n"
            "- `상승 판단 기준`을 0.55로 올리면 더 보수적으로 상승 신호를 보게 됩니다.\n"
            "- 이 값은 다음 날 방향 확률이지, 상승폭이나 수익률 크기를 뜻하는 값은 아닙니다."
        )

    summary_frame = pd.DataFrame(
        [
            {"항목": "한줄 해석", "값": f"{prediction.ticker}은(는) 현재 {signal_label} 신호입니다."},
            {"항목": "확률 해석", "값": f"상승 {prediction.probability_up:.1%} / 하락 {prediction.probability_down:.1%}"},
            {"항목": "사용 모델", "값": prediction.model_label},
            {"항목": "초보자 체크", "값": "50%에 가까울수록 애매하고, 50%에서 멀수록 방향성이 더 뚜렷합니다."},
        ]
    )
    st.dataframe(summary_frame, use_container_width=True, hide_index=True)
    render_prediction_backtest(prediction)

    with st.expander("자세한 모델 수치 보기", expanded=False):
        if metrics.accuracy < metrics.baseline_accuracy:
            st.warning(
                "이 종목에서는 모델 검증 정확도가 단순 기준보다 낮았습니다. "
                "예측은 참고용으로만 보는 편이 좋습니다."
            )

        detail_row = {
            "최근 거래일": prediction.latest_date,
            "최근 종가": round(prediction.latest_close, 6),
            "사용 모델": prediction.model_label,
            "모델 검증 정확도": round(metrics.accuracy, 4),
            "균형 정확도": round(metrics.balanced_accuracy, 4),
            "단순 기준 정확도": round(metrics.baseline_accuracy, 4),
            "기준 대비 개선폭": round(metrics.edge_vs_baseline, 4),
            "상승 precision": round(metrics.precision_up, 4),
            "상승 recall": round(metrics.recall_up, 4),
            "학습 데이터 수": metrics.train_rows,
            "검증 데이터 수": metrics.test_rows,
        }
        if metrics.walk_forward_accuracy is not None:
            detail_row["Walk-forward 정확도"] = round(metrics.walk_forward_accuracy, 4)
            detail_row["Walk-forward 균형 정확도"] = round(metrics.walk_forward_balanced_accuracy or 0.0, 4)
            detail_row["Walk-forward 단순 기준"] = round(metrics.walk_forward_baseline_accuracy or 0.0, 4)
            detail_row["Walk-forward 개선폭"] = round(metrics.walk_forward_edge_vs_baseline or 0.0, 4)
            detail_row["Walk-forward 검증 수"] = metrics.walk_forward_test_rows
        if metrics.recommended_threshold is not None:
            detail_row["자동 추천 기준"] = round(metrics.recommended_threshold, 4)
            detail_row["추천 기준 개선폭"] = round(metrics.recommended_threshold_edge or 0.0, 4)
            detail_row["추천 기준 균형 정확도"] = round(metrics.recommended_threshold_balanced_accuracy or 0.0, 4)

        detail_frame = pd.DataFrame([detail_row])
        st.dataframe(detail_frame, use_container_width=True, hide_index=True)
        if len(prediction.compared_models) > 1:
            st.caption("모델 비교 요약")
            st.dataframe(build_model_comparison_frame(prediction), use_container_width=True, hide_index=True)

        explanation_rows = [
            {
                "지표": "사용 모델",
                "의미": "이번 종목에서 최근 검증 결과가 더 나았던 모델입니다. 비교 옵션을 켰을 때 자동으로 선택됩니다.",
            },
            {
                "지표": "모델 검증 정확도",
                "의미": "마지막 검증 구간에서 전체 방향을 맞춘 비율입니다.",
            },
            {
                "지표": "균형 정확도",
                "의미": "상승과 하락을 한쪽에 치우치지 않고 얼마나 균형 있게 맞췄는지 보여줍니다.",
            },
            {
                "지표": "단순 기준 정확도",
                "의미": "모델 없이 검증 구간에서 더 많았던 방향만 계속 찍었을 때의 정확도입니다.",
            },
            {
                "지표": "기준 대비 개선폭",
                "의미": "모델 검증 정확도에서 단순 기준 정확도를 뺀 값입니다. 양수면 모델이 단순 기준보다 낫습니다.",
            },
            {
                "지표": "상승 precision",
                "의미": "모델이 상승이라고 말한 날들 중 실제로 상승한 비율입니다.",
            },
            {
                "지표": "상승 recall",
                "의미": "실제로 상승했던 날들 중 모델이 상승으로 잡아낸 비율입니다.",
            },
            {
                "지표": "학습 데이터 수",
                "의미": "모델을 학습하는 데 실제로 사용한 과거 데이터 개수입니다.",
            },
            {
                "지표": "검증 데이터 수",
                "의미": "성능 확인용으로 뒤쪽에 따로 남겨둔 데이터 개수입니다.",
            },
        ]
        if metrics.walk_forward_accuracy is not None:
            explanation_rows.extend(
                [
                    {
                        "지표": "Walk-forward 정확도",
                        "의미": "과거에서 하루씩 앞으로 전진하며 매번 그 시점 이전 데이터만으로 예측했을 때의 정확도입니다.",
                    },
                    {
                        "지표": "Walk-forward 균형 정확도",
                        "의미": "Walk-forward 방식에서 상승과 하락을 균형 있게 맞췄는지 보여줍니다.",
                    },
                    {
                        "지표": "Walk-forward 단순 기준",
                        "의미": "Walk-forward 검증 구간에서 단순히 한쪽 방향만 찍었을 때의 기준 정확도입니다.",
                    },
                    {
                        "지표": "Walk-forward 개선폭",
                        "의미": "Walk-forward 정확도가 단순 기준보다 얼마나 나은지 보여줍니다.",
                    },
                    {
                        "지표": "Walk-forward 검증 수",
                        "의미": "하루씩 전진하며 실제로 평가한 예측 횟수입니다.",
                    },
                    {
                        "지표": "자동 추천 기준",
                        "의미": "0.40~0.60 후보 중 최근 검증 성능이 가장 안정적이었던 threshold 값입니다.",
                    },
                    {
                        "지표": "추천 기준 개선폭",
                        "의미": "자동 추천 기준을 적용했을 때 단순 기준보다 얼마나 나은지 보여줍니다.",
                    },
                ]
            )
        if len(prediction.compared_models) > 1:
            explanation_rows.extend(
                [
                    {
                        "지표": "모델 비교 요약",
                        "의미": "로지스틱 회귀와 무료 트리 모델을 같은 데이터로 검증한 뒤 더 나은 모델을 고른 표입니다.",
                    },
                    {
                        "지표": "비교 기준",
                        "의미": "가능하면 walk-forward 검증으로 비교하고, 그 값이 없으면 holdout 검증 결과로 비교합니다.",
                    },
                ]
            )

        st.caption("자세한 수치 의미")
        st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)


def render_valuation_card(result: ValuationResult) -> None:
    snapshot = result.snapshot
    st.subheader(f"{result.ticker} · {snapshot.name}")
    if snapshot.valuation_profile:
        st.caption(
            f"기업 유형 보정: `{snapshot.valuation_profile}`"
            + (f" · 섹터 `{snapshot.sector}`" if snapshot.sector else "")
            + (f" · 산업 `{snapshot.industry}`" if snapshot.industry else "")
        )
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
                    "유형": snapshot.valuation_profile,
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


def render_latest_csv_preview(base_dir: str | Path, pattern: str, *, key_prefix: str) -> None:
    directory = Path(base_dir)
    if not directory.is_absolute():
        directory = ROOT / directory
    latest = latest_file(directory, pattern)
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


def list_result_files(user_id: str | None = None) -> list[Path]:
    base_dir = user_outputs_dir(user_id) if user_id else OUTPUTS
    if not base_dir.exists():
        return []
    files = [path for path in base_dir.rglob("*.csv") if path.is_file()]
    return sorted(files, key=lambda path: path.stat().st_mtime, reverse=True)


def latest_file(base_dir: Path, pattern: str) -> Path | None:
    if not base_dir.exists():
        return None
    files = [path for path in base_dir.rglob(pattern) if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def load_fear_greed_data():
    return fetch_fear_greed_data()


@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_candidates(query: str, exchange: str):
    return [
        {
            "ticker": item.ticker,
            "exchange": item.exchange,
            "name": item.name,
            "symbol": item.symbol,
            "source": item.source,
        }
        for item in search_stock_candidates(query, exchange_hint=exchange or None, limit=8)
    ]


if __name__ == "__main__":
    main()
