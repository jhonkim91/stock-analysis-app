from __future__ import annotations

import argparse
import csv

from stock_analysis.top_candidates import TopCandidateSummary, run_top_market_cap_screen


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_top_market_cap_screen(
            output_dir=args.output_dir,
            rank_limit=args.rank_limit,
            run_limit=args.run_limit,
            top=args.top,
            market=args.market,
            source=args.source,
            date=args.date,
            exclude_preferred=args.exclude_preferred,
            period=args.period,
            interval=args.interval,
            test_size=args.test_size,
            threshold=args.threshold,
            sleep_seconds=args.sleep,
            retries=args.retries,
            fail_fast=args.fail_fast,
            progress=None if args.quiet else print_progress,
        )
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")

    print_summary(summary)
    if not args.no_table:
        print_top_table(summary.top_csv_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3단계: 시가총액 상위 종목을 학습하고 다음 거래일 상승 확률 Top N을 고릅니다.",
    )
    parser.add_argument("--output-dir", default="outputs/top_market_cap", help="결과를 저장할 폴더입니다.")
    parser.add_argument("--rank-limit", type=int, default=300, help="시가총액 상위 N개 종목을 가져옵니다.")
    parser.add_argument("--run-limit", type=int, help="테스트용으로 실제 학습할 종목 수를 제한합니다.")
    parser.add_argument("--top", type=int, default=10, help="상승 확률 상위 N개 종목을 저장합니다.")
    parser.add_argument("--market", choices=["ALL", "KOSPI", "KOSDAQ"], default="ALL", help="대상 시장입니다.")
    parser.add_argument("--source", choices=["auto", "pykrx", "naver"], default="naver", help="시가총액 데이터 소스입니다.")
    parser.add_argument("--date", help="pykrx 사용 시 조회 기준일입니다. 예: 20260430")
    parser.add_argument("--exclude-preferred", action="store_true", help="우선주로 보이는 종목을 제외합니다.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance 조회 기간입니다. 예: 2y, 5y, 10y")
    parser.add_argument("--interval", default="1d", help="조회 간격입니다. 현재 모델은 1d 사용을 권장합니다.")
    parser.add_argument("--test-size", type=float, default=0.2, help="시간순 검증 데이터 비율입니다.")
    parser.add_argument("--threshold", type=float, default=0.5, help="상승으로 판단할 확률 기준입니다.")
    parser.add_argument("--sleep", type=float, default=0.0, help="종목 사이 대기 시간(초)입니다.")
    parser.add_argument("--retries", type=int, default=1, help="종목별 실패 시 재시도 횟수입니다.")
    parser.add_argument("--fail-fast", action="store_true", help="실패가 발생하면 즉시 중단합니다.")
    parser.add_argument("--quiet", action="store_true", help="종목별 진행 상황을 출력하지 않습니다.")
    parser.add_argument("--no-table", action="store_true", help="완료 후 Top 표를 출력하지 않습니다.")
    return parser


def print_progress(index: int, total: int, row: dict) -> None:
    ticker = row["ticker"] or row["ticker_input"]
    if row["status"] == "success":
        print(f"[{index}/{total}] {ticker}: success, up={float(row['probability_up']):.2%}")
    else:
        print(f"[{index}/{total}] {ticker}: failed, {row['error']}")


def print_summary(summary: TopCandidateSummary) -> None:
    print("=== 3단계: 시가총액 상위 종목 Top 후보 선별 완료 ===")
    print(f"실행 시각: {summary.run_at}")
    print(f"시가총액 universe: {summary.universe_count}")
    print(f"학습/예측 대상: {summary.evaluated_count}")
    print(f"성공: {summary.succeeded}")
    print(f"실패: {summary.failed}")
    print(f"선정 종목 수: {summary.top_count}")
    print(f"전체 결과 CSV: {summary.predictions_path}")
    print(f"Top 결과 CSV: {summary.top_csv_path}")
    print(f"Top 결과 JSON: {summary.top_json_path}")
    print()
    print("주의: 이 결과는 과거 가격 기반의 실험용 통계 모델이며 투자 조언이 아닙니다.")


def print_top_table(path: str) -> None:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print()
        print("Top 후보가 없습니다.")
        return

    print()
    print("[상승 확률 Top 후보]")
    for row in rows:
        rank = row.get("selection_rank", "")
        name = row.get("name", "")
        ticker = row.get("ticker", "")
        market_rank = row.get("market_cap_rank", "")
        probability = float(row.get("probability_up") or 0)
        accuracy = float(row.get("accuracy") or 0)
        print(f"{rank}. {name} ({ticker}) | 시총순위 {market_rank} | 상승확률 {probability:.2%} | 검증정확도 {accuracy:.2%}")
