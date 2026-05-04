from __future__ import annotations

import argparse

from stock_analysis.auto_runner import AutoRunSummary, run_watchlist


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_watchlist(
            watchlist_path=args.watchlist,
            output_dir=args.output_dir,
            period=args.period,
            interval=args.interval,
            test_size=args.test_size,
            threshold=args.threshold,
            limit=args.limit,
            sleep_seconds=args.sleep,
            retries=args.retries,
            fail_fast=args.fail_fast,
            progress=None if args.quiet else print_progress,
        )
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")

    print_summary(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="2단계: 관심종목 데이터를 자동 수집하고 1단계 예측을 일괄 실행합니다.",
    )
    parser.add_argument("--watchlist", default="watchlist.csv", help="자동 실행할 종목 CSV 경로입니다.")
    parser.add_argument("--output-dir", default="outputs", help="결과를 저장할 폴더입니다.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance 조회 기간입니다. 예: 2y, 5y, 10y")
    parser.add_argument("--interval", default="1d", help="조회 간격입니다. 현재 모델은 1d 사용을 권장합니다.")
    parser.add_argument("--test-size", type=float, default=0.2, help="시간순 검증 데이터 비율입니다.")
    parser.add_argument("--threshold", type=float, default=0.5, help="상승으로 판단할 확률 기준입니다.")
    parser.add_argument("--limit", type=int, help="앞에서부터 N개 종목만 실행합니다. 테스트용입니다.")
    parser.add_argument("--sleep", type=float, default=0.0, help="종목 사이 대기 시간(초)입니다.")
    parser.add_argument("--retries", type=int, default=1, help="종목별 실패 시 재시도 횟수입니다.")
    parser.add_argument("--fail-fast", action="store_true", help="실패가 발생하면 즉시 중단합니다.")
    parser.add_argument("--quiet", action="store_true", help="종목별 진행 상황을 출력하지 않습니다.")
    return parser


def print_progress(index: int, total: int, row: dict) -> None:
    ticker = row["ticker"] or row["ticker_input"]
    if row["status"] == "success":
        print(f"[{index}/{total}] {ticker}: success, up={float(row['probability_up']):.2%}")
    else:
        print(f"[{index}/{total}] {ticker}: failed, {row['error']}")


def print_summary(summary: AutoRunSummary) -> None:
    print("=== 2단계: 자동 예측 실행 완료 ===")
    print(f"실행 시각: {summary.run_at}")
    print(f"관심종목 파일: {summary.watchlist_path}")
    print(f"전체 처리: {summary.total}")
    print(f"성공: {summary.succeeded}")
    print(f"실패: {summary.failed}")
    print(f"결과 CSV: {summary.csv_path}")
    print(f"결과 JSON: {summary.json_path}")
    print()
    print("주의: 이 결과는 과거 가격 기반의 실험용 통계 모델이며 투자 조언이 아닙니다.")
