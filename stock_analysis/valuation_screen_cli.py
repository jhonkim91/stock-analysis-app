from __future__ import annotations

import argparse
import csv

from stock_analysis.valuation_screen import ValuationScreenSummary, run_valuation_screen


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_valuation_screen(
            output_dir=args.output_dir,
            rank_limit=args.rank_limit,
            run_limit=args.run_limit,
            top=args.top,
            market=args.market,
            source=args.source,
            date=args.date,
            exclude_preferred=args.exclude_preferred,
            target_pe=args.target_pe,
            target_pbr=args.target_pbr,
            growth=args.growth,
            discount_rate=args.discount_rate,
            terminal_growth=args.terminal_growth,
            years=args.years,
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
        description="5단계: 시가총액 상위 종목의 목표주가를 계산하고 상승여력 Top N을 고릅니다.",
    )
    parser.add_argument("--output-dir", default="outputs/target_market_cap", help="결과를 저장할 폴더입니다.")
    parser.add_argument("--rank-limit", type=int, default=300, help="시가총액 상위 N개 종목을 가져옵니다.")
    parser.add_argument("--run-limit", type=int, help="테스트용으로 실제 계산할 종목 수를 제한합니다.")
    parser.add_argument("--top", type=int, default=10, help="상승여력 상위 N개 종목을 저장합니다.")
    parser.add_argument("--market", choices=["ALL", "KOSPI", "KOSDAQ"], default="ALL", help="대상 시장입니다.")
    parser.add_argument("--source", choices=["auto", "pykrx", "naver"], default="naver", help="시가총액 데이터 소스입니다.")
    parser.add_argument("--date", help="pykrx 사용 시 조회 기준일입니다. 예: 20260430")
    parser.set_defaults(exclude_preferred=True)
    parser.add_argument("--exclude-preferred", action="store_true", help="우선주로 보이는 종목을 제외합니다. 5단계 기본값입니다.")
    parser.add_argument(
        "--include-preferred",
        action="store_false",
        dest="exclude_preferred",
        help="우선주로 보이는 종목도 포함합니다.",
    )
    parser.add_argument("--target-pe", type=float, help="모든 종목에 직접 적용할 목표 PER입니다.")
    parser.add_argument("--target-pbr", type=float, help="모든 종목에 직접 적용할 목표 PBR입니다.")
    parser.add_argument("--growth", type=float, help="모든 종목에 직접 적용할 DCF 성장률입니다. 예: 0.04")
    parser.add_argument("--discount-rate", type=float, default=0.10, help="DCF 할인율입니다. 기본값 0.10")
    parser.add_argument("--terminal-growth", type=float, default=0.02, help="DCF 영구성장률입니다. 기본값 0.02")
    parser.add_argument("--years", type=int, default=5, help="DCF 명시 예측 기간입니다.")
    parser.add_argument("--sleep", type=float, default=0.0, help="종목 사이 대기 시간(초)입니다.")
    parser.add_argument("--retries", type=int, default=1, help="종목별 실패 시 재시도 횟수입니다.")
    parser.add_argument("--fail-fast", action="store_true", help="실패가 발생하면 즉시 중단합니다.")
    parser.add_argument("--quiet", action="store_true", help="종목별 진행 상황을 출력하지 않습니다.")
    parser.add_argument("--no-table", action="store_true", help="완료 후 Top 표를 출력하지 않습니다.")
    return parser


def print_progress(index: int, total: int, row: dict) -> None:
    ticker = row["ticker"] or row["ticker_input"]
    if row["status"] == "success":
        print(
            f"[{index}/{total}] {ticker}: success, "
            f"target={float(row['target_price']):,.2f}, upside={float(row['upside']):.2%}"
        )
    else:
        print(f"[{index}/{total}] {ticker}: failed, {row['error']}")


def print_summary(summary: ValuationScreenSummary) -> None:
    print("=== 5단계: 목표주가 상승여력 Top 후보 선별 완료 ===")
    print(f"실행 시각: {summary.run_at}")
    print(f"시가총액 universe: {summary.universe_count}")
    print(f"목표주가 계산 대상: {summary.evaluated_count}")
    print(f"성공: {summary.succeeded}")
    print(f"실패: {summary.failed}")
    print(f"선정 종목 수: {summary.top_count}")
    print(f"전체 결과 CSV: {summary.valuations_csv_path}")
    print(f"Top 결과 CSV: {summary.top_csv_path}")
    print(f"Top 결과 JSON: {summary.top_json_path}")
    print()
    print("주의: 이 결과는 공개 재무 데이터와 입력 가정에 기반한 실험용 계산이며 투자 조언이 아닙니다.")


def print_top_table(path: str) -> None:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print()
        print("Top 후보가 없습니다.")
        return

    print()
    print("[목표주가 상승여력 Top 후보]")
    for row in rows:
        rank = row.get("selection_rank", "")
        name = row.get("name", "")
        ticker = row.get("ticker", "")
        market_rank = row.get("market_cap_rank", "")
        current_price = float(row.get("current_price") or 0)
        target_price = float(row.get("target_price") or 0)
        upside = float(row.get("upside") or 0)
        currency = row.get("currency", "")
        print(
            f"{rank}. {name} ({ticker}) | 시총순위 {market_rank} | "
            f"현재가 {current_price:,.2f} {currency} | 목표가 {target_price:,.2f} | 상승여력 {upside:.2%}"
        )
