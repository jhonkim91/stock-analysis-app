from __future__ import annotations

import argparse
import json

from stock_analysis.valuation import ValuationResult, calculate_target_price, save_valuation_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = calculate_target_price(
            args.ticker,
            exchange=args.exchange,
            target_pe=args.target_pe,
            target_pbr=args.target_pbr,
            growth=args.growth,
            discount_rate=args.discount_rate,
            terminal_growth=args.terminal_growth,
            years=args.years,
        )
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_human_readable(result)

    if args.output:
        save_valuation_json(args.output, result)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="4단계: 특정 종목의 재무 데이터를 이용해 목표주가를 산출합니다.",
    )
    parser.add_argument("--ticker", "-t", required=True, help="예: AAPL, MSFT, 005930.KS, 005930")
    parser.add_argument(
        "--exchange",
        choices=["KRX", "KS", "KOSPI", "KQ", "KOSDAQ"],
        help="6자리 한국 종목코드만 넣었을 때 붙일 거래소 접미사입니다. 기본값은 KS입니다.",
    )
    parser.add_argument("--target-pe", type=float, help="직접 지정할 목표 PER입니다.")
    parser.add_argument("--target-pbr", type=float, help="직접 지정할 목표 PBR입니다.")
    parser.add_argument("--growth", type=float, help="DCF 성장률입니다. 예: 0.05")
    parser.add_argument("--discount-rate", type=float, default=0.10, help="DCF 할인율입니다. 기본값 0.10")
    parser.add_argument("--terminal-growth", type=float, default=0.02, help="DCF 영구성장률입니다. 기본값 0.02")
    parser.add_argument("--years", type=int, default=5, help="DCF 명시 예측 기간입니다.")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력합니다.")
    parser.add_argument("--output", help="결과 JSON을 저장할 파일 경로입니다.")
    return parser


def print_human_readable(result: ValuationResult) -> None:
    snapshot = result.snapshot
    assumptions = result.assumptions

    print("=== 4단계: 재무 데이터 기반 목표주가 산출 ===")
    print(f"종목: {result.ticker}")
    print(f"회사명: {snapshot.name}")
    print(f"현재가: {_money(snapshot.current_price)} {snapshot.currency}")
    print(f"재무제표 기준일: {snapshot.statement_date or 'N/A'}")
    print(f"최종 목표주가: {_money(result.target_price)} {snapshot.currency}")
    print(f"현재가 대비 상승여력: {result.upside:.2%}")
    print()
    print("[핵심 재무 데이터]")
    print(f"EPS: {_optional_money(snapshot.eps)}")
    print(f"BPS: {_optional_money(snapshot.book_value_per_share)}")
    print(f"FCF/share: {_optional_money(snapshot.free_cash_flow_per_share)}")
    print(f"ROE: {_optional_percent(snapshot.roe)}")
    print(f"매출 성장률: {_optional_percent(snapshot.revenue_growth)}")
    print(f"순이익 성장률: {_optional_percent(snapshot.net_income_growth)}")
    print(f"FCF 성장률: {_optional_percent(snapshot.free_cash_flow_growth)}")
    print()
    print("[적용 가정]")
    print(f"목표 PER: {assumptions.target_pe:.2f}")
    print(f"목표 PBR: {assumptions.target_pbr:.2f}")
    print(f"DCF 성장률: {assumptions.dcf_growth:.2%}")
    print(f"DCF 할인율: {assumptions.discount_rate:.2%}")
    print(f"DCF 영구성장률: {assumptions.terminal_growth:.2%}")
    print(f"DCF 기간: {assumptions.years}년")
    print()
    print("[방법별 목표주가]")
    for method in result.methods:
        print(
            f"{method.name}: {_money(method.target_price)} "
            f"(가중치 {method.weight:.0%}) - {method.detail}"
        )
    print()
    print("주의: 이 목표주가는 입력 가정과 공개 재무 데이터에 기반한 실험용 계산이며 투자 조언이 아닙니다.")


def _money(value: float) -> str:
    return f"{value:,.2f}"


def _optional_money(value: float | None) -> str:
    return "N/A" if value is None else _money(value)


def _optional_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"
