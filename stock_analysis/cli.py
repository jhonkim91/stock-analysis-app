from __future__ import annotations

import argparse
import json
from pathlib import Path

from stock_analysis.predictor import Prediction, predict_next_day


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prediction = predict_next_day(
            args.ticker,
            exchange=args.exchange,
            period=args.period,
            interval=args.interval,
            csv_path=args.csv,
            test_size=args.test_size,
            threshold=args.threshold,
        )
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")

    if args.json:
        payload = json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2)
        print(payload)
    else:
        print_human_readable(prediction)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="1단계: 특정 종목의 다음 거래일 상승/하락을 실험적으로 예측합니다.",
    )
    parser.add_argument("--ticker", "-t", required=True, help="예: AAPL, MSFT, 005930.KS, 005930")
    parser.add_argument(
        "--exchange",
        choices=["KRX", "KS", "KOSPI", "KQ", "KOSDAQ"],
        help="6자리 한국 종목코드만 넣었을 때 붙일 거래소 접미사입니다. 기본값은 KS입니다.",
    )
    parser.add_argument("--period", default="5y", help="Yahoo Finance 조회 기간입니다. 예: 2y, 5y, 10y")
    parser.add_argument("--interval", default="1d", help="조회 간격입니다. 현재 모델은 1d 사용을 권장합니다.")
    parser.add_argument("--csv", help="Yahoo Finance 대신 사용할 CSV 파일 경로입니다.")
    parser.add_argument("--test-size", type=float, default=0.2, help="시간순 검증 데이터 비율입니다.")
    parser.add_argument("--threshold", type=float, default=0.5, help="상승으로 판단할 확률 기준입니다.")
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력합니다.")
    parser.add_argument("--output", help="결과 JSON을 저장할 파일 경로입니다.")
    return parser


def print_human_readable(prediction: Prediction) -> None:
    metrics = prediction.metrics
    signal_kr = "상승" if prediction.signal == "UP" else "하락"

    print("=== 1단계: 다음 거래일 상승/하락 예측 ===")
    print(f"종목: {prediction.ticker}")
    print(f"최근 거래일: {prediction.latest_date}")
    print(f"최근 종가: {prediction.latest_close:,.2f}")
    print(f"예측: {signal_kr}")
    print(f"상승 확률: {prediction.probability_up:.2%}")
    print(f"하락 확률: {prediction.probability_down:.2%}")
    print()
    print("[시간순 검증 결과]")
    print(f"학습 행 수: {metrics.train_rows:,}")
    print(f"검증 행 수: {metrics.test_rows:,}")
    print(f"정확도: {metrics.accuracy:.2%}")
    print(f"단순 기준 정확도: {metrics.baseline_accuracy:.2%}")
    print(f"상승 precision: {metrics.precision_up:.2%}")
    print(f"상승 recall: {metrics.recall_up:.2%}")
    print()
    print("주의: 이 결과는 과거 가격 기반의 실험용 통계 모델이며 투자 조언이 아닙니다.")
