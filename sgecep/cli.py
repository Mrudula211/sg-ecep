"""Command-line interface.

    python -m sgecep.cli "the students reads the book"
    python -m sgecep.cli --verbose "she saw the man with the pizza"
    python -m sgecep.cli --chart "the student reads book"
    python -m sgecep.cli --demo
    python -m sgecep.cli --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import DATA_DIR, CorrectionPipeline, tokenise

DEMO_FILE = Path(__file__).resolve().parent.parent / "demo" / "demo_sentences.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgecep",
        description="Semantically-Guided Error-Correcting Earley Parser",
    )
    parser.add_argument("sentence", nargs="*", help="sentence to correct")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="show candidate list, semantic report and parse tree",
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=3, help="candidates to display",
    )
    parser.add_argument(
        "--chart", action="store_true",
        help="dump the baseline Earley chart (PREDICT/SCAN/COMPLETE)",
    )
    parser.add_argument(
        "--ec-chart", action="store_true",
        help="summarise the error-correcting chart",
    )
    parser.add_argument(
        "--max-cost", type=float, default=3.0, help="edit cost budget",
    )
    parser.add_argument(
        "--no-wordnet", action="store_true",
        help="use the bundled ontology instead of WordNet",
    )
    parser.add_argument(
        "--no-lm", action="store_true", help="disable the language model",
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--demo", action="store_true", help="run the demo set")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="read sentences from stdin",
    )
    return parser


def _load(args: argparse.Namespace) -> CorrectionPipeline:
    return CorrectionPipeline.from_data_dir(
        args.data_dir,
        use_wordnet=not args.no_wordnet,
        train_lm=not args.no_lm,
        max_cost=args.max_cost,
    )


def _handle(pipeline: CorrectionPipeline, text: str, args: argparse.Namespace) -> None:
    tokens = tokenise(text)
    if not tokens:
        return
    if args.chart:
        print(pipeline.baseline.dump_chart(tokens))
        print()
    if args.ec_chart:
        print(pipeline.parser.chart_summary(tokens))
        print()
    result = pipeline.correct(tokens, top_k=args.top_k)
    print(result.report(verbose=args.verbose, max_candidates=args.top_k))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = _load(args)

    if args.demo:
        if not DEMO_FILE.exists():
            print(f"demo file not found: {DEMO_FILE}", file=sys.stderr)
            return 1
        for raw in DEMO_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            print("=" * 72)
            _handle(pipeline, line, args)
        return 0

    if args.interactive:
        print("Enter a sentence (blank line or Ctrl-D to quit).")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                break
            _handle(pipeline, line, args)
        return 0

    if not args.sentence:
        build_parser().print_help()
        return 1

    _handle(pipeline, " ".join(args.sentence), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
