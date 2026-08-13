#!/usr/bin/env python3
"""
Runs the rebuttal-vs-refusal extension experiment: does "rebuttals beat
refusals" hold across several differently-styled rebuttals, or was the
original paper's result riding on one particular phrasing?

Requires an API key in your environment, same as run_experiment.py:
    export ANTHROPIC_API_KEY=...

Usage:
    python run_rebuttal_experiment.py --provider anthropic --samples 10
"""
from __future__ import annotations

import argparse
import json

from src.chat_client import get_client
from src.rebuttal_experiment import run_all, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--samples", type=int, default=10, help="samples per condition")
    parser.add_argument("--conditions", default="data/rebuttal_variants.json")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    with open(args.conditions) as f:
        conditions = json.load(f)

    client = get_client(args.provider, args.model)
    trials, summaries = run_all(client, conditions, n_samples=args.samples)
    save_results(trials, summaries, args.out_dir)

    print(f"\n{'condition':<20}{'type':<12}{'disclosure_rate'}")
    for s in summaries:
        print(f"{s.condition_id:<20}{s.condition_type:<12}{s.disclosure_rate:.2f}")

    print(f"\nFull trial-level output: {args.out_dir}/rebuttal_trials.json")
    print(f"Per-condition summaries: {args.out_dir}/rebuttal_summaries.json")
    print(
        f"\n{args.samples * len(conditions)} total API calls made "
        f"({args.samples} samples x {len(conditions)} conditions)."
    )


if __name__ == "__main__":
    main()
