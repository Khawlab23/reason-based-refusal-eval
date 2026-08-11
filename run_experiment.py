#!/usr/bin/env python3
"""
Run the reason-consistency eval against a real model.

Requires an API key in your environment — never paste one into this repo:
    export ANTHROPIC_API_KEY=...     # for --provider anthropic
    export OPENAI_API_KEY=...        # for --provider openai

Usage:
    python run_experiment.py --provider anthropic --model claude-haiku-4-5-20251001 --resamples 5
"""
from __future__ import annotations

import argparse
import json

from src.chat_client import get_client
from src.experiment import run_all, save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--resamples", type=int, default=5)
    parser.add_argument("--prompts", default="data/prompts.json")
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    client = get_client(args.provider, args.model)
    trials, summaries = run_all(client, prompts, n_resamples=args.resamples)
    save_results(trials, summaries, args.out_dir)

    print(f"\n{'prompt_id':<8}{'bare_refuse':<14}{'expl_refuse':<14}{'mean_consistency':<18}")
    for s in summaries:
        mc = f"{s.mean_reason_consistency:.2f}" if s.mean_reason_consistency == s.mean_reason_consistency else "n/a"
        print(f"{s.prompt_id:<8}{s.bare_refusal_rate:<14.2f}{s.explained_refusal_rate:<14.2f}{mc:<18}")

    print(f"\nFull trial-level output: {args.out_dir}/trials.json")
    print(f"Per-prompt summaries: {args.out_dir}/summaries.json")


if __name__ == "__main__":
    main()
