import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

from src.chat_client import FakeClient
from src.refusal_parsing import parse_refusal
from src.reason_consistency import score_reason_consistency
from src.experiment import run_prompt


def test_parse_refusal_detects_marker_and_extracts_reason():
    text = "I can't help with that. This request involves sharing someone's private medical information without consent."
    parsed = parse_refusal(text)
    assert parsed.is_refusal
    assert "private medical information" in parsed.stated_reason


def test_parse_refusal_no_marker():
    text = "Sure, here's how to do that: first, gather your materials."
    parsed = parse_refusal(text)
    assert not parsed.is_refusal
    assert parsed.stated_reason == ""


def test_reason_consistency_identical_reasons_score_high():
    reasons = [
        "This could facilitate harm to a specific person.",
        "This could facilitate harm to a specific person.",
        "This could facilitate harm to a specific person.",
    ]
    result = score_reason_consistency(reasons)
    assert result.n_reasons == 3
    assert result.mean_pairwise_similarity > 0.9


def test_reason_consistency_different_reasons_score_low():
    reasons = [
        "This involves financial fraud against a third party.",
        "The tone of the request seems inappropriate for a general audience.",
        "I am concerned about copyright infringement in this context.",
    ]
    result = score_reason_consistency(reasons)
    assert result.n_reasons == 3
    assert result.mean_pairwise_similarity < 0.3


def test_reason_consistency_single_reason_returns_nan():
    result = score_reason_consistency(["only one reason given"])
    assert math.isnan(result.mean_pairwise_similarity)


def test_run_prompt_end_to_end_with_fake_client():
    # bare condition response, then 3 explained-condition responses with
    # deliberately inconsistent stated reasons, to check the pipeline wiring
    fake = FakeClient([
        "I can't help with that.",
        "I won't help with that. This could enable financial fraud.",
        "I won't help with that. This raises copyright concerns.",
        "I won't help with that. This seems like it could hurt someone's feelings.",
    ])
    trials, summary = run_prompt(fake, "test01", "some borderline prompt", n_resamples=3)

    assert len(trials) == 4  # 1 bare + 3 explained
    assert summary.bare_refusal_rate == 1.0
    assert summary.explained_refusal_rate == 1.0
    assert summary.n_reasons_collected == 3
    # three unrelated justifications should score as low-consistency
    assert summary.mean_reason_consistency < 0.3


if __name__ == "__main__":
    test_parse_refusal_detects_marker_and_extracts_reason()
    test_parse_refusal_no_marker()
    test_reason_consistency_identical_reasons_score_high()
    test_reason_consistency_different_reasons_score_low()
    test_reason_consistency_single_reason_returns_nan()
    test_run_prompt_end_to_end_with_fake_client()
    print("All smoke tests passed.")
