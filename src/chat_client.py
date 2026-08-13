"""
Minimal, provider-agnostic chat harness.

Deliberately thin: no framework, just enough to swap between OpenAI- and
Anthropic-style chat APIs behind one interface, so the experiment code
doesn't care which provider you point it at. Reads credentials from
environment variables only — never hardcode a key here, and never paste one
into this repo before pushing it public.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class ChatClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    def complete_conversation(self, system_prompt: str, messages: list[dict]) -> str: ...


@dataclass
class OpenAIClient:
    model: str = "gpt-4o-mini"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete_conversation(system_prompt, [{"role": "user", "content": user_prompt}])

    def complete_conversation(self, system_prompt: str, messages: list[dict]) -> str:
        from openai import OpenAI  # imported lazily so this file has no hard dependency

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=1.0,
        )
        return resp.choices[0].message.content


@dataclass
class AnthropicClient:
    model: str = "claude-haiku-4-5-20251001"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.complete_conversation(system_prompt, [{"role": "user", "content": user_prompt}])

    def complete_conversation(self, system_prompt: str, messages: list[dict]) -> str:
        import anthropic  # imported lazily so this file has no hard dependency

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system_prompt,
            messages=messages,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))


class FakeClient:
    """Deterministic stand-in used ONLY by the test suite — no network, no key,
    no real model. Lets us test the experiment plumbing (looping, scoring,
    saving) without needing API access. Never used by run_experiment.py."""

    def __init__(self, canned_responses: list[str]):
        self._responses = canned_responses
        self._i = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp

    def complete_conversation(self, system_prompt: str, messages: list[dict]) -> str:
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp


def get_client(provider: str, model: str | None = None) -> ChatClient:
    if provider == "openai":
        return OpenAIClient(model=model) if model else OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient(model=model) if model else AnthropicClient()
    raise ValueError(f"Unknown provider: {provider}")
