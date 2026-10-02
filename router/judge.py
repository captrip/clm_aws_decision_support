"""Sampled LLM judge supplying delayed reward for unverifiable turns.

DESIGN.md section 6. Runs out of the request path, so latency does not matter.
Rewards land through Router.record_reward, which uses the context vector stored
at selection time -- a reward arriving minutes later is therefore identical in
effect to an immediate one. Bandits are order-insensitive.
"""

import re

import numpy as np

from router import arms as arms_module
from router import config

_PROMPT = """You are grading an AI assistant's response.

User request:
{query}

Assistant response:
{response}

Did the response adequately and correctly answer the request? Respond with only
the word 'True' or 'False'."""


def build_judge_prompt(query, response_text):
    return _PROMPT.format(query=query or "", response=response_text or "")


_VERDICT = re.compile(r"\b(true|false)\b")
_LEADING_VERDICT = re.compile(r"^\W*(true|false)\b")


def parse_verdict(text):
    """1, 0, or None when the judge did not answer usably.

    Three rules, all guarding against scoring a bad answer as good:
      - word-boundary matched, not substring, so "untrue" is not "true"
      - the verdict word must LEAD the reply, which is what the prompt asks
        for, so a hedge like "not true" is not read as a verdict
      - a reply containing BOTH words is ambiguous, not a vote for whichever
        appeared first

    Anything else returns None, which records nothing, per the three-valued
    contract shared with verify.check.
    """
    if not text:
        return None
    lowered = text.strip().lower()
    found = set(_VERDICT.findall(lowered))
    if len(found) != 1:
        return None
    leading = _LEADING_VERDICT.match(lowered)
    if leading is None:
        return None
    return 1 if leading.group(1) == "true" else 0


def run_judge_batch(
    router,
    client,
    limit=20,
    sample_rate=config.JUDGE_SAMPLE_RATE,
    converse_fn=arms_module.bedrock_converse,
    rng=None,
):
    rng = rng if rng is not None else np.random.default_rng()
    rows = router.store.pending_judge(limit=limit)
    stats = {
        "considered": len(rows),
        "judged": 0,
        "skipped": 0,
        "ambiguous": 0,
        "errors": 0,
    }

    for row in rows:
        if rng.random() >= sample_rate:
            stats["skipped"] += 1
            continue
        prompt = build_judge_prompt(row["query"], row["response_text"])
        try:
            # Sonnet 5: no temperature, thinking disabled, or it returns no text.
            text, _usage = converse_fn(
                client,
                config.JUDGE_MODEL,
                prompt,
                max_tokens=10,
                disable_thinking=True,
            )
        except Exception:  # noqa: BLE001 - one bad row must not kill the batch
            stats["errors"] += 1
            continue
        verdict = parse_verdict(text)
        if verdict is None:
            stats["ambiguous"] += 1
            continue
        router.record_reward(row["id"], verdict)
        stats["judged"] += 1
    return stats
