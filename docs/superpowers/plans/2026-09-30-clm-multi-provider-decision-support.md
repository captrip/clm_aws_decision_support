# CLM Multi-Provider Decision Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone Python package that runs per-question model cascades with LinUCB, CLM, or hybrid selection over mixed AWS Bedrock and Azure OpenAI model deployments.

**Architecture:** The cascade coordinates policy selection, provider invocation, Sonnet 5 grading, cost accounting, logging, and feedback. Policies select provider-neutral logical model names; a registry and provider router resolve and invoke each configured deployment. CLM is an optional policy dependency, while the hybrid policy augments its structured CLM decision with LinUCB scores and cost estimates.

**Tech Stack:** Python 3.11+, boto3, Azure OpenAI SDK with optional Azure Identity authentication, PyYAML, NumPy, CLM, pytest.

**Spec:** `docs/superpowers/specs/2026-09-30-clm-multi-provider-decision-support-design.md`

## Global Constraints

- Project root is `/Users/trips/Desktop/clm_aws_decision_support`; do not initialize or use Git.
- Do not depend on OpenRouter or any OpenRouter `openai` client configuration.
- Do not commit credentials or deployment secrets; read AWS configuration from standard boto3 resolution and Azure configuration from environment-backed settings.
- Default grader is AWS Bedrock `us.anthropic.claude-sonnet-5`, with thinking disabled.
- Candidate models may be configured with either `bedrock` or `azure_openai` providers.
- Unit tests must not contact AWS, Azure, or a CLM server.

## Review Focus

- A CLM response selecting an ineligible, unknown, duplicate, or over-budget model is rejected before any provider call.
- A mixed registry routes logical model names to the correct provider while preserving per-provider token usage and configured cost rates.
- An unavailable CLM server affects only `clm` and `hybrid` modes, never `linucb` mode.
- Bedrock Sonnet 5 grading omits temperature, disables thinking, and treats text-free or ambiguous responses as incorrect.
- A failed provider invocation creates a safe zero-reward attempt and does not corrupt LinUCB/cost state.

---

### Task 1: Package foundation and provider-neutral configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/clm_aws_decision_support/__init__.py`
- Create: `src/clm_aws_decision_support/models.py`
- Create: `src/clm_aws_decision_support/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces `ModelSpec`, `GraderSpec`, `ProviderKind`, `ModelRegistry`, and `load_config(path: Path) -> AppConfig`.
- Produces typed provider-neutral `InvocationResult` and `Usage` records for Tasks 2–6.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_load_config_supports_bedrock_and_azure_openai_models(tmp_path):
    config = load_config(write_mixed_provider_config(tmp_path))
    assert config.models["nova_lite"].provider is ProviderKind.BEDROCK
    assert config.models["gpt_4o_mini"].provider is ProviderKind.AZURE_OPENAI

def test_model_registry_rejects_duplicate_logical_model_names():
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry([bedrock_spec("same"), azure_spec("same")])
```

- [ ] **Step 2: Run the configuration tests to verify failure**

Run: `pytest tests/test_config.py -v`
Expected: FAIL because package/configuration interfaces do not exist.

- [ ] **Step 3: Implement configuration and typed records**

Implement immutable dataclasses and enum-backed validation. Parse the documented YAML shape with explicit required provider fields: `model_id` for Bedrock and `deployment` for Azure OpenAI. Retain configured input/output cost per million tokens.

- [ ] **Step 4: Run configuration tests to verify passing**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

### Task 2: Provider adapters and routing

**Files:**
- Create: `src/clm_aws_decision_support/providers.py`
- Create: `src/clm_aws_decision_support/bedrock.py`
- Create: `src/clm_aws_decision_support/azure_openai.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `ModelSpec`, `InvocationResult`, and `Usage` from Task 1.
- Produces `ModelProvider.invoke(model: ModelSpec, prompt: str, max_tokens: int, temperature: float | None) -> InvocationResult` and `ProviderRouter.invoke(logical_name: str, ...) -> InvocationResult`.

- [ ] **Step 1: Write failing provider routing tests**

```python
def test_router_sends_bedrock_model_to_bedrock_provider():
    result = router_with_fakes().invoke("nova_lite", "solve", 32, 0.0)
    assert result.provider == "bedrock"
    assert result.usage.input_tokens == 7

def test_router_sends_azure_deployment_to_azure_openai_provider():
    result = router_with_fakes().invoke("gpt_4o_mini", "solve", 32, 0.0)
    assert result.provider == "azure_openai"
    assert result.usage.output_tokens == 9
```

- [ ] **Step 2: Run provider tests to verify failure**

Run: `pytest tests/test_providers.py -v`
Expected: FAIL because provider adapters/router do not exist.

- [ ] **Step 3: Implement provider protocol, Bedrock adapter, Azure OpenAI adapter, and router**

`BedrockProvider` wraps boto3 Converse and normalizes all text blocks plus `inputTokens`/`outputTokens`. It retries only transient Bedrock errors. `AzureOpenAIProvider` invokes the configured deployment through Azure OpenAI and normalizes usage. The router looks up a logical name, selects its adapter, and attaches contextual provider/model errors.

- [ ] **Step 4: Run provider tests to verify passing**

Run: `pytest tests/test_providers.py -v`
Expected: PASS.

### Task 3: Sonnet 5 grader and cost accounting

**Files:**
- Create: `src/clm_aws_decision_support/grader.py`
- Modify: `src/clm_aws_decision_support/models.py`
- Test: `tests/test_grader.py`

**Interfaces:**
- Consumes: `ProviderRouter`, `GraderSpec`, `ModelRegistry`, and `InvocationResult`.
- Produces `AnswerGrader.grade(answer: str, expected: str) -> bool` and `calculate_cost(model: ModelSpec, usage: Usage) -> float`.

- [ ] **Step 1: Write failing grader tests**

```python
def test_sonnet_grader_disables_thinking_and_omits_temperature():
    grader.grade("x = 2", "2")
    assert router.calls[0].model_id == "us.anthropic.claude-sonnet-5"
    assert router.calls[0].temperature is None
    assert router.calls[0].additional_fields == {"thinking": {"type": "disabled"}}

def test_ambiguous_grader_reply_is_incorrect():
    assert grader_returning("maybe").grade("x", "y") is False
```

- [ ] **Step 2: Run grader tests to verify failure**

Run: `pytest tests/test_grader.py -v`
Expected: FAIL because grading/cost interfaces do not exist.

- [ ] **Step 3: Implement grading and costs**

Use a short boolean-only grading prompt. Accept only clear true/false verdicts. Short-circuit exact non-empty matches. Compute dollar cost from normalized token counts and the selected arm's configured rates.

- [ ] **Step 4: Run grader tests to verify passing**

Run: `pytest tests/test_grader.py -v`
Expected: PASS.

### Task 4: LinUCB state and baseline selection policy

**Files:**
- Create: `src/clm_aws_decision_support/policies/base.py`
- Create: `src/clm_aws_decision_support/policies/linucb.py`
- Test: `tests/test_linucb_policy.py`

**Interfaces:**
- Produces `DecisionContext`, `Selection`, `SelectionPolicy`, `LinUCBState`, and `LinUCBPolicy.select(context: DecisionContext) -> Selection`.
- Produces `LinUCBState.update(logical_model: str, features: NDArray, reward: float, cost: float) -> None` for Tasks 5–6.

- [ ] **Step 1: Write failing LinUCB tests**

```python
def test_linucb_selects_the_highest_ucb_eligible_model():
    selection = policy.select(context(eligible=("cheap", "strong")))
    assert selection.models == ("strong",)

def test_linucb_update_changes_future_score_for_observed_model():
    before = state.score("cheap", features)
    state.update("cheap", features, reward=1.0, cost=0.01)
    assert state.score("cheap", features) != before
```

- [ ] **Step 2: Run LinUCB tests to verify failure**

Run: `pytest tests/test_linucb_policy.py -v`
Expected: FAIL because policy/state interfaces do not exist.

- [ ] **Step 3: Implement extracted LinUCB policy behavior**

Extract the score/update mathematics from the AWS notebooks into a focused state object. The baseline policy returns one eligible model with the largest UCB score and complete score metadata. Keep a cost-estimate hook in the context so the next task can implement budget-aware and PAKH-compatible selection without changing the interface.

- [ ] **Step 4: Run LinUCB tests to verify passing**

Run: `pytest tests/test_linucb_policy.py -v`
Expected: PASS.

### Task 5: CLM and hybrid policies

**Files:**
- Create: `src/clm_aws_decision_support/policies/clm.py`
- Create: `src/clm_aws_decision_support/policies/hybrid.py`
- Test: `tests/test_clm_policy.py`
- Test: `tests/test_hybrid_policy.py`

**Interfaces:**
- Consumes: `DecisionContext`, `Selection`, `SelectionPolicy`, and `LinUCBState`.
- Produces `CLMPolicy(client: CLMClient).select(context: DecisionContext) -> Selection` and `HybridPolicy(client: CLMClient, linucb: LinUCBState).select(context: DecisionContext) -> Selection`.

- [ ] **Step 1: Write failing CLM/hybrid tests**

```python
def test_clm_policy_uses_choice_score_and_noul_questions():
    selection = policy_with_fake_clm("nova_lite").select(context())
    assert selection.models == ("nova_lite",)
    assert fake_client.system_one_calls[0].questions.keys() == {"model", "confidence", "reason"}

def test_clm_policy_rejects_model_outside_eligible_set():
    with pytest.raises(InvalidSelectionError, match="eligible"):
        policy_with_fake_clm("unknown").select(context(eligible=("nova_lite",)))

def test_hybrid_passes_ucb_and_cost_signals_into_clm_criteria():
    policy.select(context())
    assert "UCB" in fake_client.system_one_calls[0].questions["model"].criteria["nova_lite"]
```

- [ ] **Step 2: Run CLM/hybrid tests to verify failure**

Run: `pytest tests/test_clm_policy.py tests/test_hybrid_policy.py -v`
Expected: FAIL because CLM/hybrid policy interfaces do not exist.

- [ ] **Step 3: Implement CLM and hybrid selection**

Import CLM lazily only when constructing CLM-backed policies. Build the structured `system_one` request from question state, failures, eligible models, budget, and costs. Use `Choice` for logical model choice, `Score` for confidence, and `Noul` for rationale. Validate the response before returning it. Hybrid enriches the criteria with current LinUCB UCB and estimated-cost values; it does not alter LinUCB state during selection.

- [ ] **Step 4: Run CLM/hybrid tests to verify passing**

Run: `pytest tests/test_clm_policy.py tests/test_hybrid_policy.py -v`
Expected: PASS.

### Task 6: Cascade orchestration, budget eligibility, and feedback

**Files:**
- Create: `src/clm_aws_decision_support/cascade.py`
- Test: `tests/test_cascade.py`

**Interfaces:**
- Consumes: policy interface from Tasks 4–5, `ProviderRouter`, `AnswerGrader`, `ModelRegistry`.
- Produces `Cascade.run(question: Question, expected_answer: str, budget: float | None) -> QuestionResult`.

- [ ] **Step 1: Write failing cascade tests**

```python
def test_cascade_routes_a_mixed_provider_selection_and_records_attempt():
    result = cascade.run(question("2 + 2"), "4", budget=1.0)
    assert result.attempts[0].provider == "azure_openai"
    assert result.attempts[0].reward == 1

def test_provider_failure_records_zero_reward_and_continues():
    result = cascade_with_failing_first_provider().run(question("q"), "a", budget=1.0)
    assert result.attempts[0].reward == 0
    assert len(result.attempts) == 2

def test_clm_selection_over_budget_is_rejected_before_invocation():
    with pytest.raises(InvalidSelectionError):
        cascade.run(question("q"), "a", budget=0.001)
```

- [ ] **Step 2: Run cascade tests to verify failure**

Run: `pytest tests/test_cascade.py -v`
Expected: FAIL because cascade interfaces do not exist.

- [ ] **Step 3: Implement the one-question cascade**

Build decision contexts for each attempt, calculate eligible models from current budget/cost estimates, invoke each selected model in order, grade, calculate cost, append immutable attempt records, and update LinUCB feedback whenever a LinUCB state is present. Treat provider failures as failed attempts with safe accounting and continue until success, attempt limit, or budget exhaustion.

- [ ] **Step 4: Run cascade tests to verify passing**

Run: `pytest tests/test_cascade.py -v`
Expected: PASS.

### Task 7: Executable example, documentation, and full verification

**Files:**
- Create: `examples/run_math500.py`
- Create: `examples/models.example.yaml`
- Create: `README.md`
- Modify: `pyproject.toml`
- Test: `tests/test_example_configuration.py`

**Interfaces:**
- Consumes: all package public interfaces.
- Produces a documented CLI with `--policy linucb|clm|hybrid`, a safe sample config, and a real-service smoke workflow excluded from pytest.

- [ ] **Step 1: Write failing example/configuration tests**

```python
def test_example_config_has_no_credentials_and_supports_both_providers():
    config = load_config(EXAMPLE_CONFIG)
    assert {model.provider.value for model in config.models.values()} == {"bedrock", "azure_openai"}
    assert "AKIA" not in EXAMPLE_CONFIG.read_text()
```

- [ ] **Step 2: Run example/configuration tests to verify failure**

Run: `pytest tests/test_example_configuration.py -v`
Expected: FAIL because example artifacts do not exist.

- [ ] **Step 3: Implement example and README**

Document installation extras, AWS and Azure environment setup, CLM service requirement for `clm`/`hybrid`, configuration fields, policy behavior, and commands for test versus explicitly opt-in real-service smoke execution. The example loads Math500-compatible JSON and writes results only to a user-supplied output path.

- [ ] **Step 4: Run full test suite and static import check**

Run: `pytest -v && python -m compileall src`
Expected: PASS with no network calls.

## Self-Review

- Spec coverage: Tasks 1–7 cover the project boundary, provider-neutral registry, AWS/Azure routing, all three policy modes, Sonnet 5 grading, error handling, testing, and documented real-service usage.
- Type consistency: model records and normalized invocation results originate in Task 1 and are consumed by providers, grader, and cascade; policy selection uses logical model names throughout.
- Review focus coverage: Task 5 validates invalid CLM choices and CLM isolation; Task 2 validates routing/usage; Task 3 validates Sonnet settings; Task 6 validates provider failure and budget rejection.
- Proportion: implementation details are specified by test behavior and interface boundaries; algorithm bodies remain implementation decisions.
