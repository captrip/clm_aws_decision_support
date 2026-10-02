# CLM Multi-Provider Decision Support Design

## Goal

Build a standalone, non-Git Python project that runs the repository's per-question model-selection algorithms with pluggable LinUCB, CLM, or hybrid decision policies. Candidate deployments may be hosted on AWS Bedrock or Azure OpenAI; Claude Sonnet 5 on AWS Bedrock is the default answer grader.

## Scope and constraints

- Create the project outside `Online_Multi_LLM`, at `/Users/trips/Desktop/clm_aws_decision_support`.
- Do not initialize or use Git for the new project.
- Reuse only the relevant algorithm behavior from the AWS notebooks; do not copy notebooks, datasets, cached model state, or historical results.
- Do not include OpenRouter or the `openai` OpenRouter client.
- Use `CLMClient`, `Choice`, `Noul`, and `Score` from `clm` for CLM-backed decisions.
- Use `us.anthropic.claude-sonnet-5` through AWS Bedrock as the default grader.
- Keep credentials out of source and configuration files.

## Architecture

The cascade owns the life cycle of a single question: construct state, ask a selection policy for the next logical model(s), invoke the resolved deployment, grade the answer, account for cost, log the attempt, and feed the observed reward and cost back into learning components.

Policies do not call model-provider APIs. Providers do not know selection logic. This keeps the current algorithms intact and makes CLM independently replaceable.

```text
question + prior attempts + budget
              |
              v
    SelectionPolicy (linucb | clm | hybrid)
              |
              v
  logical model name(s) from model registry
              |
              v
 Provider router (Bedrock | Azure OpenAI)
              |
              v
 answer -> Sonnet 5 Bedrock grader -> reward/cost feedback -> policy state
```

### Provider-neutral model registry

Every selectable arm has a stable logical name and provider details. Policies use only the logical name and numeric metadata such as estimated cost and score.

```yaml
models:
  nova_lite:
    provider: bedrock
    model_id: amazon.nova-lite-v1:0
    input_cost_per_million: 0.06
    output_cost_per_million: 0.24
  gpt_4o_mini:
    provider: azure_openai
    deployment: gpt-4o-mini-prod
    input_cost_per_million: 0.15
    output_cost_per_million: 0.60

grader:
  provider: bedrock
  model_id: us.anthropic.claude-sonnet-5
```

`BedrockProvider` uses boto3's Converse API. `AzureOpenAIProvider` uses a configured Azure OpenAI endpoint, API version, and either an API key or the Azure identity credential chain. The initial implementation reads the normal AWS credential chain plus explicit environment-backed Azure settings; it stores no secret values.

### Policies

All policies satisfy one interface: accept a decision context and return an ordered selection of eligible logical model names plus decision metadata for logging.

- `linucb`: extracted greedy LinUCB, budget-aware LinUCB, and PAKH-compatible selectors choose from learned reward/cost scores.
- `clm`: sends the current question, cascade step, prior failures, eligible models, costs, and budget to `CLMClient.system_one`. The request includes a `Choice` question for the model, `Score` for decision confidence, and `Noul` for rationale.
- `hybrid`: supplies LinUCB's UCB and cost estimates as CLM decision criteria. CLM makes the final choice; grades and actual costs still update LinUCB.

CLM is optional. `linucb` mode does not import or contact CLM. `clm` and `hybrid` report a clear configuration or connection error if the service cannot be reached at its configured endpoint (default `http://127.0.0.1:8700`).

### Grading and feedback

The grader is a provider-independent component configured initially with Bedrock Sonnet 5. It sends an answer and expected result, requests a boolean verdict, disables Sonnet 5 thinking content, and treats empty/ambiguous replies as incorrect. The cascade records exact provider usage and calculates costs using the selected logical model's configured per-token rates.

After each attempt, the cascade emits an immutable attempt record that includes policy decision metadata, selected logical model, resolved provider, usage, cost, answer, grade, and reward. LinUCB learning state is updated for every attempt in `linucb` and `hybrid` modes.

## Project layout

```text
clm_aws_decision_support/
  src/clm_aws_decision_support/
    bedrock.py
    azure_openai.py
    providers.py
    models.py
    grader.py
    cascade.py
    datasets.py
    policies/
      base.py
      linucb.py
      clm.py
      hybrid.py
  tests/
  examples/
  pyproject.toml
  README.md
```

## Error handling

- Reject unknown logical models, unconfigured providers, duplicate selections, and model choices outside the eligible/budget-valid set before an invocation.
- Preserve provider errors with provider and logical model context; bounded retries apply to transient Bedrock failures.
- The cascade logs failed invocations with zero reward and safe cost handling, then continues according to cascade/budget rules.
- No test invokes real AWS, Azure, or CLM. Fakes model each boundary.

## Verification

- Unit tests cover each policy mode, mixed-provider routing, provider request/usage normalization, grader request settings, budget eligibility, invalid CLM selections, unavailable CLM behavior, and LinUCB feedback updates.
- A clearly labeled smoke example is available for a configured AWS account, Azure OpenAI deployment, and running CLM server; it is not part of the test suite.
