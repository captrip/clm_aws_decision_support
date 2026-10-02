# Task 2 report — Provider adapters and routing

## Scope and status

Implemented Task 2 only. No later-task policy, grader, cascade, documentation, or example work was added. The project remains standalone and non-Git; no Git command was run and no repository was initialized.

## Rulings

- **Provider invocation fields:** `ModelProvider.invoke` and `ProviderRouter.invoke` accept `additional_fields: Mapping[str, Any] | None = None`, and both adapters pass it through to their provider-native request field. This implements the confirmed ruling required by Task 3: Sonnet 5 grading can send `{"thinking": {"type": "disabled"}}` while setting `temperature=None`. Cost if wrong: provider-specific future request controls would be dropped before reaching the selected adapter.
- **Non-Git execution record:** The standard SDD task scripts and ledger are Git-backed, which conflicts with the plan's binding global constraint not to use Git. The requested report is the task audit trail instead. Cost if wrong: Git-oriented SDD metadata is absent; this report preserves the TDD, commands, results, rulings, and changed-file record.
- **Optional Azure dependencies:** Azure SDK imports are deferred until the provider needs to create a real runtime client. An injected client works without either optional SDK. Cost if wrong: an Azure deployment without an API key requires `azure-identity` at runtime, and a real Azure client always requires the optional `openai` package.

## Files changed

- `src/clm_aws_decision_support/providers.py` — provider protocol, contextual errors, and logical-model router.
- `src/clm_aws_decision_support/bedrock.py` — boto3 Converse adapter, text/usage normalization, and transient-only retry.
- `src/clm_aws_decision_support/azure_openai.py` — lazy Azure OpenAI adapter with environment-backed endpoint, API version, API-key, and Azure-identity setup.
- `tests/test_providers.py` — provider/router routing, request construction, retry, normalization, contextual error, lazy-client, and configuration tests.
- `.superpowers/sdd/2026-09-30-clm-multi-provider-decision-support/task-2-report.md` — this report.

## Strict TDD evidence

1. Wrote `tests/test_providers.py` before adding any Task 2 production module.
2. Ran the scoped test command before implementation:

   ```text
   pytest tests/test_providers.py -v
   ```

   Result: expected RED state. Collection failed with `ModuleNotFoundError: No module named 'clm_aws_decision_support.providers'`; Task 2 did not yet exist.
3. Added the minimal adapters and router needed for the tests.
4. Re-ran the scoped suite:

   ```text
   pytest -p no:cacheprovider tests/test_providers.py -v
   ```

   Result: `8 passed in 0.04s`.

## Verification commands and results

| Command | Result |
| --- | --- |
| `pytest tests/test_providers.py -v` | Expected RED: Task 2 module absent (`ModuleNotFoundError`). |
| `pytest -p no:cacheprovider tests/test_providers.py -v` | PASS — 8 passed. |
| `pytest -p no:cacheprovider -v` | PASS — 13 passed (Task 1 plus Task 2). |
| `PYTHONPATH=src python -c '...optional Azure imports blocked...; import clm_aws_decision_support.azure_openai'` | PASS — Azure adapter imported while `openai` and `azure` imports were blocked. |

The first import probe used `python -S` and failed because it also suppresses the required PyYAML dependency imported by the package initializer. It was not an optional-Azure failure and did not prompt a code change. The final blocked-import probe isolates and validates the actual requirement.

## Concerns and follow-up boundaries

- Unit tests intentionally use fakes only and contacted no AWS, Azure, or CLM service, as required.
- The Bedrock adapter lazily creates `boto3.client("bedrock-runtime")`, preserving normal boto3 credential-chain resolution. Runtime credentials are never read from project configuration.
- Azure endpoint and API version are read from `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_VERSION` at runtime. The API key is read only from `AZURE_OPENAI_API_KEY` when creating the SDK client; no secret is written to source, YAML, tests, or this report. Without an API key, the normal Azure identity credential chain is used.
- Bedrock retries only the documented transient error-code set (`ThrottlingException`, `ServiceUnavailableException`, `InternalServerException`, `ModelNotReadyException`). Future provider-specific retry codes can be added in a later scoped task if operational evidence requires them.
