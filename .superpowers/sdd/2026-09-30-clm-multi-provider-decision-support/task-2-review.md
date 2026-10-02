# Task 2 review — Provider adapters and routing

## Verdicts

- **Specification compliance: PASS.** Task 2 implements the required provider protocol and logical-model router, normalizes Bedrock and Azure OpenAI usage into `Usage`, defers Azure optional imports, uses environment-backed Azure configuration, retains standard boto3 credential resolution, retries only the explicit transient Bedrock error-code set, and gives routed failures logical-model/provider context.
- **Code quality: PASS WITH NON-BLOCKING FINDINGS.** The implementation is compact, typed at the public boundaries, separates routing from provider-specific request construction, and has no credential persistence in the reviewed files. The findings below concern regression coverage, not an observed production defect.

## Verification performed

- Read the Task 2 plan, all Global Constraints, and the design specification before review.
- Read the Task 2 implementation report and all in-scope production/test files.
- Ran `pytest -p no:cacheprovider tests/test_providers.py -v`: **8 passed**.
- Ran an isolated import probe with `openai` and `azure` imports deliberately blocked: importing `clm_aws_decision_support.azure_openai` succeeded.
- Searched reviewed files for Azure settings and credential-like material. They reference only environment-variable names; no credential values are stored.

## Findings

### P2 — The test named “retries transient errors only” does not verify the “only” requirement

- **Path:** `tests/test_providers.py:161`
- **Issue:** The test proves a `ThrottlingException` is retried, but it does not exercise a permanent Bedrock error and assert exactly one call/no sleep. The implementation currently does the correct thing through `_is_transient`, but this test will not catch a future broad retry policy that retries authorization, validation, or arbitrary failures.
- **Recommendation:** Add a fake non-transient error case and assert that it is raised after one invocation with no delay.

### P2 — Azure optional-import behavior is verified only by the task report’s ad hoc command, not by the unit suite

- **Path:** `tests/test_providers.py:189`
- **Issue:** The injected-client test demonstrates the useful dependency-injection path, but it does not block `openai`/`azure` imports. The report records a separate import probe, and this review independently reproduced that probe successfully; nevertheless, the regression guarantee is outside `pytest`.
- **Recommendation:** Add a unit test that temporarily blocks `openai` and `azure` imports, imports the adapter, and invokes it with an injected client.

### P3 — Provider-error coverage exercises an invalid return value, not an adapter exception

- **Path:** `tests/test_providers.py:116`
- **Issue:** The router correctly wraps both invalid results and thrown provider exceptions in source, but the test reaches the wrapper by returning `None`. It does not assert that a provider-thrown error remains the exception cause while the outer error retains logical-model and provider context.
- **Recommendation:** Make the fake adapter raise a sentinel exception and assert both the contextual `ProviderInvocationError` message and `__cause__`.

## Focus-area assessment

- **Optional Azure imports:** Pass. `openai` is imported only while building a real client; `azure.identity` is imported only for no-key authentication. An injected client needs neither package.
- **Normalized usage:** Pass. Bedrock maps `inputTokens`/`outputTokens`; Azure maps `prompt_tokens`/`completion_tokens`; absent values become zero.
- **Transient Bedrock retries:** Pass in source. The retry set is bounded and limited to four named transient error codes. See P2 for test coverage.
- **Provider errors:** Pass in source. Unknown models, unconfigured adapters, and invocation failures include relevant context; adapter failures are retained as causes. See P3 for test coverage.
- **Credential persistence:** Pass. AWS resolves through boto3; Azure values are read from environment only when constructing a client. No credential values are written to reviewed source, tests, or report.
