# MAF

MAF is the multi-agent workflow product in this monorepo. Its original source,
examples, notebooks, integration modules, and archive are retained here. The
legacy `MAF.*` import paths remain available.

## Local setup

Use Python 3.13 or newer to run the Sentinel integration. From the monorepo
root:

```sh
python -m pip install -e "products/sentinel[dev]"
python -m pip install -e "products/maf[dev]"
```

Agent workflows that route prompts through MONA should set `MONA_API_URL` to
the MONA service origin and `MONA_API_KEY` to an API key created for the
signed-in user. The Sentinel `MonaClient` sends that key to MONA using
`X-API-Key`; MAF skill runners can use the same client. Keys are account-owned
and are not embedded in package downloads.

For full MAF workflows and the interactive Sentinel search-agent example,
configure provider credentials in the environment. Never copy production
credentials into this repository.

## MONA + Sentinel conversational demo

The authenticated `POST /api/maf/demo` endpoint creates a MAF conversational
agent, routes each draft through MONA using the calling user's API key, then
evaluates the first draft in Sentinel. If the configured completeness
threshold is not met, Sentinel steers one actual follow-up agent call. Open
**MAF demo** in the signed-in console to configure and chat with the agent.
Set `MONA_API_URL` when the MONA API is not reachable at the request's own
service origin (for example, in a multi-service deployment).

## Supervised mock agent

The supervision smoke test does not call a provider:

```sh
python -m pytest products/maf/tests -q
```

The wrapper and higher-level Sentinel runtime integration are separate
interfaces. `MAF.global_.sentinel_orchestrator.SentinelOrchestrator` exposes
three-thread supervision for MAF agents; `MAF.integration.MAFSentinelRuntimeBridge`
adapts a MAF workflow to Sentinel's actor runtime.

Earlier experiments and real-provider examples remain in `archive/` for
understanding. They may require credentials, external services, or optional
integrations and are not part of the offline smoke test.