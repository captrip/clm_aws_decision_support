# MONA demo and production configuration

MONA supports an explicit local demo mode and a production mode backed by AWS
Bedrock and/or Azure OpenAI. Use this guide to configure the API safely; a
successful deployment still requires real provider credentials and reviewed
model deployments.

## Current state

- `CLM_RUNTIME_MODE` defaults to `demo` for local development. The Docker image
  explicitly defaults to `production`.
- Demo mode seeds an empty registry with simulated deployments and uses
  simulated models, grader, and CLM policy client.
- Production mode uses the Bedrock and Azure OpenAI adapters, never seeds
  simulated deployments, and refuses a database marked as demo-seeded.
- `POST /api/ask` invokes a configured model provider. The `clm` and `hybrid`
  policies additionally require `CLM_BASE_URL`; they do not fall back to the
  in-process simulator in production.
- Production `POST /api/run` requires `CLM_GRADER_PROVIDER` and
  `CLM_GRADER_MODEL`, and uses the selected real provider adapter to grade.
- An empty production registry is allowed at startup but cannot route
  requests. Add and verify live deployments before enabling user traffic.
- The React client uses Supabase email-link auth. Its publishable key is
  public; the backend separately needs a Supabase service-role key to validate
  user sessions.

## 1. Keep environments and credentials separate

Create a separate persistent data directory and Supabase project (or an
intentionally isolated set of production tables/keys) for production. Do not
promote the demo SQLite database or demo API keys.

Configure secrets in your deployment platform's secret manager, not in source
control or a `VITE_*` variable:

| Setting | Where it is used | Secret? |
| --- | --- | --- |
| `VITE_API_BASE_URL` | Frontend build; public API origin | No |
| `VITE_SUPABASE_URL` | Frontend build; Supabase project URL | No |
| `VITE_SUPABASE_ANON_KEY` | Frontend build; Supabase publishable/anon key | No |
| `SUPABASE_URL` | Backend session validation | No |
| `SUPABASE_SERVICE_ROLE_KEY` | Backend Supabase admin client | **Yes** |
| AWS credentials / role | Backend Bedrock adapter | **Yes**; prefer workload identity/instance role over static keys |
| `AZURE_OPENAI_ENDPOINT` | Backend Azure OpenAI adapter | No |
| `AZURE_OPENAI_API_VERSION` | Backend Azure OpenAI adapter | No |
| `AZURE_OPENAI_API_KEY` | Backend Azure OpenAI adapter, if using API-key auth | **Yes**; alternatively use Azure identity |
| `CLM_BASE_URL` | Backend, when using a separately deployed CLM service | Depends on deployment; protect the endpoint |
| `CLM_DATA_DIR` | Backend database and LinUCB snapshot location | No; must be persistent and writable |
| `CLM_RUNTIME_MODE` | Backend runtime; use `production` outside local development | No |
| `CLM_CORS_ORIGINS` | Comma-separated allowed browser origins | No |
| `CLM_GRADER_PROVIDER` / `CLM_GRADER_MODEL` | Production `/api/run` grading adapter and model | No |

The frontend `.env` belongs to the browser build. Never put provider keys or the
Supabase service-role key there. Only `VITE_*` values are exposed to frontend
code. Keep the Supabase publishable key separate from the service-role key.

Set the Supabase Auth allowed redirect URL(s) to the deployed frontend origin
(and any local development origin you explicitly need).

## 2. Configure production providers

Set `CLM_RUNTIME_MODE=production` explicitly outside the supplied Docker image.
Configure provider credentials on the backend host using its secret manager.
Bedrock uses the standard AWS SDK credential chain; Azure requires an endpoint
and API version plus either an API key or a working Azure Identity credential
chain.

The production registry is SQLite-backed. Add reviewed deployments with the
authenticated deployment management API, or provision the registry through a
controlled administrative process. Each Bedrock deployment needs a valid
Bedrock `model_id`; each Azure OpenAI deployment needs an Azure `deployment`.
Set input/output rates in USD per million tokens using the provider's current
price sheet. `sample_accuracy` is demo/learning metadata and does not configure
a live provider.

To use `clm` or `hybrid`, configure and secure the separately hosted CLM
service, then set `CLM_BASE_URL`. To enable `/api/run`, set
`CLM_GRADER_PROVIDER` to `bedrock` or `azure_openai` and
`CLM_GRADER_MODEL` to the provider's model ID or deployment name. The grader's
provider spend is not currently included in `/api/usage` metering.

The adapters use Bedrock's Converse API and the Azure OpenAI chat-completions
API. Azure authentication can use `AZURE_OPENAI_API_KEY` or Azure Identity's
default credential chain. Azure Identity is an optional import in the adapter.

The YAML loader in `config.py` is not used by the web app. SQLite is the active
web registry source of truth; do not assume edits to YAML change deployed
models.

## 3. Install and configure runtime dependencies

The `web`, `auth`, `bedrock`, `azure`, and aggregate `production` dependency
extras are declared in `pyproject.toml`. The Docker image installs the
`production` extra. Build it from a clean checkout and configure a persistent,
writable `CLM_DATA_DIR`; do not rely on packages from a developer environment.

## 4. Replace demo data deliberately

Production mode never seeds a registry. Demo seeding marks its database, and
production startup rejects that marked database. Use an isolated production
`CLM_DATA_DIR` and populate it with reviewed live deployments; the service
starts with an empty production registry but requests cannot route until models
are configured.

Before adding real deployments:

- Confirm each logical name is unique and each provider-specific identifier is
  correct.
- Confirm rates use the expected USD-per-million unit.
- Treat any stored `sample_accuracy` values as demo-only unless they come from
  measured, documented production data.
- `sample_accuracy` validation accepts values from 0 through 1. It is not a
  provider-quality guarantee and should not be treated as measured accuracy
  unless the value is supported by documented evaluation.
- Back up and migrate existing SQLite data intentionally. Do not point the
  live app at a demo database and expect demo rows or learning state to be
  valid for production.

## 5. Prepare the service for production traffic

- Set `CLM_CORS_ORIGINS` to a comma-separated list of exact frontend origins.
  The default is limited to local development origins.
- Serve the API over HTTPS and keep it off the public internet unless access is
  protected by the intended authentication and network controls.
- Persist `CLM_DATA_DIR`, including the SQLite database and LinUCB snapshot.
  Test backup and restore. Use one application worker while learning state is
  process-local; multiple workers do not share a single synchronized LinUCB
  state.
- Review database and learning-state ownership before serving multiple
  customers. The current LinUCB state and active deployment registry are
  process-wide, not partitioned by user. The API key identifies a user, but
  that alone does not make model-learning state or model configuration
  user-isolated.
- Set request, token, provider, and per-request budget limits. Treat the
  configured model rates and budget gate as estimates, not a hard provider
  billing cap.
- Add production logs and monitoring for provider errors, throttling,
  timeouts, latency, token usage, cost, feedback, and failed grading. Never
  log credentials or full sensitive prompts/answers unless data policy permits
  it.
- The process-wide registry and LinUCB learning state are shared across API
  keys. This is not tenant-isolated model configuration or learning state and
  must be addressed before treating the service as a multi-tenant production
  control plane.
- `/api/usage` currently meters successful `/api/ask` requests only. Billing,
  quotas, hard provider spend caps, and grader usage metering are not
  implemented.
- Deployment administration currently shares the process-wide registry; do
  not issue product API keys as administrative credentials. Restrict
  deployment-management access at the network/service layer until dedicated
  account-scoped administration is implemented.

## 6. Verify in stages

1. **Unit tests:** run `pytest` with mocked providers; no real provider calls
   should be needed.
2. **Provider smoke tests:** use a restricted test account and low token
   limits to call each configured provider and verify returned text, token
   usage, error handling, and estimated cost.
3. **Grader check:** compare known matching and non-matching answer pairs.
   Confirm the configured grader returns sensible verdicts and that its spend
   is included in cost reporting.
4. **Policy check:** test `linucb`, `clm`, and `hybrid` separately. Verify that
   the selected model belongs to the active registry and that failures do not
   trigger a simulator fallback.
5. **API/auth check:** sign in through the production Supabase project, create
   an API key, call authenticated endpoints, then revoke the key and confirm
   it no longer works.
6. **Operational check:** restart the service and verify the production
   registry and learning state load from persistent storage. Confirm the
   service is reachable only on the intended origins and network paths.
7. **Gradual release:** start with a small set of users, conservative budgets,
   and provider spend alerts. Expand model availability and traffic only after
   observed latency, quality, and cost match expectations.

## Production readiness checklist

- [x] Explicit production/demo mode; production does not silently use
      simulators or accept a marked demo database.
- [x] Live provider adapters and configurable production grader are wired.
- [x] Runtime dependency extras and production Docker default are declared.
- [x] Deployment request accuracy validation accepts values from 0 to 1.
- [ ] Production model registry reviewed and provider credentials smoke-tested.
- [ ] Backend and frontend secrets stored separately and securely.
- [ ] CORS, HTTPS, authentication, API-key revocation, and user isolation
      reviewed.
- [ ] Persistent storage, backup/restore, and single-worker learning-state
      behavior tested.
- [ ] Provider cost, rate-limit, monitoring, and incident procedures in place.
- [ ] Provider, grader, policy, auth, and restart smoke tests pass.
