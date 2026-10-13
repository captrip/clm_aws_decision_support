# MONA Product Workspace

This repository is the product workspace for MONA, Sentinel, and MAF.

- **MONA** — policy-driven model selection and API-key-authenticated model
  routing. The FastAPI service lives in `src/clm_aws_decision_support`.
- **Sentinel** — the standalone agent execution runtime and its CLI/library
  examples, source, tests, and docs live in [`products/sentinel`](./products/sentinel).
- **MAF** — the multi-agent framework with its Sentinel integrations and
  examples lives in [`products/maf`](./products/maf).

Sentinel and MAF are preserved as separately packaged products instead of
being merged into MONA's Python package. This keeps their dependencies and
entry points distinct while making their complete source available in one
repository.

## Whitebat product console

After signing in to the frontend, open **Products** to see MONA, Sentinel, and
MAF together. Create an account-owned API key under **API keys**; the same
`X-API-Key` credential authenticates MONA requests, Sentinel's `MonaClient`,
and MAF skill integrations that call MONA. The console offers authenticated
source-package downloads for Sentinel and MAF. Package downloads do not embed
credentials.

### MAF Studio

The MAF card on **Products** opens **MAF Studio**, a live workspace for the
workflow and agent lifecycle. Work moves between personas (Chatbot Builder,
Workflow Builder, Agent Builder, AI Ethics & Compliance Champion, AI Governance
Manager) through shared workflow and agent catalogues. It is backed by the
`/api/maf` FastAPI router (`src/clm_aws_decision_support/maf_api`):

- State lives in SQLite (`$CLM_DATA_DIR/maf.db`). Every transition is checked
  server-side and written to an audit log.
- Identity is the Supabase bearer token. Personas are assigned per email by an
  admin in the Studio's Team tab. Admins come from `MAF_ADMIN_EMAILS`; outside
  `CLM_RUNTIME_MODE=production` the first user becomes admin when none exists.
- `MAF_SEPARATION_OF_DUTIES` (default on in production) stops a reviewer from
  approving work they submitted.
- Evaluation runs and the live feed use the [AG-UI](https://docs.ag-ui.com)
  protocol over server-sent events: `POST /api/maf/requests/{id}/agui` streams a
  MONA + Sentinel run (steps, tool calls, Sentinel events, answer) and
  `GET /api/maf/stream` pushes state snapshots to every open session.
  Evaluation also requires the account's `X-API-Key` and the MAF/Sentinel
  runtime (Python 3.13+).

The earlier single-call endpoint `POST /api/maf/demo` remains available.

## Local development

The products have independent dependency sets. Use Python 3.13+ for Sentinel
and Python 3.11+ for MONA and MAF core. MAF's Sentinel integration requires
Python 3.13+. Sentinel's concurrency tests require a free-threaded Python
3.14 build. From the repository root:

```sh
python -m pip install -e ".[web,auth,dev]"
python -m pip install -e "products/sentinel[dev]"
python -m pip install -e "products/maf[dev]"
```

For MAF/Sentinel integration imports, install Sentinel as shown above. Put
provider credentials in local environment variables or an ignored `.env` file;
never commit secrets. See
[`docs/DEMO_TO_PRODUCTION.md`](./docs/DEMO_TO_PRODUCTION.md) for MONA provider
configuration and production safeguards. The integrated MAF/Sentinel demo
requires Python 3.13+ and the `products/maf` and `products/sentinel` packages;
the provided backend Dockerfile installs these packages.

## Tests

Run each suite from the monorepo root. Set `PYTHONPATH` for MAF tests so the
source checkouts of both integrated products are importable:

```sh
python -m pytest tests -q
PYTHON_GIL=0 python -m pytest products/sentinel/tests -q
PYTHONPATH=products/sentinel:products/maf python -m pytest products/maf/tests -q
```

Run Sentinel's suite with its free-threaded 3.14 interpreter; GPU parallelism
tests additionally require Apple Silicon and MPS. Performance benchmark
assertions depend on the host hardware. MAF integration tests use mock graphs
and agents, so they do not require provider credentials. Tests and examples
under product archives may require external providers, credentials, or
services and are not part of the default test suite.
