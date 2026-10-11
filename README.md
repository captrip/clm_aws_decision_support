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
credentials. The **MAF demo** page lets a user configure an agent, chat with it,
and inspect its MONA routing and Sentinel evaluation/steering trace. The
integrated demo endpoint is `POST /api/maf/demo` and requires the account's
`X-API-Key`.

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
