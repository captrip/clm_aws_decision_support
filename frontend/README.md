# MONA frontend

React + TypeScript console for the FastAPI service in `src/clm_aws_decision_support/webapp.py`.
The interface uses Whitebat's charcoal, muted-gold, Playfair Display, and DM Mono theme.

## Configure

Copy `.env.example` to `.env` and set:

- `VITE_API_BASE_URL` to the running API (defaults to `http://localhost:8800`).
- `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to the Supabase project used by the backend.

Only the public Supabase publishable/anon key belongs in the frontend. Keep the Supabase
service-role key and provider credentials on the backend.

## Run

```sh
npm install
npm run dev
```

The frontend signs users in with a Supabase email link. Create an API key in the console
or paste an existing one under **API keys** to use models, ask questions, run evaluations,
manage deployments, and import feedback. The selected API key is held in browser
session storage. Model requests and evaluations can incur provider charges.

```sh
npm run build
```
