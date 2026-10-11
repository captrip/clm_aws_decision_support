import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import type { Session } from '@supabase/supabase-js';
import { apiRequest, currency, downloadProductPackage, type ApiKey, type AskResult, type Deployment, type MAFDemoMessage, type MAFDemoResult, type MAFDemoTraceStep, type Model, type Policy, type RunResult } from './src/api';
import { supabase } from './src/supabase';

type Page = 'overview' | 'products' | 'maf_demo' | 'playground' | 'evaluate' | 'deployments' | 'keys' | 'feedback' | 'docs';
const pages: { id: Page; label: string; glyph: string }[] = [
  { id: 'overview', label: 'Overview', glyph: '◫' },
  { id: 'products', label: 'Products', glyph: '⬡' },
  { id: 'maf_demo', label: 'MAF demo', glyph: '✳' },
  { id: 'playground', label: 'Playground', glyph: '⌁' },
  { id: 'evaluate', label: 'Evaluate', glyph: '◉' },
  { id: 'deployments', label: 'Deployments', glyph: '◈' },
  { id: 'keys', label: 'API keys', glyph: '⌘' },
  { id: 'feedback', label: 'Import feedback', glyph: '↗' },
  { id: 'docs', label: 'API guide', glyph: '▤' },
];

function money(value: number | undefined): string {
  return currency(value ?? 0);
}

function pretty(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function Brand() {
  return (
    <a className="brand" href="/" aria-label="Whitebat">
      <img src="/whitebat-logo.png" alt="" />
      <span>WHITEBAT</span>
    </a>
  );
}

function SignIn() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    setMessage('');
    if (!supabase) {
      setError('Add the Supabase URL and publishable key to frontend/.env, then restart the dev server.');
      return;
    }
    setBusy(true);
    try {
      const { error: authError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: { emailRedirectTo: window.location.origin },
      });
      if (authError) throw authError;
      setMessage('Check your inbox for a secure sign-in link.');
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not send the sign-in link.');
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="auth-shell">
      <Brand />
      <section className="auth-card">
        <span className="eyebrow">MONA / DECISION SUPPORT</span>
        <h1>Make every<br /><em>model count.</em></h1>
        <p>Sign in to manage your model portfolio, run evaluations, and guide your routing policy.</p>
        <form onSubmit={submit}>
          <label htmlFor="signin-email">Work email</label>
          <input id="signin-email" type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@company.com" />
          {error && <p className="form-error" role="alert">{error}</p>}
          {message && <p className="form-success" role="status">{message}</p>}
          <button className="button button-primary" type="submit" disabled={busy}>
            {busy ? 'Sending link…' : 'Continue with email'} <span>↗</span>
          </button>
        </form>
        <p className="auth-footnote">Secure sign-in powered by Supabase.</p>
      </section>
      <p className="auth-caption">INTELLIGENCE · AUTOMATION · IMPACT</p>
    </main>
  );
}

function PageHeading({ eyebrow, title, description, aside }: { eyebrow: string; title: string; description: string; aside?: ReactNode }) {
  return (
    <header className="view-heading">
      <div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>
      {aside}
    </header>
  );
}

function StatusPill({ children, tone = 'green' }: { children: ReactNode; tone?: 'green' | 'gold' | 'muted' }) {
  return <span className={`status-pill ${tone}`}><i />{children}</span>;
}

function EmptyState({ title, detail, action }: { title: string; detail: string; action?: ReactNode }) {
  return <div className="empty-state"><span className="empty-mark">✳</span><h3>{title}</h3><p>{detail}</p>{action}</div>;
}

function Overview({ models, deployments, apiKey, setPage, backendStatus }: {
  models: Model[]; deployments: Deployment[]; apiKey: string; setPage: (page: Page) => void; backendStatus: string;
}) {
  return (
    <section className="console-view">
      <PageHeading eyebrow="MONA / OVERVIEW" title="Choose with confidence." description="A live view of your multi-model decision system." aside={<StatusPill tone={backendStatus === 'online' ? 'green' : 'muted'}>{backendStatus === 'online' ? 'API connected' : backendStatus}</StatusPill>} />
      {!apiKey && <div className="callout">
        <div><span className="eyebrow">ONE QUICK SETUP</span><h2>Connect your workspace</h2><p>Create or enter an API key to load models and use the decision-support endpoints.</p></div>
        <button className="button button-primary" onClick={() => setPage('keys')}>Set up API access <span>→</span></button>
      </div>}
      {apiKey && <div className="metrics">
        <article><span>Active deployments</span><b>{deployments.length}</b><small>{deployments.length ? 'Available to decision policies' : 'Add your first model deployment'}</small></article>
        <article><span>Model options</span><b>{models.length}</b><small>Bedrock and Azure OpenAI</small></article>
        <article><span>Routing policies</span><b>03</b><small>LinUCB · CLM · Hybrid</small></article>
      </div>}
      <div className="home-grid">
        <article className="panel home-feature"><span className="eyebrow">ASK / ROUTE</span><h2>Explore an answer.</h2><p>Submit a real question and let the selected policy choose a model. Review its rationale, token use, and cost.</p><button className="button button-primary" onClick={() => setPage('playground')}>Open playground <span>↗</span></button></article>
        <article className="panel home-feature home-feature-gold"><span className="eyebrow">EVALUATE / LEARN</span><h2>Put routing to the test.</h2><p>Compare candidate models against an expected answer. Grading feedback helps the LinUCB policy learn.</p><button className="button button-outline" onClick={() => setPage('maf_demo')}>Open MAF agent demo <span>↗</span></button></article>
      </div>
      <div className="panel roster-panel">
        <div className="panel-title"><div><span className="eyebrow">MODEL PORTFOLIO</span><h2>Configured deployments</h2></div><button className="button-link" onClick={() => setPage('deployments')}>Manage models →</button></div>
        {deployments.length ? <div className="table-wrap"><table><thead><tr><th>Deployment</th><th>Provider</th><th>Input / 1M</th><th>Output / 1M</th></tr></thead><tbody>{deployments.slice(0, 5).map((item) => <tr key={item.logical_name}><td><strong>{item.logical_name}</strong><small>{item.model_id || item.deployment}</small></td><td><span className="provider-label">{pretty(item.provider)}</span></td><td>{money(item.input_cost_per_million)}</td><td>{money(item.output_cost_per_million)}</td></tr>)}</tbody></table></div> : <EmptyState title={apiKey ? 'No active deployments' : 'Your portfolio will appear here'} detail={apiKey ? 'Add a Bedrock model or Azure OpenAI deployment to get started.' : 'Connect your API key to load your live model registry.'} action={apiKey ? <button className="button button-outline" onClick={() => setPage('deployments')}>Add deployment</button> : undefined} />}
      </div>
      <p className="disclaimer">Provider requests and evaluations may incur costs according to your configured model rates.</p>
    </section>
  );
}

function Products({ apiKey, setPage }: { apiKey: string; setPage: (page: Page) => void }) {
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  async function download(product: 'sentinel' | 'maf') {
    setError('');
    if (!apiKey) {
      setPage('keys');
      return;
    }
    setBusy(product);
    try {
      await downloadProductPackage(product, apiKey);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Package download failed.');
    } finally {
      setBusy('');
    }
  }

  return <section className="console-view">
    <PageHeading eyebrow="WHITEBAT / PRODUCT SUITE" title="One workspace. Three products." description="MONA routes model requests; Sentinel supervises agent execution; MAF builds multi-agent workflows." />
    <div className="callout product-key-callout">
      <div><span className="eyebrow">SHARED API ACCESS</span><h2>One personal API key works across the suite.</h2><p>Your account's key authenticates MONA routing and is used by Sentinel and MAF integrations when they call MONA. Keys remain owned by the account that created them; never share a secret between users.</p></div>
      <button className="button button-primary" onClick={() => setPage('keys')}>{apiKey ? 'Manage API keys' : 'Create API key'} <span>→</span></button>
    </div>
    {error && <p className="form-error page-alert" role="alert">{error}</p>}
    <div className="product-grid">
      <article className="panel product-card product-card-featured">
        <span className="eyebrow">01 / MODEL ROUTING</span><h2>MONA</h2>
        <p>Route requests across configured model providers using policy-based selection, with API-key authentication and usage metering.</p>
        <div className="product-tags"><span>Hosted API</span><span>LinUCB · CLM · Hybrid</span></div>
        <button className="button button-primary" onClick={() => setPage('playground')}>Try MONA <span>↗</span></button>
      </article>
      <article className="panel product-card">
        <span className="eyebrow">02 / AGENT RUNTIME</span><h2>Sentinel</h2>
        <p>A Python runtime for supervised agent execution, concurrency, retries, and observability. Use its MONA client with your account API key.</p>
        <div className="product-tags"><span>Python package</span><span>CLI / library</span></div>
        <pre><code>{`export MONA_API_URL="${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8800'}"\nexport MONA_API_KEY="$CLM_API_KEY"\nunzip sentinel-package.zip\ncd sentinel\npython -m pip install -e ".[dev]"\n\nfrom sentinal import MonaClient\nclient = MonaClient()\nresult = await client.ask_async("Route this task")`}</code></pre>
        <button className="button button-outline" onClick={() => void download('sentinel')} disabled={busy === 'sentinel'}>{busy === 'sentinel' ? 'Preparing package…' : 'Download Sentinel source'} <span>↓</span></button>
      </article>
      <article className="panel product-card">
        <span className="eyebrow">03 / MULTI-AGENT WORKFLOWS</span><h2>MAF</h2>
        <p>Compose multi-agent workflows and run them under Sentinel supervision. Connect agent tasks to MONA with the same account-owned API key.</p>
        <div className="product-tags"><span>Python package</span><span>MAF + Sentinel</span></div>
        <pre><code>{`export MONA_API_URL="${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8800'}"\nexport MONA_API_KEY="$CLM_API_KEY"\nunzip sentinel-package.zip && cd sentinel\npython -m pip install -e ".[dev]"\ncd .. && unzip maf-package.zip && cd maf\npython -m pip install -e ".[dev]"\n\nfrom sentinal import MonaClient\nmona = MonaClient()\n# Use mona.ask_async(...) in an MAF skill runner.`}</code></pre>
        <button className="button button-outline" onClick={() => setPage('maf_demo')}>Launch MAF demo <span>↗</span></button>
        <button className="button button-outline" onClick={() => void download('maf')} disabled={busy === 'maf'}>{busy === 'maf' ? 'Preparing package…' : 'Download MAF source'} <span>↓</span></button>
      </article>
    </div>
    <p className="disclaimer">Sentinel and MAF downloads contain source packages, not embedded credentials. Use your own API key via environment variables. MONA remains the shared authenticated inference endpoint.</p>
  </section>;
}

function MAFDemo({ apiKey, setPage }: { apiKey: string; setPage: (page: Page) => void }) {
  const [agentName, setAgentName] = useState('Workspace Assistant');
  const [instructions, setInstructions] = useState('Be a reliable conversational assistant. Be direct, accurate, and clear. If you are unsure, say so.');
  const [policy, setPolicy] = useState<Policy>('linucb');
  const [qualityThreshold, setQualityThreshold] = useState('0.65');
  const [draft, setDraft] = useState('');
  const [messages, setMessages] = useState<MAFDemoMessage[]>([]);
  const [lastRun, setLastRun] = useState<MAFDemoResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  async function send(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiKey || !draft.trim() || busy) return;
    const question = draft.trim();
    const priorConversation = messages.slice(-10);
    setMessages([...messages, { role: 'user', content: question }]);
    setDraft('');
    setError('');
    setLastRun(null);
    setBusy(true);
    try {
      const result = await apiRequest<MAFDemoResult>('/api/maf/demo', {
        method: 'POST',
        body: JSON.stringify({
          question,
          agent_name: agentName,
          instructions,
          conversation: priorConversation,
          policy,
          budget: 0.01,
          quality_threshold: Number(qualityThreshold),
        }),
      }, apiKey);
      setMessages((current) => [...current, { role: 'assistant', content: result.answer }]);
      setLastRun(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'The MAF demo could not complete.');
    } finally {
      setBusy(false);
    }
  }

  function clearConversation() {
    setMessages([]);
    setLastRun(null);
    setError('');
  }

  return <section className="console-view">
    <PageHeading eyebrow="MAF × MONA × SENTINEL" title="Build a conversational agent." description="MAF assembles your agent, MONA selects and calls the model, and Sentinel evaluates the draft and steers a revision when needed." />
    {!apiKey && <div className="callout"><div><span className="eyebrow">API KEY REQUIRED</span><h2>Connect your account key to run the agent.</h2><p>Each routed model call uses your MONA key and is counted in your account usage.</p></div><button className="button button-primary" onClick={() => setPage('keys')}>Open API keys <span>→</span></button></div>}
    {error && <p className="form-error page-alert" role="alert">{error}</p>}
    <div className="agent-demo-layout">
      <aside className="panel agent-config">
        <span className="eyebrow">01 / MAF AGENT FACTORY</span><h2>Configure your agent</h2>
        <label htmlFor="maf-agent-name">Agent name</label>
        <input id="maf-agent-name" value={agentName} maxLength={80} onChange={(event) => setAgentName(event.target.value)} />
        <label htmlFor="maf-agent-instructions">Agent instructions</label>
        <textarea id="maf-agent-instructions" rows={6} maxLength={2000} value={instructions} onChange={(event) => setInstructions(event.target.value)} />
        <label htmlFor="maf-agent-policy">MONA routing policy</label>
        <PolicySelect id="maf-agent-policy" value={policy} onChange={setPolicy} />
        <label htmlFor="maf-quality">Sentinel quality threshold <strong>{qualityThreshold}</strong></label>
        <input id="maf-quality" type="range" min="0.25" max="1" step="0.05" value={qualityThreshold} onChange={(event) => setQualityThreshold(event.target.value)} />
        <p className="form-note">Sentinel scores draft completeness from response detail, then can issue one steering instruction and rerun the agent through MONA.</p>
        <div className="product-tags"><span>MAF agent</span><span>MONA-routed</span><span>Sentinel-governed</span></div>
      </aside>
      <div className="panel agent-chat">
        <div className="panel-title"><div><span className="eyebrow">02 / CONVERSATION</span><h2>{agentName || 'New agent'}</h2></div><button className="button-link" onClick={clearConversation} disabled={busy || !messages.length}>New conversation</button></div>
        <div className="chat-messages" aria-live="polite">
          {!messages.length && !busy && <EmptyState title="Your agent is ready" detail="Ask a question to run the configured MAF agent through MONA and Sentinel." />}
          {messages.map((message, index) => <article key={`${index}-${message.role}`} className={`chat-message ${message.role}`}><span>{message.role === 'user' ? 'YOU' : agentName.toUpperCase()}</span><p>{message.content}</p></article>)}
          {busy && <article className="chat-message assistant"><span>MAF → MONA → SENTINEL</span><p className="typing">Creating agent response, evaluating quality, and checking for steering…</p></article>}
        </div>
        <form className="agent-composer" onSubmit={(event) => void send(event)}>
          <textarea aria-label="Message your MAF agent" rows={3} maxLength={4000} value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="What should your agent help with?" />
          <button className="button button-primary" type="submit" disabled={!apiKey || busy || !draft.trim()}>{busy ? 'Running agent…' : 'Send to agent'} <span>↗</span></button>
        </form>
        {lastRun && <div className="agent-run-report">
          <div className="agent-run-summary"><span className="eyebrow">03 / SENTINEL RUNTIME REPORT</span><div className="product-tags"><span>{lastRun.sentinel.status}</span><span>{pretty(lastRun.policy)} policy</span><span>{lastRun.model}</span><span>{lastRun.elapsed_ms.toFixed(0)} ms</span><span>{currency(lastRun.cost)}</span></div><p>{lastRun.rationale}</p></div>
          <div className="agent-trace">{lastRun.sentinel.trace.map((step, index) => <MAFTraceStep key={`${step.stage}-${index}`} step={step} />)}</div>
          <p className="form-note">Execution {lastRun.execution_id} · MONA decision {lastRun.decision_id || 'not returned'}. Additional steering passes make additional MONA calls and may incur additional provider cost.</p>
        </div>}
      </div>
    </div>
  </section>;
}

function MAFTraceStep({ step }: { step: MAFDemoTraceStep }) {
  const title = step.stage === 'maf_agent'
    ? `${step.agent || 'MAF agent'} · draft ${step.attempt || 1}`
    : step.stage === 'sentinel_evaluation'
      ? `Sentinel quality evaluation · ${step.score?.toFixed(2) ?? '—'} / ${step.threshold?.toFixed(2) ?? '—'}`
      : 'Sentinel steering';
  const detail = step.stage === 'maf_agent'
    ? `${step.model || 'MONA'} · ${pretty(step.provider || 'provider')} · ${step.elapsed_ms?.toFixed(0) ?? '—'} ms · ${currency(step.cost || 0)}`
    : step.stage === 'sentinel_evaluation'
      ? step.instruction || (step.correction ? `Correction: ${step.correction}` : 'Draft met the configured threshold; no correction required.')
      : step.instruction || 'Instruction applied to the next MAF agent pass.';
  return <article className={`agent-trace-step ${step.stage}`}><i aria-hidden="true" /><div><b>{title}</b><p>{detail}</p></div><StatusPill tone={step.stage === 'sentinel_steering' ? 'gold' : 'muted'}>{step.stage === 'maf_agent' ? 'routed' : step.stage === 'sentinel_steering' ? 'steered' : step.correction || 'evaluated'}</StatusPill></article>;
}

function PolicySelect({ id, value, onChange }: { id: string; value: Policy; onChange: (policy: Policy) => void }) {
  return <select id={id} value={value} onChange={(event) => onChange(event.target.value as Policy)}><option value="linucb">LinUCB · adaptive</option><option value="clm">CLM · language model</option><option value="hybrid">Hybrid · combined</option></select>;
}

type DecisionScores = AskResult['candidates'] | RunResult['attempts'][number]['candidates'];

function formatScore(value: number | string): string {
  return typeof value === 'number' ? value.toFixed(4) : value;
}

function WorkflowNode({ icon, label, title, detail, tone, tooltip, children }: {
  icon: string;
  label: string;
  title: string;
  detail: string;
  tone: 'gold' | 'blue' | 'green' | 'red';
  tooltip: string;
  children?: ReactNode;
}) {
  return (
    <div className={`workflow-node ${tone}`} tabIndex={0} aria-label={`${label}: ${title}. ${tooltip}`}>
      <div className="workflow-node-top"><span className="workflow-icon" aria-hidden="true">{icon}</span><span className="workflow-node-label">{label}</span><span className="workflow-info" aria-hidden="true">i</span></div>
      <strong>{title}</strong>
      <span className="workflow-node-detail">{detail}</span>
      {children}
      <div className="workflow-tooltip" role="tooltip">
        <b>{title}</b>
        <p>{tooltip}</p>
      </div>
    </div>
  );
}

function CandidateChoices({ candidates, selected }: { candidates: DecisionScores; selected: string }) {
  const entries = Object.entries(candidates || {});
  if (!entries.length) return <span className="workflow-empty-detail">No candidate scores returned</span>;
  return (
    <div className="workflow-candidates" aria-label="Policy candidate scores">
      {entries.map(([name, scores]) => {
        const summary = Object.entries(scores).map(([key, value]) => `${pretty(key)} ${formatScore(value)}`).join(' · ');
        return (
          <div className={`workflow-candidate ${name === selected ? 'selected' : ''}`} key={name} tabIndex={0} aria-label={`${name}${name === selected ? ', selected' : ''}. ${summary}`}>
            <span className="candidate-state" aria-hidden="true">{name === selected ? '✓' : '·'}</span>
            <b>{name}</b>
            {name === selected && <em>selected</em>}
            <div className="workflow-tooltip candidate-tooltip" role="tooltip"><b>{name}{name === selected ? ' · selected' : ''}</b><p>{summary || 'No scoring metadata returned for this candidate.'}</p></div>
          </div>
        );
      })}
    </div>
  );
}

interface WorkflowEntry {
  step: number;
  policy: string;
  model: string;
  provider: string | null;
  rationale: string | null;
  candidates: DecisionScores;
  cost: number;
  inputTokens: number;
  outputTokens: number;
  outcome?: 'correct' | 'incorrect' | 'error' | 'pending';
  error?: string | null;
}

function DecisionWorkflow({ entries, success, totalCost, retriedModels = [] }: {
  entries: WorkflowEntry[];
  success?: boolean;
  totalCost?: number;
  retriedModels?: string[];
}) {
  return (
    <section className="workflow-panel" aria-label="Policy decision workflow">
      <div className="workflow-heading">
        <div><span className="eyebrow">ROUTING TRACE</span><h3>Decision workflow</h3></div>
        <div className="workflow-legend"><span><i className="legend-selection" /> Policy selection</span><span><i className="legend-invocation" /> Model run</span><span><i className="legend-verdict" /> Verdict</span></div>
      </div>
      <div className="workflow-timeline">
        {entries.map((entry, index) => {
          const failed = entry.outcome === 'error' || entry.outcome === 'incorrect';
          const outcomeTone = entry.outcome === 'correct' ? 'green' : failed ? 'red' : 'blue';
          const outcomeLabel = entry.outcome === 'correct' ? 'Reference matched' : entry.outcome === 'incorrect' ? 'Reference mismatch' : entry.outcome === 'error' ? 'Invocation failed' : 'Answer returned';
          const outcomeTip = entry.error || (entry.outcome === 'correct'
            ? 'The grader marked this attempt as correct. The cascade stopped here.'
            : entry.outcome === 'incorrect'
              ? 'The grader marked this attempt as incorrect. The cascade continued to the next eligible model, if available.'
              : 'The model returned a response. No reference-answer grade is part of this request.');
          return (
            <div className={`workflow-run ${failed ? 'failed' : ''}`} key={`${entry.step}-${entry.model}`}>
              <div className="workflow-run-index"><span>{String(index + 1).padStart(2, '0')}</span><i /></div>
              <div className="workflow-run-content">
                <div className="workflow-run-heading"><div><span className="eyebrow">DECISION {String(entry.step + 1).padStart(2, '0')}</span><strong>{pretty(entry.policy)} policy</strong></div><span className={`workflow-run-status ${entry.outcome || 'complete'}`}>{entry.outcome === 'correct' ? '● Passed' : failed && index < entries.length - 1 ? '● Retried' : entry.outcome === 'incorrect' ? '● No match' : entry.outcome === 'error' ? '● Failed' : '● Complete'}</span></div>
                <div className="workflow-scroll">
                  <div className="workflow-steps">
                    <WorkflowNode icon="◇" label="Policy selection" title={pretty(entry.policy)} detail={`Selected ${entry.model}`} tone="gold" tooltip={entry.rationale || 'The policy selected a model from the eligible candidates. Hover over a candidate below to inspect its scores.'}>
                      <CandidateChoices candidates={entry.candidates} selected={entry.model} />
                    </WorkflowNode>
                    <span className="workflow-connector" aria-hidden="true"><i>›</i></span>
                    <WorkflowNode icon="⌁" label="Model invocation" title={entry.model} detail={entry.provider ? pretty(entry.provider) : 'Provider unavailable'} tone={entry.error ? 'red' : 'blue'} tooltip={entry.error || `Provider: ${entry.provider ? pretty(entry.provider) : 'not reported'} · Estimated cost: ${money(entry.cost)} · Input tokens: ${entry.inputTokens.toLocaleString()} · Output tokens: ${entry.outputTokens.toLocaleString()}`}>
                      <span className="workflow-node-metric">{money(entry.cost)} <small>est. cost</small></span>
                    </WorkflowNode>
                    <span className="workflow-connector" aria-hidden="true"><i>›</i></span>
                    <WorkflowNode icon={entry.outcome === 'correct' ? '✓' : entry.outcome === 'incorrect' ? '×' : entry.outcome === 'error' ? '!' : '○'} label={entry.outcome === 'error' ? 'Invocation status' : 'Evaluation'} title={outcomeLabel} detail={entry.outcome === 'correct' ? 'Cascade complete' : entry.outcome === 'incorrect' ? 'Next model selected' : entry.outcome === 'error' ? 'Next model selected' : 'Answer ready'} tone={outcomeTone} tooltip={outcomeTip} />
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      {entries.length === 0 && <p className="workflow-empty">No policy decisions were recorded for this run.</p>}
      {retriedModels.length > 0 && <p className="workflow-retry-note"><b>Models retried:</b> {retriedModels.join(' → ')} <span>Detailed retry decisions are not returned by this endpoint.</span></p>}
      <div className="workflow-summary">
        <span>{entries.length} {entries.length === 1 ? 'decision' : 'decisions'} · {new Set(entries.map((entry) => entry.model)).size} {new Set(entries.map((entry) => entry.model)).size === 1 ? 'model' : 'models'} selected</span>
        {totalCost !== undefined && <b>{money(totalCost)} total estimated cost</b>}
        {success !== undefined && <StatusPill tone={success ? 'green' : 'gold'}>{success ? 'Evaluation passed' : 'Evaluation did not match'}</StatusPill>}
      </div>
    </section>
  );
}

function Playground({ apiKey }: { apiKey: string }) {
  const [question, setQuestion] = useState('');
  const [policy, setPolicy] = useState<Policy>('linucb');
  const [budget, setBudget] = useState('0.01');
  const [result, setResult] = useState<AskResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState('');

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setFeedback(''); setResult(null); setBusy(true);
    try {
      const numericBudget = budget.trim() === '' ? null : Number(budget);
      if (numericBudget !== null && (!Number.isFinite(numericBudget) || numericBudget < 0)) throw new Error('Budget must be zero or a positive number.');
      const data = await apiRequest<AskResult>('/api/ask', { method: 'POST', body: JSON.stringify({ question, policy, budget: numericBudget }) }, apiKey);
      setResult(data);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'The request failed.'); }
    finally { setBusy(false); }
  }

  async function rate(rating: 'up' | 'down') {
    if (!result) return;
    setError('');
    try {
      const reply = await apiRequest<{ applied: boolean; reason?: string }>('/api/feedback', { method: 'POST', body: JSON.stringify({ decision_id: result.decision_id, rating }) }, apiKey);
      setFeedback(reply.applied ? 'Feedback saved. The LinUCB policy has been updated.' : reply.reason || 'Feedback was not applied.');
      if (!reply.applied) setError(reply.reason || 'This decision could not be rated.');
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Feedback could not be saved.'); }
  }

  return <section className="console-view">
    <PageHeading eyebrow="MONA / PLAYGROUND" title="Ask, then inspect." description="Route a single question and see why a model was selected." />
    <div className="workspace-grid">
      <form className="panel form-panel" onSubmit={submit}>
        <div className="form-title"><span className="eyebrow">01 / INPUT</span><h2>What are you working on?</h2></div>
        <label htmlFor="ask-question">Question</label>
        <textarea id="ask-question" required minLength={1} rows={7} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Describe the task you want a model to solve…" />
        <div className="form-grid">
          <div><label htmlFor="ask-policy">Decision policy</label><PolicySelect id="ask-policy" value={policy} onChange={setPolicy} /></div>
          <div><label htmlFor="ask-budget">Budget · USD</label><input id="ask-budget" type="number" min="0" step="0.001" value={budget} onChange={(event) => setBudget(event.target.value)} /></div>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="button button-primary" type="submit" disabled={busy || !apiKey}>{busy ? 'Routing question…' : 'Run decision'} <span>↗</span></button>
        <p className="form-note">Uses your configured API key. Model usage may incur provider charges.</p>
      </form>
      <div className="result-column">
        {result ? <article className="panel result-panel">
          <div className="result-top"><span className="eyebrow">02 / RESPONSE</span><StatusPill>{pretty(result.provider)}</StatusPill></div>
          <div className="result-model"><span>Selected model</span><h2>{result.model}</h2></div>
          <DecisionWorkflow entries={[{
            step: 0,
            policy: result.policy,
            model: result.model,
            provider: result.provider,
            rationale: result.rationale,
            candidates: result.candidates,
            cost: result.cost,
            inputTokens: result.usage.input_tokens,
            outputTokens: result.usage.output_tokens,
            outcome: 'pending',
          }]} retriedModels={result.retried_models} />
          <div className="answer-copy">{result.answer}</div>
          <div className="result-stats"><div><span>Estimated cost</span><b>{money(result.cost)}</b></div><div><span>Input tokens</span><b>{result.usage.input_tokens.toLocaleString()}</b></div><div><span>Output tokens</span><b>{result.usage.output_tokens.toLocaleString()}</b></div></div>
          <div className="rationale"><span className="eyebrow">POLICY RATIONALE</span><p>{result.rationale || 'No rationale was returned.'}</p></div>
          {result.retried_models.length > 0 && <p className="form-note">Retried: {result.retried_models.join(', ')}</p>}
          {result.feedback_eligible && <div className="feedback-actions"><span>Was this answer useful?</span><button aria-label="Rate answer as useful" onClick={() => void rate('up')}>↑</button><button aria-label="Rate answer as not useful" onClick={() => void rate('down')}>↓</button></div>}
          {feedback && <p className="form-success" role="status">{feedback}</p>}
        </article> : <EmptyState title="Your next decision, in view." detail="The selected model, rationale, candidates, token usage, and estimated cost will appear here." />}
      </div>
    </div>
  </section>;
}

function Evaluation({ apiKey }: { apiKey: string }) {
  const [question, setQuestion] = useState('');
  const [expected, setExpected] = useState('');
  const [policy, setPolicy] = useState<Policy>('linucb');
  const [budget, setBudget] = useState('0.01');
  const [maxAttempts, setMaxAttempts] = useState('3');
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setResult(null); setBusy(true);
    try {
      const data = await apiRequest<RunResult>('/api/run', { method: 'POST', body: JSON.stringify({
        question, expected_answer: expected, policy, budget: budget.trim() === '' ? null : Number(budget), max_attempts: Number(maxAttempts),
      }) }, apiKey);
      setResult(data);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Evaluation failed.'); }
    finally { setBusy(false); }
  }

  return <section className="console-view">
    <PageHeading eyebrow="MONA / EVALUATE" title="Measure the answer." description="Run the cascade against a reference answer and observe how policy choices perform." />
    <div className="workspace-grid">
      <form className="panel form-panel" onSubmit={submit}>
        <div className="form-title"><span className="eyebrow">EVALUATION SETUP</span><h2>Define a test case.</h2></div>
        <label htmlFor="run-question">Question</label><textarea id="run-question" required rows={4} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Enter the task…" />
        <label htmlFor="expected-answer">Expected answer</label><textarea id="expected-answer" required rows={3} value={expected} onChange={(event) => setExpected(event.target.value)} placeholder="What should a correct answer contain?" />
        <div className="form-grid">
          <div><label htmlFor="run-policy">Decision policy</label><PolicySelect id="run-policy" value={policy} onChange={setPolicy} /></div>
          <div><label htmlFor="run-budget">Budget · USD</label><input id="run-budget" type="number" min="0" step="0.001" value={budget} onChange={(event) => setBudget(event.target.value)} /></div>
          <div><label htmlFor="max-attempts">Maximum attempts</label><select id="max-attempts" value={maxAttempts} onChange={(event) => setMaxAttempts(event.target.value)}>{[1, 2, 3, 4, 5].map((value) => <option key={value}>{value}</option>)}</select></div>
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="button button-primary" disabled={busy || !apiKey} type="submit">{busy ? 'Evaluating…' : 'Run evaluation'} <span>↗</span></button>
        <p className="form-note">This calls the model cascade and configured grader. Provider usage may incur charges.</p>
      </form>
      <div className="result-column">
        {result ? <article className="panel result-panel">
          <div className="result-top"><span className="eyebrow">EVALUATION RESULT</span><StatusPill tone={result.success ? 'green' : 'gold'}>{result.success ? 'Reference matched' : 'No match'}</StatusPill></div>
          <div className="result-stats"><div><span>Attempts</span><b>{result.attempts.length}</b></div><div><span>Total cost</span><b>{money(result.total_cost)}</b></div><div><span>Policy</span><b>{pretty(result.policy)}</b></div></div>
          <DecisionWorkflow
            entries={result.attempts.map((attempt) => ({
              step: attempt.step,
              policy: attempt.policy,
              model: attempt.model,
              provider: attempt.provider,
              rationale: attempt.rationale,
              candidates: attempt.candidates,
              cost: attempt.cost,
              inputTokens: attempt.usage.input_tokens,
              outputTokens: attempt.usage.output_tokens,
              outcome: attempt.error ? 'error' : attempt.reward === null ? 'pending' : attempt.reward > 0 ? 'correct' : 'incorrect',
              error: attempt.error,
            }))}
            success={result.success}
            totalCost={result.total_cost}
          />
        </article> : <EmptyState title="Evidence, not guesswork." detail="Every attempt, grader verdict, selection rationale, and cost will be recorded here." />}
      </div>
    </div>
  </section>;
}

function Deployments({ apiKey, models, deployments, onRefresh }: { apiKey: string; models: Model[]; deployments: Deployment[]; onRefresh: () => Promise<void> }) {
  const [name, setName] = useState('');
  const [provider, setProvider] = useState<'bedrock' | 'azure_openai'>('bedrock');
  const [modelId, setModelId] = useState('');
  const [deployment, setDeployment] = useState('');
  const [inputCost, setInputCost] = useState('0');
  const [outputCost, setOutputCost] = useState('0');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);

  async function add(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setNotice(''); setBusy(true);
    try {
      await apiRequest('/api/deployments', { method: 'POST', body: JSON.stringify({
        logical_name: name.trim(), provider, model_id: provider === 'bedrock' ? modelId.trim() : null,
        deployment: provider === 'azure_openai' ? deployment.trim() : null,
        input_cost_per_million: Number(inputCost), output_cost_per_million: Number(outputCost),
      }) }, apiKey);
      setName(''); setModelId(''); setDeployment(''); setNotice('Deployment added to your active model registry.');
      await onRefresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Deployment could not be added.'); }
    finally { setBusy(false); }
  }

  async function remove(logicalName: string) {
    if (!window.confirm(`Deactivate ${logicalName}? This removes it from new policy selections.`)) return;
    setError(''); setNotice('');
    try {
      await apiRequest(`/api/deployments/${encodeURIComponent(logicalName)}`, { method: 'DELETE' }, apiKey);
      setNotice(`${logicalName} was deactivated.`);
      await onRefresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Deployment could not be removed.'); }
  }

  return <section className="console-view">
    <PageHeading eyebrow="MONA / MODEL PORTFOLIO" title="Shape the portfolio." description="Manage provider deployments and the cost profile used by routing decisions." />
    <div className="keys-layout deployment-layout">
      <form className="panel key-form" onSubmit={add}>
        <div><span className="eyebrow">NEW MODEL</span><h2>Add deployment</h2></div>
        <label htmlFor="logical-name">Logical name</label><input id="logical-name" required value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. fast_reasoner" />
        <label htmlFor="provider">Provider</label><select id="provider" value={provider} onChange={(event) => setProvider(event.target.value as 'bedrock' | 'azure_openai')}><option value="bedrock">AWS Bedrock</option><option value="azure_openai">Azure OpenAI</option></select>
        {provider === 'bedrock' ? <><label htmlFor="model-id">Bedrock model ID</label><input id="model-id" required value={modelId} onChange={(event) => setModelId(event.target.value)} placeholder="anthropic.claude…" /></> : <><label htmlFor="azure-deployment">Azure deployment name</label><input id="azure-deployment" required value={deployment} onChange={(event) => setDeployment(event.target.value)} placeholder="deployment-name" /></>}
        <div className="form-grid"><div><label htmlFor="input-cost">Input · USD / 1M tokens</label><input id="input-cost" type="number" required min="0" step="any" value={inputCost} onChange={(event) => setInputCost(event.target.value)} /></div><div><label htmlFor="output-cost">Output · USD / 1M tokens</label><input id="output-cost" type="number" required min="0" step="any" value={outputCost} onChange={(event) => setOutputCost(event.target.value)} /></div></div>
        {error && <p className="form-error" role="alert">{error}</p>}{notice && <p className="form-success" role="status">{notice}</p>}
        <button className="button button-primary" type="submit" disabled={busy || !apiKey}>{busy ? 'Saving…' : 'Add deployment'} <span>+</span></button>
      </form>
      <div className="panel key-list">
        <div className="panel-title"><div><span className="eyebrow">ACTIVE REGISTRY</span><h2>Deployments <small>{deployments.length}</small></h2></div></div>
        {deployments.length ? deployments.map((item) => <article className="deployment-row" key={item.logical_name}><div><b>{item.logical_name}</b><code>{item.model_id || item.deployment}</code><span>{pretty(item.provider)} · Input {money(item.input_cost_per_million)} / Output {money(item.output_cost_per_million)} per 1M</span></div><button className="button-link danger" onClick={() => void remove(item.logical_name)}>Deactivate</button></article>) : <EmptyState title="No deployments yet" detail="Add a provider deployment to make it available to the model registry." />}
        {models.length > 0 && <p className="form-note">{models.length} model{models.length === 1 ? '' : 's'} available in the routing registry.</p>}
      </div>
    </div>
  </section>;
}

function ApiKeys({ session, apiKey, onApiKeyChange }: { session: Session; apiKey: string; onApiKeyChange: (key: string) => void }) {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [label, setLabel] = useState('');
  const [newSecret, setNewSecret] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const data = await apiRequest<ApiKey[]>('/api/auth/keys', {}, undefined, session.access_token);
    setKeys(data);
  }
  useEffect(() => { void refresh().catch((caught: unknown) => setError(caught instanceof Error ? caught.message : 'Could not load API keys.')); }, [session.access_token]);

  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setNotice(''); setBusy(true);
    try {
      const created = await apiRequest<ApiKey & { api_key: string }>('/api/auth/keys', { method: 'POST', body: JSON.stringify({ label: label.trim() || null }) }, undefined, session.access_token);
      setNewSecret(created.api_key);
      onApiKeyChange(created.api_key);
      setLabel('');
      setNotice('Key created and selected for this browser session.');
      await refresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not create API key.'); }
    finally { setBusy(false); }
  }

  async function revoke(key: ApiKey) {
    if (!window.confirm(`Revoke the ${key.label || 'unnamed'} key? Existing clients using it will stop working.`)) return;
    setError(''); setNotice('');
    try {
      await apiRequest(`/api/auth/keys/${encodeURIComponent(key.id)}`, { method: 'DELETE' }, undefined, session.access_token);
      if (key.key_prefix && apiKey.startsWith(key.key_prefix)) onApiKeyChange('');
      setNewSecret(''); setNotice('API key revoked.');
      await refresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Could not revoke API key.'); }
  }

  async function copySecret() {
    try { await navigator.clipboard.writeText(newSecret); setNotice('API key copied to clipboard.'); }
    catch { setError('Clipboard access failed. Select and copy the key directly from the notice.'); }
  }

  return <section className="console-view">
    <PageHeading eyebrow="MONA / DEVELOPER ACCESS" title="Keys with intention." description="Create and revoke API credentials for your integrations." />
    {error && <p className="form-error page-alert" role="alert">{error}</p>}
    {notice && <p className="flash" role="status">{notice}</p>}
    {newSecret && <div className="secret-notice"><div><b>Copy this key now.</b><code>{newSecret}</code><span>The full secret is only shown once. It is selected for this browser session.</span></div><button className="button button-primary" onClick={() => void copySecret()}>Copy key</button></div>}
    <div className="keys-layout">
      <form className="panel key-form" onSubmit={create}><span className="eyebrow">NEW CREDENTIAL</span><h2>Create an API key</h2><p>Use a descriptive label for the environment or application this key will serve.</p><label htmlFor="key-label">Key label</label><input id="key-label" value={label} onChange={(event) => setLabel(event.target.value)} placeholder="e.g. Production" /><button className="button button-primary" type="submit" disabled={busy}>{busy ? 'Creating…' : 'Create key'} <span>+</span></button></form>
      <div className="panel key-list"><h2>API credentials <small>{keys.filter((key) => !key.revoked).length} active</small></h2>
        {keys.length ? keys.map((key) => <article className="key-row" key={key.id}><div><b>{key.label || 'Unlabelled key'}</b><code>{key.key_prefix}••••••••</code><span>Created {new Date(key.created_at).toLocaleDateString()}</span></div><div>{key.revoked ? <em className="status revoked-status">Revoked</em> : <><em className="status active-status">Active</em><button className="button-link danger" onClick={() => void revoke(key)}>Revoke</button></>}</div></article>) : <p className="empty">No API keys yet. Create one to connect your workspace.</p>}
      </div>
    </div>
    <div className="panel key-entry"><div><span className="eyebrow">CLIENT CONNECTION</span><h2>Use an existing key</h2><p>The selected key stays in this browser tab session; it is never sent to Supabase.</p></div><div className="key-entry-controls"><input aria-label="API key" type="password" autoComplete="off" placeholder="Paste an API key" value={apiKey} onChange={(event) => onApiKeyChange(event.target.value)} /><StatusPill tone={apiKey ? 'green' : 'muted'}>{apiKey ? 'Key selected' : 'Not connected'}</StatusPill></div></div>
  </section>;
}

function FeedbackImport({ apiKey, models }: { apiKey: string; models: Model[] }) {
  const [question, setQuestion] = useState('');
  const [model, setModel] = useState('');
  const [reward, setReward] = useState('1');
  const [cost, setCost] = useState('0');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => { if (!model && models.length) setModel(models[0].name); }, [model, models]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setError(''); setMessage(''); setBusy(true);
    try {
      const result = await apiRequest<{ applied: number; skipped: { index: number; reason: string }[] }>('/api/feedback/import', {
        method: 'POST', body: JSON.stringify({ records: [{ question, model, reward: Number(reward), cost: Number(cost) }] }),
      }, apiKey);
      setMessage(`${result.applied} record${result.applied === 1 ? '' : 's'} imported.${result.skipped.length ? ` ${result.skipped.map((item) => item.reason).join('; ')}` : ''}`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : 'Feedback could not be imported.'); }
    finally { setBusy(false); }
  }

  return <section className="console-view">
    <PageHeading eyebrow="MONA / LEARNING SIGNALS" title="Bring your evidence." description="Import a known result to seed feedback for the adaptive routing policy." />
    <form className="panel import-panel" onSubmit={submit}><span className="eyebrow">FEEDBACK RECORD</span><h2>One record at a time.</h2><p>Provide the original question, the logical model name, whether its result was correct, and its observed cost.</p>
      <label htmlFor="import-question">Question</label><textarea id="import-question" required rows={5} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="The original question…" />
      <div className="form-grid"><div><label htmlFor="import-model">Logical model</label><select id="import-model" required value={model} onChange={(event) => setModel(event.target.value)}><option value="" disabled>Select a model</option>{models.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></div><div><label htmlFor="import-reward">Outcome</label><select id="import-reward" value={reward} onChange={(event) => setReward(event.target.value)}><option value="1">Correct · positive reward</option><option value="0">Incorrect · negative reward</option></select></div><div><label htmlFor="import-cost">Observed cost · USD</label><input id="import-cost" type="number" min="0" step="any" value={cost} onChange={(event) => setCost(event.target.value)} /></div></div>
      {error && <p className="form-error" role="alert">{error}</p>}{message && <p className="form-success" role="status">{message}</p>}
      <button className="button button-primary" type="submit" disabled={busy || !apiKey || !models.length}>{busy ? 'Importing…' : 'Import feedback'} <span>↗</span></button>
    </form>
  </section>;
}

function Docs({ apiKey }: { apiKey: string }) {
  return <section className="console-view docs-view">
    <PageHeading eyebrow="MONA / INTEGRATION GUIDE" title="A clear path to your API." description="Use your API key to call the decision-support endpoints from your service." />
    <article className="panel doc-block"><span className="eyebrow">01 / AUTHENTICATE</span><h2>Keep the API key server-side.</h2><p>Send the key as <code>X-API-Key</code>. Do not embed a production key in a public client bundle.</p><pre><code>{`curl ${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8800'}/api/models \\\n  -H "X-API-Key: $CLM_API_KEY"`}</code></pre></article>
    <article className="panel doc-block"><span className="eyebrow">02 / ASK</span><h2>Route one question.</h2><pre><code>{`curl -X POST ${import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8800'}/api/ask \\\n  -H "Content-Type: application/json" \\\n  -H "X-API-Key: $CLM_API_KEY" \\\n  -d '{"question":"Summarize the request","policy":"linucb","budget":0.01}'`}</code></pre><p>Policies: <code>linucb</code>, <code>clm</code>, or <code>hybrid</code>. Rate an eligible answer at <code>POST /api/feedback</code> using its <code>decision_id</code>.</p></article>
    <article className="panel doc-block"><span className="eyebrow">03 / EVALUATE</span><h2>Compare against a reference.</h2><pre><code>{`POST /api/run\n{\n  "question": "…",\n  "expected_answer": "…",\n  "policy": "hybrid",\n  "budget": 0.01,\n  "max_attempts": 3\n}`}</code></pre><p>Evaluation calls use your configured providers and grader. Budget and model rates are estimates; actual charges depend on provider billing.</p></article>
    <p className="disclaimer">The key entered in the console is held in session storage only. {apiKey ? 'A key is currently selected.' : 'No key is currently selected.'}</p>
  </section>;
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [page, setPage] = useState<Page>('overview');
  const [apiKey, setApiKeyState] = useState(() => sessionStorage.getItem('mona-api-key') || '');
  const [models, setModels] = useState<Model[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [dataError, setDataError] = useState('');

  function setApiKey(value: string) {
    setApiKeyState(value);
    if (value) sessionStorage.setItem('mona-api-key', value);
    else sessionStorage.removeItem('mona-api-key');
  }

  useEffect(() => {
    if (!supabase) { setAuthLoading(false); return; }
    void supabase.auth.getSession().then(({ data, error }) => {
      if (error) setDataError(error.message);
      setSession(data.session);
      setAuthLoading(false);
    });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, currentSession) => {
      setSession(currentSession);
      if (!currentSession) setApiKey('');
    });
    return () => subscription.unsubscribe();
  }, []);

  useEffect(() => {
    void apiRequest<{ status?: string; service?: string }>('/').then(() => setBackendStatus('online')).catch(() => setBackendStatus('offline'));
  }, []);

  async function refreshWorkspace() {
    if (!apiKey) { setModels([]); setDeployments([]); setDataError(''); return; }
    setDataError('');
    try {
      const [nextModels, nextDeployments] = await Promise.all([
        apiRequest<Model[]>('/api/models', {}, apiKey),
        apiRequest<Deployment[]>('/api/deployments', {}, apiKey),
      ]);
      setModels(nextModels);
      setDeployments(nextDeployments);
    } catch (caught) {
      setDataError(caught instanceof Error ? caught.message : 'Could not load workspace data.');
      setModels([]);
      setDeployments([]);
    }
  }

  useEffect(() => { void refreshWorkspace(); }, [apiKey]);

  async function signOut() {
    if (!supabase) return;
    const { error } = await supabase.auth.signOut();
    if (error) setDataError(error.message);
  }

  if (authLoading) return <main className="loading-shell"><span className="eyebrow">MONA / WHITEBAT</span><p>Preparing your workspace…</p></main>;
  if (!session) return <SignIn />;

  const activePage = pages.find((item) => item.id === page)?.label || 'Overview';
  let content: ReactNode;
  if (page === 'overview') content = <Overview models={models} deployments={deployments} apiKey={apiKey} setPage={setPage} backendStatus={backendStatus} />;
  else if (page === 'products') content = <Products apiKey={apiKey} setPage={setPage} />;
  else if (page === 'maf_demo') content = <MAFDemo apiKey={apiKey} setPage={setPage} />;
  else if (page === 'playground') content = <Playground apiKey={apiKey} />;
  else if (page === 'evaluate') content = <Evaluation apiKey={apiKey} />;
  else if (page === 'deployments') content = <Deployments apiKey={apiKey} models={models} deployments={deployments} onRefresh={refreshWorkspace} />;
  else if (page === 'keys') content = <ApiKeys session={session} apiKey={apiKey} onApiKeyChange={setApiKey} />;
  else if (page === 'feedback') content = <FeedbackImport apiKey={apiKey} models={models} />;
  else content = <Docs apiKey={apiKey} />;

  return (
    <main className="console-layout">
      <aside className="sidebar">
        <Brand />
        <div className="console-product"><span>PRODUCT SUITE</span><small>MONA · Sentinel · MAF</small></div>
        <nav aria-label="Main navigation">{pages.map((item) => <button key={item.id} className={page === item.id ? 'selected' : ''} onClick={() => setPage(item.id)} aria-current={page === item.id ? 'page' : undefined}><span className="nav-glyph" aria-hidden="true">{item.glyph}</span><span>{item.label}</span></button>)}</nav>
        <div className="sidebar-bottom"><span className="user-dot">{(session.user.email || 'U').charAt(0).toUpperCase()}</span><div className="sidebar-user"><b title={session.user.email || ''}>{session.user.email}</b><button onClick={() => void signOut()}>Sign out</button></div><span className={`connection-dot ${backendStatus === 'online' ? 'online' : ''}`} title={`API ${backendStatus}`} /></div>
      </aside>
      <div className="console-main">
        <div className="mobile-bar"><Brand /><span>{activePage}</span></div>
        {dataError && <div className="global-error" role="alert"><span>{dataError}</span><button aria-label="Dismiss error" onClick={() => setDataError('')}>×</button></div>}
        {content}
      </div>
    </main>
  );
}
