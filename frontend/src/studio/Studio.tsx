import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { currency } from '../api';
import {
  openStudioFeed, roleLabels, runEvaluation, studioApi,
  type AguiEvent, type Catalogue, type EvalProps, type MafRequest, type Me, type Member, type Role, type Status,
} from './api';

type StageId =
  | 'requirement' | 'wf_discovery' | 'wf_build' | 'wf_review' | 'wf_register'
  | 'agent_discovery' | 'agent_build' | 'agent_review' | 'agent_register'
  | 'compose' | 'evaluate' | 'release' | 'handover' | 'observe';
type NodeState = 'done' | 'active' | 'skipped' | 'pending';

const personaDuty: Record<Role, string> = {
  admin: 'Assign personas to your team and oversee access.',
  chatbot_builder: 'Raise requirements, run live evaluations, hand released work to UX.',
  workflow_builder: 'Build workflows that are not in the catalogue yet.',
  agent_builder: 'Build agents that are not in the catalogue yet.',
  ethics_champion: 'Approve workflows and authorise releases.',
  governance_manager: 'Validate, certify and approve agents.',
};
const tabOrder: Role[] = ['chatbot_builder', 'workflow_builder', 'agent_builder', 'ethics_champion', 'governance_manager', 'admin'];
const assignable: Role[] = tabOrder;

const lanes: { title: string; nodes: { stage: StageId; label: string; role: string; kind: 'process' | 'decision' | 'registry' }[] }[] = [
  { title: 'Workflow lifecycle', nodes: [
    { stage: 'requirement', label: 'Requirement', role: 'Chatbot Builder', kind: 'process' },
    { stage: 'wf_discovery', label: 'Workflow available?', role: 'Workflow Discovery', kind: 'decision' },
    { stage: 'wf_build', label: 'Create workflow', role: 'Workflow Builder', kind: 'process' },
    { stage: 'wf_review', label: 'Validate & approve', role: 'Ethics & Compliance Champion', kind: 'decision' },
    { stage: 'wf_register', label: 'Workflow registry', role: 'Workflow Catalogue', kind: 'registry' },
  ] },
  { title: 'Agent lifecycle', nodes: [
    { stage: 'agent_discovery', label: 'Agents available?', role: 'Agent Discovery', kind: 'decision' },
    { stage: 'agent_build', label: 'Create agent', role: 'Agent Builder', kind: 'process' },
    { stage: 'agent_review', label: 'Certify & approve', role: 'AI Governance Manager', kind: 'decision' },
    { stage: 'agent_register', label: 'Agent registry', role: 'Agent Catalogue', kind: 'registry' },
    { stage: 'compose', label: 'Add agents to workflow', role: 'MAF', kind: 'process' },
  ] },
  { title: 'Evaluate, release & observe', nodes: [
    { stage: 'evaluate', label: 'Evaluation', role: 'Chatbot Builder · MAF · Sentinel', kind: 'process' },
    { stage: 'release', label: 'Release', role: 'Ethics & Compliance Champion', kind: 'decision' },
    { stage: 'handover', label: 'Handover to UX', role: 'Chatbot Builder', kind: 'process' },
    { stage: 'observe', label: 'Observability', role: 'Complete', kind: 'registry' },
  ] },
];

const phaseText: Record<MafRequest['phase'], string> = {
  building: 'Building',
  ready_to_evaluate: 'Ready to evaluate',
  awaiting_release: 'Awaiting release',
  ready_for_handover: 'Ready for UX handover',
  complete: 'Complete',
};

function stageStates(r: MafRequest): Record<StageId, NodeState> {
  const any = (status: Status) => r.agents.some((agent) => agent.status === status);
  const allCatalogue = r.agents.every((agent) => agent.from_catalogue);
  const agentsReady = r.agents.every((agent) => agent.status === 'published');
  const ready = r.workflow.status === 'published' && agentsReady;
  const wf = r.workflow;
  return {
    requirement: 'done',
    wf_discovery: 'done',
    wf_build: wf.from_catalogue ? 'skipped' : wf.status === 'todo' ? 'active' : 'done',
    wf_review: wf.from_catalogue ? 'skipped' : wf.status === 'in_review' ? 'active' : wf.status === 'published' ? 'done' : 'pending',
    wf_register: wf.from_catalogue ? 'skipped' : wf.status === 'published' ? 'done' : 'pending',
    agent_discovery: 'done',
    agent_build: allCatalogue ? 'skipped' : any('todo') ? 'active' : 'done',
    agent_review: allCatalogue ? 'skipped' : any('in_review') ? 'active' : agentsReady ? 'done' : 'pending',
    agent_register: allCatalogue ? 'skipped' : agentsReady ? 'done' : 'pending',
    compose: ready ? 'done' : 'pending',
    evaluate: !ready ? 'pending' : r.evaluation ? 'done' : 'active',
    release: !r.evaluation ? 'pending' : r.released ? 'done' : 'active',
    handover: !r.released ? 'pending' : r.handed_over ? 'done' : 'active',
    observe: r.handed_over ? 'done' : 'pending',
  };
}

const splitList = (value: string) => [...new Set(value.split(',').map((item) => item.trim()).filter(Boolean))];
const errorText = (caught: unknown, fallback: string) => (caught instanceof Error ? caught.message : fallback);

interface RunView {
  status: 'idle' | 'running' | 'done' | 'error';
  stage: string;
  attempts: number;
  quality: number | null;
  steps: { name: string; done: boolean }[];
  tools: { id: string; name: string; args: string; result: string }[];
  custom: { name: string; value: unknown }[];
  text: string;
  error: string;
}
const idleRun: RunView = { status: 'idle', stage: '', attempts: 0, quality: null, steps: [], tools: [], custom: [], text: '', error: '' };

function applyEvent(view: RunView, event: AguiEvent): RunView {
  const str = (key: string) => String(event[key] ?? '');
  switch (event.type) {
    case 'RUN_STARTED': return { ...idleRun, status: 'running', stage: 'starting' };
    case 'STATE_SNAPSHOT': {
      const snap = event.snapshot as { stage?: string; attempts?: number } | undefined;
      return snap?.stage ? { ...view, stage: snap.stage, attempts: snap.attempts ?? view.attempts } : view;
    }
    case 'STATE_DELTA': {
      const next = { ...view };
      for (const op of (event.delta as { path: string; value: unknown }[]) ?? []) {
        if (op.path === '/stage') next.stage = String(op.value);
        if (op.path === '/attempts') next.attempts = Number(op.value);
        if (op.path === '/quality' && typeof op.value === 'number') next.quality = op.value;
      }
      return next;
    }
    case 'STEP_STARTED': return { ...view, steps: [...view.steps, { name: str('stepName'), done: false }] };
    case 'STEP_FINISHED': return { ...view, steps: view.steps.map((s) => (s.name === str('stepName') ? { ...s, done: true } : s)) };
    case 'TOOL_CALL_START': return { ...view, tools: [...view.tools, { id: str('toolCallId'), name: str('toolCallName'), args: '', result: '' }] };
    case 'TOOL_CALL_ARGS': return { ...view, tools: view.tools.map((t) => (t.id === str('toolCallId') ? { ...t, args: t.args + str('delta') } : t)) };
    case 'TOOL_CALL_RESULT': return { ...view, tools: view.tools.map((t) => (t.id === str('toolCallId') ? { ...t, result: str('content') } : t)) };
    case 'CUSTOM': return { ...view, custom: [...view.custom, { name: str('name'), value: event.value }] };
    case 'TEXT_MESSAGE_CONTENT': return { ...view, text: view.text + str('delta') };
    case 'RUN_FINISHED': return { ...view, status: 'done', stage: 'complete' };
    case 'RUN_ERROR': return { ...view, status: 'error', error: str('message') };
    default: return view;
  }
}

function Checklist({ items, checked, onToggle }: { items: string[]; checked: boolean[]; onToggle: (index: number) => void }) {
  return <div className="lifecycle-checks">{items.map((item, index) => <label key={item}><input type="checkbox" checked={checked[index]} onChange={() => onToggle(index)} /><span>{item}</span></label>)}</div>;
}

function ReviewCard({ title, subtitle, body, items, approveLabel, busy, onApprove, onReject }: {
  title: string; subtitle: string; body?: string; items: string[]; approveLabel: string; busy: boolean;
  onApprove: () => void; onReject: (note: string) => void;
}) {
  const [checked, setChecked] = useState(items.map(() => false));
  const [note, setNote] = useState('');
  return <article className="lifecycle-task">
    <div><b>{title}</b><small>{subtitle}</small></div>
    {body && <pre className="lifecycle-preview"><code>{body}</code></pre>}
    <Checklist items={items} checked={checked} onToggle={(index) => setChecked((current) => current.map((value, i) => (i === index ? !value : value)))} />
    <input aria-label="Rejection note" value={note} onChange={(event) => setNote(event.target.value)} placeholder="Rejection note (required to send back)" />
    <div className="lifecycle-actions">
      <button className="button button-primary" disabled={busy || !checked.every(Boolean)} onClick={onApprove}>{approveLabel} <span>→</span></button>
      <button className="button button-outline" disabled={busy || !note.trim()} onClick={() => onReject(note.trim())}>Reject · send back</button>
    </div>
  </article>;
}

function BuildCard({ title, subtitle, note, label, placeholder, initial, busy, onSubmit }: {
  title: string; subtitle: string; note: string; label: string; placeholder: string; initial: string; busy: boolean; onSubmit: (text: string) => void;
}) {
  const [text, setText] = useState(initial);
  return <article className="lifecycle-task">
    <div><b>{title}</b><small>{subtitle}</small></div>
    {note && <p className="form-error">Sent back: {note}</p>}
    <label>{label}</label>
    <textarea rows={6} maxLength={2000} value={text} onChange={(event) => setText(event.target.value)} placeholder={placeholder} />
    <button className="button button-primary" disabled={busy || !text.trim()} onClick={() => onSubmit(text.trim())}>Submit for review <span>→</span></button>
  </article>;
}

const Empty = ({ children }: { children: ReactNode }) => <p className="form-note lifecycle-empty">{children}</p>;

function RunConsole({ view }: { view: RunView }) {
  if (view.status === 'idle') return null;
  return <div className="agui-console" aria-live="polite">
    <div className="agui-head">
      <span className={`agui-dot ${view.status}`} />
      <b>{view.status === 'running' ? 'Live run' : view.status === 'done' ? 'Run finished' : 'Run failed'}</b>
      <span className="form-note">{view.stage || '—'}{view.attempts ? ` · attempt ${view.attempts}` : ''}{view.quality !== null ? ` · quality ${view.quality.toFixed(2)}` : ''}</span>
    </div>
    {view.steps.map((step) => <div className="agui-line" key={step.name}><i className={step.done ? 'done' : 'active'} /><span>{step.name}</span></div>)}
    {view.tools.map((tool) => <div className="agui-tool" key={tool.id}><b>{tool.name}</b><code>{tool.args}</code>{tool.result && <code>{tool.result}</code>}</div>)}
    {view.custom.map((item, index) => <div className="agui-tool sentinel" key={`${item.name}-${index}`}><b>{item.name.replace('.', ' · ')}</b><code>{JSON.stringify(item.value)}</code></div>)}
    {view.text && <div className="chat-message assistant"><span>ANSWER</span><p>{view.text}</p></div>}
    {view.error && <p className="form-error" role="alert">{view.error}</p>}
  </div>;
}

function EvaluateCard({ request, apiKey, openKeys, busy, onRun }: {
  request: MafRequest; apiKey: string; openKeys: () => void; busy: boolean;
  onRun: (prompt: string, props: EvalProps) => void;
}) {
  const [prompt, setPrompt] = useState('Help me with a refund for an order that arrived damaged.');
  const [policy, setPolicy] = useState<EvalProps['policy']>('linucb');
  const [threshold, setThreshold] = useState(0.65);
  const [budget, setBudget] = useState(0.01);
  return <>
    {!apiKey && <div className="callout"><div><p>Evaluation runs through MONA with your account key.</p></div><button className="button button-primary" onClick={openKeys}>API keys <span>→</span></button></div>}
    <label htmlFor={`ev-prompt-${request.id}`}>Evaluation prompt</label>
    <textarea id={`ev-prompt-${request.id}`} rows={2} maxLength={1000} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
    <div className="agui-params">
      <label>Policy<select value={policy} onChange={(event) => setPolicy(event.target.value as EvalProps['policy'])}><option value="linucb">LinUCB</option><option value="clm">CLM</option><option value="hybrid">Hybrid</option></select></label>
      <label>Quality threshold<input type="number" min={0} max={1} step={0.05} value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>
      <label>Budget (USD)<input type="number" min={0} step={0.005} value={budget} onChange={(event) => setBudget(Number(event.target.value))} /></label>
    </div>
    <button className="button button-primary" disabled={!apiKey || busy || !prompt.trim()} onClick={() => onRun(prompt.trim(), { policy, quality_threshold: threshold, budget })}>{busy ? 'Running…' : 'Run live evaluation'} <span>→</span></button>
  </>;
}

function TeamPanel({ members, busy, onSave, onRemove }: {
  members: Member[]; busy: boolean; onSave: (email: string, roles: Role[]) => void; onRemove: (email: string) => void;
}) {
  const [email, setEmail] = useState('');
  const [roles, setRoles] = useState<Role[]>([]);
  const toggle = (role: Role) => setRoles((current) => (current.includes(role) ? current.filter((item) => item !== role) : [...current, role]));
  return <>
    <article className="lifecycle-task">
      <div><b>Add or update a team member</b><small>Roles apply the next time they open the Studio.</small></div>
      <label htmlFor="team-email">Email</label>
      <input id="team-email" type="email" value={email} onChange={(event) => setEmail(event.target.value)} placeholder="name@company.com" />
      <div className="lifecycle-checks">{assignable.map((role) => <label key={role}><input type="checkbox" checked={roles.includes(role)} onChange={() => toggle(role)} /><span>{roleLabels[role]}</span></label>)}</div>
      <button className="button button-primary" disabled={busy || !email.includes('@') || !roles.length} onClick={() => { onSave(email.trim(), roles); setEmail(''); setRoles([]); }}>Save member <span>→</span></button>
    </article>
    {members.map((member) => <article className="lifecycle-task" key={member.email}>
      <div><b>{member.email}</b><small>{member.roles.map((role) => roleLabels[role as Role] ?? role).join(' · ')}</small></div>
      <div className="lifecycle-actions">
        <button className="button button-outline" onClick={() => { setEmail(member.email); setRoles(member.roles as Role[]); }}>Edit</button>
        <button className="button button-outline" disabled={busy} onClick={() => onRemove(member.email)}>Remove access</button>
      </div>
    </article>)}
  </>;
}

export default function Studio({ accessToken, apiKey, openKeys }: { accessToken: string; apiKey: string; openKeys: () => void }) {
  const api = useMemo(() => studioApi(accessToken, apiKey), [accessToken, apiKey]);
  const [me, setMe] = useState<Me | null>(null);
  const [requests, setRequests] = useState<MafRequest[]>([]);
  const [catalogue, setCatalogue] = useState<Catalogue>({ workflows: [], agents: [] });
  const [members, setMembers] = useState<Member[]>([]);
  const [live, setLive] = useState(false);
  const [tab, setTab] = useState<Role | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [detail, setDetail] = useState<MafRequest | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [loadError, setLoadError] = useState('');
  const [run, setRun] = useState<RunView>(idleRun);
  const abortRef = useRef<AbortController | null>(null);
  const [name, setName] = useState('');
  const [requirement, setRequirement] = useState('');
  const [agentList, setAgentList] = useState('');

  useEffect(() => {
    let cancelled = false;
    api.me().then((value) => { if (!cancelled) { setMe(value); setTab((current) => current ?? tabOrder.find((role) => value.roles.includes(role)) ?? null); } })
      .catch((caught: unknown) => { if (!cancelled) setLoadError(errorText(caught, 'Could not load your profile.')); });
    return () => { cancelled = true; };
  }, [api]);

  const hasAccess = Boolean(me && me.roles.length);
  useEffect(() => {
    if (!hasAccess) return undefined;
    return openStudioFeed(accessToken, (state) => { setRequests(state.requests); setCatalogue(state.catalogue); }, setLive);
  }, [accessToken, hasAccess]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const isAdmin = me?.roles.includes('admin') ?? false;
  const loadMembers = useCallback(() => { if (isAdmin) api.members().then(setMembers).catch((caught: unknown) => setError(errorText(caught, 'Could not load team.'))); }, [api, isAdmin]);
  useEffect(loadMembers, [loadMembers]);

  const selected = requests.find((item) => item.id === selectedId) ?? requests[0];
  const selectedStamp = selected?.updated_at;
  useEffect(() => {
    if (!selected) { setDetail(null); return; }
    let cancelled = false;
    api.request(selected.id).then((value) => { if (!cancelled) setDetail(value); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [api, selected?.id, selectedStamp]);

  async function act(task: () => Promise<MafRequest | Member[] | void>) {
    if (busy) return;
    setBusy(true);
    setError('');
    try {
      const result = await task();
      if (result && !Array.isArray(result)) setRequests((current) => current.map((item) => (item.id === result.id ? result : item)));
    } catch (caught) {
      setError(errorText(caught, 'The action could not be completed.'));
    } finally {
      setBusy(false);
    }
  }

  async function startRun(request: MafRequest, prompt: string, props: EvalProps) {
    if (busy) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError('');
    setRun({ ...idleRun, status: 'running', stage: 'starting' });
    try {
      await runEvaluation(accessToken, apiKey, request.id, prompt, props, (event) => setRun((current) => applyEvent(current, event)), controller.signal);
    } catch (caught) {
      setRun((current) => ({ ...current, status: 'error', error: errorText(caught, 'The evaluation could not complete.') }));
    } finally {
      setBusy(false);
    }
  }

  if (loadError) return <section className="console-view"><p className="form-error page-alert" role="alert">{loadError}</p></section>;
  if (!me) return <section className="console-view"><p className="form-note">Loading MAF Studio…</p></section>;
  if (!me.roles.length) {
    return <section className="console-view">
      <header className="view-heading"><div><span className="eyebrow">MAF STUDIO</span><h1>Request access.</h1><p>You are signed in as {me.email}, but no persona has been assigned yet. Ask a MAF admin to add you to a persona.</p></div></header>
    </section>;
  }

  const visibleTabs = tabOrder.filter((role) => me.roles.includes(role));
  const active: Role = tab && visibleTabs.includes(tab) ? tab : visibleTabs[0];
  const counts = (role: Role) => (role === 'admin' ? 0 : requests.filter((r) => r.waiting_on.includes(role)).length);
  const states = selected ? stageStates(selected) : null;

  let queue: ReactNode;
  if (active === 'chatbot_builder') {
    queue = <>
      <article className="lifecycle-task">
        <div><b>New requirement</b><small>Describe the chatbot. Discovery runs against both catalogues.</small></div>
        <label htmlFor="st-name">Workflow name</label>
        <input id="st-name" value={name} maxLength={80} onChange={(event) => setName(event.target.value)} placeholder="Refund assistant" />
        <label htmlFor="st-req">Requirement</label>
        <textarea id="st-req" rows={3} maxLength={1000} value={requirement} onChange={(event) => setRequirement(event.target.value)} placeholder="Answer refund questions and escalate disputes" />
        <label htmlFor="st-agents">Required agents (comma separated)</label>
        <input id="st-agents" value={agentList} onChange={(event) => setAgentList(event.target.value)} placeholder="Intent classifier, Policy checker" />
        <button className="button button-primary" disabled={busy || !name.trim() || !requirement.trim() || !splitList(agentList).length}
          onClick={() => void act(async () => { const created = await api.create({ name: name.trim(), requirement: requirement.trim(), agents: splitList(agentList) }); setSelectedId(created.id); setName(''); setRequirement(''); setAgentList(''); setRequests((current) => [created, ...current.filter((item) => item.id !== created.id)]); })}>Submit requirement <span>→</span></button>
      </article>
      {requests.map((request) => <article className="lifecycle-task" key={request.id}>
        <div><b>{request.name}</b><small>{phaseText[request.phase]} · {request.created_by}</small></div>
        {request.phase === 'ready_to_evaluate' && <EvaluateCard request={request} apiKey={apiKey} openKeys={openKeys} busy={busy} onRun={(prompt, props) => { setSelectedId(request.id); void startRun(request, prompt, props); }} />}
        {selected?.id === request.id && <RunConsole view={run} />}
        {request.evaluation && <div className="agent-run-report">
          <div className="product-tags"><span>{request.evaluation.status}</span><span>{request.evaluation.model}</span><span>{request.evaluation.elapsed_ms.toFixed(0)} ms</span><span>{currency(request.evaluation.cost)}</span></div>
          <div className="chat-message assistant"><span>{request.name.toUpperCase()}</span><p>{request.evaluation.answer}</p></div>
        </div>}
        {request.phase === 'ready_for_handover' && <button className="button button-primary" disabled={busy} onClick={() => void act(() => api.handover(request.id))}>Hand over to UX <span>→</span></button>}
      </article>)}
    </>;
  } else if (active === 'workflow_builder') {
    const tasks = requests.filter((r) => r.workflow.status === 'todo');
    queue = tasks.length ? tasks.map((r) => <BuildCard key={`${r.id}-${r.workflow.note}`} title={r.name} subtitle={r.requirement} note={r.workflow.note} label="Workflow steps" placeholder={'1. Classify intent\n2. Draft answer\n3. Escalate if needed'} initial={r.workflow.steps} busy={busy} onSubmit={(steps) => void act(() => api.submitWorkflow(r.id, steps))} />) : <Empty>No workflows waiting to be built.</Empty>;
  } else if (active === 'agent_builder') {
    const tasks = requests.flatMap((r) => r.agents.filter((a) => a.status === 'todo').map((a) => ({ r, a })));
    queue = tasks.length ? tasks.map(({ r, a }) => <BuildCard key={`${r.id}-${a.agent}-${a.note}`} title={a.agent} subtitle={`For “${r.name}”`} note={a.note} label="Agent instructions" placeholder="What should this agent do, and what must it never do?" initial={a.instructions} busy={busy} onSubmit={(text) => void act(() => api.submitAgent(r.id, a.agent, text))} />) : <Empty>No agents waiting to be built.</Empty>;
  } else if (active === 'ethics_champion') {
    const workflows = requests.filter((r) => r.workflow.status === 'in_review');
    const releases = requests.filter((r) => r.phase === 'awaiting_release');
    queue = workflows.length || releases.length ? <>
      {workflows.map((r) => <ReviewCard key={`wf-${r.id}`} title={`Workflow · ${r.name}`} subtitle={`Validation & approval · submitted by ${r.workflow.submitted_by ?? 'unknown'}`} body={r.workflow.steps} items={['No unsafe or discriminatory behaviour', 'Personal data handled per policy', 'Human escalation is defined']} approveLabel="Approve workflow" busy={busy} onApprove={() => void act(() => api.reviewWorkflow(r.id, true))} onReject={(note) => void act(() => api.reviewWorkflow(r.id, false, note))} />)}
      {releases.map((r) => <ReviewCard key={`rel-${r.id}`} title={`Release · ${r.name}`} subtitle={`Evaluation ${r.evaluation?.execution_id} · ${r.evaluation?.status} · run by ${r.evaluation?.run_by}`} body={r.evaluation?.answer} items={['Evaluation reviewed', 'Residual risks accepted', 'Cleared for release']} approveLabel="Release" busy={busy} onApprove={() => void act(() => api.reviewRelease(r.id, true))} onReject={(note) => void act(() => api.reviewRelease(r.id, false, note))} />)}
    </> : <Empty>No workflow approvals or releases pending.</Empty>;
  } else if (active === 'governance_manager') {
    const tasks = requests.flatMap((r) => r.agents.filter((a) => a.status === 'in_review').map((a) => ({ r, a })));
    queue = tasks.length ? tasks.map(({ r, a }) => <ReviewCard key={`${r.id}-${a.agent}`} title={a.agent} subtitle={`Agent validation & model certification · “${r.name}”`} body={a.instructions} items={['Instructions are in scope and unambiguous', 'Model certification and benchmarking passed', 'Approved for catalogue publication']} approveLabel="Approve & publish" busy={busy} onApprove={() => void act(() => api.reviewAgent(r.id, a.agent, true))} onReject={(note) => void act(() => api.reviewAgent(r.id, a.agent, false, note))} />) : <Empty>No agents awaiting certification.</Empty>;
  } else {
    queue = <TeamPanel members={members} busy={busy}
      onSave={(email, roles) => void act(async () => { setMembers(await api.setMember(email, roles)); })}
      onRemove={(email) => void act(async () => { setMembers(await api.removeMember(email)); })} />;
  }

  return <section className="console-view">
    <header className="view-heading">
      <div><span className="eyebrow">MAF STUDIO</span><h1>Build, govern, release.</h1><p>One live workspace for chatbots, workflows and agents. Every action is enforced and recorded by the server.</p></div>
      <span className={`studio-live ${live ? 'on' : ''}`}><i />{live ? 'Live' : 'Reconnecting…'}</span>
    </header>

    <div className="persona-tabs" role="tablist" aria-label="Persona">
      {visibleTabs.map((role) => <button key={role} role="tab" aria-selected={active === role} className={active === role ? 'selected' : ''} onClick={() => { setTab(role); setError(''); }}>
        <b>{roleLabels[role]}</b>{counts(role) > 0 && <span className="persona-count">{counts(role)}</span>}
      </button>)}
    </div>

    <div className="lifecycle-map" aria-label="High level design">
      <div className="lifecycle-map-bar">
        <label htmlFor="st-request">Tracking request</label>
        <select id="st-request" value={selected?.id || ''} onChange={(event) => setSelectedId(event.target.value)} disabled={!requests.length}>
          {!requests.length && <option value="">No requests yet</option>}
          {requests.map((request) => <option key={request.id} value={request.id}>{request.name} · {phaseText[request.phase]}</option>)}
        </select>
      </div>
      {lanes.map((lane) => <div className="lifecycle-lane" key={lane.title}>
        <div className="lifecycle-lane-head"><h3>{lane.title}</h3></div>
        <ol>{lane.nodes.map((node) => <li key={node.stage} className={`lifecycle-node ${node.kind} ${states ? states[node.stage] : 'pending'}`}><i aria-hidden="true" /><b>{node.label}</b><small>{node.role}</small></li>)}</ol>
      </div>)}
    </div>

    {error && <p className="form-error page-alert" role="alert">{error}</p>}
    <div className="lifecycle-layout">
      <div className="panel lifecycle-panel">
        <span className="eyebrow">YOUR QUEUE</span>
        <h2>{roleLabels[active]}</h2>
        <p className="form-note">{personaDuty[active]}</p>
        {queue}
      </div>
      <aside className="lifecycle-side">
        <div className="panel">
          <div className="panel-title"><div><span className="eyebrow">CATALOGUES</span><h2>Registry</h2></div></div>
          <span className="eyebrow">Workflows · {catalogue.workflows.length}</span>
          <ul className="lifecycle-list">{catalogue.workflows.map((item) => <li key={item.name}><b>{item.name} <small>v{item.version}</small></b><span>{item.agents.length} agents</span></li>)}{!catalogue.workflows.length && <li><span>None published yet</span></li>}</ul>
          <span className="eyebrow">Agents · {catalogue.agents.length}</span>
          <ul className="lifecycle-list">{catalogue.agents.map((item) => <li key={item.name}><b>{item.name} <small>v{item.version}</small></b><span>{item.approved_by}</span></li>)}{!catalogue.agents.length && <li><span>None published yet</span></li>}</ul>
        </div>
        <div className="panel">
          <span className="eyebrow">AUDIT TRAIL{selected ? ` · ${selected.name}` : ''}</span>
          <div className="agent-trace">{detail?.id === selected?.id && detail?.events?.map((entry, index) => <article className="agent-trace-step sentinel_evaluation" key={`${entry.at}-${index}`}><i aria-hidden="true" /><div><b>{entry.actor}</b><p>{entry.action} · {entry.detail}</p></div><span className="form-note">{entry.at.replace('T', ' ').slice(0, 16)}</span></article>)}{!detail?.events?.length && <p className="form-note">Actions appear here as the request moves between personas.</p>}</div>
        </div>
      </aside>
    </div>
  </section>;
}
