import { apiRequest, baseUrl } from '../api';

export type Role = 'admin' | 'chatbot_builder' | 'workflow_builder' | 'agent_builder' | 'ethics_champion' | 'governance_manager';
export type Status = 'todo' | 'in_review' | 'published';
export type Phase = 'building' | 'ready_to_evaluate' | 'awaiting_release' | 'ready_for_handover' | 'complete';

export interface Me { email: string; roles: Role[]; separation_of_duties: boolean }
export interface Member { email: string; roles: Role[] }
export interface WorkflowWork { status: Status; from_catalogue: boolean; steps: string; note: string; submitted_by: string | null }
export interface AgentWork { agent: string; status: Status; from_catalogue: boolean; instructions: string; note: string; submitted_by: string | null }
export interface Evaluation {
  execution_id: string; status: string; model: string; provider: string | null; answer: string; prompt: string;
  cost: number; elapsed_ms: number; run_by: string; run_at: string;
}
export interface AuditEvent { at: string; actor: string; role: string | null; action: string; detail: string }
export interface MafRequest {
  id: string; name: string; requirement: string; created_by: string; created_at: string; updated_at: string;
  workflow: WorkflowWork; agents: AgentWork[]; evaluation: Evaluation | null; released: boolean; handed_over: boolean;
  phase: Phase; waiting_on: Role[]; events?: AuditEvent[];
}
export interface Catalogue {
  workflows: { name: string; steps: string; agents: string[]; version: number; approved_by: string }[];
  agents: { name: string; instructions: string; version: number; approved_by: string }[];
}
export interface StudioState { requests: MafRequest[]; catalogue: Catalogue }

export interface AguiEvent { type: string; [key: string]: unknown }

export const roleLabels: Record<Role, string> = {
  admin: 'Admin',
  chatbot_builder: 'Chatbot Builder',
  workflow_builder: 'Workflow Builder',
  agent_builder: 'Agent Builder',
  ethics_champion: 'AI Ethics & Compliance Champion',
  governance_manager: 'AI Governance Manager',
};

export function studioApi(accessToken: string, apiKey: string) {
  const call = <T,>(path: string, method = 'GET', body?: unknown) =>
    apiRequest<T>(`/api/maf${path}`, { method, body: body === undefined ? undefined : JSON.stringify(body) }, apiKey || undefined, accessToken);
  const enc = encodeURIComponent;
  return {
    me: () => call<Me>('/me'),
    members: () => call<Member[]>('/members'),
    setMember: (email: string, roles: Role[]) => call<Member[]>('/members', 'PUT', { email, roles }),
    removeMember: (email: string) => call<Member[]>(`/members/${enc(email)}`, 'DELETE'),
    request: (id: string) => call<MafRequest>(`/requests/${enc(id)}`),
    create: (body: { name: string; requirement: string; agents: string[] }) => call<MafRequest>('/requests', 'POST', body),
    submitWorkflow: (id: string, steps: string) => call<MafRequest>(`/requests/${enc(id)}/workflow/submit`, 'POST', { steps }),
    reviewWorkflow: (id: string, approve: boolean, note?: string) => call<MafRequest>(`/requests/${enc(id)}/workflow/review`, 'POST', { approve, note }),
    submitAgent: (id: string, agent: string, instructions: string) => call<MafRequest>(`/requests/${enc(id)}/agents/${enc(agent)}/submit`, 'POST', { instructions }),
    reviewAgent: (id: string, agent: string, approve: boolean, note?: string) => call<MafRequest>(`/requests/${enc(id)}/agents/${enc(agent)}/review`, 'POST', { approve, note }),
    reviewRelease: (id: string, approve: boolean, note?: string) => call<MafRequest>(`/requests/${enc(id)}/release/review`, 'POST', { approve, note }),
    handover: (id: string) => call<MafRequest>(`/requests/${enc(id)}/handover`, 'POST'),
  };
}

async function failure(response: Response): Promise<Error> {
  const payload: unknown = await response.json().catch(() => null);
  const detail = payload && typeof payload === 'object' && 'detail' in payload ? String(payload.detail) : response.statusText;
  return new Error(`${response.status}: ${detail || 'The request failed.'}`);
}

/** Read an AG-UI server-sent-event stream. fetch is used because EventSource cannot send auth headers. */
export async function readAguiStream(
  path: string,
  init: RequestInit,
  headers: Record<string, string>,
  onEvent: (event: AguiEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, { ...init, headers: { Accept: 'text/event-stream', ...headers }, signal });
  } catch (error) {
    if (signal.aborted) return;
    throw new Error(`Could not reach the API at ${baseUrl}.`, { cause: error });
  }
  if (!response.ok) throw await failure(response);
  if (!response.body) throw new Error('The server did not return a stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
    let end = buffer.indexOf('\n\n');
    while (end !== -1) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = frame.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
      if (data) {
        try { onEvent(JSON.parse(data) as AguiEvent); } catch { /* ignore a malformed frame */ }
      }
      end = buffer.indexOf('\n\n');
    }
  }
}

export function openStudioFeed(accessToken: string, onState: (state: StudioState) => void, onStatus: (live: boolean) => void): () => void {
  const controller = new AbortController();
  void (async () => {
    let delay = 1000;
    while (!controller.signal.aborted) {
      try {
        await readAguiStream('/api/maf/stream', {}, { Authorization: `Bearer ${accessToken}` }, (event) => {
          if (event.type === 'STATE_SNAPSHOT') {
            onStatus(true);
            delay = 1000;
            onState(event.snapshot as StudioState);
          }
        }, controller.signal);
      } catch { /* reconnect below */ }
      if (controller.signal.aborted) return;
      onStatus(false);
      await new Promise((resolve) => window.setTimeout(resolve, delay));
      delay = Math.min(delay * 2, 15000);
    }
  })();
  return () => controller.abort();
}

export interface EvalProps { policy: 'linucb' | 'clm' | 'hybrid'; quality_threshold: number; budget: number }

export function runEvaluation(
  accessToken: string, apiKey: string, requestId: string, prompt: string, props: EvalProps,
  onEvent: (event: AguiEvent) => void, signal: AbortSignal,
): Promise<void> {
  const body = {
    threadId: requestId,
    runId: `run_${Date.now().toString(36)}`,
    state: {},
    messages: [{ id: `m_${Date.now().toString(36)}`, role: 'user', content: prompt }],
    tools: [],
    context: [],
    forwardedProps: props,
  };
  return readAguiStream(
    `/api/maf/requests/${encodeURIComponent(requestId)}/agui`,
    { method: 'POST', body: JSON.stringify(body) },
    { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}`, 'X-API-Key': apiKey },
    onEvent, signal,
  );
}
