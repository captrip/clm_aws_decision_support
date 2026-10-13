export type Policy = 'linucb' | 'clm' | 'hybrid';

export interface Model {
  name: string;
  provider: 'bedrock' | 'azure_openai';
  input_cost_per_million: number;
  output_cost_per_million: number;
}

export interface Deployment {
  logical_name: string;
  provider: 'bedrock' | 'azure_openai';
  model_id: string | null;
  deployment: string | null;
  input_cost_per_million: number;
  output_cost_per_million: number;
  created_at: string;
}

export interface ApiKey {
  id: string;
  label: string | null;
  key_prefix: string;
  revoked: boolean;
  created_at: string;
}

export interface AskResult {
  decision_id: string;
  question: string;
  policy: Policy;
  model: string;
  provider: string;
  answer: string;
  cost: number;
  usage: { input_tokens: number; output_tokens: number };
  rationale: string;
  candidates: Record<string, Record<string, number | string>>;
  retried_models: string[];
  feedback_eligible: boolean;
}

export interface RunAttempt {
  step: number;
  model: string;
  provider: string;
  policy: string;
  cost: number;
  usage: { input_tokens: number; output_tokens: number };
  reward: number | null;
  error: string | null;
  rationale: string;
  candidates: Record<string, Record<string, number | string>>;
}

export interface RunResult {
  policy: Policy;
  question: string;
  expected_answer: string;
  budget: number | null;
  success: boolean;
  total_cost: number;
  attempts: RunAttempt[];
}

export interface MAFDemoMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface MAFDemoTraceStep {
  stage: string;
  status?: string;
  agent?: string;
  attempt?: number;
  answer?: string;
  decision_id?: string | null;
  model?: string;
  provider?: string;
  policy?: string;
  cost?: number;
  elapsed_ms?: number;
  score?: number;
  threshold?: number;
  characters?: number;
  correction?: string | null;
  instruction?: string;
}

export interface MAFDemoResult {
  answer: string;
  agent: string;
  decision_id: string | null;
  model: string;
  provider: string;
  policy: Policy;
  cost: number;
  usage: { input_tokens: number; output_tokens: number };
  rationale: string;
  elapsed_ms: number;
  execution_id: string;
  sentinel: {
    status: string;
    quality: { score: number; threshold: number; characters: number; step: number }[];
    trace: MAFDemoTraceStep[];
  };
}

export const baseUrl = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8800').replace(/\/+$/, '');

export async function downloadProductPackage(
  product: 'sentinel' | 'maf',
  apiKey: string,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(`${baseUrl}/api/products/${product}/download`, {
      headers: { 'X-API-Key': apiKey },
    });
  } catch (error) {
    throw new Error(
      `Could not reach the API at ${baseUrl}. Check that the backend is running.`,
      { cause: error },
    );
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const detail =
      payload && typeof payload === 'object' && 'detail' in payload
        ? String(payload.detail)
        : response.statusText;
    throw new Error(`${response.status}: ${detail || 'Package download failed.'}`);
  }

  const packageBlob = await response.blob();
  const objectUrl = URL.createObjectURL(packageBlob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = `${product}-package.zip`;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

export async function apiRequest<T>(
  path: string,
  options: RequestInit = {},
  apiKey?: string,
  accessToken?: string,
): Promise<T> {
  const headers = new Headers(options.headers);
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  if (apiKey) headers.set('X-API-Key', apiKey);
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);

  let response: Response;
  try {
    response = await fetch(`${baseUrl}${path}`, { ...options, headers });
  } catch (error) {
    throw new Error(
      `Could not reach the API at ${baseUrl}. Check that the backend is running and VITE_API_BASE_URL is correct.`,
      { cause: error },
    );
  }

  if (!response.ok) {
    const payload: unknown = await response.json().catch(() => null);
    const detail =
      payload && typeof payload === 'object' && 'detail' in payload
        ? String(payload.detail)
        : response.statusText;
    throw new Error(`${response.status}: ${detail || 'The request failed.'}`);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function currency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: value < 0.01 ? 6 : 4,
  }).format(value);
}
