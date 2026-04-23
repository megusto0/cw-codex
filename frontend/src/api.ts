import type { ModelKey } from './types';

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBase = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, '') ?? '';
  const response = await fetch(`${apiBase}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Запрос завершился ошибкой ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  return fetchJson<T>(path, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export function withModelParam(path: string, model: ModelKey) {
  const [pathname, rawQuery = ''] = path.split('?');
  const params = new URLSearchParams(rawQuery);
  params.set('model', model);
  const query = params.toString();
  return query ? `${pathname}?${query}` : pathname;
}

export async function fetchModelJson<T>(path: string, model: ModelKey): Promise<T> {
  return fetchJson<T>(withModelParam(path, model));
}

export async function retrieveWindow<T>(model: ModelKey, windowId: string, k = 5): Promise<T> {
  return postJson<T>('/api/retrieve', {
    model,
    window_id: windowId,
    k,
  });
}
