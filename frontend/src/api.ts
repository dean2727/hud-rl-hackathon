// REST + SSE client for the backend. Dev proxy in vite.config.ts forwards /api
// to http://localhost:8000, so these are same-origin from the browser's view.

export type StageStatus = 'started' | 'completed' | 'failed'

export interface TimelineEvent {
  // The SSE `event:` name. One of:
  //   stage | gizmo | rollout | train_round | train_further_done |
  //   train_further_error | error | done
  event: string
  data: Record<string, unknown>
  ts: number
}

export interface CreateRunResult {
  run_id: string
}

export async function createRun(
  images: File[],
  activities: string[],
): Promise<CreateRunResult> {
  const form = new FormData()
  for (const img of images) form.append('images', img)
  for (const a of activities) form.append('activities', a)

  const resp = await fetch('/api/runs', { method: 'POST', body: form })
  if (!resp.ok) {
    const detail = await resp.text()
    throw new Error(`createRun failed (${resp.status}): ${detail}`)
  }
  return resp.json()
}

export async function trainFurther(
  runId: string,
  activityIndex: number,
): Promise<void> {
  const resp = await fetch(
    `/api/runs/${runId}/train-further?activity_index=${activityIndex}`,
    { method: 'POST' },
  )
  if (!resp.ok) {
    throw new Error(`trainFurther failed (${resp.status}): ${await resp.text()}`)
  }
}

// Opens the SSE stream for a run. Returns the EventSource so the caller can close
// it. `onEvent` fires for every named event the backend emits.
export function openEventStream(
  runId: string,
  onEvent: (e: TimelineEvent) => void,
): EventSource {
  const es = new EventSource(`/api/runs/${runId}/events`)
  const NAMED = [
    'stage',
    'gizmo',
    'rollout',
    'train_round',
    'train_further_done',
    'train_further_error',
    'error',
    'done',
  ]
  for (const name of NAMED) {
    es.addEventListener(name, (ev) => {
      let data: Record<string, unknown> = {}
      try {
        data = JSON.parse((ev as MessageEvent).data)
      } catch {
        data = { raw: (ev as MessageEvent).data }
      }
      onEvent({ event: name, data, ts: Date.now() })
      if (name === 'done' || name === 'error') es.close()
    })
  }
  return es
}
