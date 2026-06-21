// REST + SSE client for the backend. Dev proxy in vite.config.ts forwards /api
// to http://localhost:8000, so these are same-origin from the browser's view.

export type StageStatus = 'started' | 'completed' | 'failed'

export interface TimelineEvent {
  // The SSE `event:` name. One of:
  //   stage | awaiting_confirmation | gizmo | rollout | train_round |
  //   train_further_done | train_further_error | error | done |
//   train_stage | eval_rollout | eval_summary | curate |
//   train_modal_done | train_modal_error |
//   video_generating | video_ready | video_error
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

// Confirm (and optionally edit) the vision model's scene description, which
// resumes the pipeline into Gizmo scene generation. The run must be in its
// awaiting_confirmation stage.
export async function confirmScene(
  runId: string,
  scenePrompt: string,
): Promise<void> {
  const resp = await fetch(`/api/runs/${runId}/confirm-scene`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scene_prompt: scenePrompt }),
  })
  if (!resp.ok) {
    throw new Error(`confirmScene failed (${resp.status}): ${await resp.text()}`)
  }
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

// Kick off the pi0.5 BC loop (eval -> curate -> finetune) on Modal A100s with live
// reward streaming, then build a rollout video from the fine-tuned checkpoint.
export async function trainModal(
  runId: string,
  activityIndex: number,
  dryRun?: boolean,
): Promise<void> {
  const params = new URLSearchParams({ activity_index: String(activityIndex) })
  if (dryRun !== undefined) params.set('dry_run', String(dryRun))
  const resp = await fetch(`/api/runs/${runId}/train-modal?${params}`, { method: 'POST' })
  if (!resp.ok) {
    throw new Error(`trainModal failed (${resp.status}): ${await resp.text()}`)
  }
}

export async function rolloutVideo(
  runId: string,
  activityIndex: number,
): Promise<void> {
  const resp = await fetch(
    `/api/runs/${runId}/rollout-video?activity_index=${activityIndex}`,
    { method: 'POST' },
  )
  if (!resp.ok) {
    throw new Error(`rolloutVideo failed (${resp.status}): ${await resp.text()}`)
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
    'awaiting_confirmation',
    'gizmo',
    'rollout',
    'train_round',
    'train_further_done',
    'train_further_error',
    'train_stage',
    'eval_rollout',
    'eval_summary',
    'curate',
    'train_modal_done',
    'train_modal_error',
    'video_generating',
    'video_ready',
    'video_error',
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
      if (name === 'error') es.close()
    })
  }
  return es
}
