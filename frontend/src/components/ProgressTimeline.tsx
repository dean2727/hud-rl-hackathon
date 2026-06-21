import type { ReactNode } from 'react'
import type { TimelineEvent } from '../api'

const STAGE_LABELS: Record<string, string> = {
  describing_photos: 'Describing photos (DeepMind vision)',
  generating_scene: 'Generating 3D scene (Gizmo)',
  composing_scene: 'Composing scene + robot gripper',
  mapping_activities: 'Mapping activities to tasks',
  running_rollouts: 'Running RL rollouts',
}

// Gizmo emits hundreds of fine-grained events per scene; collapse them to these
// ordered meta-stages and show a single advancing bar instead of a raw-JSON flood.
const GIZMO_STAGES: { label: string; test: RegExp }[] = [
  { label: 'Queued', test: /job_queued|modal_dispatch|job_claimed|job_started/ },
  { label: 'Planning the scene', test: /scene_director|asset_inventory|floorplan|task_plan|placement_plan|structure_path/ },
  { label: 'Generating assets', test: /asset_/ },
  { label: 'Scripting the build', test: /scripting_/ },
  { label: 'Assembling structure', test: /structure_|placement_|lighting|acceptance/ },
  { label: 'Texturing', test: /texture/ },
  { label: 'Scene ready', test: /job_succeeded/ },
]

function gizmoStageIndex(type: string): number {
  // latest matching stage wins, so the bar only ever moves forward
  for (let i = GIZMO_STAGES.length - 1; i >= 0; i--) {
    if (GIZMO_STAGES[i].test.test(type)) return i
  }
  return 0
}

function gizmoProgress(gizmoEvents: TimelineEvent[]): { label: string; pct: number; done: boolean } {
  let maxIdx = 0
  for (const e of gizmoEvents) {
    const t = String((e.data as { type?: unknown })?.type ?? '')
    if (t === 'ping') continue
    maxIdx = Math.max(maxIdx, gizmoStageIndex(t))
  }
  const last = GIZMO_STAGES.length - 1
  const done = maxIdx >= last
  return { label: GIZMO_STAGES[maxIdx].label, pct: done ? 100 : Math.max(8, Math.round((maxIdx / last) * 100)), done }
}

interface Rendered {
  dotClass: string
  title: string
  detail?: string
}

function renderEvent(e: TimelineEvent): Rendered | null {
  const d = e.data
  switch (e.event) {
    case 'stage': {
      const stage = String(d.stage)
      const status = String(d.status)
      const label = STAGE_LABELS[stage] ?? stage
      let detail: string | undefined
      const det = d.detail as Record<string, unknown> | undefined
      if (det) {
        if (det.scene_prompt) detail = `"${det.scene_prompt}"`
        else if (det.scene_id) detail = `scene ${det.scene_id}`
        else if (Array.isArray(det.objects))
          detail = `objects: ${(det.objects as string[]).join(', ') || '(none)'}`
        else if (Array.isArray(det.mappings))
          detail = (det.mappings as { activity: string; task: string }[])
            .map((m) => `${m.activity} → ${m.task}`)
            .join('  •  ')
      }
      return { dotClass: status, title: `${label} — ${status}`, detail }
    }
    case 'awaiting_confirmation':
      return {
        dotClass: 'started',
        title: 'Waiting for your review of the scene description',
        detail: d.scene_prompt ? `"${d.scene_prompt}"` : undefined,
      }
    case 'rollout': {
      const status = String(d.status)
      const activity = String(d.activity)
      const task = String(d.task)
      let detail = `task: ${task}`
      if (status === 'completed' && d.reward != null)
        detail += ` — reward ${Number(d.reward).toFixed(3)}${d.success ? ' (SUCCESS)' : ''}`
      if (status === 'failed' && d.content) detail += ` — ${String(d.content)}`
      return { dotClass: status, title: `Rollout: ${activity}`, detail }
    }
    case 'train_round':
      return {
        dotClass: 'completed',
        title: `Train-further round ${Number(d.round) + 1} (activity ${Number(d.activity_index) + 1})`,
        detail: `best ${Number(d.best_reward).toFixed(3)} · mean ${Number(d.mean_reward).toFixed(3)}`,
      }
    case 'train_further_done':
      return { dotClass: 'completed', title: `Train-further complete (activity ${Number(d.activity_index) + 1})` }
    case 'train_further_error':
      return { dotClass: 'error', title: 'Train-further error', detail: String(d.message) }
    case 'error':
      return { dotClass: 'error', title: 'Error', detail: `${d.message}${d.stage ? ` (at ${d.stage})` : ''}` }
    case 'done':
      return { dotClass: 'completed', title: 'All done ✓' }
    default:
      return null
  }
}

export function ProgressTimeline({ events, loading }: { events: TimelineEvent[]; loading?: boolean }) {
  const gz = events.some((e) => e.event === 'gizmo')
    ? gizmoProgress(events.filter((e) => e.event === 'gizmo'))
    : null

  // The entire Gizmo run collapses into ONE progress-bar row, placed where its
  // first event landed; everything else renders as a normal timeline item.
  const rows: ReactNode[] = []
  let gizmoShown = false
  events.forEach((e, i) => {
    if (e.event === 'gizmo') {
      if (!gizmoShown && gz) {
        gizmoShown = true
        rows.push(
          <div className="tl-item" key={`gz-${i}`}>
            <span className={`tl-dot ${gz.done ? 'completed' : 'gizmo'}`} />
            <div className="tl-body" style={{ flex: 1 }}>
              <div className="tl-title">Generating 3D scene — {gz.label}</div>
              <div className="gizmo-bar">
                <div style={{ width: `${gz.pct}%` }} />
              </div>
            </div>
          </div>,
        )
      }
      return
    }
    const r = renderEvent(e)
    if (!r) return
    rows.push(
      <div className="tl-item" key={i}>
        <span className={`tl-dot ${r.dotClass}`} />
        <div className="tl-body">
          <div className="tl-title">{r.title}</div>
          {r.detail && <div className="tl-detail">{r.detail}</div>}
        </div>
      </div>,
    )
  })

  return (
    <div className={`panel ${loading ? 'loading' : ''}`}>
      <h2>Progress</h2>
      {events.length === 0 ? (
        <p className="empty">
          Upload photos, list activities, and hit Start Learning to watch the pipeline run.
        </p>
      ) : (
        <div className="timeline">{rows}</div>
      )}
    </div>
  )
}
