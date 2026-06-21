import type { TimelineEvent } from '../api'

const STAGE_LABELS: Record<string, string> = {
  describing_photos: 'Describing photos (Gemma vision)',
  generating_scene: 'Generating 3D scene (Gizmo)',
  composing_scene: 'Composing scene + robot gripper',
  mapping_activities: 'Mapping activities to tasks',
  running_rollouts: 'Running RL rollouts',
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
    case 'gizmo': {
      const type = String(d.type)
      const raw =
        typeof d.data === 'string'
          ? d.data
          : d.data
            ? JSON.stringify(d.data)
            : ''
      return {
        dotClass: 'gizmo',
        title: `Gizmo: ${type}`,
        detail: raw ? raw.slice(0, 160) : undefined,
      }
    }
    case 'rollout': {
      const status = String(d.status)
      const activity = String(d.activity)
      const task = String(d.task)
      let detail = `task: ${task}`
      if (status === 'completed' && d.reward != null)
        detail += ` — reward ${Number(d.reward).toFixed(3)}${
          d.success ? ' (SUCCESS)' : ''
        }`
      if (status === 'failed' && d.content) detail += ` — ${String(d.content)}`
      return { dotClass: status, title: `Rollout: ${activity}`, detail }
    }
    case 'train_round': {
      return {
        dotClass: 'completed',
        title: `Train-further round ${Number(d.round) + 1} (activity ${
          Number(d.activity_index) + 1
        })`,
        detail: `best ${Number(d.best_reward).toFixed(3)} · mean ${Number(
          d.mean_reward,
        ).toFixed(3)}`,
      }
    }
    case 'train_further_done':
      return {
        dotClass: 'completed',
        title: `Train-further complete (activity ${
          Number(d.activity_index) + 1
        })`,
      }
    case 'train_further_error':
      return {
        dotClass: 'error',
        title: 'Train-further error',
        detail: String(d.message),
      }
    case 'error':
      return {
        dotClass: 'error',
        title: 'Error',
        detail: `${d.message}${d.stage ? ` (at ${d.stage})` : ''}`,
      }
    case 'done':
      return { dotClass: 'completed', title: 'All done ✓' }
    default:
      return null
  }
}

export function ProgressTimeline({ events }: { events: TimelineEvent[] }) {
  return (
    <div className="panel">
      <h2>Progress</h2>
      {events.length === 0 ? (
        <p className="empty">
          Upload photos, list activities, and hit Start Learning to watch the
          pipeline run.
        </p>
      ) : (
        <div className="timeline">
          {events.map((e, i) => {
            const r = renderEvent(e)
            if (!r) return null
            return (
              <div className="tl-item" key={i}>
                <span className={`tl-dot ${r.dotClass}`} />
                <div className="tl-body">
                  <div className="tl-title">{r.title}</div>
                  {r.detail && <div className="tl-detail">{r.detail}</div>}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
