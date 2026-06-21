import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  createRun,
  openEventStream,
  trainFurther,
  type TimelineEvent,
} from './api'
import { PhotoUpload } from './components/PhotoUpload'
import { ActivityList } from './components/ActivityList'
import { ProgressTimeline } from './components/ProgressTimeline'
import {
  ResultsPanel,
  type ActivityResult,
  type TrainRound,
} from './components/ResultsPanel'

type Phase = 'idle' | 'running' | 'done' | 'failed'

function App() {
  const [photos, setPhotos] = useState<File[]>([])
  const [activities, setActivities] = useState<string[]>([''])
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [trainingIndices, setTrainingIndices] = useState<Set<number>>(new Set())
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  const cleanActivities = activities.map((a) => a.trim()).filter(Boolean)
  const canStart =
    phase !== 'running' &&
    photos.length >= 2 &&
    photos.length <= 3 &&
    cleanActivities.length > 0

  async function handleStart() {
    setError(null)
    setEvents([])
    setTrainingIndices(new Set())
    setPhase('running')
    setRunId(null)
    try {
      const { run_id } = await createRun(photos, cleanActivities)
      setRunId(run_id)
      esRef.current = openEventStream(run_id, (e) => {
        setEvents((prev) => [...prev, e])
        if (e.event === 'done') setPhase('done')
        if (e.event === 'error') {
          setPhase('failed')
          setError(String(e.data.message ?? 'pipeline failed'))
        }
        if (e.event === 'train_further_done' || e.event === 'train_further_error') {
          const idx = Number(e.data.activity_index)
          setTrainingIndices((prev) => {
            const next = new Set(prev)
            next.delete(idx)
            return next
          })
        }
      })
    } catch (err) {
      setPhase('failed')
      setError(err instanceof Error ? err.message : String(err))
    }
  }

  async function handleTrainFurther(activityIndex: number) {
    if (!runId) return
    setTrainingIndices((prev) => new Set(prev).add(activityIndex))
    try {
      await trainFurther(runId, activityIndex)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setTrainingIndices((prev) => {
        const next = new Set(prev)
        next.delete(activityIndex)
        return next
      })
    }
  }

  const { results, trainRounds } = useMemo(
    () => deriveResults(events),
    [events],
  )

  const locked = phase === 'running'

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Robot Skill Learner</h1>
        <p>
          Show the robot where it lives, tell it what to do, and watch it learn
          each skill in simulation.
        </p>
      </header>

      <div className="layout">
        <div>
          <PhotoUpload photos={photos} onChange={setPhotos} disabled={locked} />
          <div style={{ height: 24 }} />
          <ActivityList
            activities={activities}
            onChange={setActivities}
            disabled={locked}
          />
          <div className="start-bar">
            <button
              className="primary"
              disabled={!canStart}
              onClick={handleStart}
            >
              {phase === 'running' ? 'Learning…' : 'Start Learning'}
            </button>
            {phase !== 'running' && !canStart && (
              <span className="hint">Add 2-3 photos and at least one activity.</span>
            )}
          </div>
          {error && <div className="error-banner">{error}</div>}
        </div>

        <div>
          <ProgressTimeline events={events} />
          <ResultsPanel
            results={results}
            trainRounds={trainRounds}
            trainingIndices={trainingIndices}
            onTrainFurther={handleTrainFurther}
          />
        </div>
      </div>
    </div>
  )
}

// Build the per-activity results + train rounds from the raw event stream. The
// backend's rollout/train_round events carry everything needed, so the UI stays a
// pure projection of the SSE log (no separate fetch).
function deriveResults(events: TimelineEvent[]): {
  results: ActivityResult[]
  trainRounds: Record<number, TrainRound[]>
} {
  const byIndex = new Map<number, ActivityResult>()
  const trainRounds: Record<number, TrainRound[]> = {}

  for (const e of events) {
    if (e.event === 'rollout') {
      const d = e.data
      const idx = Number(d.activity_index)
      byIndex.set(idx, {
        activity_index: idx,
        activity: String(d.activity),
        task: String(d.task),
        target: String(d.target),
        status: String(d.status),
        reward: d.reward != null ? Number(d.reward) : null,
        success: d.success != null ? Boolean(d.success) : null,
        content: d.content != null ? String(d.content) : null,
        can_train_further: Boolean(d.can_train_further),
      })
    } else if (e.event === 'train_round') {
      const idx = Number(e.data.activity_index)
      ;(trainRounds[idx] ??= []).push({
        round: Number(e.data.round),
        best_reward: Number(e.data.best_reward),
        mean_reward: Number(e.data.mean_reward),
      })
    }
  }

  const results = [...byIndex.values()].sort(
    (a, b) => a.activity_index - b.activity_index,
  )
  return { results, trainRounds }
}

export default App
