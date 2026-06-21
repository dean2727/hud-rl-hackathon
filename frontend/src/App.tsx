import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  confirmScene,
  createRun,
  openEventStream,
  trainFurther,
  trainModal,
  type TimelineEvent,
} from './api'
import { PhotoUpload } from './components/PhotoUpload'
import { ActivityList } from './components/ActivityList'
import { SceneReview } from './components/SceneReview'
import { ProgressTimeline } from './components/ProgressTimeline'
import {
  ResultsPanel,
  type ActivityResult,
  type ModalTraining,
  type TrainRound,
} from './components/ResultsPanel'

// 'awaiting' = paused at the scene-description review checkpoint.
type Phase = 'idle' | 'running' | 'awaiting' | 'done' | 'failed'

function App() {
  const [photos, setPhotos] = useState<File[]>([])
  const [activities, setActivities] = useState<string[]>([''])
  const [phase, setPhase] = useState<Phase>('idle')
  const [error, setError] = useState<string | null>(null)
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [runId, setRunId] = useState<string | null>(null)
  const [sceneDraft, setSceneDraft] = useState('')
  const [detectedObjects, setDetectedObjects] = useState<string[]>([])
  const [trainingIndices, setTrainingIndices] = useState<Set<number>>(new Set())
  const [modalIndices, setModalIndices] = useState<Set<number>>(new Set())
  const esRef = useRef<EventSource | null>(null)

  useEffect(() => () => esRef.current?.close(), [])

  const cleanActivities = activities.map((a) => a.trim()).filter(Boolean)
  const busy = phase === 'running' || phase === 'awaiting'
  // Once a run starts the robot slides off and the progress panels slide in from the top.
  const started = phase !== 'idle'
  const canStart =
    !busy &&
    photos.length >= 1 &&
    photos.length <= 3 &&
    cleanActivities.length > 0

  async function handleStart() {
    setError(null)
    setEvents([])
    setSceneDraft('')
    setDetectedObjects([])
    setTrainingIndices(new Set())
    setModalIndices(new Set())
    setPhase('running')
    setRunId(null)
    try {
      const { run_id } = await createRun(photos, cleanActivities)
      setRunId(run_id)
      esRef.current = openEventStream(run_id, (e) => {
        setEvents((prev) => [...prev, e])
        if (e.event === 'awaiting_confirmation') {
          setSceneDraft(String(e.data.scene_prompt ?? ''))
          setDetectedObjects(
            Array.isArray(e.data.objects) ? (e.data.objects as string[]) : [],
          )
          setPhase('awaiting')
        }
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
        if (e.event === 'train_modal_done' || e.event === 'train_modal_error') {
          const idx = Number(e.data.activity_index)
          setModalIndices((prev) => {
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

  async function handleConfirmScene() {
    if (!runId) return
    setPhase('running')
    try {
      await confirmScene(runId, sceneDraft)
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

  async function handleTrainModal(activityIndex: number) {
    if (!runId) return
    setModalIndices((prev) => new Set(prev).add(activityIndex))
    try {
      await trainModal(runId, activityIndex) // dry_run by default: streams reward, no GPU spend
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setModalIndices((prev) => {
        const next = new Set(prev)
        next.delete(activityIndex)
        return next
      })
    }
  }

  const { results, trainRounds, modalTraining } = useMemo(
    () => deriveResults(events),
    [events],
  )

  const locked = busy

  return (
    <div className="app">
      <header className="app-header">
        <div className="badge-row">
          <span className="pill">Gizmo</span>
          <span className="pill">π₀.₅ VLA</span>
          <span className="pill">Newton sim</span>
        </div>
        <h1>
          <span className="logo">🤖</span>
          <span className="title-grad">Robot Skill Learner</span>
        </h1>
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
          {!started && (
            <div className="start-bar">
              <button
                className="primary"
                disabled={!canStart}
                onClick={handleStart}
              >
                Start Learning
              </button>
              {!busy && !canStart && (
                <span className="hint">Add 1-3 photos and at least one activity.</span>
              )}
            </div>
          )}
          {error && <div className="error-banner">{error}</div>}
        </div>

        <div className={`right-col ${started ? 'started' : ''}`}>
          <div className={`robot-hero ${started ? 'robot-exit' : ''}`} aria-hidden={started}>
            <div className="robot-stage">
              <div className="robot-bubble">
                Add photos &amp; activities on the left, then hit{' '}
                <strong>Start Learning</strong> and I'll get to work →
              </div>
              <img src="/robot.jpg" alt="A friendly robot" className="robot-img" />
            </div>
          </div>

          {started && (
            <div className="run-panels">
              {phase === 'awaiting' && (
                <>
                  <SceneReview
                    scenePrompt={sceneDraft}
                    objects={detectedObjects}
                    onChange={setSceneDraft}
                    onConfirm={handleConfirmScene}
                    submitting={false}
                  />
                  <div style={{ height: 24 }} />
                </>
              )}
              <ProgressTimeline events={events} loading={busy} />
            </div>
          )}
        </div>
      </div>

      {started && results.length > 0 && (
        <div className="results-row">
          <ResultsPanel
            results={results}
            trainRounds={trainRounds}
            modalTraining={modalTraining}
            trainingIndices={trainingIndices}
            modalIndices={modalIndices}
            onTrainFurther={handleTrainFurther}
            onTrainModal={handleTrainModal}
          />
        </div>
      )}
    </div>
  )
}

// Build the per-activity results + train rounds from the raw event stream. The
// backend's rollout/train_round events carry everything needed, so the UI stays a
// pure projection of the SSE log (no separate fetch).
function deriveResults(events: TimelineEvent[]): {
  results: ActivityResult[]
  trainRounds: Record<number, TrainRound[]>
  modalTraining: Record<number, ModalTraining>
} {
  const byIndex = new Map<number, ActivityResult>()
  const trainRounds: Record<number, TrainRound[]> = {}
  const modalTraining: Record<number, ModalTraining> = {}

  const mt = (idx: number): ModalTraining =>
    (modalTraining[idx] ??= {
      rollouts: [],
      rounds: [],
      curate: [],
      stages: [],
      status: 'running',
    })

  for (const e of events) {
    const d = e.data
    if (e.event === 'rollout') {
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
      const idx = Number(d.activity_index)
      ;(trainRounds[idx] ??= []).push({
        round: Number(d.round),
        best_reward: Number(d.best_reward),
        mean_reward: Number(d.mean_reward),
      })
    } else if (e.event === 'eval_rollout') {
      mt(Number(d.activity_index)).rollouts.push({
        round: Number(d.round),
        index: Number(d.index),
        reward: Number(d.reward),
        success: Boolean(d.success),
      })
    } else if (e.event === 'eval_summary') {
      mt(Number(d.activity_index)).rounds.push({
        round: Number(d.round),
        mean_reward: Number(d.mean_reward),
        success_rate: Number(d.success_rate),
      })
    } else if (e.event === 'curate') {
      mt(Number(d.activity_index)).curate.push({
        round: Number(d.round),
        threshold: Number(d.threshold),
        selected: Number(d.selected),
        available: Number(d.available),
        mean_selected_reward: Number(d.mean_selected_reward),
      })
    } else if (e.event === 'train_stage') {
      mt(Number(d.activity_index)).stages.push({
        round: Number(d.round ?? 0),
        stage: String(d.stage),
        status: String(d.status),
      })
    } else if (e.event === 'train_modal_done') {
      mt(Number(d.activity_index)).status = 'done'
    } else if (e.event === 'train_modal_error') {
      const m = mt(Number(d.activity_index))
      m.status = 'error'
      m.message = d.message != null ? String(d.message) : 'training failed'
    }
  }

  const results = [...byIndex.values()].sort(
    (a, b) => a.activity_index - b.activity_index,
  )
  return { results, trainRounds, modalTraining }
}

export default App
