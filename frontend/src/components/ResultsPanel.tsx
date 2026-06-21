import { RewardChart, type ChartDot, type ChartSeries } from './RewardChart'

export interface ActivityResult {
  activity_index: number
  activity: string
  task: string
  target: string
  status: string
  reward: number | null
  success: boolean | null
  content: string | null
  can_train_further: boolean
}

export interface TrainRound {
  round: number
  best_reward: number
  mean_reward: number
}

export interface ModalRollout {
  round: number
  index: number
  reward: number
  success: boolean
}

export interface ModalTraining {
  rollouts: ModalRollout[]
  rounds: { round: number; mean_reward: number; success_rate: number }[]
  curate: {
    round: number
    threshold: number
    selected: number
    available: number
    mean_selected_reward: number
  }[]
  stages: { round: number; stage: string; status: string }[]
  status: 'running' | 'done' | 'error'
  message?: string
}

interface Props {
  results: ActivityResult[]
  trainRounds: Record<number, TrainRound[]>
  modalTraining: Record<number, ModalTraining>
  trainingIndices: Set<number>
  modalIndices: Set<number>
  onTrainFurther: (activityIndex: number) => void
  onTrainModal: (activityIndex: number) => void
}

const STAGE_ORDER = ['serve', 'eval', 'curate', 'finetune']

function statusBadge(r: ActivityResult, mt?: ModalTraining) {
  if (r.status === 'running')
    return <span className="badge running">running…</span>
  if (r.status === 'failed') return <span className="badge fail">failed</span>
  if (mt?.status === 'done') return <span className="badge success">fine-tuned</span>
  if (r.success) return <span className="badge success">success</span>
  if (r.status === 'completed')
    return <span className="badge fail">incomplete</span>
  return <span className="badge">pending</span>
}

function ModalTrainingView({ mt }: { mt: ModalTraining }) {
  // Scatter: every collected rollout, green when it succeeded.
  const dots: ChartDot[] = mt.rollouts.map((r, i) => ({
    x: i,
    y: r.reward,
    color: r.success ? '#34d399' : '#60a5fa',
  }))

  // Mean line: one point per round, placed at the centre of that round's rollouts.
  const roundXs: Record<number, number[]> = {}
  mt.rollouts.forEach((r, i) => (roundXs[r.round] ??= []).push(i))
  const meanPoints = mt.rounds
    .map((rd) => {
      const xs = roundXs[rd.round] ?? []
      const x = xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : rd.round
      return { x, y: rd.mean_reward }
    })
    .sort((a, b) => a.x - b.x)
  const series: ChartSeries[] = [
    { label: 'round mean', color: '#fbbf24', points: meanPoints },
  ]

  const threshold = mt.curate.at(-1)?.threshold
  const lastStage = mt.stages.at(-1)
  const lastCurate = mt.curate.at(-1)
  const lastRound = mt.rounds.at(-1)

  return (
    <div className="modal-train">
      <div className="mt-head">
        <strong>VLA fine-tune (collect → curate → train)</strong>
        {mt.status === 'running' && lastStage && (
          <span className="mt-stage">
            {STAGE_ORDER.map((s) => (
              <span
                key={s}
                className={`mt-step ${lastStage.stage === s ? 'active' : ''}`}
              >
                {s}
              </span>
            ))}
          </span>
        )}
        {mt.status === 'done' && <span className="badge success">done</span>}
        {mt.status === 'error' && <span className="badge fail">failed</span>}
      </div>

      {dots.length > 0 && (
        <RewardChart
          dots={dots}
          series={series}
          yRef={
            threshold != null
              ? { value: threshold, label: `curate ≥ ${threshold}` }
              : undefined
          }
          xLabel="rollout # (collected in order)"
        />
      )}

      <div className="mt-stats">
        {lastRound && (
          <span>
            mean {lastRound.mean_reward.toFixed(3)} · success{' '}
            {(lastRound.success_rate * 100).toFixed(0)}%
          </span>
        )}
        {lastCurate && (
          <span>
            kept {lastCurate.selected}/{lastCurate.available} for training
          </span>
        )}
      </div>
      {mt.status === 'error' && mt.message && (
        <div className="rc-content">{mt.message}</div>
      )}
    </div>
  )
}

export function ResultsPanel({
  results,
  trainRounds,
  modalTraining,
  trainingIndices,
  modalIndices,
  onTrainFurther,
  onTrainModal,
}: Props) {
  if (results.length === 0) return null

  return (
    <div className="panel">
      <h2>Results</h2>
      <div className="results-cards">
      {results.map((r) => {
        const rounds = trainRounds[r.activity_index] ?? []
        const mt = modalTraining[r.activity_index]
        const isTraining = trainingIndices.has(r.activity_index)
        const isModalTraining =
          modalIndices.has(r.activity_index) || mt?.status === 'running'
        // After fine-tune, use the latest success rate; otherwise use initial reward.
        const lastRoundSummary = mt?.rounds.at(-1)
        const barPct = lastRoundSummary != null
          ? lastRoundSummary.success_rate * 100
          : r.reward != null ? Math.max(0, Math.min(1, r.reward)) * 100 : 0
        return (
          <div className="result-card" key={r.activity_index}>
            <div className="rc-head">
              <span className="rc-activity">{r.activity}</span>
              {statusBadge(r, mt)}
            </div>
            <div className="rc-meta">
              {r.task}
              {r.target ? ` · ${r.target}` : ''}
              {r.reward != null ? ` · reward ${r.reward.toFixed(3)}` : ''}
            </div>
            {(r.reward != null || lastRoundSummary != null) && (
              <div className="reward-bar">
                <div style={{ width: `${barPct}%` }} />
              </div>
            )}
            {r.content && <div className="rc-content">{r.content}</div>}

            {rounds.length > 0 && (
              <div className="train-rounds">
                <strong>Train-further (best-of-N search):</strong>
                {rounds.map((tr) => (
                  <div className="tr-row" key={tr.round}>
                    <span>Round {tr.round + 1}</span>
                    <span>
                      best {tr.best_reward.toFixed(3)} · mean{' '}
                      {tr.mean_reward.toFixed(3)}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {mt && <ModalTrainingView mt={mt} />}

            {r.can_train_further && (
              <div className="rc-actions">
                <button
                  disabled={isTraining}
                  onClick={() => onTrainFurther(r.activity_index)}
                >
                  {isTraining ? 'Training…' : 'Train further'}
                </button>
                <button
                  disabled={isModalTraining}
                  onClick={() => onTrainModal(r.activity_index)}
                >
                  {isModalTraining ? 'Fine-tuning…' : 'Fine-tune (VLA)'}
                </button>
              </div>
            )}
          </div>
        )
      })}
      </div>
    </div>
  )
}
