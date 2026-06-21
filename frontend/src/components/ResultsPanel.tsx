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

interface Props {
  results: ActivityResult[]
  trainRounds: Record<number, TrainRound[]>
  trainingIndices: Set<number>
  onTrainFurther: (activityIndex: number) => void
}

function statusBadge(r: ActivityResult) {
  if (r.status === 'running')
    return <span className="badge running">running…</span>
  if (r.status === 'failed') return <span className="badge fail">failed</span>
  if (r.success) return <span className="badge success">success</span>
  if (r.status === 'completed')
    return <span className="badge fail">incomplete</span>
  return <span className="badge">pending</span>
}

export function ResultsPanel({
  results,
  trainRounds,
  trainingIndices,
  onTrainFurther,
}: Props) {
  if (results.length === 0) return null

  return (
    <div className="panel" style={{ marginTop: 24 }}>
      <h2>Results</h2>
      {results.map((r) => {
        const rounds = trainRounds[r.activity_index] ?? []
        const isTraining = trainingIndices.has(r.activity_index)
        const pct =
          r.reward != null ? Math.max(0, Math.min(1, r.reward)) * 100 : 0
        return (
          <div className="result-card" key={r.activity_index}>
            <div className="rc-head">
              <span className="rc-activity">{r.activity}</span>
              {statusBadge(r)}
            </div>
            <div className="rc-meta">
              {r.task}
              {r.target ? ` · ${r.target}` : ''}
              {r.reward != null ? ` · reward ${r.reward.toFixed(3)}` : ''}
            </div>
            {r.reward != null && (
              <div className="reward-bar">
                <div style={{ width: `${pct}%` }} />
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

            {r.can_train_further && (
              <button
                style={{ marginTop: 10 }}
                disabled={isTraining}
                onClick={() => onTrainFurther(r.activity_index)}
              >
                {isTraining ? 'Training…' : 'Train further'}
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}
