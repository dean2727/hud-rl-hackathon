interface Props {
  activities: string[]
  onChange: (activities: string[]) => void
  disabled?: boolean
}

export function ActivityList({ activities, onChange, disabled }: Props) {
  function update(i: number, value: string) {
    onChange(activities.map((a, idx) => (idx === i ? value : a)))
  }
  function add() {
    onChange([...activities, ''])
  }
  function remove(i: number) {
    const next = activities.filter((_, idx) => idx !== i)
    onChange(next.length ? next : [''])
  }

  return (
    <div className="panel">
      <h2>
        <span className="section-num">2</span>Activities to learn
      </h2>

      {activities.map((a, i) => (
        <div className="activity-row" key={i}>
          <input
            type="text"
            placeholder="e.g. Pick up the mug and put it in the sink"
            value={a}
            disabled={disabled}
            onChange={(e) => update(i, e.target.value)}
          />
          {!disabled && (
            <button onClick={() => remove(i)} title="remove">
              ×
            </button>
          )}
        </div>
      ))}

      {!disabled && (
        <button onClick={add} style={{ marginTop: 4 }}>
          + Add activity
        </button>
      )}
      <p className="hint">
        Each activity becomes one graded task the robot attempts in simulation.
      </p>
    </div>
  )
}
