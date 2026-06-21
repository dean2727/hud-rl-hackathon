interface Props {
  scenePrompt: string
  objects: string[]
  onChange: (value: string) => void
  onConfirm: () => void
  submitting: boolean
}

// Shown when the run pauses at awaiting_confirmation: the user reviews/edits the
// vision model's scene description before it's sent to the 3D scene generator.
export function SceneReview({
  scenePrompt,
  objects,
  onChange,
  onConfirm,
  submitting,
}: Props) {
  return (
    <div className="panel scene-review">
      <h2>Review the scene description</h2>
      <p className="hint">
        This is what the vision model saw. Edit it so it names every object you
        want the robot to work with, then generate the 3D scene.
      </p>
      <textarea
        className="scene-textarea"
        value={scenePrompt}
        rows={4}
        disabled={submitting}
        onChange={(e) => onChange(e.target.value)}
      />
      {objects.length > 0 && (
        <div className="object-chips">
          {objects.map((o) => (
            <span className="chip" key={o}>
              {o}
            </span>
          ))}
        </div>
      )}
      <button
        className="primary"
        style={{ marginTop: 12 }}
        disabled={submitting || !scenePrompt.trim()}
        onClick={onConfirm}
      >
        {submitting ? 'Generating…' : 'Generate 3D Scene →'}
      </button>
    </div>
  )
}
