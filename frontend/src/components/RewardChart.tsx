// Dependency-free inline-SVG reward chart. No charting library: a single
// responsive <svg> (viewBox + width:100%) drawing gridlines, line series, scatter
// dots, and an optional dashed reference line (e.g. the curation threshold).
// y is fixed to the reward range [0, 1]; x is caller-defined (rollout # or round #).

export interface ChartPoint {
  x: number
  y: number
}

export interface ChartSeries {
  label: string
  color: string
  points: ChartPoint[]
  dashed?: boolean
}

export interface ChartDot {
  x: number
  y: number
  color: string
}

interface Props {
  series?: ChartSeries[]
  dots?: ChartDot[]
  yRef?: { value: number; label: string }
  xLabel?: string
  height?: number
}

const W = 320 // viewBox units; CSS scales width to the container
const PAD = { top: 12, right: 12, bottom: 22, left: 30 }

export function RewardChart({
  series = [],
  dots = [],
  yRef,
  xLabel,
  height = 160,
}: Props) {
  const H = height
  const allX = [
    ...series.flatMap((s) => s.points.map((p) => p.x)),
    ...dots.map((d) => d.x),
  ]
  if (allX.length === 0) return null

  const minX = Math.min(...allX)
  const maxX = Math.max(...allX, minX + 1) // avoid divide-by-zero for a single x
  const plotW = W - PAD.left - PAD.right
  const plotH = H - PAD.top - PAD.bottom

  const sx = (x: number) =>
    PAD.left + (maxX === minX ? plotW / 2 : ((x - minX) / (maxX - minX)) * plotW)
  const sy = (y: number) => PAD.top + (1 - Math.max(0, Math.min(1, y))) * plotH

  const yTicks = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="reward-chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img">
        {/* y gridlines + labels */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              y1={sy(t)}
              x2={W - PAD.right}
              y2={sy(t)}
              className="rcx-grid"
            />
            <text x={PAD.left - 5} y={sy(t) + 3} className="rcx-ytick">
              {t}
            </text>
          </g>
        ))}

        {/* reference line (curation threshold) */}
        {yRef && (
          <g>
            <line
              x1={PAD.left}
              y1={sy(yRef.value)}
              x2={W - PAD.right}
              y2={sy(yRef.value)}
              className="rcx-ref"
            />
            <text x={W - PAD.right} y={sy(yRef.value) - 3} className="rcx-reflabel">
              {yRef.label}
            </text>
          </g>
        )}

        {/* scatter dots (per-rollout reward) */}
        {dots.map((d, i) => (
          <circle key={i} cx={sx(d.x)} cy={sy(d.y)} r={2.6} fill={d.color} className="rcx-dot" />
        ))}

        {/* line series (per-round mean, etc.) */}
        {series.map((s) =>
          s.points.length === 0 ? null : (
            <polyline
              key={s.label}
              points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={s.dashed ? '4 3' : undefined}
              className="rcx-line"
            />
          ),
        )}
        {/* endpoint markers for line series */}
        {series.flatMap((s) =>
          s.points.map((p, i) => (
            <circle key={`${s.label}-${i}`} cx={sx(p.x)} cy={sy(p.y)} r={2.4} fill={s.color} />
          )),
        )}

        {xLabel && (
          <text x={(PAD.left + W - PAD.right) / 2} y={H - 4} className="rcx-xlabel">
            {xLabel}
          </text>
        )}
      </svg>

      {(series.length > 0 || yRef) && (
        <div className="rcx-legend">
          {series.map((s) => (
            <span key={s.label} className="rcx-leg">
              <span className="rcx-swatch" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
          {yRef && (
            <span className="rcx-leg">
              <span className="rcx-swatch rcx-swatch-ref" />
              {yRef.label}
            </span>
          )}
        </div>
      )}
    </div>
  )
}
