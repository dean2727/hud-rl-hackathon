import { useEffect, useRef, useState } from 'react'

interface Props {
  photos: File[]
  onChange: (photos: File[]) => void
  disabled?: boolean
}

const MAX = 3

export function PhotoUpload({ photos, onChange, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [previews, setPreviews] = useState<string[]>([])

  useEffect(() => {
    const urls = photos.map((f) => URL.createObjectURL(f))
    setPreviews(urls)
    return () => urls.forEach((u) => URL.revokeObjectURL(u))
  }, [photos])

  function addFiles(files: FileList | null) {
    if (!files) return
    const incoming = Array.from(files).filter((f) => f.type.startsWith('image/'))
    onChange([...photos, ...incoming].slice(0, MAX))
  }

  function removeAt(i: number) {
    onChange(photos.filter((_, idx) => idx !== i))
  }

  return (
    <div className="panel">
      <h2>
        <span className="section-num">1</span>Upload photos
      </h2>

      <div className="photo-grid">
        {previews.map((src, i) => (
          <div className="photo-thumb" key={i}>
            <img src={src} alt={`photo ${i + 1}`} />
            {!disabled && <button onClick={() => removeAt(i)}>×</button>}
          </div>
        ))}
      </div>

      {photos.length < MAX && !disabled && (
        <div
          className="dropzone"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            addFiles(e.dataTransfer.files)
          }}
        >
          Click or drop photos here ({photos.length}/{MAX})
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => addFiles(e.target.files)}
      />
      <p className="hint">
        2-3 photos of where the robot arm would go, and the things it should work
        with.
      </p>
    </div>
  )
}
