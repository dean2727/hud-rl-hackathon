import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // backend runs on :8000 (uv run uvicorn backend.main:app --port 8000)
      '/api': {
        target: 'http://localhost:8000',
        // SSE responses must not be buffered/rewritten by the proxy.
        ws: false,
      },
    },
  },
})
