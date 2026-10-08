import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 防：前端直连后端跨域 → dev 代理把 /chat 转到 FastAPI（cli.py serve 默认 8000）
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://127.0.0.1:8000',
    },
  },
})
