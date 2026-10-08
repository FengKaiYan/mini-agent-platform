import { useRef, useState } from 'react'
import './App.css'

// SSE 协议类型化解析：与后端 fetchSse 生产协议同源
// data: {"result":…} 流式正文 / {"meta":…} 路由·RAG·工具·trace / [DONE] 收尾哨兵
type SsePayload = { result?: string; meta?: string }
type MetaLine = { id: number; text: string }

function App() {
  const [input, setInput] = useState('这个候选人和 JD-001 岗位匹配吗')
  const [answer, setAnswer] = useState('')
  const [metas, setMetas] = useState<MetaLine[]>([])
  const [running, setRunning] = useState(false)
  const [aborted, setAborted] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const metaId = useRef(0)
  const logRef = useRef<HTMLDivElement>(null)

  const pushMeta = (text: string) =>
    setMetas((m) => [...m, { id: ++metaId.current, text }])

  const scroll = () => {
    requestAnimationFrame(() => {
      logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
    })
  }

  // 中断：AbortController 断流 + 状态机标记，与生产流式状态机的"用户主动停止"降级同构
  const stop = () => {
    abortRef.current?.abort()
    setAborted(true)
    setRunning(false)
    pushMeta('⏹ 已中断（AbortController 断流，状态机置 stopped）')
  }

  const send = async () => {
    const text = input.trim()
    if (!text || running) return
    setRunning(true)
    setAborted(false)
    setAnswer('')
    setMetas([])
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, thread_id: 'web-' + Math.random().toString(36).slice(2, 8) }),
        signal: ctrl.signal,
      })
      const reader = res.body!.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let idx: number
        while ((idx = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6)
          if (payload === '[DONE]') {
            pushMeta('✔ [DONE] 流结束')
            continue
          }
          try {
            const obj: SsePayload = JSON.parse(payload)
            if (obj.result) {
              setAnswer((a) => a + obj.result)
              scroll()
            } else if (obj.meta) {
              pushMeta(obj.meta)
              scroll()
            }
          } catch {
            /* 非 JSON 帧忽略 */
          }
        }
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') pushMeta('[错误] ' + (e as Error).message)
    } finally {
      setRunning(false)
      abortRef.current = null
    }
  }

  return (
    <div className="wrap">
      <h1>mini Agent 平台 · React 聊天前端</h1>
      <div className="sub">
        消费同一 SSE 协议（result / meta / [DONE]）· 流式渲染 · AbortController 中断 · 工具调用与 trace 可视化
      </div>

      <div className="log" ref={logRef}>
        {metas.map((m) => (
          <div className="meta" key={m.id}>· {m.text}</div>
        ))}
        {answer && <div className="ai">{answer}</div>}
        {aborted && <div className="meta">（本次会话被用户中断，历史消息保留）</div>}
      </div>

      <div className="bar">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="例：这个候选人和 JD-001 岗位匹配吗（触发工具调用）/ 帮我筛下简历"
        />
        {running ? (
          <button className="stop" onClick={stop}>中断</button>
        ) : (
          <button onClick={send}>发送</button>
        )}
      </div>
    </div>
  )
}

export default App
