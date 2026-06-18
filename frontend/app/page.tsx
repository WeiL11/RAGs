"use client";

import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Strategy = { name: string; description: string };
type Context = {
  text: string;
  episode_id: string;
  ep_number: number;
  publish_date: string | null;
  start_s: number | null;
  end_s: number | null;
  score: number;
  source: string;
};
type Trace = Record<string, unknown>;

export default function Home() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [strategy, setStrategy] = useState<string>("");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [contexts, setContexts] = useState<Context[]>([]);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const answerRef = useRef("");

  // Populate the strategy dropdown from the backend — the ONLY coupling the UI
  // has to the RAG layer. New strategies appear here automatically.
  useEffect(() => {
    fetch(`${API}/strategies`)
      .then((r) => r.json())
      .then((d) => {
        setStrategies(d.strategies ?? []);
        setStrategy(d.default ?? d.strategies?.[0]?.name ?? "");
      })
      .catch((e) => setError(`Cannot reach backend at ${API}: ${e}`));
  }, []);

  async function ask() {
    if (!query.trim() || !strategy || busy) return;
    setBusy(true);
    setError("");
    setAnswer("");
    setContexts([]);
    setTrace(null);
    answerRef.current = "";

    try {
      const resp = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, strategy, history: [], filters: null }),
      });
      if (!resp.ok || !resp.body) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const frames = buf.split("\n\n");
        buf = frames.pop() ?? "";
        for (const frame of frames) {
          const line = frame.trim();
          if (!line.startsWith("data:")) continue;
          const evt = JSON.parse(line.slice(5).trim());
          if (evt.type === "token") {
            answerRef.current += evt.delta ?? "";
            setAnswer(answerRef.current);
          } else if (evt.type === "contexts") {
            setContexts(evt.contexts ?? []);
          } else if (evt.type === "done") {
            setTrace(evt.trace ?? null);
          } else if (evt.type === "error") {
            setError(evt.delta ?? "stream error");
          }
        }
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main style={{ maxWidth: 820, margin: "0 auto", padding: "32px 20px" }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>股癌 RAG Playground</h1>
      <p style={{ color: "#9aa0a6", marginTop: 0, fontSize: 14 }}>
        Ask the Gooaye podcast. Switch RAG strategies to compare answers.
      </p>

      <div style={{ display: "flex", gap: 8, alignItems: "center", margin: "16px 0" }}>
        <label style={{ fontSize: 13, color: "#9aa0a6" }}>Strategy</label>
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          style={selectStyle}
        >
          {strategies.map((s) => (
            <option key={s.name} value={s.name} title={s.description}>
              {s.name}
            </option>
          ))}
        </select>
        {strategy && (
          <span style={{ fontSize: 12, color: "#6b7177" }}>
            {strategies.find((s) => s.name === strategy)?.description}
          </span>
        )}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="例如：股癌怎麼看輝達 NVDA？"
          style={inputStyle}
        />
        <button onClick={ask} disabled={busy} style={buttonStyle}>
          {busy ? "…" : "問"}
        </button>
      </div>

      {error && <p style={{ color: "#f28b82", fontSize: 13 }}>{error}</p>}

      {answer && (
        <section style={cardStyle}>
          <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{answer}</div>
        </section>
      )}

      {contexts.length > 0 && (
        <section style={{ marginTop: 20 }}>
          <h3 style={{ fontSize: 14, color: "#9aa0a6" }}>Sources ({contexts.length})</h3>
          {contexts.map((c, i) => (
            <div key={i} style={sourceStyle}>
              <div style={{ fontSize: 12, color: "#8ab4f8" }}>
                EP{c.ep_number} {c.publish_date ?? ""}{" "}
                {c.start_s != null ? `· ${fmt(c.start_s)}` : ""} · {c.source} ·{" "}
                {c.score.toFixed(3)}
              </div>
              <div style={{ fontSize: 13, color: "#c9ccd1" }}>{c.text}</div>
            </div>
          ))}
        </section>
      )}

      {trace && (
        <pre style={traceStyle}>{JSON.stringify(trace, null, 2)}</pre>
      )}
    </main>
  );
}

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

const selectStyle: React.CSSProperties = {
  background: "#1a1c20",
  color: "#e8eaed",
  border: "1px solid #2c2f36",
  borderRadius: 6,
  padding: "6px 10px",
};
const inputStyle: React.CSSProperties = {
  flex: 1,
  background: "#1a1c20",
  color: "#e8eaed",
  border: "1px solid #2c2f36",
  borderRadius: 8,
  padding: "10px 12px",
  fontSize: 15,
};
const buttonStyle: React.CSSProperties = {
  background: "#8ab4f8",
  color: "#0b0c0f",
  border: "none",
  borderRadius: 8,
  padding: "0 18px",
  fontWeight: 600,
  cursor: "pointer",
};
const cardStyle: React.CSSProperties = {
  marginTop: 20,
  background: "#15171b",
  border: "1px solid #2c2f36",
  borderRadius: 10,
  padding: 16,
};
const sourceStyle: React.CSSProperties = {
  background: "#15171b",
  border: "1px solid #23262c",
  borderRadius: 8,
  padding: "8px 12px",
  marginBottom: 8,
};
const traceStyle: React.CSSProperties = {
  marginTop: 16,
  background: "#101216",
  border: "1px solid #23262c",
  borderRadius: 8,
  padding: 12,
  fontSize: 12,
  color: "#9aa0a6",
  overflowX: "auto",
};
