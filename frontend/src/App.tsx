import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service: string;
  environment: string;
};

type Source = {
  document_id: string;
  title: string;
  page_start: number;
  page_end: number;
  excerpt: string;
  relevance_score?: number | null;
};

type AnalysisResponse = {
  status: "supported" | "insufficient_evidence" | "assessment_unavailable";
  assessment?: string | null;
  supporting_points: string[];
  suggested_next_steps: string[];
  missing_information: string[];
  sources: Source[];
  warning?: string | null;
  disclaimer: string;
};

const API = "http://localhost:8001";

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "#f5f7fb",
    color: "#182230",
    fontFamily: "Inter, Segoe UI, Arial, sans-serif",
  },
  header: {
    background: "#14243b",
    color: "white",
    padding: "22px clamp(20px, 5vw, 72px)",
  },
  brand: { fontSize: 28, fontWeight: 800, letterSpacing: 2 },
  subtitle: { margin: "7px 0 0", color: "#c8d4e5", fontSize: 14 },
  main: {
    maxWidth: 1000,
    margin: "0 auto",
    padding: "36px 20px 56px",
  },
  eyebrow: {
    color: "#3765a6",
    fontWeight: 700,
    fontSize: 12,
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
  heading: {
    fontSize: "clamp(28px, 5vw, 40px)",
    lineHeight: 1.15,
    margin: "10px 0",
  },
  muted: { color: "#627086", lineHeight: 1.6 },
  card: {
    background: "white",
    border: "1px solid #e0e6ef",
    borderRadius: 14,
    padding: 24,
    marginTop: 22,
    boxShadow: "0 4px 18px rgba(20,36,59,0.04)",
  },
  label: { display: "block", fontWeight: 700, marginBottom: 10 },
  textarea: {
    display: "block",
    boxSizing: "border-box",
    width: "100%",
    minHeight: 150,
    resize: "vertical",
    padding: 14,
    border: "1px solid #ccd5e2",
    borderRadius: 9,
    font: "inherit",
    lineHeight: 1.6,
    background: "#fff",
    color: "#182230",
  },
  button: {
    marginTop: 14,
    background: "#2459a9",
    color: "white",
    padding: "12px 20px",
    border: 0,
    borderRadius: 8,
    fontSize: 15,
    fontWeight: 700,
    cursor: "pointer",
  },
  secondary: {
    background: "#edf3fc",
    color: "#2459a9",
    border: "1px solid #d7e3f6",
    borderRadius: 7,
    padding: "7px 10px",
    cursor: "pointer",
    margin: "5px 6px 0 0",
    textAlign: "left",
  },
  sectionTitle: { fontSize: 19, margin: "0 0 14px" },
  item: { margin: "10px 0", lineHeight: 1.65 },
  source: {
    border: "1px solid #e0e6ef",
    borderRadius: 10,
    padding: 16,
    marginTop: 12,
    background: "#fbfcfe",
  },
  small: { fontSize: 12, color: "#627086", lineHeight: 1.5 },
};

const examples = [
  "I purchased a defective product, and the seller refuses to replace it or refund my payment.",
  "A company charged me for a service I did not agree to.",
  "An online order was paid for but never delivered.",
];

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [situation, setSituation] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((res) => {
        if (!res.ok) throw new Error("Backend unavailable");
        return res.json();
      })
      .then((data: HealthResponse) => {
        setHealth(data);
        setHealthError(false);
      })
      .catch(() => setHealthError(true));
  }, []);

  async function analyze() {
    if (situation.trim().length < 10) {
      setError("Please describe your situation in at least 10 characters.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/situation/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ situation: situation.trim(), top_k: 5 }),
      });

      if (!response.ok) {
        let message = `Request failed (${response.status}).`;
        try {
          const body = await response.json();
          if (typeof body.detail === "string") message = body.detail;
        } catch {
          // Keep the default error message.
        }
        throw new Error(message);
      }

      const data: AnalysisResponse = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not analyze your situation. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  const statusInfo = {
    supported: {
      title: "Evidence-supported assessment",
      color: "#176b45",
      background: "#eaf7ef",
      description: "The response identifies supporting source passages. Review them before relying on the findings.",
    },
    insufficient_evidence: {
      title: "Insufficient evidence",
      color: "#8a5700",
      background: "#fff5df",
      description: "The retrieved documents do not establish enough support for a reliable assessment.",
    },
    assessment_unavailable: {
      title: "Assessment unavailable",
      color: "#8a5700",
      background: "#fff5df",
      description: "Sources may have been retrieved, but a validated assessment could not be generated.",
    },
  };

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div style={styles.brand}>JURIS</div>
        <p style={styles.subtitle}>
          Legal document text mining · Consumer-law information support
        </p>
      </header>

      <main style={styles.main}>
        <div style={styles.eyebrow}>Consumer situation assistant</div>
        <h1 style={styles.heading}>Understand your situation.<br />Find supporting evidence.</h1>
        <p style={{ ...styles.muted, maxWidth: 720 }}>
          Describe your consumer problem in everyday language. JURIS searches
          the available legal documents and presents relevant passages,
          possible next steps, and information that may still be missing.
        </p>

        <div style={styles.card}>
          <label htmlFor="situation" style={styles.label}>
            What happened?
          </label>
          <textarea
            id="situation"
            style={styles.textarea}
            value={situation}
            maxLength={4000}
            onChange={(event) => setSituation(event.target.value)}
            placeholder="Explain what you purchased, what went wrong, what the seller said, and what outcome you are seeking..."
          />
          <div style={{ ...styles.small, textAlign: "right", marginTop: 6 }}>
            {situation.length}/4000 characters
          </div>

          <div style={{ marginTop: 12 }}>
            <span style={styles.small}>Try an example:</span>
            <div>
              {examples.map((example) => (
                <button
                  type="button"
                  key={example}
                  style={styles.secondary}
                  onClick={() => setSituation(example)}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          <button
            type="button"
            style={{
              ...styles.button,
              opacity: loading ? 0.65 : 1,
              cursor: loading ? "wait" : "pointer",
            }}
            disabled={loading}
            onClick={analyze}
          >
            {loading ? "Searching documents and analyzing..." : "Analyze my situation"}
          </button>

          {loading && (
            <p role="status" style={styles.muted}>
              Searching the available legal documents. This may take a little time.
            </p>
          )}
          {error && (
            <p role="alert" style={{ color: "#b42318", lineHeight: 1.5 }}>
              {error}
            </p>
          )}
        </div>

        <div style={{ ...styles.small, marginTop: 16 }}>
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: health && !healthError ? "#219653" : "#d99a25",
              marginRight: 7,
            }}
          />
          {health && !healthError
            ? `Backend connected · ${health.environment}`
            : healthError
              ? "Backend connection unavailable"
              : "Checking backend connection..."}
        </div>

        {result && (
          <section aria-live="polite" style={styles.card}>
            <div
              style={{
                padding: 16,
                borderRadius: 9,
                background: statusInfo[result.status].background,
                color: statusInfo[result.status].color,
              }}
            >
              <div style={{ fontWeight: 800, fontSize: 18 }}>
                {statusInfo[result.status].title}
              </div>
              <p style={{ marginBottom: 0, lineHeight: 1.6 }}>
                {statusInfo[result.status].description}
              </p>
            </div>

            <section style={{ marginTop: 24 }}>
              <h2 style={styles.sectionTitle}>Assessment</h2>
              <p style={{ lineHeight: 1.75 }}>
                {result.assessment || "No assessment was returned."}
              </p>
            </section>

            {result.supporting_points?.length > 0 && (
              <section style={{ marginTop: 24 }}>
                <h2 style={styles.sectionTitle}>Supporting points</h2>
                <ul>
                  {result.supporting_points.map((point, index) => (
                    <li key={index} style={styles.item}>{point}</li>
                  ))}
                </ul>
              </section>
            )}

            {result.suggested_next_steps?.length > 0 && (
              <section style={{ marginTop: 24 }}>
                <h2 style={styles.sectionTitle}>Possible next steps</h2>
                <ul>
                  {result.suggested_next_steps.map((step, index) => (
                    <li key={index} style={styles.item}>{step}</li>
                  ))}
                </ul>
              </section>
            )}

            {result.missing_information?.length > 0 && (
              <section style={{ marginTop: 24 }}>
                <h2 style={styles.sectionTitle}>What may still be needed</h2>
                <ul>
                  {result.missing_information.map((item, index) => (
                    <li key={index} style={styles.item}>{item}</li>
                  ))}
                </ul>
              </section>
            )}

            <section style={{ marginTop: 28 }}>
              <h2 style={styles.sectionTitle}>
                Source passages ({result.sources?.length ?? 0})
              </h2>
              {result.sources?.length ? (
                result.sources.map((source, index) => (
                  <article
                    key={`${source.document_id}-${source.page_start}-${index}`}
                    style={styles.source}
                  >
                    <div style={{ fontWeight: 700, lineHeight: 1.5 }}>
                      {source.title}
                    </div>
                    <div style={{ ...styles.small, margin: "5px 0 10px" }}>
                      Pages {source.page_start === source.page_end
                        ? source.page_start
                        : `${source.page_start}–${source.page_end}`}
                      {source.relevance_score != null
                        ? ` · Retrieval score: ${source.relevance_score.toFixed(3)}`
                        : ""}
                    </div>
                    <p style={{ whiteSpace: "pre-wrap", lineHeight: 1.7, marginBottom: 10 }}>
                      {source.excerpt}
                    </p>
                    <div style={styles.small}>
                      Document ID: {source.document_id}
                    </div>
                  </article>
                ))
              ) : (
                <p style={styles.muted}>No source passages were returned.</p>
              )}
            </section>

            {result.warning && (
              <p style={{ ...styles.small, marginTop: 20 }}>
                System warning: {result.warning}
              </p>
            )}

            <p style={{
              ...styles.small,
              borderTop: "1px solid #e0e6ef",
              paddingTop: 16,
              marginTop: 24,
            }}>
              {result.disclaimer}
            </p>
          </section>
        )}

        <footer style={{
          ...styles.small,
          textAlign: "center",
          marginTop: 36,
        }}>
          JURIS is an educational information tool, not a substitute for advice
          from a qualified legal professional. Results depend on the documents
          available in the system.
        </footer>
      </main>
    </div>
  );
}

export default App;
