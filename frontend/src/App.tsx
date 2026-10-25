
import { useEffect, useState } from "react";

type HealthResponse = {
  status: string;
  service: string;
  environment: string;
};

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://localhost:8001/health")
      .then((res) => {
        if (!res.ok) throw new Error(`Request failed: ${res.status}`);
        return res.json();
      })
      .then((data: HealthResponse) => setHealth(data))
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>JURIS</h1>
      <p>AI-Powered Legal Knowledge Management and Text Mining System</p>
      <h2>Backend connection status</h2>
      {error && <p style={{ color: "red" }}>Error: {error}</p>}
      {!error && !health && <p>Connecting to backend...</p>}
      {health && (
        <pre style={{ background: "#f4f4f4", padding: "1rem" }}>
          {JSON.stringify(health, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default App;