import React, { useEffect, useState } from "react";
import "./App.css";
import { fetchTickers, predictMovement } from "./api";

function App() {
  const [tickers, setTickers] = useState({});
  const [selectedTicker, setSelectedTicker] = useState("2010");
  const [textInput, setTextInput] = useState("");
  const [loadingTickers, setLoadingTickers] = useState(false);
  const [loadingPredict, setLoadingPredict] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    async function loadTickers() {
      setLoadingTickers(true);
      setErrorMsg("");
      try {
        const data = await fetchTickers();
        setTickers(data);
        const keys = Object.keys(data);
        if (keys.length > 0) {
          setSelectedTicker(keys[0]);
        }
      } catch (err) {
        console.error(err);
        setErrorMsg("Failed to load tickers from API.");
      } finally {
        setLoadingTickers(false);
      }
    }
    loadTickers();
  }, []);

  const handlePredict = async () => {
    const cleaned = textInput
      .split("\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (!cleaned.length) {
      setErrorMsg("Please enter at least one news sentence.");
      return;
    }

    setLoadingPredict(true);
    setErrorMsg("");
    setResult(null);

    try {
      const data = await predictMovement(selectedTicker, cleaned);
      setResult({
        ...data,
        inputTexts: cleaned,
      });
    } catch (err) {
      console.error(err);
      setErrorMsg("Prediction failed. Check backend is running.");
    } finally {
      setLoadingPredict(false);
    }
  };

  return (
    <div className="App">
      <h1>Tadawul News-Based Stock Movement Predictor</h1>

      <div className="panel">
        <h2>1. Select Ticker</h2>
        {loadingTickers ? (
          <p>Loading tickers...</p>
        ) : (
          <select
            value={selectedTicker}
            onChange={(e) => setSelectedTicker(e.target.value)}
          >
            {Object.entries(tickers).map(([code, info]) => (
              <option key={code} value={code}>
                {code} - {info.name}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="panel">
        <h2>2. Enter News Text</h2>
        <p>Write one or more news sentences (each on a new line):</p>
        <textarea
          rows={6}
          value={textInput}
          onChange={(e) => setTextInput(e.target.value)}
          placeholder="Example:
SABIC reports strong quarterly earnings and positive outlook.
SABIC faces higher costs and weaker demand in international markets."
        />
        <button onClick={handlePredict} disabled={loadingPredict}>
          {loadingPredict ? "Predicting..." : "Predict Movement"}
        </button>
      </div>

      {errorMsg && (
        <div className="panel error">
          <strong>Error: </strong> {errorMsg}
        </div>
      )}

      {result && (
        <div className="panel">
          <h2>3. Prediction Result</h2>
          <p>
            Ticker: <strong>{result.ticker}</strong>
          </p>
          <table className="result-table">
            <thead>
              <tr>
                <th>#</th>
                <th>News Text</th>
                <th>Label</th>
                <th>Probability Up</th>
              </tr>
            </thead>
            <tbody>
              {result.labels.map((label, idx) => (
                <tr key={idx}>
                  <td>{idx + 1}</td>
                  <td>{result.inputTexts[idx]}</td>
                  <td>{label}</td>
                  <td>
                    {result.probabilities_up &&
                    result.probabilities_up.length > idx
                      ? result.probabilities_up[idx].toFixed(2)
                      : "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="footer">
        <p>
          Backend: FastAPI (http://localhost:8000) | Frontend: React
          (http://localhost:3000)
        </p>
      </div>
    </div>
  );
}

export default App;
