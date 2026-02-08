import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export async function fetchTickers() {
  const res = await client.get("/tickers");
  return res.data; // { "2010": { name: ..., sector: ... }, ... }
}

export async function predictMovement(ticker, texts) {
  const res = await client.post("/predict", {
    ticker,
    texts,
  });
  return res.data; // { ticker, labels, predictions, probabilities_up }
}
