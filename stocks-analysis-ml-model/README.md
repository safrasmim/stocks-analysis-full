# Tadawul Stock Movement Prediction System

MSc Thesis Project – News-Driven Stock Price Movement Prediction

This project predicts Up/Down movements for selected Tadawul stocks based on
synthetic financial news data.

## Tech Stack

- Backend: Python, FastAPI, scikit-learn
- Models: Random Forest (classification)
- Data: CSV (generated synthetic news)
- Target stocks: 1120, 2010, 7010, 1150, 4325

## Project Structure

stocks-analysis-ml-model/
├── backend/
│   ├── src/
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   └── app.py
│   ├── scripts/
│   │   ├── generate_sample_data.py
│   │   └── train_models.py
│   ├── data/
│   ├── models/
│   └── requirements.txt

## How to Run (Windows)

1. Open PowerShell and go to the project folder:

   cd stocks-analysis-ml-model\backend

2. Create and activate virtual environment:

   python -m venv venv
   .\venv\Scripts\activate

3. Install dependencies:

   pip install -r requirements.txt

4. Generate sample data:

   python scripts\generate_sample_data.py

5. Train model:

   python scripts\train_models.py

6. Start API:

   uvicorn src.app:app --reload --port 8000

7. Open in browser:

   http://localhost:8000/docs

Use the `/predict` endpoint with a ticker and list of news texts to get
predicted Up/Down labels and probabilities.
