# SentimentScope — Movie Review Sentiment Analysis with Transformers

SentimentScope fine-tunes a transformer encoder (e.g., BERT) to classify IMDB movie reviews as positive or negative.

## Project Structure
- `src/` — configuration, dataset/dataloader helpers, model definition, training and evaluation utilities.
- `notebooks/` — exploratory analysis and training notebooks.
- `requirements.txt` — Python dependencies.

## Getting Started
1) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```
2) Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3) Run training from a Python session or notebook:
   ```python
   from src.config import Config
   from src.train import train_model

   cfg = Config()
   model = train_model(cfg)
   ```
Use the returned metrics and saved checkpoint (`best_model.pt`) for downstream evaluation or visualization in notebooks.
