# AI Portfolio Assistant

An AI-powered helper for **financial analysis**. It recommends diversified portfolios, allocates capital by ticker, and projects expected returns over a chosen horizon. It can also pull **news by ticker via Tiingo** and (optionally) summarize it for context.

> ⚠️ **Disclaimer:** Educational purposes only. Not investment advice.

---

## ✨ Features

* **Portfolio construction**: Equal-weight, inverse-volatility, or **PyPortfolioOpt**-based optimization (e.g., Max Sharpe) with position/sector caps.
* **Capital allocation**: Convert weights → dollar targets with min-trade thresholds and optional fractional-share handling.
* **Return projections**: Simple CAGR/expected-return projections over user-selected horizons.
* **News integration (Tiingo)**: Fetch the latest headlines tagged to your tickers; optional LLM summaries.
* **(Optional) RL engine**: Hooks to **FinRL** + Stable-Baselines3 for research experiments.

---

## 🧰 Tech Stack

* **Python** 3.10+
* **Core**: `pandas`, `numpy`, `pydantic`
* **Optimization**: `PyPortfolioOpt` (`pypfopt`)
* **News**: `tiingo` (News API)
* **(Optional) RL**: `finrl`, `stable-baselines3`, `gymnasium`
* **(Optional) LLM**: provider of your choice for summarization

---

## 🗂️ Data Sources

* **Prices**: Bring your own (CSV), or integrate any source you prefer. Example utilities are provided for quick CSV/Yahoo usage; swap in Tiingo EOD if you have access.
* **News**: **Tiingo News API** (requires API key). Only the news endpoint is used by default.

---

## 🚀 Getting Started

### 1) Install

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt` example:

```text
pandas
numpy
pypfopt
tiingo
requests
matplotlib
# optional
finrl
stable-baselines3
gymnasium<1.0
```

### 2) Configure environment

Create a `.env` or set environment variables:

```bash
# Windows (PowerShell)
setx TIINGO_API_KEY "YOUR_KEY"
# macOS/Linux (bash)
export TIINGO_API_KEY="YOUR_KEY"
```

### 3) Quickstart (Python)

```python
from datetime import date, timedelta
import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# --- Example prices (replace with your loader) ---
# Load your own CSV or use a helper like get_prices_df(...)
prices = pd.DataFrame({
    "AAPL": [190, 191, 189, 194, 197],
    "MSFT": [420, 418, 423, 425, 430],
    "NVDA": [125, 127, 128, 129, 131],
}).astype(float)

# --- Optimize weights (Max Sharpe with a 13% cap per name) ---
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.13))
w = ef.max_sharpe()
weights = ef.clean_weights()  # dict summing to 1.0

# --- Convert weights to dollars ---
portfolio_value = 5000
min_trade = 5.0  # dollars

alloc_dollars = {t: round(portfolio_value * wt, 2) for t, wt in weights.items()}
alloc_dollars = {t: v for t, v in alloc_dollars.items() if v >= min_trade}

print("Weights:", weights)
print("Dollar targets:", alloc_dollars)
```

### 4) Tiingo News (only)

```python
from tiingo import TiingoClient
from datetime import date, timedelta

client = TiingoClient()  # reads TIINGO_API_KEY

end = date.today()
start = end - timedelta(days=7)
articles = client.get_news(
    tickers=["AAPL", "MSFT", "NVDA"],
    startDate=start.isoformat(),
    endDate=end.isoformat(),
)

# Simple printout
for a in sorted(articles, key=lambda x: x.get("publishedDate", ""), reverse=True)[:10]:
    print(a["publishedDate"], a["source"], "-", a["title"])  # and a["url"] if needed
```

---

## ⚙️ Configuration

Create a `config.yaml` (or `.env`) to tweak behavior:

```yaml
# weights & thresholds
max_weight_per_name: 0.13     # 13% cap per ticker
min_trade_dollars: 5.0        # drop tiny trades

# sector comparison
underweight_threshold_pp: 0.5  # percentage points; 0.5 == 0.5%
weights_in_percent: true

# news
news_days: 7
news_top_n: 25
```

---

## 🧪 Project Structure (suggested)

```
.
├─ src/
│  └─ ai_portfolio/
│     ├─ optimizers.py         # PyPortfolioOpt helpers
│     ├─ allocation.py         # dollars ↔ weights utilities
│     ├─ news.py               # Tiingo news wrappers
│     ├─ config.py             # pydantic settings
│     └─ __init__.py
├─ notebooks/
│  └─ 01_quickstart.ipynb
├─ examples/
│  └─ news_demo.py
├─ requirements.txt
├─ README.md
└─ LICENSE
```

---

## 🔌 Optional: FinRL integration

If you enable RL experiments, install `finrl`, `stable-baselines3`, and `gymnasium`. Expect version differences across FinRL releases; a small import shim is included in `src/ai_portfolio/rl_shim.py`.

---

## 🗺️ Roadmap

* [ ] Add FastAPI endpoint for on-demand allocation & news
* [ ] Per-sector caps & exposure checks
* [ ] Caching layer for news & prices
* [ ] CLI commands (e.g., `ai-portfolio optimize --tickers ...`)

---

## 📝 License

MIT (see `LICENSE`).

---

## 🙌 Acknowledgments

* PyPortfolioOpt for efficient frontier tools
* Tiingo for News API access
* (Optional) FinRL for RL finance research utilities
