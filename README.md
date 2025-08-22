# AI Portfolio Assistant 

An AI-powered helper for **financial analysis**. It recommends diversified portfolios, allocates capital by ticker, and projects expected returns over a chosen horizon. It can also pull **news by ticker via Tiingo** and (optionally) summarize it for context.

> ⚠️ **Disclaimer:** Educational purposes only. Not investment advice. ⚠️

> 🚧 **Status:** In active development (alpha). Expect breaking changes and incomplete features.
> 
This project is a work in progress. APIs and outputs may change. Feedback and PRs are welcome.
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
* **Core**: `pandas`, `numpy`
* **Optimization**: `PyPortfolioOpt` (`pypfopt`)
* **News**: `tiingo` (News API)
* **(Optional) RL**: `finrl`, `stable-baselines3`
* **(Optional) LLM**: OpenAI

---

## 🗂️ Data Sources

* **Prices**: Prices are found using YFinance as well as parsing through your portfolio if avaiable on it
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

If you enable RL experiments, install `finrl`, `stable-baselines3`. Expect version differences across FinRL releases; a small import shim is included in `src/ai_portfolio/rl_shim.py`.

---

## 🗺️ Roadmap

* [ ] Add FastAPI endpoint for on-demand allocation & news
* [ ] Per-sector caps & exposure checks
* [ ] Caching layer for news & prices
* [ ] CLI commands (e.g., `ai-portfolio optimize --tickers ...`)

---

## 🙌 Acknowledgments

* PyPortfolioOpt for efficient frontier tools
* Tiingo for News API access
* FinRL for RL finance research utilities
