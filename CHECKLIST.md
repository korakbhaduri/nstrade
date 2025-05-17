# =================  NSTRADE CONTEST CHECKLIST  ==================

## 0. ENVIRONMENT
- Python ≥3.12, pandas ≥2.2 installed.
- Data files provided by contest:
    • development.csv  (use for all research)
    • holdout.csv      (never load while developing)

## 1. BASELINES
- Run Buy-and-Hold and SMA-Crossover on development.csv.
- Record these metrics: Sharpe, total return, annualised return, max drawdown,
  number of trades, win rate.
  (They are your "score to beat".)

## 2. BACKTESTER QUALITY
- Fix equity calculation:
    equity = initial_capital + (close - entry_price)/entry_price * initial_capital
  (removes unintended compounding).
- Use loop index `i` for entry_idx / exit_idx to avoid off-by-one errors.
- Trading fee parameter `fee` (fraction, e.g. 0.001 for 0.1 %) must be applied
- on **every** buy and sell.  Set default `fee=0.0` so notebooks can override.
- Make `verbose=False` default; print only when needed.
- Return equity and returns as pandas Series indexed by timestamp.

## 3. SPEED IMPROVEMENT (after bugs are gone)
- Vectorise indicator calculations: add SMA/EMA/Rolling RSI columns instead of
  recalculating inside loops.
- Aim for <1 s runtime on full development set.
- If runtime is still slow: profile with `%timeit` or the built-in profiler.

## 4. STRATEGY TEMPLATE  (each new idea lives in its own file)
# src/strategies/my_strategy.py
from strategy import Strategy

class MyStrategy(Strategy):
    NAME   = "my_strategy"   # shows on leaderboard
    AUTHOR = "Korak"

    def __init__(self, initial_capital=10_000, **params):
        super().__init__(initial_capital)
        self.params = params
        # preload any arrays / indicators here

    def process_bar(self, bar):
        # update internal state for this bar
        ...

    def get_signal(self):
        # return 'buy', 'sell', or 'hold'
        ...

- Add a tiny test in `tests/test_my_strategy.py` that calls `run_backtest`
  and asserts the function returns without error.

## 5. INCREMENTAL STRATEGY ROAD-MAP
Work top-to-bottom; move forward only if Sharpe improves on development set.

1. **Parameter sweep for SMA-Crossover**  
   Fast window 10–50, slow window 80–300; walk-forward validate.
2. **EMA-Crossover** (classic 12/26 or sweep similar ranges).
3. **Single-indicator rules**  
   • RSI(14): enter when RSI<30, exit when RSI>70  
   • Bollinger breakout: close crosses upper/lower band  
   • ATR-based stop-and-reverse.
4. **Rule ensembles**  
   Combine two or three signals with AND/OR logic; weight signals by their
   recent Sharpe.
5. **Tabular ML**  
   Build a feature table (past returns, volume %, TA-Lib indicators, hour-of-day),
   train XGBoost / LightGBM classifier to predict next-hour direction.  
   Trade long when prob_up > 0.55, flat otherwise.
6. **Online / adaptive model**  
   River (online logistic regression) that updates each bar.
7. **Reinforcement learning (optional, time-permitting)**  
   PPO agent that decides position sizing; reward = risk-adjusted returns.

## 6. VALIDATION RULE
- Inside development.csv, use walk-forward evaluation:
    • Train (or pick parameters) on 2017-2021  
    • Test on 2022  
    • Slide window; repeat for 2023 and 2024.  
- Choose parameters that have the best **average test Sharpe** across folds.
- **Freeze** parameters before final submission.

## 7. SUBMISSION READINESS
- Every strategy class sits in `src/strategies/*.py`.  
- No notebook logic, no `holdout.csv` access anywhere.  
- `pytest` passes.  
- Update top-level README with:
    • Strategy name  
    • Author (Korak)  
    • Expected Sharpe on development set  
    • Brief description of idea.

## 8. TROUBLESHOOTING TIPS
- Sharpe unexpectedly low? -> verify `fee` value passed to `run_backtest`.
- Index errors in metrics? -> verify entry_idx and exit_idx within equity length.
- Slow run? -> check loops, convert to vectorised operations.
- CI leak warning? -> search repo for 'holdout'.

# Commit small, test often.  Good luck!
# ================================================================