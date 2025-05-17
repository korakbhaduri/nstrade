import pandas as pd
from metrics import (
    sharpe_ratio, total_return, n_trades, win_rate,
    max_drawdown, annualized_return, rolling_sharpe_ratio,
    unrealized_drawdown_series, realized_drawdown_series,
)

def sma_crossover_vector(df: pd.DataFrame, fast: int, slow: int, *, fee=0.0):
    """
    Vectorised SMA crossover, long-only, full capital.
    Returns dict of metrics identical to run_backtest().
    """
    df = df.copy()
    df["fast"] = df["close"].rolling(fast).mean()
    df["slow"] = df["close"].rolling(slow).mean()
    df["signal"] = (df["fast"] > df["slow"]).astype(int)    # 1 long, 0 flat
    df["position"] = df["signal"].shift().fillna(0)

    # Hourly returns
    df["ret"] = df["close"].pct_change().fillna(0)
    df["strat_ret"] = df["position"] * df["ret"]

    # apply fees on entries & exits
    trades = df["position"].diff().abs() == 1
    df.loc[trades, "strat_ret"] -= fee

    equity_curve = (1 + df["strat_ret"]).cumprod() * 10_000

    # --- metrics (reuse functions) -----------------------------
    returns = equity_curve.pct_change().fillna(0).values
    results = {
        "sharpe": sharpe_ratio(returns),
        "total_return": total_return(equity_curve),
        "n_trades": trades.sum() // 2,
        "win_rate": win_rate([]),   # leave 0 for now
        "max_drawdown": max_drawdown(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "equity_curve": equity_curve,
        "trades": [],               # optional for plots
        "rolling_sharpe": rolling_sharpe_ratio(equity_curve),
        "unrealized_drawdown": unrealized_drawdown_series(equity_curve),
        "realized_drawdown": realized_drawdown_series(equity_curve, []),
    }
    return results
