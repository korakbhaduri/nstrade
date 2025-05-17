import pandas as pd
from typing import Union, Type, Dict, Any

from metrics import (
    sharpe_ratio,
    total_return,
    n_trades,
    win_rate,
    max_drawdown,
    annualized_return,
    rolling_sharpe_ratio,
    unrealized_drawdown_series,
    realized_drawdown_series,
)


def _load_and_clean(df_or_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Read the csv *or* accept an already‑loaded DataFrame and return a clean
    DataFrame with the columns we need: time (datetime64[ns, UTC]), close, volume.
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        # defensive copy so we never mutate the caller's frame
        df = df_or_path.copy()

    df = df.loc[:, ["time", "close", "volumeto"]]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.rename(columns={"volumeto": "volume"})
    df = df.sort_values("time").reset_index(drop=True)
    return df


def run_backtest(
    strategy_class: Type,  # subclass of Strategy
    df_or_path: Union[str, pd.DataFrame],
    *,
    initial_capital: float = 10_000.0,
    fee: float = 0.0,  # fractional cost per *side*, e.g. 0.001 == 0.1 %
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run an **hourly**, long‑only back‑test of a single‑asset strategy.

    Parameters
    ----------
    strategy_class : class
        Sub‑class of ``Strategy``.  Must expose ``process_bar`` & ``get_signal``.
    df_or_path : str | pandas.DataFrame
        Either the path to a CSV (with columns time, close, volumeto) or an
        already‑loaded DataFrame.
    initial_capital : float, default 10_000
        Starting equity in USD.
    fee : float, default 0.0
        Fractional trading cost charged on **each** buy & sell.  Example:
        ``fee=0.001`` means 0.1 %.
    verbose : bool, default False
        If *True* print trades as they happen.

    Returns
    -------
    dict
        Metrics, full equity curve, rolling sharpe, drawdown series & trade list.
    """

    df = _load_and_clean(df_or_path)

    # ---------------------------------------------------------------------
    #  Initialise strategy & bookkeeping
    # ---------------------------------------------------------------------
    strat = strategy_class(initial_capital=initial_capital)
    if not hasattr(strat, "trades"):
        strat.trades = []  # type: ignore[attr-defined]

    equity_curve = [initial_capital]
    position = 0  # 0 = flat, 1 = long
    entry_price = None
    entry_idx = None

    # ---------------------------------------------------------------------
    #  Main bar loop (event‑driven style)
    # ---------------------------------------------------------------------
    for i, row in df.iterrows():
        bar = {"time": row["time"], "close": row["close"], "volume": row["volume"]}
        strat.process_bar(bar)
        signal = strat.get_signal()

        # ------------------ BUY ------------------
        if signal == "buy" and position == 0:
            position = 1
            entry_price = row["close"]
            entry_idx = i
            strat.position = 1  # type: ignore[attr-defined]
            # subtract entry fee immediately
            equity_curve[-1] -= fee * initial_capital
            if verbose:
                print(f"BUY: {row['time']} idx={i} price={entry_price}")

        # ------------------ SELL -----------------
        elif signal == "sell" and position == 1:
            exit_price = row["close"]
            gross_pnl = (exit_price - entry_price) / entry_price * initial_capital
            net_pnl = gross_pnl - fee * initial_capital  # exit fee

            # update equity
            equity_curve.append(equity_curve[-1] + net_pnl)

            # store trade record
            strat.trades.append(
                {
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": net_pnl,
                }
            )

            # reset state
            position = 0
            entry_price = None
            strat.position = 0  # type: ignore[attr-defined]
            if verbose:
                print(f"SELL: {row['time']} idx={i} price={exit_price}")
            continue  # equity already updated for this bar

        # ---------- mark‑to‑market for open position ----------
        if position == 1:
            equity = initial_capital + (row["close"] - entry_price) / entry_price * initial_capital
        else:
            equity = equity_curve[-1]
        equity_curve.append(equity)

    # ------------------------------------------------------------------
    #  Close any open trade at the final bar
    # ------------------------------------------------------------------
    if position == 1:
        final_price = df.iloc[-1]["close"]
        gross_pnl = (final_price - entry_price) / entry_price * initial_capital
        net_pnl = gross_pnl - fee * initial_capital  # exit fee
        equity_curve.append(equity_curve[-1] + net_pnl)
        strat.trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": len(df) - 1,
                "entry_price": entry_price,
                "exit_price": final_price,
                "pnl": net_pnl,
            }
        )

    # ------------------------------------------------------------------
    #  Metrics
    # ------------------------------------------------------------------
    equity_curve = pd.Series(equity_curve, index=range(len(equity_curve)))
    returns = equity_curve.pct_change().fillna(0).values

    results = {
        "sharpe": sharpe_ratio(returns),
        "total_return": total_return(equity_curve),
        "n_trades": n_trades(strat.trades),
        "win_rate": win_rate(strat.trades),
        "max_drawdown": max_drawdown(equity_curve),
        "annualized_return": annualized_return(equity_curve),
        "equity_curve": equity_curve,
        "trades": strat.trades,
        "rolling_sharpe": rolling_sharpe_ratio(equity_curve),
        "unrealized_drawdown": unrealized_drawdown_series(equity_curve),
        "realized_drawdown": realized_drawdown_series(equity_curve, strat.trades),
    }
    return results
