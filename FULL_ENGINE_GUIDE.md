# Full event-driven backtesting engine

This is the guide to the finished reusable engine. Read the main `README.md`
first if terms such as ETF, return, moving average, volatility or drawdown are
new to you.

## What changed from the simple version?

The compact version applies portfolio weights directly to daily percentage
returns. That is excellent for learning and fast research, but it skips what
happens between deciding to trade and actually owning shares.

The full engine models this sequence:

```text
new market session
→ corporate actions and cash interest
→ old orders attempt to execute
→ cash and share quantities change
→ positions are valued at the close
→ strategy sees the close
→ strategy requests target weights
→ risk manager checks those targets
→ target weights become share orders
→ orders wait for a later session
```

An order created after Monday's close cannot fill on Monday. Its earliest
possible fill is Tuesday's session. This is enforced by the broker rather than
left to the strategy author to remember.

## The components

### `events.py`

Events are plain data objects passed between parts of the system.

- `MarketBar` stores one ticker's open, high, low, close, adjusted close,
  volume, dividend and split for one session.
- `MarketEvent` groups all available bars on one date.
- `SignalEvent` contains the portfolio weights requested by a strategy.
- `Order` contains a precise buy or sell instruction.
- `FillEvent` describes what the simulated broker actually executed.
- `Position` stores quantity, average cost and realised profit or loss.
- `PortfolioSnapshot` records the account at the end of a session.

Using separate objects prevents a vague “buy signal” from being confused with
an executed trade.

### `data.py`

`MarketDataPortal` validates and exposes historical OHLCV data in chronological
order. OHLCV means open, high, low, close and volume.

It rejects duplicate dates, impossible high/low relationships, non-positive
prices and negative volume. The real-data loader requests raw prices for trade
accounting, adjusted closes for indicators, and dividends/splits for corporate
actions. The demo generator creates deterministic artificial data.

### `strategy.py`

Every strategy implements one method:

```python
on_market_close(event, adjusted_close_history)
```

It may return desired portfolio weights or no new decision. It cannot directly
change cash, shares or fill prices.

The package includes:

- `MovingAverageMomentumStrategy`: fixed short/long moving averages;
- `BuyAndHoldStrategy`: one initial equal-weight purchase;
- `WalkForwardMomentumStrategy`: periodically selects parameters using only a
  trailing training window.

### `risk.py`

`RiskManager` rejects target portfolios that violate rules such as:

- negative weights in long-only mode;
- too much money in one asset;
- gross exposure above 100%;
- an unapproved ticker;
- insufficient required cash.

The broker independently prevents purchases that the cash account cannot fund.

### `sizing.py`

The strategy requests percentages; the market trades shares. The position
sizer performs this conversion:

```text
target weight × portfolio equity
= desired monetary value

desired monetary value ÷ current price
= desired number of shares

desired shares − existing shares
= order quantity
```

By default it rounds down to whole shares. Sell orders are placed before buy
orders so sales can release cash.

### `costs.py`

`CostModel` handles three distinct frictions:

- commission, charged separately to cash;
- half of the bid-ask spread, included adversely in the fill price;
- slippage, also included adversely in the fill price.

For a buy, the simulated fill is above the reference price. For a sell, it is
below the reference price.

The command-line defaults are:

```text
commission: 1 basis point of trade value
full spread: 3 basis points
slippage:   1 basis point
```

These are transparent research assumptions, not universal facts.

### `broker.py`

`SimulatedBroker` owns the order lifecycle. It supports:

- market orders;
- limit orders;
- stop orders;
- Good-Til-Cancelled and Day duration;
- accepted, partially filled, filled, rejected and cancelled statuses;
- next-session execution;
- maximum participation in daily volume;
- cash and holdings checks;
- split adjustment of unfilled orders.

Market orders use the next available opening price before costs. Limit and stop
orders use daily open/high/low information to determine whether their condition
could have been reached. Daily bars cannot reveal the exact intraday sequence,
so this remains an approximation.

### `portfolio.py`

`Portfolio` is the accounting ledger. It tracks:

- cash;
- whole share quantities;
- average purchase cost;
- realised profit and loss;
- current market value;
- commissions, spread and slippage;
- dividend income;
- cash interest;
- total equity.

Every end-of-day snapshot must satisfy:

```text
equity = cash + sum of all position market values
```

That identity is tested automatically.

### `engine.py`

`BacktestEngine` is the chronological coordinator. It is deliberately unaware
of moving-average mathematics: any strategy implementing the interface can use
the same engine.

It produces a `BacktestResult` containing the equity curve, order lifecycle,
fills, signals, final positions and performance metrics.

### `analytics.py`

The reporting layer calculates:

- total and annualised return;
- annualised volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- Calmar ratio;
- best and worst day;
- fraction of positive days;
- number of fills;
- traded notional;
- commissions and estimated execution costs;
- monthly and yearly returns.

The risk-free rate used for the Sharpe ratio is configurable. It is separate
from the cash rate actually credited to the portfolio.

### `research.py`

This layer runs:

- metrics over named market regimes;
- every valid short/long parameter combination as an independent event-driven
  simulation;
- walk-forward parameter selection through the special strategy.

The default grid contains 8 short and 8 long windows, or 64 valid combinations.

## Walk-forward implementation

The walk-forward strategy remains in cash until it has:

- five full trading years for training; and
- enough earlier rows to warm up the longest moving average.

It then performs the following process:

1. Calculate every candidate's net training returns using only data available
   on that date.
2. Select the pair with the highest training Sharpe ratio.
3. Trade that pair through the event-driven broker.
4. Wait approximately one trading year.
5. Repeat using the newest trailing five years.

Parameter changes cause new target weights, real orders and new transaction
costs. The selection log records the training dates and chosen pair, making the
process auditable.

## Run it

From this directory:

```bash
python run_event_driven.py --demo
```

Useful options include:

```bash
python run_event_driven.py --demo --short 20 --long 100
python run_event_driven.py --demo --commission-bps 2 --spread-bps 5
python run_event_driven.py --demo --cash-rate 0.03 --risk-free-rate 0.03
python run_event_driven.py --real-data --output-dir outputs_real
```

The full research suite is intentionally slower:

```bash
python run_event_research.py --demo
python run_event_research.py --real-data --output-dir outputs_research_real
```

Install the real-data and chart dependencies first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Audit outputs

Each normal run creates separate strategy and benchmark folders containing:

- `configuration.json`: every major run assumption;
- `equity_curve.csv`: cash, holdings values, exposure and costs per day;
- `order_lifecycle.csv`: every order-state transition;
- `fills.csv`: every completed or partial execution;
- `signals.csv`: every strategy decision and reason;
- `final_positions.csv`: final quantities, values and realised P&L;
- `metrics.csv`: headline statistics;
- `monthly_returns.csv` and `yearly_returns.csv`.

The research run additionally creates:

- `parameter_grid.csv` with 64 independent results;
- `parameter_selections.csv` with each walk-forward choice;
- `walk_forward_regimes.csv`.

## Verification

Run:

```bash
python -m unittest -v
```

The tests cover:

- next-session rather than same-session fills;
- exact buy and sell cash accounting;
- spread, slippage and commission calculations;
- limit and stop conditions;
- partial fills caused by volume limits;
- Day-order expiration;
- dividends and splits;
- cash-interest compounding;
- position risk limits;
- no change to earlier trades when a future price changes;
- daily equity reconciliation;
- all requested parameter-grid combinations;
- walk-forward warm-up and timing;
- the original vectorised research functions.

## Honest scope

This is now a genuine reusable event-driven backtesting engine for long-only,
daily, single-currency strategies. It is not an exchange simulator or an
institutional multi-asset trading platform.

It intentionally does not yet model:

- short selling and margin calls;
- foreign-exchange conversion between account currencies;
- options or futures contracts;
- an intraday order book and millisecond ordering;
- taxes;
- market impact estimated from a calibrated empirical model;
- live brokerage connectivity.

Those are separate projects, not necessary features for a daily ETF momentum
study. Claiming this narrower scope is both accurate and technically credible.

## Defensible CV wording

> Built a reusable event-driven, multi-asset backtesting engine in Python,
> separating strategy signals, target sizing, risk controls, simulated order
> execution and cash/share accounting, with commissions, spread, slippage,
> volume limits and corporate actions.

Then describe the research separately:

> Applied the engine to a six-ETF momentum strategy, testing 64 parameter
> combinations and rolling walk-forward selection across distinct market
> regimes, with auditable net-of-cost performance and risk reporting.

