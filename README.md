# Moving-average momentum research project

## Which implementation should I use?

There are now two implementations:

- `run_event_driven.py` is the main, realistic engine. It processes one market
  session at a time and separates signals, orders, fills, costs, cash and share
  accounting. Start here for the finished project.
- `momentum_project.py` is the original compact vectorised teaching version. It
  is useful because the core mathematics is visible in one file, but it is not
  as realistic.

The detailed guide to the full engine is in `FULL_ENGINE_GUIDE.md`.

Run the complete event-driven demonstration with:

```bash
python run_event_driven.py --demo
```

Run all 64 full-engine parameter combinations and walk-forward selection with:

```bash
python run_event_research.py --demo
```

These use artificial data unless `--real-data` is supplied. Artificial results
demonstrate software behaviour, not investment performance.

This project implements every technical claim in the “Quantitative Research
Project” section of the CV, except that the CV's exact headline results cannot
be reproduced from the CV alone. It does not specify the original tickers,
dates, parameters, costs, data vendor or calculation conventions. This program
therefore uses explicit, editable assumptions and reports whatever the chosen
data actually produce.

## Start here

Open a terminal in this folder and run:

```bash
python momentum_project.py --demo
```

Demo mode uses repeatable fake data, so it proves that the complete program
works without an internet connection. Fake results are never investment
evidence.

For actual historical ETF data, create an isolated environment and install the
four packages:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python momentum_project.py --real-data
```

On Windows, replace the activation line with `.venv\Scripts\activate`.

To run the correctness checks:

```bash
python -m unittest -v
```

## The finance ideas, from zero

### What is an ETF?

An exchange-traded fund (ETF) is one tradeable object that owns a collection of
investments. For example, SPY is designed to track a large group of US
companies. Using ETFs lets us study broad markets instead of betting on one
company.

The six defaults are SPY (large US companies), QQQ (large Nasdaq companies),
IWM (smaller US companies), EFA (developed markets outside the US/Canada), EEM
(emerging markets), and AGG (US bonds). They are examples, not recommendations.

### Price, return, and compounding

Price is the amount an ETF costs. Return measures the proportional change:

```text
daily return = today's price / yesterday's price - 1
```

If a price goes from £100 to £102, the return is `102/100 - 1 = 0.02 = 2%`.
Returns compound by multiplication. Starting with £1, a +10% day followed by a
-10% day leaves `1 × 1.10 × 0.90 = £0.99`, not £1.

“Adjusted” price data rewrite old prices to account for dividends and share
splits. Without this adjustment, part of the investor's gain can disappear from
the data or a split can look like a crash.

### Moving averages and momentum

An `N`-day simple moving average is the ordinary mean of the last `N` prices.
For prices 10, 12 and 14, the three-day average is `(10+12+14)/3 = 12`.

The strategy compares:

- a short average (default: 50 trading days), which reacts fairly quickly;
- a long average (default: 200 trading days), which reacts slowly.

When the short average is above the long average, recent prices have been
stronger than their longer history. The program calls that positive momentum
and produces signal `1` (own the ETF). Otherwise it produces signal `0` (hold
cash for that ETF's portion). This is not a prediction that must be right; it is
just a precise rule that can be tested.

### Portfolio weights

There are six ETFs, so each is allowed at most `1/6` of the money. If four have
signal 1, the strategy invests `4/6` and leaves `2/6` as cash. This avoids
borrowing and stops the portfolio from secretly taking more risk when fewer
signals are active.

The benchmark spends `1/6` of the starting money on every ETF and then leaves
the number of shares untouched. Its percentage weights naturally drift as the
six prices move. That is genuine buy-and-hold, rather than silently rebalancing
back to equal weights every day. “Outperformance” is the strategy's annualised
return minus the benchmark's annualised return.

### Why the signal is shifted one day

Today's closing price is not known until today closes. A backtest must not act
as though it knew that closing price earlier. The program calculates the signal
at today's close, shifts it one row, and lets it affect tomorrow's return. This
prevents look-ahead bias (using future information by accident).

### Transaction costs and basis points

Real trading is not perfectly free. One basis point (bp) is `0.01%`, or
`0.0001` as a decimal. The default cost is 5 bps: `5/10,000 = 0.0005 = 0.05%`
of the value traded.

Turnover is the total absolute change in weights. Moving one ETF from weight 0
to `1/6` creates turnover `1/6`; the program subtracts `(1/6) × 0.0005` from
that day's portfolio return. Results are therefore net of the assumed costs.

## What “backtesting” means

A backtest is a simulation of a precise rule on old data. On each date the
program:

1. calculates each ETF's percentage return;
2. calculates moving averages and raw signals;
3. shifts signals by one day and turns them into portfolio weights;
4. multiplies each asset return by its weight and adds the six contributions;
5. subtracts the cost caused by changing weights; and
6. compounds daily returns into a wealth curve.

Backtesting is an experiment, not proof of future profit. Historical data
contains only one realised version of history, and apparently good patterns can
occur by luck.

## Understanding every reported statistic

### Total and annualised return

Total return is final wealth minus starting wealth. Annualised return converts
the whole result into the constant yearly growth rate that would lead to the
same final value:

```text
annualised return = final wealth ** (1 / years) - 1
```

The `**` operator means “raise to a power.” Annualisation makes tests of
different lengths easier to compare.

### Volatility

Standard deviation measures how spread out daily returns are around their
average. Finance calls this volatility. The program multiplies daily standard
deviation by `sqrt(252)` because there are roughly 252 trading days per year.
The square-root rule is the standard approximation when daily changes are not
strongly dependent on one another.

### Sharpe ratio

Here the risk-free return is assumed to be zero, so:

```text
Sharpe ratio = annualised average return / annualised volatility
```

It asks how much average return was earned per unit of fluctuation. Higher is
better, but it is not a guarantee and can hide unusual extreme risks.

### Maximum drawdown

On every date, drawdown compares current wealth with its highest earlier value:

```text
drawdown = current wealth / previous peak wealth - 1
```

If wealth peaks at £1.00 and later reaches £0.80, drawdown is `0.80/1.00 - 1 =
-20%`. The reported maximum drawdown is the positive size of the worst fall,
20% in this example.

## Robustness and overfitting

Overfitting means choosing a rule that matches accidental quirks in old data
rather than a repeatable relationship. A complicated strategy can resemble a
student memorising practice answers without understanding the topic.

The parameter stress test tries 8 short windows and 8 long windows, producing
64 valid combinations. A robust idea should not depend on one magical pair.
Inspect `parameter_stress_test.csv`: a broad area of reasonable results is more
reassuring than one isolated winner.

The three regime rows ask how the fixed strategy behaved during:

- the global financial crisis;
- the long expansion after it; and
- the pandemic/inflation shock.

The dates are researcher choices, not laws of nature. They are editable in
`REGIMES` near the top of the program.

## Walk-forward validation

“Out of sample” means data not used to choose the parameters. Each fold does
this:

```text
five past years: choose the pair with the best training Sharpe
next one year:   freeze that pair and measure it on unseen data
move one year forward and repeat
```

Only the one-year test pieces are joined into the final walk-forward result.
When a new fold chooses different parameters, the live simulation calculates
the change from the previous fold's final weights and charges the resulting
transaction cost.
This imitates what a researcher could have known as time passed. It is much
more honest than choosing parameters on the entire history and judging them on
that same history.

It still has limitations: repeatedly trying many research ideas can overfit the
whole research process, and the future can differ structurally from the past.

## Map of the Python program

The file is deliberately divided into numbered sections.

- `Settings` is a frozen data class: a simple container whose values cannot be
  changed accidentally after creation. Type hints such as `int` document the
  expected kind of value; Python does not use them as finance logic.
- `download_adjusted_prices` imports `yfinance` only when real data is wanted.
- `make_demo_prices` uses a seeded random-number generator. The same seed gives
  the same fake history on every run.
- `validate_prices` sorts dates, fills isolated missing values, rejects bad
  rows and requires positive prices.
- `moving_average_signal` uses pandas `rolling(...).mean()` to perform the same
  average independently down every ETF column.
- `performance_statistics` calculates compounding, annualisation, volatility,
  Sharpe and drawdown.
- `backtest` creates shifted weights, turnover, costs, returns and wealth for
  both strategy and benchmark.
- `regime_analysis` uses date slices such as `series.loc[start:end]`.
- `parameter_stress_test` uses nested loops to test every short/long pair.
- `walk_forward_validation` uses rolling integer row slices for training and
  testing. A dictionary stores each parameter pair's return series so the
  expensive backtest is not repeated inside every fold.
- `save_chart` imports plotting code only when needed.
- `main` connects the steps and saves the outputs.
- `if __name__ == "__main__"` means “call `main()` only when this file is run
  directly,” so tests can import its functions without running the experiment.

Useful Python syntax in the program:

- `def` creates a function; inputs are inside parentheses and `return` sends a
  result back.
- Indentation groups the statements belonging to a function, loop or `if`.
- `=` assigns a value; `==` compares two values.
- `.` accesses an object's method or attribute, such as `prices.mean()`.
- Square brackets select a list item, dictionary item, DataFrame column or
  slice.
- A tuple uses parentheses and is an ordered collection; a dictionary uses
  `{key: value}` pairs.
- `for` repeats work; `if` selects work conditionally.
- `raise` stops with a clear error rather than silently producing nonsense.
- `try/except` handles an expected failure, here a missing optional package.
- `f"...{value}..."` inserts a value into formatted text.
- `None` means no value; `np.nan` is the numerical “not available” marker.

## Output files

The chosen output folder receives:

- `prices.csv`: input adjusted prices;
- `daily_backtest.csv`: returns, wealth, turnover and costs on every date;
- `regime_results.csv`: statistics in the three market regimes;
- `parameter_stress_test.csv`: all 64 parameter pairs, best Sharpe first;
- `walk_forward_folds.csv`: the parameter choice and test result per fold;
- `walk_forward_returns.csv`: only the joined unseen test returns; and
- `wealth_chart.png`: strategy versus benchmark growth, if matplotlib exists.

CSV is plain tabular text. It opens in Excel or Google Sheets and keeps the
calculation trail inspectable.

## Sensible experiments

Change one assumption at a time and predict what should happen before running:

```bash
python momentum_project.py --demo --short 20 --long 100
python momentum_project.py --demo --cost-bps 20
python momentum_project.py --real-data --output-dir outputs_real
```

Questions worth answering are: Does a higher cost hurt faster strategies more?
Are nearby window pairs similar? Does success come from only one regime? Are
walk-forward results much weaker than the full-history backtest? Those questions
matter more than finding the prettiest headline number.
