# R76 Backtesting Framework

## Overview

This is a complete backtesting framework for the **R76 MA5/BB Cross Strategy** that mirrors the real trading logic in `r003_trade_live_execute.py`.

### Key Features

- **Unified Strategy**: `shared/` modules contain all strategy logic shared between real trading and backtesting
- **r76 Logic Extracted**: All indicator calculations, buy/sell conditions are extracted from r76
- **3-Minute Bar Replay**: Backtester replays historical 3-minute OHLCV data
- **Risk Management**: Implements take-profit, stop-loss, and trailing-stop
- **Session-Aware**: Respects trading session rules (regular, morning NXT, afternoon NXT)
- **Modular Design**: Easy to modify strategy parameters without touching core logic

## Architecture

```
xgraph/auto_trading/
├── shared/                          # ← Shared logic (r76 + backtester)
│   ├── __init__.py
│   ├── config.py                    # All parameters/thresholds
│   ├── indicators.py                # MA5, BB, RSI, Stoch, Williams, MACD, ADX
│   └── strategy.py                  # Buy/sell conditions
│
├── backtester/                      # ← Backtesting framework
│   ├── __init__.py
│   ├── data_loader.py               # CSV data loading
│   ├── portfolio.py                 # Position/cash management
│   └── backtest.py                  # Main simulation loop
│
├── r003_trade_live_execute.py       # Real trading (uses shared)
├── run_backtest.py                  # Backtest runner
└── data/                            # Historical OHLCV data
    ├── 20260422/
    │   ├── 003490.csv
    │   ├── 018880.csv
    │   └── ...
    └── ...
```

## Usage

### Basic Backtest

```bash
# Test on 2026-04-22 using all available codes
python3 run_backtest.py --date 20260422
```

### With Specific Codes

```bash
# Test only 003490 and 018880
python3 run_backtest.py --date 20260422 --codes 003490 018880
```

### Custom Capital

```bash
# Start with 5M KRW instead of default 10M
python3 run_backtest.py --date 20260422 --capital 5000000
```

### Combined

```bash
python3 run_backtest.py --date 20260422 --codes 003490 018880 --capital 5000000
```

## Data Format

Place 3-minute bar CSV files in `data/YYYYMMDD/` directories:

**Required columns**: `open`, `high`, `low`, `close`, `volume`

Example structure:
```
data/20260422/003490.csv
data/20260422/018880.csv
...
```

Auto-detected time columns: `timestamp`, `datetime`, `time`, `date`

## Configuration

Edit `shared/config.py` to change:

- **Indicator periods**: `MA_PERIOD`, `RSI_PERIOD`, `MACD_FAST`, `ADX_PERIOD`, etc.
- **Trading parameters**: `MAX_ORDER_AMOUNT_KRW`, `TAKE_PROFIT_PERCENT`, `STOP_LOSS_PERCENT`
- **Thresholds**: `RSI_BUY_MAX`, `STOCH_BUY_MAX`, `ADX_MIN_TREND`, etc.
- **Volume ratios**: `VOLUME_RATIO_OPEN`, `VOLUME_RATIO_MIDDAY`, etc.

## Output

Backtest produces:

1. **Console Log**: Trade-by-trade details and summary statistics
2. **Trade Records**: List of all BUY/SELL actions with P&L
3. **Summary Stats**:
   - Total P&L (KRW & %)
   - Realized P&L
   - Win rate
   - Number of traded codes

## Strategy Logic

### Buy Signal

Buy when:
1. **MA5 crosses above BB Middle** (core signal)
2. **BB Middle and MA5 are rising**
3. **Close > Open** (bullish candle)
4. **Not overbought** (Stoch < 80, Williams > -20, BB proximity < 85%)
5. **Sufficient volume** (session-dependent ratio)
6. **Strong trend** (ADX >= 20)
7. **Support score >= 2** (Stoch, RSI, Williams, MACD indicators align)

### Sell Signal

Sell when:
1. **MA5 crosses below BB Middle** with support score >= 1, OR
2. **Multiple indicators reverse** (support score >= 2)

### Risk Management

- **Take Profit**: +3.5%
- **Stop Loss**: -1.5%
- **Trailing Stop**: -1.2% from peak
- **Trade Cooldown**: 3 minutes between trades for same code

## Modifying the Strategy

### Change Buy/Sell Logic

Edit `shared/strategy.py`:
- `check_buy_condition()`: Modify entry requirements
- `check_sell_condition()`: Modify exit requirements
- `_buy_support_score()`: Adjust indicator weights

### Change Indicator Parameters

Edit `shared/config.py`:
```python
MA_PERIOD = 5              # Change MA5 to MA10
RSI_BUY_MAX = 72.0         # Change from 72 to 80
TAKE_PROFIT_PERCENT = 0.05 # Change from 3.5% to 5%
```

### Change Risk Management

Edit `backtester/backtest.py` or modify `shared/config.py`:
```python
TAKE_PROFIT_PERCENT = 0.035
STOP_LOSS_PERCENT = -0.015
TRAILING_STOP_FROM_PEAK = 0.012
```

## Comparison with Real Trading (r76)

- **Shared Code**: Both use identical `shared/` modules
- **Data Source**: r76 gets real-time 3-minute bars → backtester replays historical 3-minute bars
- **Execution**: r76 places real orders → backtester simulates at close prices
- **Result**: Same strategy logic, different data source → validate strategy performance

## Troubleshooting

### "No data found" Error

1. Check data directory path: `xgraph/auto_trading/data/`
2. Verify date folder exists: `data/YYYYMMDD/`
3. Verify CSV files exist: `data/YYYYMMDD/XXXXXX.csv`

### No Trade Signals

Possible causes:
- ADX < 20 (weak trend)
- Support score < 2 (indicators not aligned)
- Volume filter failed (low volume)
- Stoch/Williams overbought
- BB proximity too high

Check detailed logs to see which filter triggered.

### High Slippage

Backtester assumes execution at bar close price. Real execution may vary.

## Future Enhancements

- [ ] Warm-up period from prior trading days
- [ ] Multi-symbol parallel processing
- [ ] Parameter optimization (sweep)
- [ ] Monte Carlo simulation
- [ ] Drawdown analysis
- [ ] Sharpe ratio calculation
- [ ] Walk-forward testing
- [ ] Output to CSV/JSON

---

**Status**: Production-ready for backtesting. Shares code with r76 real trading.
