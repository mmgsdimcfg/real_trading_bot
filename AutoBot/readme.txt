Ubuntu quick start

1) Go to repository root.
2) Create and activate virtual environment.
3) Install dependencies.
4) Run simulation script.

Commands:
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 xgraph/auto_trading/g003_trade_simulate_by_date.py --date 20260422

Optional:
python3 xgraph/auto_trading/g003_trade_simulate_by_date.py --date 20260422 --codes 003490 018880

Shortcut launcher:
bash xgraph/auto_trading/r004_run_simulation.sh 20260422 --codes 003490 018880

