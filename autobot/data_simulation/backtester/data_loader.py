# -*- coding: utf-8 -*-
"""Data loading and preprocessing for backtesting."""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage OHLCV data for backtesting."""

    def __init__(self, data_root: Path):
        """Initialize DataLoader.
        
        Args:
            data_root: Root directory containing date subdirectories (YYYYMMDD format)
        """
        self.data_root = Path(data_root)

    def load_csv(self, csv_path: Path) -> pd.DataFrame | None:
        """Load single CSV file.
        
        Args:
            csv_path: Path to CSV file
        
        Returns:
            DataFrame with normalized columns or None if failed
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning(f"Failed to load CSV {csv_path}: {exc}")
            return None

        if df.empty:
            return None

        # Detect time column
        time_col = None
        for col in ("timestamp", "datetime", "Time", "time", "date", "Date"):
            if col in df.columns:
                time_col = col
                break

        # Normalize to datetime index
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.iloc[:, 1:]

        # Ensure required columns
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            logger.warning(f"Missing required columns in {csv_path}")
            return None

        df = df[~df.index.isna()].sort_index()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close"])
        return df if not df.empty else None

    def load_date_folder(self, date_str: str) -> dict[str, pd.DataFrame]:
        """Load all CSV files from a date folder.
        
        Args:
            date_str: Date in YYYYMMDD format
        
        Returns:
            Dict of {code: DataFrame}
        """
        date_dir = self.data_root / date_str
        if not date_dir.is_dir():
            logger.error(f"Date directory not found: {date_dir}")
            return {}

        result = {}
        for csv_file in sorted(date_dir.glob("*.csv")):
            code = csv_file.stem.zfill(6)
            df = self.load_csv(csv_file)
            if df is not None and not df.empty:
                result[code] = df
                logger.info(f"Loaded {code}: {len(df)} bars")

        return result

    def find_prior_trading_day_csv(self, as_of: date, code: str, max_days: int = 10) -> Path | None:
        """Find prior trading day CSV for warmup period.
        
        Args:
            as_of: Target date
            code: Stock code
            max_days: Maximum days back to search
        
        Returns:
            Path to found CSV or None
        """
        cursor = as_of - timedelta(days=1)
        for _ in range(max_days):
            candidate = self.data_root / cursor.strftime("%Y%m%d") / f"{code}.csv"
            if candidate.is_file():
                return candidate
            cursor -= timedelta(days=1)
        return None

    def build_simulation_frame(
        self,
        date_str: str,
        code: str,
        warmup_bars: int = 30,
        max_warmup_days: int = 10,
    ) -> pd.DataFrame | None:
        """Build complete frame with warmup period.
        
        Args:
            date_str: Target simulation date (YYYYMMDD)
            code: Stock code
            warmup_bars: Number of bars to load from prior day
            max_warmup_days: Max days to search for prior data
        
        Returns:
            Combined DataFrame or None if failed
        """
        date_dir = self.data_root / date_str
        today_path = date_dir / f"{code}.csv"

        if not today_path.is_file():
            return None

        today = self.load_csv(today_path)
        if today is None or today.empty:
            return None

        # Try to load prior day for warmup
        target_date = datetime.strptime(date_str, "%Y%m%d").date()
        prior_path = self.find_prior_trading_day_csv(target_date, code, max_warmup_days)

        if prior_path is not None:
            prior = self.load_csv(prior_path)
            if prior is not None and not prior.empty:
                # Concat and remove duplicates
                merged = pd.concat([prior.tail(warmup_bars), today])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                return merged

        return today
