"""数据层 —— yfinance / Stooq 双源, GitHub Actions 上可切 AKShare (A股)."""
import io
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import requests


def _a_yf_symbol(symbol: str) -> str:
    """A股代码转 yfinance 格式: 600519 -> 600519.SS, 002594 -> 002594.SZ"""
    if symbol.startswith("6"):
        return symbol + ".SS"
    return symbol + ".SZ"


def _fetch_yfinance(symbol: str, lookback_days: int) -> pd.DataFrame:
    import yfinance as yf
    yf_symbol = _a_yf_symbol(symbol)
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    strategies = [
        lambda: yf.Ticker(yf_symbol).history(start=start_s, end=end_s, interval="1d", auto_adjust=True),
        lambda: yf.Ticker(yf_symbol).history(period="2y" if lookback_days > 365 else "1y", interval="1d", auto_adjust=True),
        lambda: yf.download(yf_symbol, start=start_s, end=end_s, interval="1d", auto_adjust=True, progress=False, threads=False),
    ]

    df = None
    last_err = None
    for attempt in range(2):
        for strat in strategies:
            try:
                candidate = strat()
                if candidate is not None and not candidate.empty:
                    df = candidate
                    break
            except Exception as e:
                last_err = e
        if df is not None and not df.empty:
            break
        time.sleep(1 + attempt * 2)

    if df is None or df.empty:
        raise RuntimeError(f"yfinance no data for {symbol} ({yf_symbol}); last_err={last_err}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change().fillna(0) * 100
    return df.tail(lookback_days).reset_index(drop=True)


def _fetch_stooq(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Stooq 兜底, A股用 .ss/.sz 后缀."""
    suffix = ".ss" if symbol.startswith("6") else ".sz"
    stooq_symbol = symbol + suffix
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)
    url = (
        f"https://stooq.com/q/d/l/?s={stooq_symbol}"
        f"&i=d&d1={start.strftime('%Y%m%d')}&d2={end.strftime('%Y%m%d')}"
    )
    resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    if "No data" in resp.text or len(resp.text) < 100:
        raise RuntimeError(f"stooq no data for {stooq_symbol}")

    df = pd.read_csv(io.StringIO(resp.text))
    df = df.rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change().fillna(0) * 100
    return df.tail(lookback_days).reset_index(drop=True)


def _fetch_akshare(symbol: str, lookback_days: int) -> pd.DataFrame:
    import akshare as ak
    end = datetime.now()
    start = end - timedelta(days=lookback_days + 30)
    df = ak.stock_zh_a_hist(
        symbol=symbol, period="daily",
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
        adjust="qfq",
    )
    if df is None or df.empty:
        raise RuntimeError(f"akshare no data for {symbol}")
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "涨跌幅": "pct_change",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.tail(lookback_days).reset_index(drop=True)


def fetch_daily(symbol: str, lookback_days: int = 365) -> pd.DataFrame:
    source = os.environ.get("DATA_SOURCE", "yfinance").lower()
    errors = []

    sources = {
        "akshare": [_fetch_akshare, _fetch_yfinance, _fetch_stooq],
        "yfinance": [_fetch_yfinance, _fetch_stooq, _fetch_akshare],
        "stooq": [_fetch_stooq, _fetch_yfinance, _fetch_akshare],
    }.get(source, [_fetch_yfinance, _fetch_stooq, _fetch_akshare])

    for fn in sources:
        try:
            df = fn(symbol, lookback_days)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError(f"all data sources failed for {symbol}: {' | '.join(errors)}")


def fetch_spot(symbol: str) -> dict | None:
    import yfinance as yf
    ticker = yf.Ticker(_a_yf_symbol(symbol))
    df = ticker.history(period="2d", interval="1d")
    if df is None or df.empty:
        return None
    last = df.iloc[-1]
    prev_close = float(df.iloc[-2]["Close"]) if len(df) >= 2 else float(last["Close"])
    price = float(last["Close"])
    return {
        "symbol": symbol, "name": None, "price": price,
        "pct_change": (price - prev_close) / prev_close * 100 if prev_close else 0,
        "high": float(last["High"]), "low": float(last["Low"]),
    }
