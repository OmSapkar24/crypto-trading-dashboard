"""
Crypto Trading Dashboard Pipeline
- Data ingestion from public crypto APIs
- ETL and feature engineering
- Trend analysis and simple signals
- Interactive Plotly Dash dashboard

Note: This script is self-contained and uses free, unauthenticated endpoints when possible.
"""
from __future__ import annotations

import os
import time
import math
import json
import textwrap
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import requests

import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_SYMBOLS = [
    "bitcoin",
    "ethereum",
    "solana",
    "cardano",
    "binancecoin",
]

DEFAULT_VS_CURRENCY = "usd"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

# -----------------------------
# Utilities
# -----------------------------

def to_dt(ts: int | float) -> datetime:
    return datetime.fromtimestamp(ts / 1000 if ts > 10**12 else ts, tz=timezone.utc)


def pct_change(a: float, b: float) -> float:
    if b == 0 or b is None or a is None:
        return np.nan
    return (a - b) / b * 100.0


# -----------------------------
# Data Ingestion
# -----------------------------

def fetch_market_chart(symbol: str, vs_currency: str = DEFAULT_VS_CURRENCY, days: int = 90) -> pd.DataFrame:
    """Fetch OHLC-like time series using CoinGecko market_chart endpoint.
    Falls back with retry if rate-limited.
    """
    url = f"{COINGECKO_BASE}/coins/{symbol}/market_chart"
    params = {"vs_currency": vs_currency, "days": days, "interval": "hourly"}
    for attempt in range(3):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            df = pd.DataFrame(data.get("prices", []), columns=["ts", "price"])  # [ms, price]
            vol = pd.DataFrame(data.get("total_volumes", []), columns=["ts", "volume"])
            mcap = pd.DataFrame(data.get("market_caps", []), columns=["ts", "market_cap"])
            df = df.merge(vol, on="ts", how="left").merge(mcap, on="ts", how="left")
            df["symbol"] = symbol
            df["datetime"] = df["ts"].apply(to_dt)
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        # rate limit or transient error
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch data for {symbol}: {r.status_code} {r.text[:200]}")


def ingest_all(symbols: List[str], vs_currency: str, days: int = 90) -> pd.DataFrame:
    frames = []
    for s in symbols:
        try:
            frames.append(fetch_market_chart(s, vs_currency, days))
        except Exception as e:
            print(f"Ingest error for {s}: {e}")
    if not frames:
        return pd.DataFrame(columns=["datetime", "price", "volume", "market_cap", "symbol"])    
    return pd.concat(frames, ignore_index=True)


# -----------------------------
# ETL / Feature Engineering
# -----------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out = out.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    def grp_apply(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ret_1h"] = g["price"].pct_change() * 100
        g["ret_24h"] = g["price"].pct_change(24) * 100
        g["ret_7d"] = g["price"].pct_change(24 * 7) * 100
        
        # Moving averages
        for w in [7, 24, 72, 168]:
            g[f"sma_{w}h"] = g["price"].rolling(w).mean()
        # Exponential moving averages
        for w in [12, 26, 50, 200]:
            g[f"ema_{w}"] = g["price"].ewm(span=w, adjust=False).mean()

        # Volatility (rolling std of hourly returns)
        g["vol_24h"] = g["ret_1h"].rolling(24).std()
        g["vol_7d"] = g["ret_1h"].rolling(24 * 7).std()

        # RSI (14)
        delta = g["price"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(span=14, adjust=False).mean()
        roll_down = down.ewm(span=14, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-9)
        g["rsi14"] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = g["price"].ewm(span=12, adjust=False).mean()
        ema26 = g["price"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        g["macd"] = macd
        g["macd_signal"] = signal
        g["macd_hist"] = macd - signal

        # Volume z-score (24h)
        g["vol_z24"] = (g["volume"] - g["volume"].rolling(24).mean()) / (g["volume"].rolling(24).std() + 1e-9)

        return g

    out = out.groupby("symbol", group_keys=False).apply(grp_apply)
    return out


# -----------------------------
# Trend Analysis / Simple Signals
# -----------------------------

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    last = df.sort_values("datetime").groupby("symbol").tail(1).copy()
    last["trend"] = np.select(
        [last["ema_12"] > last["ema_26"], last["ema_12"] < last["ema_26"]],
        ["bullish", "bearish"],
        default="neutral",
    )
    last["rsi_state"] = pd.cut(
        last["rsi14"], bins=[-np.inf, 30, 70, np.inf], labels=["oversold", "neutral", "overbought"]
    )
    last["signal"] = np.select(
        [
            (last["macd_hist"] > 0) & (last["trend"] == "bullish") & (last["rsi_state"] != "overbought"),
            (last["macd_hist"] < 0) & (last["trend"] == "bearish") & (last["rsi_state"] != "oversold"),
        ],
        ["potential buy", "potential sell"],
        default="hold",
    )
    return last


# -----------------------------
# Dashboard
# -----------------------------

def build_dashboard(df: pd.DataFrame) -> Dash:
    app = Dash(__name__)
    symbols = sorted(df["symbol"].unique().tolist()) if not df.empty else DEFAULT_SYMBOLS

    app.layout = html.Div([
        html.H2("Crypto Trading Dashboard"),
        html.Div([
            html.Label("Symbols"),
            dcc.Dropdown(
                id="symbols-dd",
                options=[{"label": s.title(), "value": s} for s in symbols],
                value=symbols[:3],
                multi=True,
            ),
            html.Label("History (days)"),
            dcc.Slider(id="days-slider", min=7, max=365, value=90, step=1,
                       marks={7: "7", 30: "30", 90: "90", 180: "180", 365: "365"}),
            html.Button("Refresh Data", id="refresh-btn"),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"}),
        html.Br(),
        dcc.Tabs(id="tabs", value="prices", children=[
            dcc.Tab(label="Prices", value="prices"),
            dcc.Tab(label="Indicators", value="indicators"),
            dcc.Tab(label="Signals", value="signals"),
            dcc.Tab(label="Table", value="table"),
        ]),
        html.Div(id="tab-content"),
        html.Div(id="hidden-store", style={"display": "none"}),
    ], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "20px"})

    @app.callback(
        Output("hidden-store", "children"),
        Input("refresh-btn", "n_clicks"),
        State("symbols-dd", "value"),
        State("days-slider", "value"),
        prevent_initial_call=False,
    )
    def refresh_data(n_clicks, symbols_sel, days_val):
        try:
            raw = ingest_all(symbols_sel or DEFAULT_SYMBOLS, DEFAULT_VS_CURRENCY, int(days_val or 90))
            feat = add_features(raw)
            sigs = compute_signals(feat)
            payload = {
                "raw": raw.to_json(date_format="iso", orient="records"),
                "feat": feat.to_json(date_format="iso", orient="records"),
                "sigs": sigs.to_json(date_format="iso", orient="records"),
            }
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("hidden-store", "children"),
    )
    def render_tab(tab, store_json):
        if not store_json:
            return html.Div("Loading...")
        data = json.loads(store_json)
        if "error" in data:
            return html.Div(f"Error: {data['error']}", style={"color": "red"})

        raw = pd.read_json(data["raw"], orient="records")
        feat = pd.read_json(data["feat"], orient="records")
        sigs = pd.read_json(data["sigs"], orient="records")

        if tab == "prices":
            figs = []
            for s in sorted((feat["symbol"].unique() if not feat.empty else [])):
                sub = feat[feat["symbol"] == s]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["price"], name=f"{s} price"))
                fig.add_trace(go.Scatter(x=sub["datetime"], y=sub.get("sma_24h"), name="SMA 24h"))
                fig.add_trace(go.Scatter(x=sub["datetime"], y=sub.get("sma_168h"), name="SMA 168h"))
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                figs.append(dcc.Graph(figure=fig))
            return html.Div(figs)

        if tab == "indicators":
            comps = []
            for s in sorted((feat["symbol"].unique() if not feat.empty else [])):
                sub = feat[feat["symbol"] == s]
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["macd"], name="MACD"))
                macd_fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["macd_signal"], name="Signal"))
                macd_fig.add_trace(go.Bar(x=sub["datetime"], y=sub["macd_hist"], name="Hist"))
                macd_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))

                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=sub["datetime"], y=sub["rsi14"], name="RSI"))
                rsi_fig.add_hrect(y0=30, y1=70, fillcolor="lightgray", opacity=0.2, line_width=0)
                rsi_fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))

                comps.append(html.Div([
                    html.H4(s.title()),
                    dcc.Graph(figure=macd_fig),
                    dcc.Graph(figure=rsi_fig),
                ]))
            return html.Div(comps)

        if tab == "signals":
            cols = ["symbol", "price", "ret_24h", "vol_24h", "rsi14", "trend", "signal"]
            merged = feat.sort_values("datetime").groupby("symbol").tail(1).merge(
                sigs[["symbol", "trend", "signal"]], on="symbol", how="left", suffixes=("", "_s")
            )
            merged = merged[cols]
            merged = merged.sort_values("signal")
            return dash_table.DataTable(
                data=merged.round(3).to_dict("records"),
                columns=[{"name": c, "id": c} for c in merged.columns],
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
            )

        # table
        latest = feat.sort_values(["symbol", "datetime"]).groupby("symbol").tail(1)
        latest["ret_7d"] = latest["ret_7d"].round(2)
        latest["ret_24h"] = latest["ret_24h"].round(2)
        latest["vol_24h"] = latest["vol_24h"].round(2)
        return dash_table.DataTable(
            data=latest.round(4).to_dict("records"),
            columns=[{"name": c, "id": c} for c in latest.columns],
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_table={"overflowX": "auto"},
        )

    return app


# -----------------------------
# Main for local run
# -----------------------------
if __name__ == "__main__":
    # Initial ingest for faster first render
    raw_df = ingest_all(DEFAULT_SYMBOLS, DEFAULT_VS_CURRENCY, days=90)
    feat_df = add_features(raw_df)
    _ = compute_signals(feat_df)
    app = build_dashboard(feat_df)
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
