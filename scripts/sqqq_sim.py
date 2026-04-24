#!/usr/bin/env python3
"""
SQQQ Entry Strategy Simulation

Compares 4 strategies:
  A) Baseline:       TQQQ-only, go flat when gates fail (current production behavior)
  B) KNN-60:         + SQQQ when k-NN SHORT ≥ 0.60 (current config)
  C) KNN-55:         + SQQQ when k-NN SHORT ≥ 0.55 (lowered threshold)
  D) Trend-override: + SQQQ when QQQ bearish trend even without strong k-NN

Uses gate-based TQQQ entry (matching production) + walk-forward k-NN.
"""

import sys
sys.path.insert(0, "/home/ubuntu/leveraged-etf")

import numpy as np
from db.models import get_connection
from strategy.regime import detect_regime, get_regime_target_pct
from strategy.signals import calculate_momentum, calculate_realized_vol, classify_vol_regime, get_vol_adjustment
from strategy.sizing import check_sideways, check_overextended, check_consecutive_down_days, check_rsi_overbought
from config import LEVERAGE_CONFIG

# ── Load data ────────────────────────────────────────────────────────────────

conn = get_connection()

def load_cached(sym):
    rows = conn.execute(
        "SELECT date, open, high, low, close, volume FROM bar_cache WHERE symbol=? ORDER BY date",
        (sym,),
    ).fetchall()
    return [dict(r) for r in rows]

qqq_bars = load_cached("QQQ")
tqqq_bars = load_cached("TQQQ")
sqqq_bars = load_cached("SQQQ")
cross_asset_bars = {}
for sym in ("TLT", "GLD", "IWM"):
    cross_asset_bars[sym] = load_cached(sym)

import json
from pathlib import Path
vix_path = Path("/home/ubuntu/leveraged-etf/data/vix_cache.json")
vix_by_date = {}
if vix_path.exists():
    with open(vix_path) as f:
        vix_by_date = json.load(f)

conn.close()

print(f"Data loaded: QQQ={len(qqq_bars)} TQQQ={len(tqqq_bars)} SQQQ={len(sqqq_bars)} bars")

qqq_by_date = {b["date"]: b for b in qqq_bars}
tqqq_by_date = {b["date"]: b for b in tqqq_bars}
sqqq_by_date = {b["date"]: b for b in sqqq_bars}

# ── Adjust SQQQ for reverse splits ──────────────────────────────────────────
# Detect splits: any day with >2x price change from previous close
# Normalize all prices to post-split basis (multiply pre-split prices by ratio)
sqqq_sorted = sorted(sqqq_bars, key=lambda b: b["date"])
splits = []
for i in range(1, len(sqqq_sorted)):
    ratio = sqqq_sorted[i]["close"] / sqqq_sorted[i-1]["close"]
    if ratio > 2:
        split_ratio = round(ratio)
        splits.append((sqqq_sorted[i]["date"], split_ratio))
        print(f"  SQQQ reverse split detected: {sqqq_sorted[i]['date']} (approx {split_ratio}:1)")

# Apply cumulative adjustment: multiply all pre-split prices forward
# so all prices are on the same (most recent) basis
cumulative_factor = 1.0
# Process splits from latest to earliest
for split_date, ratio in reversed(splits):
    cumulative_factor = 1.0
    for split_dt, split_r in splits:
        if split_dt > split_date:
            pass  # already on correct basis
    # Simpler approach: just walk forward and track the cumulative factor
    pass

# Cleaner approach: walk through sorted bars, track cumulative split factor
adj_factor = 1.0
for split_date, ratio in splits:
    adj_factor *= ratio  # total factor from earliest to now

# Now apply: bars before first split get full factor, between splits get partial, etc.
current_factor = adj_factor
split_idx = 0
for bar in sqqq_sorted:
    if split_idx < len(splits) and bar["date"] >= splits[split_idx][0]:
        current_factor /= splits[split_idx][1]
        split_idx += 1
    if current_factor != 1.0:
        for key in ("open", "high", "low", "close"):
            bar[key] *= current_factor

# Rebuild lookup after adjustment
sqqq_by_date = {b["date"]: b for b in sqqq_sorted}
print(f"  SQQQ prices adjusted ({len(splits)} splits, final factor check: "
      f"first={sqqq_sorted[0]['close']:.2f}, last={sqqq_sorted[-1]['close']:.2f})")

common_dates = sorted(set(qqq_by_date) & set(tqqq_by_date) & set(sqqq_by_date))
print(f"Common dates: {len(common_dates)} ({common_dates[0]} to {common_dates[-1]})")

qqq_aligned = [qqq_by_date[d] for d in common_dates]

# ── Train k-NN (walk-forward) ────────────────────────────────────────────────

from strategy.knn_signal import KNNSignal, FeatureCalculator

KNN_TRAIN_CUTOFF = 400
knn_model = KNNSignal(n_neighbors=LEVERAGE_CONFIG.get("knn_neighbors", 7))

if not knn_model.fit_from_bars(
    qqq_aligned[:KNN_TRAIN_CUTOFF + 1],
    vix_by_date=vix_by_date,
    cross_asset_bars=cross_asset_bars,
):
    print("ERROR: k-NN training failed")
    sys.exit(1)

print(f"k-NN trained on {knn_model.training_samples} samples")

# Precompute k-NN predictions
knn_predictions = {}
for i in range(KNN_TRAIN_CUTOFF, len(common_dates)):
    features = FeatureCalculator.compute_features(
        qqq_aligned, i, vix_by_date=vix_by_date, cross_asset_bars=cross_asset_bars,
    )
    if features is None:
        continue
    X_scaled = knn_model.scaler.transform(features.reshape(1, -1))
    probs = knn_model.model.predict_proba(X_scaled)[0]
    p_up = float(probs[1]) if len(probs) > 1 else 0.5
    p_down = float(probs[0])
    confidence = max(p_up, p_down)
    if confidence < knn_model.min_confidence:
        direction = "FLAT"
    else:
        direction = "LONG" if p_up > p_down else "SHORT"
    knn_predictions[common_dates[i]] = {
        "direction": direction, "confidence": confidence,
        "p_down": p_down, "p_up": p_up,
    }

print(f"k-NN predictions: {len(knn_predictions)}")

# ── Config constants ─────────────────────────────────────────────────────────

WARMUP = 250
INITIAL_CAPITAL = 100_000.0
MAX_PCT = LEVERAGE_CONFIG["max_portfolio_pct"]
MIN_TRADE = LEVERAGE_CONFIG["min_trade_value"]
MIN_MOM = LEVERAGE_CONFIG["min_momentum_score"]
MEAN_REV = LEVERAGE_CONFIG["mean_reversion_threshold"]
SQQQ_MAX_PCT = LEVERAGE_CONFIG.get("sqqq_max_position_pct", 0.40)
MIN_TREND = LEVERAGE_CONFIG["min_trend_strength"]


def run_tqqq_gates(qqq_close, sma_50, sma_250, mom_score, vol, closes, regime):
    """Simplified production gate check for TQQQ entry (skipping live-only gates)."""
    failed = []

    if regime in ("RISK_OFF", "BREAKDOWN"):
        failed.append("regime")

    # trend_strength: QQQ above SMA-50 or SMA-250 by min_trend_strength
    if sma_50 > 0 and sma_250 > 0:
        pct_50 = (qqq_close - sma_50) / sma_50
        pct_250 = (qqq_close - sma_250) / sma_250
        if pct_50 < MIN_TREND and pct_250 < MIN_TREND:
            failed.append("trend_strength")

    if mom_score < MIN_MOM:
        failed.append("momentum")

    if vol >= LEVERAGE_CONFIG["vol_high_threshold"]:
        failed.append("vol_extreme")

    if check_sideways(closes):
        failed.append("sideways")

    if check_overextended(qqq_close, sma_50):
        failed.append("overextended")

    if check_consecutive_down_days(closes):
        failed.append("consecutive_down")

    if check_rsi_overbought(closes):
        failed.append("rsi_overbought")

    return len(failed) == 0, failed


def run_simulation(name, sqqq_entry_fn):
    """Run one simulation variant with gate-based TQQQ logic."""
    cash = INITIAL_CAPITAL
    tqqq_shares = 0
    sqqq_shares = 0
    prev_regime = None
    regime_change_day = 0
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    trades = 0
    tqqq_days = 0
    sqqq_days = 0
    sqqq_trade_count = 0
    sqqq_wins = 0
    sqqq_losses = 0
    sqqq_total_pnl = 0.0
    sqqq_entry_price = None
    sqqq_entry_shares = 0

    daily = []
    sqqq_trade_log = []

    for i, dt in enumerate(common_dates):
        if i < WARMUP:
            continue

        qqq_bar = qqq_by_date[dt]
        tqqq_bar = tqqq_by_date[dt]
        sqqq_bar = sqqq_by_date[dt]

        window_dates = common_dates[max(0, i - 260):i + 1]
        closes = [qqq_by_date[d]["close"] for d in window_dates]
        if len(closes) < 250:
            continue

        qqq_close = closes[-1]
        sma_50 = float(np.mean(closes[-50:]))
        sma_250 = float(np.mean(closes[-250:]))

        # Regime with oscillation protection
        regime = detect_regime(qqq_close, sma_50, sma_250)
        if prev_regime and regime != prev_regime:
            if regime not in ("RISK_OFF", "BREAKDOWN"):
                if (i - regime_change_day) < LEVERAGE_CONFIG["min_regime_hold_days"]:
                    regime = prev_regime
        if regime != prev_regime:
            regime_change_day = i
            prev_regime = regime

        # Signals
        mom = calculate_momentum(closes)
        vol = calculate_realized_vol(closes)
        vol_regime = classify_vol_regime(vol)
        vol_adj = get_vol_adjustment(vol_regime)

        portfolio_value = cash + tqqq_shares * tqqq_bar["close"] + sqqq_shares * sqqq_bar["close"]
        allocated = portfolio_value * MAX_PCT

        # ── TQQQ gate check ──
        tqqq_gates_pass, tqqq_failed = run_tqqq_gates(
            qqq_close, sma_50, sma_250, mom["score"], vol, closes, regime
        )

        knn_pred = knn_predictions.get(dt)

        # ── SQQQ EXIT (before TQQQ entry) ──
        if sqqq_shares > 0:
            should_exit = False
            reason = ""
            if knn_pred and knn_pred["direction"] != "SHORT":
                should_exit = True
                reason = "knn_flipped"
            if regime in ("RISK_OFF", "BREAKDOWN"):
                should_exit = True
                reason = "regime"
            if tqqq_gates_pass:
                should_exit = True
                reason = "tqqq_gates_pass"

            if should_exit:
                proceeds = sqqq_shares * sqqq_bar["close"]
                pnl = (sqqq_bar["close"] - sqqq_entry_price) * sqqq_entry_shares if sqqq_entry_price else 0
                sqqq_total_pnl += pnl
                if pnl > 0:
                    sqqq_wins += 1
                else:
                    sqqq_losses += 1
                sqqq_trade_log.append({
                    "entry_date": sqqq_trade_log[-1]["entry_date"] if sqqq_trade_log and "exit_date" not in sqqq_trade_log[-1] else "?",
                    "exit_date": dt,
                    "entry_price": sqqq_entry_price,
                    "exit_price": sqqq_bar["close"],
                    "shares": sqqq_entry_shares,
                    "pnl": pnl,
                    "reason": reason,
                })
                cash += proceeds
                sqqq_shares = 0
                sqqq_entry_price = None
                sqqq_entry_shares = 0
                trades += 1
                sqqq_trade_count += 1

        # ── TQQQ logic ──
        if tqqq_gates_pass and sqqq_shares == 0:
            # Size the TQQQ position
            regime_pct = get_regime_target_pct(regime)
            target_pct = regime_pct
            if mom["score"] < MIN_MOM:
                target_pct = LEVERAGE_CONFIG["min_position_pct"]
            elif mom["score"] < 0.8:
                min_pct = LEVERAGE_CONFIG["min_position_pct"]
                scale = (mom["score"] - MIN_MOM) / (0.8 - MIN_MOM)
                target_pct = min_pct + (regime_pct - min_pct) * scale
            target_pct *= vol_adj
            if sma_50 > 0 and (qqq_close - sma_50) / sma_50 > MEAN_REV:
                target_pct *= 0.5

            target_value = allocated * target_pct
            target_shares = max(0, int(target_value / tqqq_bar["close"])) if tqqq_bar["close"] > 0 else 0
            delta = target_shares - tqqq_shares

            if abs(delta * tqqq_bar["close"]) >= MIN_TRADE:
                if delta > 0:
                    cost = delta * tqqq_bar["close"]
                    if cost <= cash:
                        tqqq_shares += delta
                        cash -= cost
                        trades += 1
                elif delta < 0:
                    proceeds = abs(delta) * tqqq_bar["close"]
                    tqqq_shares += delta
                    cash += proceeds
                    trades += 1
                    if tqqq_shares < 0:
                        cash += abs(tqqq_shares) * tqqq_bar["close"]
                        tqqq_shares = 0

        elif not tqqq_gates_pass and tqqq_shares > 0:
            # Gates failed — exit TQQQ
            proceeds = tqqq_shares * tqqq_bar["close"]
            cash += proceeds
            tqqq_shares = 0
            trades += 1

        # ── SQQQ ENTRY (only when TQQQ gates fail and flat) ──
        if (not tqqq_gates_pass and tqqq_shares == 0 and sqqq_shares == 0
                and vol_regime != "EXTREME" and regime not in ("RISK_OFF", "BREAKDOWN")):

            enter, size_pct = sqqq_entry_fn(dt, regime, mom, closes, knn_pred)
            if enter and size_pct > 0:
                sqqq_target_value = allocated * size_pct
                sqqq_target_shares = max(0, int(sqqq_target_value / sqqq_bar["close"])) if sqqq_bar["close"] > 0 else 0
                if sqqq_target_shares > 0:
                    cost = sqqq_target_shares * sqqq_bar["close"]
                    if cost <= cash and cost >= MIN_TRADE:
                        sqqq_shares = sqqq_target_shares
                        sqqq_entry_price = sqqq_bar["close"]
                        sqqq_entry_shares = sqqq_target_shares
                        cash -= cost
                        trades += 1
                        sqqq_trade_count += 1
                        sqqq_trade_log.append({"entry_date": dt, "entry_price": sqqq_bar["close"], "shares": sqqq_target_shares})

        # Track
        portfolio_value = cash + tqqq_shares * tqqq_bar["close"] + sqqq_shares * sqqq_bar["close"]
        peak = max(peak, portfolio_value)
        dd = (peak - portfolio_value) / peak * 100
        max_dd = max(max_dd, dd)
        if tqqq_shares > 0:
            tqqq_days += 1
        if sqqq_shares > 0:
            sqqq_days += 1

        daily.append({
            "date": dt, "portfolio": portfolio_value,
            "tqqq_shares": tqqq_shares, "sqqq_shares": sqqq_shares,
            "regime": regime, "dd": dd,
        })

    final_val = daily[-1]["portfolio"] if daily else INITIAL_CAPITAL
    total_return = (final_val / INITIAL_CAPITAL - 1) * 100

    start_dt = daily[0]["date"]
    end_dt = daily[-1]["date"]
    qqq_ret = (qqq_by_date[end_dt]["close"] / qqq_by_date[start_dt]["close"] - 1) * 100
    tqqq_ret = (tqqq_by_date[end_dt]["close"] / tqqq_by_date[start_dt]["close"] - 1) * 100

    return {
        "name": name, "start": start_dt, "end": end_dt,
        "total_return": total_return, "final_value": final_val,
        "max_drawdown": max_dd, "trades": trades,
        "tqqq_days": tqqq_days, "sqqq_days": sqqq_days,
        "sqqq_trades": sqqq_trade_count,
        "sqqq_wins": sqqq_wins, "sqqq_losses": sqqq_losses,
        "sqqq_total_pnl": sqqq_total_pnl,
        "total_days": len(daily),
        "qqq_buy_hold": qqq_ret, "tqqq_buy_hold": tqqq_ret,
        "daily": daily, "sqqq_trade_log": sqqq_trade_log,
    }


# ── Strategy variants ────────────────────────────────────────────────────────

def strategy_a_no_sqqq(dt, regime, mom, closes, knn_pred):
    return False, 0.0


def strategy_b_knn60(dt, regime, mom, closes, knn_pred):
    """Current config: k-NN SHORT ≥ 0.60."""
    if knn_pred is None:
        return False, 0.0
    if knn_pred["direction"] == "SHORT" and knn_pred["confidence"] >= 0.60:
        conf = knn_pred["confidence"]
        if conf >= 0.80:
            pct = SQQQ_MAX_PCT
        else:
            scale = (conf - 0.60) / (0.80 - 0.60)
            pct = SQQQ_MAX_PCT * (0.5 + 0.5 * scale)
        return True, pct
    return False, 0.0


def strategy_c_knn55(dt, regime, mom, closes, knn_pred):
    """Lowered threshold: k-NN SHORT ≥ 0.55."""
    if knn_pred is None:
        return False, 0.0
    if knn_pred["direction"] == "SHORT" and knn_pred["confidence"] >= 0.55:
        conf = knn_pred["confidence"]
        if conf >= 0.80:
            pct = SQQQ_MAX_PCT
        else:
            scale = (conf - 0.55) / (0.80 - 0.55)
            pct = SQQQ_MAX_PCT * (0.5 + 0.5 * scale)
        return True, pct
    return False, 0.0


def strategy_d_trend_override(dt, regime, mom, closes, knn_pred):
    """
    Trend-based: enter SQQQ when QQQ is in a clear downtrend.

    Triggers:
    1. k-NN SHORT ≥ 0.55 (same as C), OR
    2. QQQ > 4% below SMA-50 AND both ROC-5 and ROC-20 negative
       (bearish trend confirmation without requiring k-NN)

    Sizing: k-NN driven when available, conservative 20% of max on trend-only.
    """
    if knn_pred is None:
        return False, 0.0

    sma_50 = float(np.mean(closes[-50:]))
    qqq_close = closes[-1]
    pct_below = (qqq_close - sma_50) / sma_50 if sma_50 > 0 else 0

    # Path 1: k-NN SHORT with lowered threshold
    if knn_pred["direction"] == "SHORT" and knn_pred["confidence"] >= 0.55:
        conf = knn_pred["confidence"]
        if conf >= 0.80:
            pct = SQQQ_MAX_PCT
        else:
            scale = (conf - 0.55) / (0.80 - 0.55)
            pct = SQQQ_MAX_PCT * (0.5 + 0.5 * scale)
        return True, pct

    # Path 2: Trend override — QQQ deeply below SMA-50 with negative momentum
    if (pct_below < -0.04
            and mom["roc_fast"] < -0.01
            and mom["roc_slow"] < -0.01):
        return True, SQQQ_MAX_PCT * 0.20

    return False, 0.0


# Also test a more aggressive trend variant
def strategy_e_trend_aggressive(dt, regime, mom, closes, knn_pred):
    """
    More aggressive trend-based: enter SQQQ earlier in downtrends.

    Triggers:
    1. k-NN SHORT ≥ 0.55, OR
    2. QQQ > 3% below SMA-50 AND ROC-20 < -2%

    Sizing: larger allocation on trend entries (30% of max).
    """
    if knn_pred is None:
        return False, 0.0

    sma_50 = float(np.mean(closes[-50:]))
    qqq_close = closes[-1]
    pct_below = (qqq_close - sma_50) / sma_50 if sma_50 > 0 else 0

    # Path 1: k-NN
    if knn_pred["direction"] == "SHORT" and knn_pred["confidence"] >= 0.55:
        conf = knn_pred["confidence"]
        if conf >= 0.80:
            pct = SQQQ_MAX_PCT
        else:
            scale = (conf - 0.55) / (0.80 - 0.55)
            pct = SQQQ_MAX_PCT * (0.5 + 0.5 * scale)
        return True, pct

    # Path 2: Earlier trend entry
    if pct_below < -0.03 and mom["roc_slow"] < -0.02:
        return True, SQQQ_MAX_PCT * 0.30

    return False, 0.0


# ── Run all strategies ───────────────────────────────────────────────────────

strategies = [
    ("A) No SQQQ (baseline)", strategy_a_no_sqqq),
    ("B) KNN SHORT >= 0.60", strategy_b_knn60),
    ("C) KNN SHORT >= 0.55", strategy_c_knn55),
    ("D) Trend conservative", strategy_d_trend_override),
    ("E) Trend aggressive", strategy_e_trend_aggressive),
]

results = []
for name, fn in strategies:
    r = run_simulation(name, fn)
    results.append(r)

# ── Print results ────────────────────────────────────────────────────────────

print(f"\n{'='*90}")
print(f"  SQQQ ENTRY STRATEGY COMPARISON")
print(f"  Period: {results[0]['start']} to {results[0]['end']} ({results[0]['total_days']} trading days)")
print(f"{'='*90}")
print(f"\n  Benchmarks:")
print(f"    QQQ Buy & Hold:  {results[0]['qqq_buy_hold']:+.1f}%")
print(f"    TQQQ Buy & Hold: {results[0]['tqqq_buy_hold']:+.1f}%")

print(f"\n  {'Strategy':<28} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'TQQQ d':>7} {'SQQQ d':>7} {'SQQQ#':>6} {'W/L':>7} {'SQQQ P&L':>10}")
print(f"  {'-'*28} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*7} {'-'*10}")

for r in results:
    wl = f"{r['sqqq_wins']}/{r['sqqq_losses']}" if r['sqqq_trades'] > 0 else "-"
    pnl = f"${r['sqqq_total_pnl']:+,.0f}" if r['sqqq_trades'] > 0 else "-"
    print(f"  {r['name']:<28} {r['total_return']:>+7.1f}% {r['max_drawdown']:>7.1f}% {r['trades']:>7} {r['tqqq_days']:>7} {r['sqqq_days']:>7} {r['sqqq_trades']:>6} {wl:>7} {pnl:>10}")

# ── Monthly comparison ───────────────────────────────────────────────────────

from collections import defaultdict

print(f"\n{'='*90}")
print(f"  MONTHLY RETURNS — ALL STRATEGIES")
print(f"{'='*90}")

header = f"  {'Month':<10}"
for r in results:
    label = r["name"].split(")")[0] + ")"
    header += f" {label:>10}"
print(header)
print(f"  {'-'*10}" + f" {'-'*10}" * len(results))

monthly_data = []
for r in results:
    by_month = defaultdict(list)
    for d in r["daily"]:
        by_month[d["date"][:7]].append(d["portfolio"])
    monthly_data.append(by_month)

all_months = sorted(set().union(*[m.keys() for m in monthly_data]))
prev_vals = [INITIAL_CAPITAL] * len(results)

for month in all_months:
    line = f"  {month:<10}"
    for j, md in enumerate(monthly_data):
        if month in md:
            end_val = md[month][-1]
            ret = (end_val / prev_vals[j] - 1) * 100
            prev_vals[j] = end_val
            line += f" {ret:>+9.1f}%"
        else:
            line += f" {'N/A':>10}"
    print(line)

# ── Trade log for each SQQQ strategy ─────────────────────────────────────────

for r in results[1:]:
    if not r["sqqq_trade_log"]:
        continue

    print(f"\n{'='*90}")
    print(f"  SQQQ TRADE LOG — {r['name']}")
    print(f"{'='*90}")
    print(f"  {'Entry':<12} {'Exit':<12} {'Days':>5} {'Entry$':>8} {'Exit$':>8} {'Shares':>7} {'P&L':>9} {'Reason':<15}")
    print(f"  {'-'*12} {'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*9} {'-'*15}")

    # Pair up entries and exits
    entries = [t for t in r["sqqq_trade_log"] if "exit_date" not in t or "entry_date" in t]
    i = 0
    while i < len(r["sqqq_trade_log"]):
        t = r["sqqq_trade_log"][i]
        if "exit_date" not in t and i + 1 < len(r["sqqq_trade_log"]):
            # Entry record, next should be exit
            entry = t
            exit_t = r["sqqq_trade_log"][i + 1]
            if "exit_date" in exit_t:
                days_held = sum(1 for d in r["daily"]
                              if d["date"] >= entry["entry_date"]
                              and d["date"] <= exit_t["exit_date"]
                              and d["sqqq_shares"] > 0)
                print(f"  {entry['entry_date']:<12} {exit_t['exit_date']:<12} {days_held:>5} "
                      f"${entry['entry_price']:>7.2f} ${exit_t['exit_price']:>7.2f} "
                      f"{entry['shares']:>7} ${exit_t['pnl']:>+8.0f} {exit_t['reason']:<15}")
                i += 2
                continue
        i += 1

print(f"\n{'='*90}")
