import numpy as np
from math import log, sqrt, isfinite
from scipy.stats import norm

# -------------------------- Core Constant Definition --------------------------
MULTIPLIER = 100  # Number of underlying assets per option contract (typically 100 shares for stock options)

# -------------------------- Numerical Stability Utility Function --------------------------
def _safe(val, eps=1e-12, min_val=None):
    """
    Ensure numerical calculation stability
    """
    # ===== Modified Position (optimized value handling with clip) =====
    if val is None or not isfinite(val):
        return eps
    v = float(val)
    if min_val is not None:
        v = max(v, min_val)
    return v if abs(v) >= eps else (eps if v >= 0 else -eps)

# -------------------------- Greek Calculation (Black-Scholes Model) --------------------------
def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculate option Greeks (Delta/Vega/Gamma)
    """
    S = _safe(S, min_val=1e-8)
    K = _safe(K, min_val=1e-8)
    T = max(float(T), 1e-8)
    sigma = max(float(sigma), 1e-8)
    r = float(r)

    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Nd1 = norm.cdf(d1)
    nd1 = norm.pdf(d1)

    if str(option_type).lower().startswith("c"):
        delta = Nd1
    else:
        delta = Nd1 - 1.0

    # ===== Modified Position (added standard vega definition & vega_pct) =====
    vega = S * nd1 * sqrt_T  # Standard definition (sensitivity to 100% volatility change)
    vega_pct = vega * 0.01   # Sensitivity to 1% volatility change

    gamma = nd1 / (S * sigma * sqrt_T)

    return {
        "delta": float(delta),
        "vega": float(vega),        # Standard definition
        "vega_pct": float(vega_pct),# Additional field (original习惯用法)
        "gamma": float(gamma)
    }

# -------------------------- Position Sizing Functions --------------------------
def calculate_trade_size_by_vega(target_vega_abs, option_vega_per_1pct):
    vega_per_contract = float(option_vega_per_1pct)
    if vega_per_contract <= 0 or not isfinite(vega_per_contract):
        return 0
    target = float(target_vega_abs)
    if target <= 0:
        return 0
    return int(np.ceil(target / vega_per_contract))

def size_pair_by_vega(long_leg_vega, short_leg_vega, target_gross_vega=2000.0):
    long_qty = calculate_trade_size_by_vega(target_gross_vega, abs(long_leg_vega))
    short_qty = calculate_trade_size_by_vega(target_gross_vega, abs(short_leg_vega))
    return (long_qty, short_qty)

# -------------------------- Transaction Cost Model --------------------------
def apply_transaction_cost(price_per_contract, qty_signed,
                           per_contract_fee=1.0, pct_fee=0.0002, slippage_bps=0.0005):
    qty = int(qty_signed)
    if qty == 0:
        return (float(price_per_contract), 0.0)

    side = 1 if qty > 0 else -1
    slippage = side * float(slippage_bps)
    exec_price = float(price_per_contract) * (1.0 + slippage)
    exec_price = round(exec_price, 8)

    qty_abs = abs(qty)
    # ===== Modified Position (consider contract multiplier for fees) =====
    notional = exec_price * qty_abs * MULTIPLIER
    fixed_fee = qty_abs * float(per_contract_fee)
    proportional_fee = notional * float(pct_fee)
    total_fee = round(fixed_fee + proportional_fee, 6)

    return (exec_price, total_fee)

# -------------------------- Margin Calculation Function --------------------------
def calculate_margin(option_row, qty_signed, margin_rate=0.15):
    """
    Simplified margin model (naked short selling only)
    """
    if qty_signed >= 0:
        return 0.0

    qty_abs = abs(int(qty_signed))
    underlying_price = float(option_row["underlying_price"])
    option_price = float(option_row.get("option_price", 0.0))  # ===== Modified Position (consider premium) =====

    # Simplified model: max(underlying market value × margin rate, premium × multiplier)
    margin = max(
        underlying_price * MULTIPLIER * qty_abs * float(margin_rate),
        option_price * MULTIPLIER * qty_abs
    )
    return round(max(margin, 0.0), 2)
