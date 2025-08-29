import pandas as pd
import numpy as np
from datetime import datetime

# Import core functions from risk_manager
from risk_manager import (
    black_scholes_greeks,
    apply_transaction_cost,
    MULTIPLIER as OPTION_MULTIPLIER
)

# Hedging configuration constants
FUTURE_MULTIPLIER = 300
DEFAULT_HEDGE_TOOL = "FUTURE"
DAILY_HEDGE_TIME = "14:50"
VOL_TRIGGER_THRESHOLD = 0.03
MIN_HEDGE_ADJUST = 1

# Internal tolerance/safety lower limit constants: Avoid meaningless frequent small hedging and crashes caused by extreme values
_HEDGE_DELTA_TOLERANCE_FUT = FUTURE_MULTIPLIER * 0.25   # Futures hedge tolerance (quarter contract)
_HEDGE_DELTA_TOLERANCE_SPOT = OPTION_MULTIPLIER * 0.25 # Spot hedge tolerance (quarter contract multiplier)
_MIN_T_ANNUAL = 1e-8                                    # Minimum annualized T to prevent expiration or anomalies
_MIN_IV = 1e-6                                          # Minimum IV clipping to prevent zero or negative values


def calculate_portfolio_net_delta(portfolio_positions, current_market_data, r=0.03):
    net_delta = 0.0
    current_market_data = current_market_data.copy()

    # First convert date/expiry columns to datetime type
    current_market_data["date"] = pd.to_datetime(current_market_data["date"])
    current_market_data["expiry"] = pd.to_datetime(current_market_data["expiry"])
    if current_market_data.empty:
        return round(net_delta, 2)

    current_date = current_market_data["date"].iloc[0]

    # Pre-indexing for speed & basic validation
    current_market_data = current_market_data.set_index("option_code", drop=False)

    for option_code, pos in portfolio_positions.items():
        qty = pos.get("qty", 0)
        if qty == 0:
            continue

        # Use index positioning to reduce filtering overhead
        if option_code not in current_market_data.index:
            continue
        row = current_market_data.loc[option_code]
        # Deduplication before setting index
        current_market_data = current_market_data.sort_values("date").drop_duplicates(subset="option_code", keep="last")
        current_market_data = current_market_data.set_index("option_code", drop=False)

        # Avoid negative T/zero IV
        S = float(row.get("underlying_price", np.nan))
        K = float(row.get("strike", np.nan))
        expiry = row.get("expiry", pd.NaT)
        T = ((expiry - current_date).days) / 365.0 if pd.notna(expiry) else _MIN_T_ANNUAL
        T = max(float(T), _MIN_T_ANNUAL)
        sigma = float(row.get("iv", np.nan))
        sigma = max(sigma if np.isfinite(sigma) else _MIN_IV, _MIN_IV)

        if not np.isfinite(S) or not np.isfinite(K):
            continue

        option_type = row.get("option_type", "call")

        try:
            greeks = black_scholes_greeks(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
        except Exception:
            # Single leg failure should not interrupt the whole process
            continue

        leg_delta = greeks.get("delta", 0.0)
        if not np.isfinite(leg_delta):
            continue

        net_delta += qty * leg_delta * OPTION_MULTIPLIER

    return round(net_delta, 2)


def _should_trigger_hedge(prev_hedge_info, current_time, current_underlying_price):
    if not prev_hedge_info or "time" not in prev_hedge_info or "price" not in prev_hedge_info:
        return True, "First hedge, trigger adjustment"

    try:
        prev_hedge_time = datetime.strptime(prev_hedge_info["time"], "%Y-%m-%d %H:%M")
        current_time_dt = datetime.strptime(current_time, "%Y-%m-%d %H:%M")
        prev_hedge_price = float(prev_hedge_info["price"])
    except (ValueError, TypeError):
        return False, "Previous hedge information format error, not triggering"

    current_time_str = current_time_dt.strftime("%H:%M")
    if current_time_str >= DAILY_HEDGE_TIME:
        return True, f"Reached daily hedge time ({DAILY_HEDGE_TIME}), triggering adjustment"

    if prev_hedge_price <= 0:
        return False, "Previous hedge price is invalid, not triggering"

    price_volatility = abs(current_underlying_price - prev_hedge_price) / prev_hedge_price
    if price_volatility > VOL_TRIGGER_THRESHOLD:
        return True, f"Underlying volatility exceeds {VOL_TRIGGER_THRESHOLD*100:.1f}% (current {price_volatility*100:.2f}%), triggering adjustment"

    return False, f"Hedge not triggered (time: {current_time_str}, volatility: {price_volatility*100:.2f}%)"


def execute_delta_hedge(
    portfolio_positions,
    current_market_data,
    current_hedge_pos,
    prev_hedge_info,
    hedge_tool=DEFAULT_HEDGE_TOOL,
    future_multiplier=FUTURE_MULTIPLIER,
    r=0.03,
    hedge_fee_params=None
):
    result = {
        "adjusted": False,
        "new_hedge_pos": current_hedge_pos,
        "hedge_trades": [],
        "net_delta_after_hedge": None,
        "msg": ""
    }

    current_market_data = current_market_data.copy()
    if current_market_data.empty:
        result["msg"] = "Error: Current market data is empty, cannot execute hedge"
        return result
    if "underlying_price" not in current_market_data.columns:
        result["msg"] = "Error: Market data missing 'underlying_price' field"
        return result

    # First convert date column to datetime type, then format
    current_market_data["date"] = pd.to_datetime(current_market_data["date"])

    # If there are multiple records, select the latest daily quote as hedge reference
    current_market_data = current_market_data.sort_values("date")
    current_date_dt = current_market_data["date"].iloc[-1]
    current_date = current_date_dt.strftime("%Y-%m-%d")

    # If there are multiple options with the same underlying price, take the mean of consistent underlying_price for the day
    try:
        current_underlying_price = float(current_market_data["underlying_price"].astype(float).mean())
    except Exception:
        current_underlying_price = float(current_market_data["underlying_price"].iloc[-1])

    current_time = f"{current_date} {DAILY_HEDGE_TIME}"

    default_fee = {
        "FUTURE": {"per_contract_fee": 2.0, "pct_fee": 0.0001, "slippage_bps": 0.0002},
        "SPOT": {"per_contract_fee": 1.0, "pct_fee": 0.0003, "slippage_bps": 0.0005}
    }
    hedge_fee_params = hedge_fee_params or default_fee.get(hedge_tool, default_fee["FUTURE"])

    trigger_flag, trigger_msg = _should_trigger_hedge(
        prev_hedge_info=prev_hedge_info,
        current_time=current_time,
        current_underlying_price=current_underlying_price
    )
    result["msg"] = trigger_msg
    if not trigger_flag:
        return result

    net_delta_before = calculate_portfolio_net_delta(
        portfolio_positions=portfolio_positions,
        current_market_data=current_market_data,
        r=r
    )

    # Hedge trigger threshold uses tool-related tolerance
    delta_tol = _HEDGE_DELTA_TOLERANCE_FUT if hedge_tool == "FUTURE" else _HEDGE_DELTA_TOLERANCE_SPOT
    if abs(net_delta_before) < max(delta_tol, 1.0):
        result["msg"] = f"Portfolio net Delta is within tolerance ({net_delta_before:.2f}), no hedge needed"
        result["net_delta_after_hedge"] = net_delta_before
        return result

    # Target position (integer)
    if hedge_tool == "FUTURE":
        target_hedge_pos = int(np.round(-net_delta_before / future_multiplier))
    else:
        # Spot hedge: Each share has Delta=1, directly round to integer shares
        target_hedge_pos = int(np.round(-net_delta_before))

    hedge_adjust_qty = target_hedge_pos - current_hedge_pos

    # Minimum adjustment amount + post-hedge residual tolerance
    if abs(hedge_adjust_qty) < max(MIN_HEDGE_ADJUST, 1):
        # Calculate residual net Delta if this minor adjustment is executed, skip if sufficiently small
        unit = (future_multiplier if hedge_tool == "FUTURE" else 1)
        residual = net_delta_before + hedge_adjust_qty * unit
        if abs(residual) < delta_tol:
            result["msg"] = f"Hedge adjustment amount is too small ({hedge_adjust_qty}), residual within tolerance ({residual:.2f}), skipping"
            result["net_delta_after_hedge"] = round(residual, 2)
            return result

    # Execute hedge
    hedge_price = current_underlying_price

    # Fall back to underlying price itself if cost calculation is abnormal
    try:
        exec_price, total_fee = apply_transaction_cost(
            price_per_contract=float(hedge_price),
            qty_signed=int(hedge_adjust_qty),
            **hedge_fee_params
        )

    except Exception as e:
        exec_price = float(hedge_price)
        total_fee = 0.0
        print(f"Warning: Transaction cost calculation error ({str(e)}), fee temporarily set to 0, please check parameters")

    hedge_trade = {
        "date": current_date,
        "time": current_time.split(" ")[1],
        "option_code": f"HEDGE_{hedge_tool}",
        "qty": int(hedge_adjust_qty),
        "exec_price": round(float(exec_price), 4),
        "fee": round(float(total_fee), 4),
        "trade_type": "delta_hedge",
        "hedge_tool": hedge_tool,
        "net_delta_before": round(float(net_delta_before), 2),
        "target_hedge_pos": int(target_hedge_pos)
    }

    result["adjusted"] = True
    result["new_hedge_pos"] = int(target_hedge_pos)
    result["hedge_trades"] = [hedge_trade]

    # Calculate net Delta after hedge
    unit = (future_multiplier if hedge_tool == "FUTURE" else 1)
    hedge_delta_offset = int(hedge_adjust_qty) * unit
    result["net_delta_after_hedge"] = round(net_delta_before + hedge_delta_offset, 2)
    result["msg"] = (
        f"Hedge executed successfully: adjusted {hedge_adjust_qty} contracts/shares of {hedge_tool}, "
        f"net Delta after hedge: {result['net_delta_after_hedge']:.2f} (tolerance: ±{delta_tol:.2f})"
    )

    return result
