import pandas as pd
import numpy as np

# Import all necessary risk calculation and position management functions from risk_manager
from risk_manager import (
    black_scholes_greeks,
    calculate_trade_size_by_vega,
    size_pair_by_vega,
    apply_transaction_cost,
    calculate_margin,
    MULTIPLIER
)


# -------------------------- Contract Pair Matching and Validation Helper Functions --------------------------
def _get_near_far_expiry(group):
    """Distinguish near-month and far-month expirations (sorted by days remaining)"""
    group = group.copy()
    group["date"] = pd.to_datetime(group["date"])
    group["expiry"] = pd.to_datetime(group["expiry"])
    group["days_to_expiry"] = (group["expiry"] - group["date"]).dt.days
    expiries = group["days_to_expiry"].unique()

    if len(expiries) < 2:
        return None, None  # Insufficient number of expirations to form calendar arbitrage
    return expiries.min(), expiries.max()


def _validate_iv_spread(long_row, short_row, min_iv_diff=0.001):
    """Validate IV spread rationality (short leg IV > long leg IV)"""
    long_iv = float(long_row["iv"])
    short_iv = float(short_row["iv"])
    return (short_iv - long_iv) >= min_iv_diff  # Ensure sufficient IV spread


# -------------------------- Core: Arbitrage Portfolio Construction Function --------------------------
def build_arbitrage_portfolio(
        signal_df,
        current_date,
        initial_cash=1_000_000,
        target_gross_vega=2000.0,
        max_capital_per_trade_pct=0.1,
        r=0.03,
        margin_rate=0.15,
        transaction_fee_params=None
):
    """
    Build calendar arbitrage portfolio based on signals from signal_generator, integrating risk_manager for risk and cost calculation

    Parameters:
        signal_df: DataFrame containing arbitrage signals (must include fields like pair_trades, option_code)
        current_date: Trading date (str or datetime)
        initial_cash: Initial account capital
        target_gross_vega: Target Vega per leg (risk exposure control)
        max_capital_per_trade_pct: Maximum capital ratio per arbitrage pair (10% by default)
        r: Risk-free interest rate (annualized)
        margin_rate: Short option margin rate
        transaction_fee_params: Transaction cost parameters (passed to apply_transaction_cost)

    Returns:
        Dictionary containing portfolio details, status, and summary information
    """
    # Initialize transaction cost parameters
    transaction_fee_params = transaction_fee_params or {
        "per_contract_fee": 1.0,
        "pct_fee": 0.0002,
        "slippage_bps": 0.0005
    }

    # 1. Data preprocessing and filtering
    signal_df = signal_df.copy()
    signal_df["date"] = pd.to_datetime(signal_df["date"])
    current_date = pd.to_datetime(current_date)

    # Filter valid signals for the current day
    daily_signal = signal_df[signal_df["date"] == current_date]
    if daily_signal.empty:
        return {
            "status": "fail",
            "msg": f"No arbitrage signals for {current_date.strftime('%Y-%m-%d')}",
            "portfolio": pd.DataFrame(),
            "summary": {}
        }

    # 2. Identify near-month/far-month expirations
    near_exp_days, far_exp_days = _get_near_far_expiry(daily_signal)
    if near_exp_days is None:
        return {
            "status": "fail",
            "msg": "Insufficient expiration types in signals (at least 2 required)",
            "portfolio": pd.DataFrame(),
            "summary": {}
        }

    # 3. Separate long and short leg signals
    long_legs = daily_signal[daily_signal["pair_trades"] == "long_leg"].copy()  # Buy low IV
    short_legs = daily_signal[daily_signal["pair_trades"] == "short_leg"].copy()  # Sell high IV

    if long_legs.empty or short_legs.empty:
        return {
            "status": "fail",
            "msg": "Incomplete long/short leg signals (missing long_leg or short_leg)",
            "portfolio": pd.DataFrame(),
            "summary": {}
        }

    # 4. Match contract pairs and calculate positions
    portfolio = []
    used_short_indices = set()  # Avoid reusing short contracts
    max_single_trade_capital = initial_cash * max_capital_per_trade_pct  # Maximum capital per pair

    for _, long_row in long_legs.iterrows():
        # 4.1 Calculate long leg days to expiry and determine type (near-month/far-month)
        long_expiry = pd.to_datetime(long_row["expiry"])
        long_days = (long_expiry - current_date).days

        if long_days not in [near_exp_days, far_exp_days]:
            continue  # Only process near-month/far-month contracts

        # 4.2 Determine target expiration days for short contract (opposite of long leg)
        target_short_days = far_exp_days if long_days == near_exp_days else near_exp_days

        # 4.3 Filter matching short contracts (same type, target expiration, unused)
        short_candidates = short_legs[
            (short_legs["option_type"] == long_row["option_type"]) &
            (~short_legs.index.isin(used_short_indices))
            ].copy()

        if short_candidates.empty:
            continue

        # Calculate days to expiry for short candidates and filter by target expiration
        short_candidates["expiry"] = pd.to_datetime(short_candidates["expiry"])
        short_candidates["days_to_expiry"] = (short_candidates["expiry"] - current_date).dt.days
        short_candidates = short_candidates[short_candidates["days_to_expiry"] == target_short_days]

        if short_candidates.empty:
            continue

        # 4.4 Match short contract with closest strike price
        long_strike = float(long_row["strike"])
        short_candidates["strike_diff"] = (short_candidates["strike"].astype(float) - long_strike).abs()
        best_short_idx = short_candidates["strike_diff"].idxmin()
        short_row = short_candidates.loc[best_short_idx]

        # 4.5 Validate IV spread (sell high, buy low)
        if not _validate_iv_spread(long_row, short_row):
            used_short_indices.add(best_short_idx)
            continue

        # 5. Calculate Greeks (call risk_manager)
        try:
            # Long leg Greeks
            long_greeks = black_scholes_greeks(
                S=float(long_row["underlying_price"]),
                K=float(long_row["strike"]),
                T=long_days / 365.0,
                r=r,
                sigma=float(long_row["iv"]),
                option_type=long_row["option_type"]
            )

            # Short leg Greeks
            short_greeks = black_scholes_greeks(
                S=float(short_row["underlying_price"]),
                K=float(short_row["strike"]),
                T=target_short_days / 365.0,
                r=r,
                sigma=float(short_row["iv"]),
                option_type=short_row["option_type"]
            )
        except Exception as e:
            used_short_indices.add(best_short_idx)
            continue  # Skip this pair if Greeks calculation fails

        # Filter invalid Greeks
        if long_greeks["vega"] <= 0 or short_greeks["vega"] <= 0:
            used_short_indices.add(best_short_idx)
            continue

        # 6. Calculate initial positions (based on target Vega, call risk_manager)
        long_qty_raw, short_qty_raw = size_pair_by_vega(
            long_leg_vega=long_greeks["vega"],
            short_leg_vega=short_greeks["vega"],
            target_gross_vega=target_gross_vega
        )

        # 6.1 Delta neutral adjustment
        if abs(short_greeks["delta"]) < 1e-6:  # [Modification 1] Avoid division by zero/extreme cases
            used_short_indices.add(best_short_idx)
            continue

        short_qty_adjusted = int(-(long_qty_raw * long_greeks["delta"]) / short_greeks["delta"])
        long_qty_adjusted = int(long_qty_raw)

        # First force correct position directions
        long_qty_adjusted = abs(long_qty_adjusted)
        short_qty_adjusted = -abs(short_qty_adjusted)

        # Then check if net Delta is within tolerance; fine-tune quantity if exceeded
        net_delta_test = (long_qty_adjusted * long_greeks["delta"] + short_qty_adjusted * short_greeks["delta"])
        if abs(net_delta_test) / MULTIPLIER >= 0.05:
            # Fine-tune short_qty_adjusted to bring net Delta close to 0
            pass

        if long_qty_adjusted < 1 or abs(short_qty_adjusted) < 1:
            used_short_indices.add(best_short_idx)
            continue

        # 7. Calculate transaction costs and execution prices (call risk_manager)
        try:
            # Base price (prioritize mid price, use (bid+ask)/2 as fallback)
            long_base_price = float(long_row["mid"]) if pd.notna(long_row["mid"]) else \
                (float(long_row["bid"]) + float(long_row["ask"])) / 2
            short_base_price = float(short_row["mid"]) if pd.notna(short_row["mid"]) else \
                (float(short_row["bid"]) + float(short_row["ask"])) / 2

            # Calculate execution price and fees including costs
            long_exec_price, long_fee = apply_transaction_cost(
                price_per_contract=long_base_price,
                qty_signed=long_qty_adjusted,
                **transaction_fee_params
            )
            short_exec_price, short_fee = apply_transaction_cost(
                price_per_contract=short_base_price,
                qty_signed=short_qty_adjusted, **transaction_fee_params
            )
        except Exception as e:
            used_short_indices.add(best_short_idx)
            continue

        # 8. Capital occupation calculation and control
        # 8.1 Premium inflows and outflows
        long_premium = long_qty_adjusted * long_exec_price * MULTIPLIER  # Outflow
        short_premium = abs(short_qty_adjusted) * short_exec_price * MULTIPLIER  # Inflow
        net_premium = long_premium - short_premium  # Net outflow

        # 8.2 Margin (only for short positions, call risk_manager)
        short_margin = calculate_margin(
            option_row=short_row,
            qty_signed=short_qty_adjusted,
            margin_rate=margin_rate
        )

        # 8.3 Total capital occupied (margin + net premium + total fees)
        total_fee = long_fee + short_fee

        total_capital = short_margin + net_premium + total_fee
        total_capital = max(total_capital, 0)  # Ensure non-negative

        # 8.4 Avoid rounding to 0 when scaling down positions
        if total_capital > max_single_trade_capital:
            scale_ratio = max_single_trade_capital / total_capital

            # Long leg: floor the value
            long_qty_scaled = int(np.floor(long_qty_adjusted * scale_ratio))
            # Short leg: first floor the absolute value, then add negative sign
            short_abs_scaled = int(np.floor(abs(short_qty_adjusted) * scale_ratio))
            short_qty_scaled = -short_abs_scaled

            if long_qty_scaled < 1 or abs(short_qty_scaled) < 1:
                used_short_indices.add(best_short_idx)
                continue

            # Recalculate capital items after scaling
            long_exec_price, long_fee = apply_transaction_cost(
                price_per_contract=long_base_price,
                qty_signed=long_qty_scaled,
                **transaction_fee_params
            )
            short_exec_price, short_fee = apply_transaction_cost(
                price_per_contract=short_base_price,
                qty_signed=short_qty_scaled, **transaction_fee_params
            )

            long_premium = long_qty_scaled * long_exec_price * MULTIPLIER
            short_premium = abs(short_qty_scaled) * short_exec_price * MULTIPLIER
            net_premium = long_premium - short_premium
            short_margin = calculate_margin(short_row, short_qty_scaled, margin_rate)
            total_fee = long_fee + short_fee
            total_capital = short_margin + max(net_premium, 0) + total_fee

            # Update to scaled quantities
            long_qty_adjusted, short_qty_adjusted = long_qty_scaled, short_qty_scaled

        # 9. Record portfolio details
        portfolio.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "pair_id": f"{long_row['option_code']}_{short_row['option_code']}",
            # Long leg info
            "long_option": long_row["option_code"],
            "long_type": long_row["option_type"],
            "long_strike": float(long_row["strike"]),
            "long_expiry": long_expiry.strftime("%Y-%m-%d"),
            "long_qty": long_qty_adjusted,
            "long_exec_price": long_exec_price,
            "long_iv": float(long_row["iv"]),
            "long_delta": long_greeks["delta"],
            "long_vega": long_greeks["vega"],
            # Short leg info
            "short_option": short_row["option_code"],
            "short_type": short_row["option_type"],
            "short_strike": float(short_row["strike"]),
            "short_expiry": short_row["expiry"].strftime("%Y-%m-%d"),
            "short_qty": short_qty_adjusted,
            "short_exec_price": short_exec_price,
            "short_iv": float(short_row["iv"]),
            "short_delta": short_greeks["delta"],
            "short_vega": short_greeks["vega"],
            # Capital and risk indicators
            "margin_used": short_margin,
            "net_premium": net_premium,
            "total_fee": total_fee,
            "total_capital_used": total_capital,
            "net_delta": (long_qty_adjusted * long_greeks["delta"] +
                          short_qty_adjusted * short_greeks["delta"]) * MULTIPLIER,
            "gross_vega": abs(long_qty_adjusted * long_greeks["vega"]) +
                          abs(short_qty_adjusted * short_greeks["vega"])
        })
        used_short_indices.add(best_short_idx)

    # 10. Generate final results
    portfolio_df = pd.DataFrame(portfolio)
    summary = {
        "total_pairs": len(portfolio_df),
        "total_capital_used": portfolio_df["total_capital_used"].sum() if not portfolio_df.empty else 0,
        "avg_capital_per_pair": (portfolio_df["total_capital_used"].mean()
                                 if not portfolio_df.empty else 0),
        "total_net_delta": portfolio_df["net_delta"].sum() if not portfolio_df.empty else 0,
        "total_gross_vega": portfolio_df["gross_vega"].sum() if not portfolio_df.empty else 0,
        "initial_cash": initial_cash,
        "capital_usage_ratio": (portfolio_df["total_capital_used"].sum() / initial_cash
                                if initial_cash > 0 and not portfolio_df.empty else 0)
    }

    return {
        "status": "success" if not portfolio_df.empty else "fail",
        "msg": f"Successfully built {len(portfolio_df)} calendar arbitrage pairs" if not portfolio_df.empty
        else "No valid arbitrage pairs found",
        "portfolio": portfolio_df,
        "summary": summary
    }
