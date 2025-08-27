import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


def generate_vol_surface_pairs(df, lookback=20, z_open=1.5, min_periods=10, cost_threshold=0.0):
    """
    Filters volatility arbitrage signals using only statistical methods (Z-Score):
    1. Daily construction of "near-month ATM + (next) far-month ATM" option pairs, balancing liquidity and time difference
    2. Calculate rolling Z-Score based on absolute IV spread (near-month - far-month)
    3. Generate trading signals when Z-Score exceeds threshold and covers costs
    4. Large-sample normality test + visual analysis to verify statistical method effectiveness
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out.sort_values(["date", "expiry", "strike"], inplace=True)
    out["pair_trades"] = None  # Mark arbitrage legs (short_leg=sell, long_leg=buy)

    # 1. Construct daily "near-month + (next) far-month" ATM contract pairs (core optimization: flexible far-month selection to improve liquidity)
    pairs = []
    for date, day_data in out.groupby("date"):
        expiries = sorted(day_data["expiry"].unique())
        if len(expiries) < 2:
            continue  # Need at least 2 expiration dates to form a pair

        # Optimization: Use farthest expiry when only 2 expiries exist, use second-farthest when more than 2 (avoids poor liquidity in farthest)
        short_expiry = expiries[0]  # Fixed to nearest month
        long_expiry = expiries[-1] if len(expiries) == 2 else expiries[-2]

        # Underlying price for the day (take first non-null value to ensure consistency)
        underlying_price = float(day_data["underlying_price"].dropna().iloc[0])

        # Pair separately by option type (call/put)
        for opt_type in day_data["option_type"].dropna().unique():
            # Filter options for near-month/far-month of corresponding type
            short_contracts = day_data[
                (day_data["expiry"] == short_expiry) &
                (day_data["option_type"] == opt_type)
            ]
            long_contracts = day_data[
                (day_data["expiry"] == long_expiry) &
                (day_data["option_type"] == opt_type)
            ]

            if short_contracts.empty or long_contracts.empty:
                continue  # Skip if contracts missing on either side

            # Select contract closest to ATM (abs() ensures "minimum difference" regardless of sign)
            short_atm_idx = (short_contracts["strike"] - underlying_price).abs().idxmin()
            long_atm_idx = (long_contracts["strike"] - underlying_price).abs().idxmin()

            # Extract IV with enhanced outlier filtering
            short_iv = float(out.loc[short_atm_idx, "iv"])
            long_iv = float(out.loc[long_atm_idx, "iv"])

            # Optimization 1: First filter non-finite values (NaN/inf)
            if not np.isfinite(short_iv) or not np.isfinite(long_iv):
                continue
            # Optimization 2: Filter IV ≤ 0 (numerically meaningless values)
            if short_iv <= 0 or long_iv <= 0:
                continue

            # Calculate spread metrics (Optimization 3: metric_rel avoids division by zero)
            metric_abs = short_iv - long_iv  # Absolute spread (core indicator)
            metric_rel = np.log(short_iv / long_iv) if long_iv > 0 else np.nan  # Relative spread (prevents division by zero)

            # Save this contract pair information
            pairs.append({
                "date": date, "option_type": opt_type,
                "short_code": out.loc[short_atm_idx, "option_code"],
                "long_code": out.loc[long_atm_idx, "option_code"],
                "metric": metric_rel,
                "metric_abs": metric_abs
            })

    # Return original data directly when no valid contract pairs exist
    if not pairs:
        print("No eligible option pairs constructed.")
        return out

    # Convert to DataFrame for subsequent calculations
    sp = pd.DataFrame(pairs)
    sp.sort_values("date", inplace=True)
    daily_avg = len(sp) / sp["date"].nunique()
    print(f"Successfully constructed {len(sp)} contract pairs (average {daily_avg:.1f} pairs per day)")

    # 2. Spread distribution analysis (verify normality to justify Z-Score usage)
    def analyze_distribution():
        # Ensure output folder exists
        os.makedirs("spread_analysis", exist_ok=True)
        for opt_type in sp["option_type"].unique():
            # Filter absolute spread data for this type
            spread_data = sp[sp["option_type"] == opt_type]["metric_abs"].dropna()
            if len(spread_data) < min_periods:
                print(f"Skipping {opt_type} distribution analysis: Insufficient samples ({len(spread_data)} < {min_periods})")
                continue

            # Print key statistical indicators
            print(f"\n=== {opt_type} Option Absolute Spread (Near-month IV - Far-month IV) Statistics ===")
            print(f"Mean: {spread_data.mean():.4f}, Standard Deviation: {spread_data.std():.4f}")
            print(f"Mean ± 2σ Range: [{spread_data.mean() - 2*spread_data.std():.4f}, "
                  f"{spread_data.mean() + 2*spread_data.std():.4f}]")
            print(f"Minimum: {spread_data.min():.4f}, Maximum: {spread_data.max():.4f}")

            # Optimization 4: Use normaltest (large-sample friendly) instead of Shapiro-Wilk
            stat, p_value = stats.normaltest(spread_data)
            norm_result = "conforms to normal distribution" if p_value > 0.05 else "does not conform to normal distribution"
            print(f"Normality Test (Normaltest): p-value = {p_value:.4f} ({norm_result}, α = 0.05)")

            # Visualization: Histogram (spread distribution) + Q-Q plot (visual normality verification)
            plt.figure(figsize=(12, 5))
            # Histogram (mark mean and ±2σ)
            plt.subplot(1, 2, 1)
            plt.hist(spread_data, bins=30, edgecolor="black", alpha=0.7)
            plt.axvline(spread_data.mean(), color="blue", linestyle="-", label=f"Mean = {spread_data.mean():.4f}")
            plt.axvline(spread_data.mean() - 2*spread_data.std(), color="red", linestyle="--", label="Mean - 2σ")
            plt.axvline(spread_data.mean() + 2*spread_data.std(), color="red", linestyle="--", label="Mean + 2σ")
            plt.title(f"{opt_type} Absolute Spread Distribution Histogram")
            plt.xlabel("IV Near-month - IV Far-month")
            plt.ylabel("Frequency")
            plt.legend()

            # Q-Q plot
            plt.subplot(1, 2, 2)
            stats.probplot(spread_data, plot=plt)
            plt.title(f"{opt_type} Spread Q-Q Plot (Normality Verification)")

            # Save charts
            plt.tight_layout()
            save_path = f"spread_analysis/{opt_type}_spread_distribution.png"
            plt.savefig(save_path)
            plt.close()
            print(f"Distribution chart saved to: {os.path.abspath(save_path)}")

    # Execute distribution analysis
    analyze_distribution()

    # 3. Calculate Z-Score (core indicator of statistical method)
    # Group by option type to avoid mixing different types
    grouped = sp.groupby("option_type")["metric_abs"]

    # Rolling mean (average spread over past lookback days)
    sp["metric_abs_mean"] = grouped.transform(
        lambda x: x.rolling(window=lookback, min_periods=min_periods).mean()
    )
    # Rolling standard deviation (spread volatility over past lookback days)
    sp["metric_abs_std"] = grouped.transform(
        lambda x: x.rolling(window=lookback, min_periods=min_periods).std()
    )

    # Optimization 5: Pre-handle cases where standard deviation = 0 (avoids division by zero resulting in inf)
    sp["metric_abs_std"].replace(0, np.nan, inplace=True)

    # Calculate Z-Score: (current spread - rolling mean) / rolling standard deviation
    sp["z_abs"] = (sp["metric_abs"] - sp["metric_abs_mean"]) / sp["metric_abs_std"]

    # Filter invalid values (inf/-inf/NaN)
    sp = sp.replace([np.inf, -np.inf], np.nan).dropna(subset=["z_abs"])

    # Debug output: Show first 5 Z-Score results
    print("\n=== Z-Score and Absolute Spread Comparison (First 5 Rows) ===")
    print(sp[["date", "option_type", "metric_abs", "metric_abs_mean", "metric_abs_std", "z_abs"]].head())

    # 4. Generate arbitrage signals (Z-Score exceeds threshold + covers costs)
    # Signal rule 1: Near-month IV is overvalued (sell near-month, buy far-month)
    overvalued = (sp["z_abs"] > z_open) & (sp["metric_abs"] > cost_threshold)
    # Signal rule 2: Near-month IV is undervalued (buy near-month, sell far-month)
    undervalued = (sp["z_abs"] < -z_open) & (abs(sp["metric_abs"]) > cost_threshold)

    # Mark "sell near-month, buy far-month" signals
    for _, row in sp[overvalued].iterrows():
        out.loc[
            (out["date"] == row["date"]) & (out["option_code"] == row["short_code"]),
            "pair_trades"
        ] = "short_leg"  # Sell near-month (overvalued)
        out.loc[
            (out["date"] == row["date"]) & (out["option_code"] == row["long_code"]),
            "pair_trades"
        ] = "long_leg"  # Buy far-month (relatively undervalued)

    # Mark "buy near-month, sell far-month" signals
    for _, row in sp[undervalued].iterrows():
        out.loc[
            (out["date"] == row["date"]) & (out["option_code"] == row["short_code"]),
            "pair_trades"
        ] = "long_leg"  # Buy near-month (undervalued)
        out.loc[
            (out["date"] == row["date"]) & (out["option_code"] == row["long_code"]),
            "pair_trades"
        ] = "short_leg"  # Sell far-month (relatively overvalued)

    # Signal statistics (each signal group contains 1 short_leg + 1 long_leg, hence divide by 2)
    signal_counts = out["pair_trades"].value_counts().to_dict()
    short_long_pairs = signal_counts.get("short_leg", 0) // 2
    long_short_pairs = signal_counts.get("long_leg", 0) // 2
    total_pairs = short_long_pairs + long_short_pairs

    print("\n=== Arbitrage Signal Statistics ===")
    print(f"Sell near-month buy far-month (near-month IV overvalued): {short_long_pairs} groups")
    print(f"Buy near-month sell far-month (near-month IV undervalued): {long_short_pairs} groups")
    print(f"Total valid arbitrage signal groups: {total_pairs} groups")

    return out
