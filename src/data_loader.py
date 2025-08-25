import pandas as pd
import numpy as np
import logging

# Configure logging for easier debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_spread_option_data(
        path="data/option_data.csv",
        option_type="call",
        atm_threshold=0.05,  # At-the-money option threshold
        target_days_diff=90,  # Target days difference: 3 months
        diff_tolerance=15,  # Days difference tolerance
        min_liquidity=None,  # Absolute spread threshold
        spread_ratio_threshold=0.02,  # Dynamic spread ratio
        show_diagnosis=True,
        debug_mode=False,
        strict=True  # Control exception handling (True=raise error, False=return empty DF)
):
    """
    Load and filter contract pairs (near-month + far-month) required for calendar spread strategies

    Returns:
        DataFrame: Contains information of near-month and far-month contracts
    """

    # Relax all conditions in debug mode
    if debug_mode:
        logger.info("Debug mode enabled: all filtering conditions are relaxed")
        atm_threshold = 0.20
        diff_tolerance = 30
        spread_ratio_threshold = 0.1
        min_liquidity = None

    try:
        # 1. Load data
        logger.info(f"Loading option data from {path}...")
        df = pd.read_csv(path, parse_dates=["date", "expiry"])

        required = {"date", "option_code", "underlying_price", "strike", "option_type",
                    "bid", "ask", "expiry", "iv"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        # mid changed to (bid+ask)/2 to avoid zero or missing values
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df = df[df["mid"] > 0].copy()

        df["days_to_expiry_abs"] = (df["expiry"] - df["date"]).dt.days
        df = df[df["days_to_expiry_abs"] > 0].copy()

        # 2. Filter option types
        df = df[df["option_type"].str.lower() == option_type.lower()]
        if df.empty:
            msg = f"No option data for {option_type} type"
            return (pd.DataFrame() if not strict else (_ for _ in ()).throw(ValueError(msg)))

        # 3. Filter at-the-money options
        df["strike_diff_ratio"] = abs(df["strike"] - df["underlying_price"]) / df["underlying_price"]
        df = df[df["strike_diff_ratio"] <= atm_threshold]
        if df.empty:
            msg = f"No options meeting at-the-money criteria (atm_threshold={atm_threshold})"
            return (pd.DataFrame() if not strict else (_ for _ in ()).throw(ValueError(msg)))

        # 4. Filter liquidity
        df["bid_ask_spread"] = df["ask"] - df["bid"]
        df["spread_ratio"] = df["bid_ask_spread"] / df["mid"]

        if show_diagnosis:
            print("\n" + "=" * 50)
            print("=== Data Diagnosis: Liquidity Distribution of ATM Contracts ===")
            print(f"Sample size: {len(df)} records")
            print(f"Absolute spread (yuan): Median={df['bid_ask_spread'].median():.6f}, "
                  f"90th percentile={df['bid_ask_spread'].quantile(0.9):.6f}")
            print(f"Spread ratio (%): Median={df['spread_ratio'].median()*100:.2f}%, "
                  f"90th percentile={df['spread_ratio'].quantile(0.9)*100:.2f}%")
            print("=" * 50 + "\n")

        if min_liquidity is not None:
            df = df[df["bid_ask_spread"] <= min_liquidity]
        else:
            df = df[df["spread_ratio"] <= spread_ratio_threshold]

        if df.empty:
            msg = "No options meeting liquidity criteria"
            return (pd.DataFrame() if not strict else (_ for _ in ()).throw(ValueError(msg)))

        # 5. Calendar spread pairing (Optimization: use merge instead of double loops)
        logger.info(f"Starting to filter contract pairs with days difference of {target_days_diff}±{diff_tolerance} days...")
        df_near = df.rename(columns=lambda x: f"near_{x}" if x not in ["date"] else x)
        df_far = df.rename(columns=lambda x: f"far_{x}" if x not in ["date"] else x)

        pairs = df_near.merge(df_far, on="date")
        pairs = pairs[pairs["far_days_to_expiry_abs"] > pairs["near_days_to_expiry_abs"]]
        pairs["days_diff"] = pairs["far_days_to_expiry_abs"] - pairs["near_days_to_expiry_abs"]

        pairs = pairs[pairs["days_diff"].between(target_days_diff - diff_tolerance,
                                                 target_days_diff + diff_tolerance)]

        if pairs.empty:
            msg = "No contract pairs meeting days difference criteria"
            return (pd.DataFrame() if not strict else (_ for _ in ()).throw(ValueError(msg)))

        # Add additional indicators (for arbitrage analysis)
        pairs["iv_diff"] = pairs["far_iv"] - pairs["near_iv"]
        pairs["relative_price_diff"] = (pairs["far_mid"] - pairs["near_mid"]) / pairs["near_underlying_price"]
        pairs["moneyness"] = pairs["near_strike"] / pairs["near_underlying_price"]

        result_df = pairs.drop_duplicates(subset=["date", "near_option_code", "far_option_code"])
        result_df.sort_values(["date", "days_diff"], inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        logger.info(f"Final filtering result: {len(result_df)} valid contract pairs")
        return result_df

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        if strict:
            raise
        else:
            return pd.DataFrame()
