import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("trade_execution.log"), logging.StreamHandler()]
)
logger = logging.getLogger("TradeExecutorMonitor")


class TradeExecutorMonitor:
    def __init__(self, initial_capital=1000000):
        """
        Initialize Trade Execution and Monitor
        :param initial_capital: Initial account capital (default: 1,000,000 RMB)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital  # Current available capital
        self.holdings = None  # Current position (assigned after entry)
        self.trade_records = []  # Trade records
        self.position_opened = False  # Position status flag
        self.slippage_limit = 0.005  # Maximum slippage (RMB per contract)
        self.price_tick = 0.001  # Minimum price tick size for uniform price constraints/rounding

    # Align price to minimum tick size
    def _round_to_tick(self, price: float) -> float:
        if price is None or np.isnan(price):
            return price
        # Prevent floating-point errors: divide by tick first, then round
        tick = self.price_tick
        return max(tick, round(round(price / tick) * tick, 3))

    def check_margin_requirement(self, arbitrage_plan):
        """
        Check if margin requirement is met (Account capital ≥ Required margin × 1.2)
        :param arbitrage_plan: Arbitrage portfolio plan (from portfolio_builder)
        :return: (Is satisfied, Capital check message)
        """
        # Extract margin information from arbitrage plan
        margin_requirement = arbitrage_plan.get("margin_requirement", {})
        total_margin = margin_requirement.get("total_margin", 0)
        reserved_margin = margin_requirement.get("reserved_margin", total_margin * 1.2)

        # Additional check for hedge instrument margin (result from hedge_manager)
        hedge_margin = arbitrage_plan.get("hedge_info", {}).get("margin", 0)
        total_required = reserved_margin + hedge_margin  # Total required capital (including 20% buffer)

        # Capital check logic
        if self.current_capital < total_required:
            msg = (f"Insufficient funds! Current capital {self.current_capital:.2f}, "
                   f"total required margin (including 20% buffer) {total_required:.2f}")
            logger.warning(msg)
            return False, msg

        msg = (f"Capital check passed! Current capital {self.current_capital:.2f}, "
               f"total required margin (including 20% buffer) {total_required:.2f}, remaining capital {self.current_capital - total_required:.2f}")
        logger.info(msg)
        return True, msg

    def execute_entry_trades(self, arbitrage_plan, market_data):
        """
        Execute position entry: synchronize long-short portfolio orders (limit orders + batch placement)
        :param arbitrage_plan: Arbitrage portfolio plan
        :param market_data: Daily market data (including bid/ask prices)
        :return: (Entry success status, Entry information)
        """
        if self.position_opened:
            msg = "Position already open, cannot re-enter position"
            logger.warning(msg)
            return False, msg

        # Check required columns and standardize date format
        required_cols = {"date", "option_code", "mid"}
        if not required_cols.issubset(set(market_data.columns)):
            miss = list(required_cols - set(market_data.columns))
            msg = f"Market data missing required columns: {miss}"
            logger.error(msg)
            return False, msg

        market_data = market_data.copy()
        try:
            market_data["date"] = pd.to_datetime(market_data["date"]).dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Failed to parse 'date' column, using original values: {e}")

        # 1. Check margin requirement
        margin_ok, margin_msg = self.check_margin_requirement(arbitrage_plan)
        if not margin_ok:
            return False, margin_msg

        # 2. Extract contract information from arbitrage portfolio
        near_leg = arbitrage_plan.get("near_month_option", {})
        far_leg = arbitrage_plan.get("far_month_option", {})
        legs = [near_leg, far_leg]

        # 3. Generate limit order prices (control slippage ≤ 0.005 RMB)
        entry_trades = []
        for leg in legs:
            # Verify required fields
            if not all(k in leg for k in ["code", "action", "quantity", "ideal_price"]):
                msg = f"Incomplete contract information: {leg}"
                logger.error(msg)
                return False, msg

            # Validate quantity is positive integer
            try:
                quantity = int(leg["quantity"])
                if quantity <= 0:
                    raise ValueError("Quantity must be a positive integer")
            except Exception as e:
                msg = f"Invalid quantity: {leg.get('quantity')} ({e})"
                logger.error(msg)
                return False, msg

            # Set limit price based on buy/sell direction (ensure slippage control)
            action = leg["action"]
            ideal_price = float(leg["ideal_price"])

            # Find corresponding market data and retrieve bid/ask/mid prices
            leg_data = market_data[market_data["option_code"] == leg["code"]]
            if leg_data.empty:
                msg = f"Contract not found in market data: {leg['code']}"
                logger.error(msg)
                return False, msg
            row = leg_data.iloc[0]
            bid = float(row["bid"]) if "bid" in row and not pd.isna(row["bid"]) else None
            ask = float(row["ask"]) if "ask" in row and not pd.isna(row["ask"]) else None
            mid = float(row["mid"])

            if action == "sell":
                # Sell order: limit price ≤ ideal price
                limit_price = ideal_price - self.slippage_limit
                # Align with market quotes to ensure execution (sell ≤ bid; if no bid, ≤ mid)
                ref_price = bid if bid is not None else mid
                limit_price = min(limit_price, ref_price)
            elif action == "buy":
                # Buy order: limit price ≥ ideal price
                limit_price = ideal_price + self.slippage_limit
                # Align with market quotes to ensure execution (buy ≥ ask; if no ask, ≥ mid)
                ref_price = ask if ask is not None else mid
                limit_price = max(limit_price, ref_price)
            else:
                msg = f"Invalid action direction: {action}, only 'sell'/'buy' supported"
                logger.error(msg)
                return False, msg

            # Align to price tick and set minimum price floor
            limit_price = max(self.price_tick, self._round_to_tick(limit_price))

            # Record order information
            trade = {
                "trade_id": f"ENTRY_{datetime.now().strftime('%Y%m%d%H%M%S')}_{leg['code']}",
                "date": market_data["date"].iloc[0],
                "time": datetime.now().strftime("%H:%M:%S"),
                "option_code": leg["code"],
                "action": action,
                "quantity": quantity,
                "ideal_price": ideal_price,
                "limit_price": limit_price,
                "exec_price": limit_price,  # Assume limit order is fully executed (for backtesting)
                "fee": self._calculate_fee(quantity, limit_price),  # Calculate transaction fee
                "trade_type": "entry"
            }
            entry_trades.append(trade)
            self.trade_records.append(trade)
            logger.info(f"Order placed successfully: {trade['option_code']} {action} {quantity} contracts, limit price {limit_price} RMB (bid={bid}, ask={ask}, mid={mid})")

        # 4. Update account capital (deduct premiums/margin)
        self._update_capital(entry_trades, arbitrage_plan)

        # 5. Record position information
        self.holdings = {
            "near_month": near_leg,
            "far_month": far_leg,
            "entry_trades": entry_trades,  # Save entry trade records
            "entry_date": market_data["date"].iloc[0],
            "entry_capital": self.current_capital,
            "take_profit": arbitrage_plan["iv_spread"]["take_profit_threshold"],
            "stop_loss": arbitrage_plan["iv_spread"]["stop_loss_threshold"]
        }
        self.position_opened = True

        msg = f"Position entry completed! Long-short contracts: {near_leg['code']} ({near_leg['action']}), {far_leg['code']} ({far_leg['action']})"
        logger.info(msg)
        return True, msg

    def monitor_and_execute_exit(self, current_market_data):
        """Monitor market status, trigger take-profit/stop-loss, and execute position exit (including time stop-loss + forced liquidation on expiry)"""
        if not self.position_opened or not self.holdings:
            msg = "No open position, no need for exit"
            logger.info(msg)
            return False, msg

        # Standardize date format for current market data
        current_market_data = current_market_data.copy()
        try:
            current_market_data["date"] = pd.to_datetime(current_market_data["date"]).dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Failed to parse 'date' column, using original values: {e}")

        # 1. Extract basic data (add date format fault tolerance)
        try:
            entry_date_str = self.holdings["entry_date"]
            current_date_str = current_market_data["date"].iloc[0]
            # Support multiple date formats (YYYY-MM-DD or YYYYMMDD)
            entry_date = datetime.strptime(entry_date_str, "%Y-%m-%d") if "-" in entry_date_str else datetime.strptime(
                entry_date_str, "%Y%m%d")
            current_date = datetime.strptime(current_date_str,
                                             "%Y-%m-%d") if "-" in current_date_str else datetime.strptime(
                current_date_str, "%Y%m%d")
        except ValueError as e:
            msg = f"Invalid date format: {str(e)}, supported formats: YYYY-MM-DD or YYYYMMDD"
            logger.error(msg)
            return False, msg

        # 2. Calculate holding days
        holding_days = (current_date - entry_date).days

        # 3. Extract IV spread
        current_spread = current_market_data["current_iv_spread"].iloc[
            0] if "current_iv_spread" in current_market_data.columns else 0.0
        take_profit = self.holdings["take_profit"]
        stop_loss = self.holdings["stop_loss"]

        # 4. Determine exit triggers (Priority: Forced liquidation on expiry > Time stop-loss > Take-profit/stop-loss)
        exit_reason = None

        # 4.1 Forced liquidation on expiry (parse expiry date from contract code, e.g., "C25MAR3000" → March 2025)
        near_leg_code = self.holdings["near_month"]["code"]
        try:
            year = 2000 + int(near_leg_code[1:3])  # Extract 2nd-3rd characters for year
            month_str = near_leg_code[3:6]  # Extract 4th-6th characters for month (e.g., MAR)
            month = {
                "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
                "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
                "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
            }[month_str]
            expiry_date = datetime(year, month, 15)  # Assume expiry on 15th of each month (adjustable)
        except (IndexError, KeyError, ValueError):
            logger.warning(f"Failed to parse contract expiry date: {near_leg_code}, skipping expiry liquidation check")
            expiry_date = None

        # Force liquidation 3 days before expiry to avoid low liquidity on expiry date
        if expiry_date and (expiry_date - current_date).days <= 3:
            exit_reason = f"Forced liquidation on expiry triggered: {((expiry_date - current_date).days)} days remaining until expiry"

        # 4.2 Time stop-loss (holding period exceeds 30 days)
        elif holding_days >= 30:
            exit_reason = f"Time stop-loss triggered: Holding period {holding_days} days (exceeds 30-day limit)"

        # 4.3 Take-profit/stop-loss (triggered by IV spread)
        elif current_spread <= take_profit:
            exit_reason = f"Take-profit triggered: Current IV spread {current_spread:.2%} ≤ Take-profit threshold {take_profit:.2%}"
        elif current_spread >= stop_loss:
            exit_reason = f"Stop-loss triggered: Current IV spread {current_spread:.2%} ≥ Stop-loss threshold {stop_loss:.2%}"

        # 5. No exit trigger activated
        if not exit_reason:
            msg = (f"No exit triggered - Holding period {holding_days} days, current IV spread {current_spread:.2%}, "
                   f"take-profit {take_profit:.2%}/stop-loss {stop_loss:.2%}, days until expiry {(expiry_date - current_date).days if expiry_date else 'Unknown'}")
            logger.info(msg)
            return False, msg

        # 6. Execute position exit
        exit_trades = []
        for leg_name in ["near_month", "far_month"]:
            leg = self.holdings[leg_name]
            exit_action = "buy" if leg["action"] == "sell" else "sell"
            quantity = int(leg["quantity"])  # Force integer type

            # Retrieve current market prices (mid price)
            leg_data = current_market_data[current_market_data["option_code"] == leg["code"]]
            if leg_data.empty:
                msg = f"Exit failed: Contract {leg['code']} not found in market data"
                logger.error(msg)
                return False, msg
            row = leg_data.iloc[0]
            current_ideal_price = float(row["mid"])
            bid = float(row["bid"]) if "bid" in row and not pd.isna(row["bid"]) else None
            ask = float(row["ask"]) if "ask" in row and not pd.isna(row["ask"]) else None

            # Calculate exit limit price (control slippage + align with market quotes)
            if exit_action == "sell":
                # Sell exit: price ≤ bid (or ≤ mid if no bid)
                limit_price = current_ideal_price - self.slippage_limit
                ref_price = bid if bid is not None else current_ideal_price
                limit_price = min(limit_price, ref_price)
            else:
                # Buy exit: price ≥ ask (or ≥ mid if no ask)
                limit_price = current_ideal_price + self.slippage_limit
                ref_price = ask if ask is not None else current_ideal_price
                limit_price = max(limit_price, ref_price)

            # Align to price tick
            limit_price = max(self.price_tick, self._round_to_tick(limit_price))

            # Record exit trade information
            trade = {
                "trade_id": f"EXIT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{leg['code']}",
                "date": current_market_data["date"].iloc[0],
                "time": datetime.now().strftime("%H:%M:%S"),
                "option_code": leg["code"],
                "action": exit_action,
                "quantity": quantity,
                "ideal_price": current_ideal_price,
                "limit_price": limit_price,
                "exec_price": limit_price,
                "fee": self._calculate_fee(quantity, limit_price),
                "trade_type": "exit",
                "exit_reason": exit_reason
            }
            exit_trades.append(trade)
            self.trade_records.append(trade)
            logger.info(f"Exit order placed successfully: {trade['option_code']} {exit_action} {quantity} contracts, limit price {limit_price} RMB (bid={bid}, ask={ask}, mid={current_ideal_price})")

        # 7. Update account capital
        self._update_capital(exit_trades, is_exit=True)

        # 8. Reset position status
        self.position_opened = False
        final_msg = (f"{exit_reason}, position exit completed! Final capital: {self.current_capital:.2f}, "
                     f"Total P&L: {self.current_capital - self.initial_capital:.2f}")
        logger.info(final_msg)
        return True, final_msg

    def _calculate_fee(self, quantity, price, fee_rate=0.0003, per_contract_fee=1.0):
        """Calculate transaction fee (fixed + proportional)"""
        fee = quantity * price * fee_rate + quantity * per_contract_fee
        return max(per_contract_fee, round(fee, 2))  # Keep minimum per-contract fee

    def _update_capital(self, trades, arbitrage_plan=None, is_exit=False):
        """Update account capital"""
        for trade in trades:
            amount = trade["quantity"] * trade["exec_price"]  # Total transaction amount
            fee = trade["fee"]  # Transaction fee

            if is_exit:
                # Exit: Calculate inversely based on entry type
                if trade["action"] == "sell":
                    # Sell exit → corresponds to "buy entry"
                    # Logic: Recover premium paid at entry - Exit fee
                    self.current_capital += amount - fee
                else:
                    # Buy exit → corresponds to "sell entry"
                    # 1) Find corresponding sell entry record; 2) Unfreeze margin proportionally; 3) Pay exit premium and fee
                    entry_trades = self.holdings.get("entry_trades", [])
                    candidates = [t for t in entry_trades
                                  if t["option_code"] == trade["option_code"] and t["trade_type"] == "entry" and t["action"] == "sell"]
                    if not candidates:
                        logger.error(f"Corresponding sell entry record not found: {trade['option_code']}")
                        continue
                    matching_entry = candidates[0]
                    entry_amount = matching_entry["quantity"] * matching_entry["exec_price"]
                    entry_margin_freeze_full = entry_amount * 20 * 0.2  # Consistent with entry margin freeze logic
                    # Proportional unfreezing (supports partial exit)
                    proportion = min(1.0, trade["quantity"] / max(1, matching_entry["quantity"]))
                    entry_margin_freeze = entry_margin_freeze_full * proportion
                    # Exit capital calculation: Unfrozen margin - Paid exit premium - Fee
                    self.current_capital += entry_margin_freeze - amount - fee
            else:
                # Entry: Symmetric to exit logic
                if trade["action"] == "buy":
                    # Buy entry: Pay premium + Fee
                    self.current_capital -= (amount + fee)
                else:
                    # Sell entry: Receive premium + Freeze margin + Pay fee
                    margin_freeze = amount * 20 * 0.2  # Margin freeze amount
                    self.current_capital += amount  # Receive premium
                    self.current_capital -= (margin_freeze + fee)  # Freeze margin + Pay fee

        # Handle hedge instrument margin (only for entry)
        if not is_exit and arbitrage_plan:
            hedge_margin = arbitrage_plan.get("hedge_info", {}).get("margin", 0)
            self.current_capital -= hedge_margin
