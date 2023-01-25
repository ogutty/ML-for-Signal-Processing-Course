import math
import numpy as np
import pandas as pd
from random import gauss, randint

# -------------------------------------------------------------------------------------
# Generate daily stock prices.
# This function generates a sequence of segments, each comprising Brownian geometric
# "motion" with a given mean & variance and num. of steps.
# -------------------------------------------------------------------------------------
def generate_stock_prices(params, delay, stock_name, seed=None):

    if seed is not None:
        np.random.seed(seed)
    init_price = 100
    curr_price = init_price
    price_seq = []
    
    # Insert a few random delays (before the actual simulation).
    for _ in range(delay):
        price_seq.append(curr_price + gauss(0, 0.2))
    price_seq.append(init_price)
    
    # Outer loop over regimes:
    for [mu, sigma, steps] in params:
        
        # Inner loop over days in the regime.
        # WARNING: absolutely inefficient, only for demonstration! 
        for _ in range(steps):
            daily_move = mu + np.random.normal(0, sigma)
            next_price = curr_price * daily_move
            price_seq.append(next_price)
            curr_price = next_price

    # Turn the whole thing into a Pandas dataframe and return it.
    # NOTICE we set the dataframe index to the day seq. num. to enable merges between stocks.
    out_df = pd.DataFrame()
    out_df['day_seqno'] = range(len(price_seq))
    out_df[stock_name] = price_seq
    out_df.set_index('day_seqno')
    return(out_df)

# -------------------------------------------------------------------------------------
# A pure Python (and therefore SLOW) rollout simulator.
# We assume the arriving prices & predictions are synced and have no holes.
# States are enumerated as follows:
# NEUT  <--> 0,
# LONG  <--> 1.
# -------------------------------------------------------------------------------------
def rollout(prices, preds, buy_thresh, sell_thresh):

    rollout_reward = 0         # We assign the complete series a "reward function".
    curr_state_num = 0         # Start in 'NEUT'.
    min_days_enter_trade = 10  # If not enough days are left, don't bother.

    trades = []
    days_in_market = 0
    for day_idx in range(len(prices)):

        # --------------------------------------------------------------------
        # In NEUT state, decide whether or not to go LONG.
        # --------------------------------------------------------------------
        if curr_state_num == 0:
            if preds[day_idx] < buy_thresh or day_idx > len(prices) - min_days_enter_trade:
                # No BUY signal (or not enough time left).
                next_state_num = 0
                continue

            # We've just entered a trade, record everything.
            trade_entry_idx = day_idx + 1 # We buy on tomorrow's price.
            trade_entry_price = prices[trade_entry_idx]
            next_state_num = 1
            # print("Entered long trade on:", day_idx, "entry price:", trade_entry_price)

        # --------------------------------------------------------------------
        # When in LONG state, check whether we've hit an exit or the rollout end.
        # --------------------------------------------------------------------
        elif curr_state_num == 1:
            if preds[day_idx] < sell_thresh or day_idx >= len(prices) - 2:

                # We've just received a SELL signal (or arrived at rollout end).
                # We should leave (assume we're getting tomorrow's price).
                trade_exit_idx = day_idx + 1
                trade_exit_price = prices[trade_exit_idx]
                trade_reward = 100 * (math.log(trade_exit_price) - math.log(trade_entry_price))
                rollout_reward += trade_reward
                days_in_market += trade_exit_idx - trade_entry_idx
                next_state_num = 0
                trades.append([trade_entry_idx+1, trade_exit_idx+1, trade_reward])
                # print("Exited trade on:", day_idx, "exit price:", trade_exit_price)
                # print("Trade gain (pct):", 100 * (math.exp(trade_reward) - 1))

        curr_state_num = next_state_num
    return(trades, days_in_market, rollout_reward)
