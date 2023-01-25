import math
import numpy as np

# -------------------------------------------------------------------------------------
# Ronni Roshbakir's original solution: check each pair using Python for loops.
# -------------------------------------------------------------------------------------
def py_naive_profit(prices):
    xmin = xmax = pmin = pmax = max_profit = -1
    for i in range(len(prices)):
        for j in range(i):
            if prices[i] - prices[j] > max_profit:
                xmin=j
                xmax=i
                pmin=prices[j]
                pmax=prices[i]
                max_profit = prices[i] - prices[j]
    return xmin, xmax, pmin, pmax, max_profit

