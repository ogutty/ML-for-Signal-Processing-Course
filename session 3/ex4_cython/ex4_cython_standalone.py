"""
Test a few Cython variations on the Stock Trader exercise.
"""
import time
import numpy as np
import pyximport
pyximport.install(
    setup_args={"include_dirs" : np.get_include()},
    language_level=3
)

from super_ronny import numpy_profit, py_naive_profit, cy_naive_profit
from super_ronny import cy_naive_profit_no_bounds, cy_naive_profit_parallel

# -------------------------------------------------------------------------------------
# Generate daily stock prices (same as exercise 1).
# -------------------------------------------------------------------------------------
def generate_stock_prices():

    # Create a semi-realistic prices array.
    # Start off with a mostly NaN array with a few 'turning points' (local min/max).
    prices = np.full(5000, fill_value=np.nan)
    prices[[0, 1250, 3000, -1]] = [80., 30., 75., 50.]

    # Linearly interpolate the missing values and add some noise.
    # NOTICE how the turning (valid) points are selected and all others are interpolated.
    x = np.arange(len(prices))
    is_valid = ~np.isnan(prices)                                  # Only look at valid numbers.
    prices = np.interp(x=x, xp=x[is_valid], fp=prices[is_valid])  # Interpolate between them.
    prices += np.random.randn(len(prices)) * 2                    # Add normally distributed noise.

    return(prices)

prices = generate_stock_prices()

# -------------------------------------------------------------------------------------
# Compare running times between naive Python, smart Numpy & various Cython routines.
# -------------------------------------------------------------------------------------

time1 = time.perf_counter()
xmin, xmax, pmin, pmax, max_profit = py_naive_profit(prices)
time2 = time.perf_counter()
print("Python naive profit:", max_profit, "time:", time2 - time1)

time1 = time.perf_counter()
max_profit = numpy_profit(prices)
time2 = time.perf_counter()
print("Numpy profit:", numpy_profit(prices), "time:", time2 - time1)

time1 = time.perf_counter()
super_ronny_profit = cy_naive_profit(prices)
time2 = time.perf_counter()
print("Cython naive profit:", super_ronny_profit, "time:", time2 - time1)

time1 = time.perf_counter()
super_ronny_profit_nb = cy_naive_profit_no_bounds(prices)
time2 = time.perf_counter()
print("Cython naive profit (no bounds checking):", super_ronny_profit_nb, "time:", time2 - time1)

time1 = time.perf_counter()
super_ronny_profit_parallel = cy_naive_profit_parallel(prices)
time2 = time.perf_counter()
print("Cython naive profit (parallel):", super_ronny_profit_parallel, "time:", time2 - time1)
