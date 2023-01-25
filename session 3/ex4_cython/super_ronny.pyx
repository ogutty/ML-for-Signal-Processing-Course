"""
Give Ronny compiled C head-banging power via Cython.
"""
cimport cython
cimport numpy as np
import numpy as np
from cython.parallel import parallel, prange
import os

# -------------------------------------------------------------------------------------
# Our interface to the outside world must be "Pythonic", declared using def.
# -------------------------------------------------------------------------------------

# The (smart & vectorized) Numpy solution.
def numpy_profit(prices):
    cum_min_prices = np.minimum.accumulate(prices)
    return(np.max(prices - cum_min_prices))

# Ronni Roshbakir's original solution: check each pair using Python for loops.
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

def cy_naive_profit(numpy_prices):
    return(cy_max_profit(numpy_prices))

def cy_naive_profit_no_bounds(numpy_prices):
    return(cy_max_profit_no_bounds(numpy_prices))

def cy_naive_profit_parallel(numpy_prices):

    # Make sure to declare types for each and every variable.
    cdef int i
    cdef int len_prices = len(numpy_prices)
    cdef int num_cpus = os.cpu_count()
    cdef double global_max_profit = 0.0

    # Set up memoryviews to hold the maximal profits possible up to each price index.
    # NOTE: the "double[::1]" means we're declaring a contiguous array.
    cdef double[::1] max_profits = np.zeros(len(numpy_prices))
    cdef double[::1] memview_prices = numpy_prices

    # In order to run loops in parallel, we instruct Python to disable the GIL (Global Interpreter Lock).
    # We also provide the number of threads we want to run in parallel.
    # In this case, we define one thread per core (note that some CPUs can run two threads per core).
    with nogil, parallel(num_threads=num_cpus):

        # NOTE: inside this loop, nothing Pythonic will work (as the GIL is disabled).
        # NOTE: we use prange() to implement the OUTER loop in parallel.
        # for i in prange(len_prices, schedule='guided'):
        for i in prange(len_prices):
            max_profits[i] = inner_loop_max_profit(i, memview_prices)

    # With max_profits populated, simply iterate to find the max.
    for i in range(len(max_profits)):
        if max_profits[i] > global_max_profit:
            global_max_profit = max_profits[i]

    return(global_max_profit)
    
# -------------------------------------------------------------------------------------
# Implement a simple Cython version of Ronny's naive algorithm.
# WARNING: a numpy "float" is actually a C "double".
# WARNING: this is a C function - nothing Pythonic will work inside!
# WARNING: make sure to type each variable inside, otherwise things will be SLOW.
# NOTE: the Numpy array is passed into this C function as is.
# NOTE: the "double[::1]" declaration means we guarantee the input is a contiguous array,
#       which (by default) Numpy arrays are.
#       In the general (>1d) case, consult the Numpy tutorial page in the Cython docs.
# -------------------------------------------------------------------------------------
cdef double cy_max_profit(double[::1] prices):
    cdef double max_profit = -1.0
    cdef int i,j
    for i in range(len(prices)):
        for j in range(i):
            if prices[i] - prices[j] > max_profit:
                max_profit = prices[i] - prices[j]
    return max_profit

# -------------------------------------------------------------------------------------
# Implement a version of Ronny's naive algorithm with array bounds checking disabled.
# SUPER WARNING: disabling array bounds checks is DANGEROUS!
#                Only do this *AFTER* the code as been *FULLY* DEBUGGED!!!
#                Don't say I didn't warn you!
# -------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double cy_max_profit_no_bounds(double[::1] prices):
    cdef double max_profit = -1.0
    cdef int i,j
    for i in range(len(prices)):
        for j in range(i):
            if prices[i] - prices[j] > max_profit:
                max_profit = prices[i] - prices[j]
    return max_profit

# -------------------------------------------------------------------------------------
# Implement *only* the (naive algorithm's) INNER loop.
# NOTE: the nogil keyword does NOT release Python's infamous GIL, it only
# instructs Cython that the function *may* be called without the GIL.
# -------------------------------------------------------------------------------------
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef double inner_loop_max_profit(int i, double[::1] prices) nogil:
    cdef double max_profit = -1.0
    cdef double final_price = prices[i]
    cdef int j
    for j in range(i):
        if final_price - prices[j] > max_profit:
            max_profit = final_price - prices[j]
    return max_profit
