"""
Spring 2021 ML Course.
Exercise 1: Basic Numpy (plus a few tricks).
Section A: drunken walk simulation.
Starting at position 0, simulate n drunk moves, all in {-1, +1}.
"""
import numpy as np
import random
import timeit

# Ronni Roshbakir's solution: pure Python code.
def random_walk_pure_python(n):
    position = 0
    walk = [position]  # This list will hold our random walk.
    for i in range(n):
        position += 2*random.randint(0,1) - 1
        walk.append(position)
    return walk

# Can you beat Ronni's solution by using vectorized Numpy?
# Batya Bingo's solution: use np.random.choice() and np.cumsum().
def random_walk_vectorized(n):
###
###
###

# Write a few stats.
print("Drunken walk: pure Python implementation:",
      timeit.timeit(
          number=100,
          stmt="random_walk_pure_python(10000)",
          setup="from __main__ import random_walk_pure_python"))

print("Drunken walk: vectorized Numpy implementation:",
      timeit.timeit(
          number=100,
          stmt="random_walk_vectorized(10000)",
          setup="from __main__ import random_walk_vectorized"))

