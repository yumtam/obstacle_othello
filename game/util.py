import random
import numpy as np
import game.bitboard as bitop


PLAYER_BLACK = 0
PLAYER_WHITE = 1
OBSTACLE = 2


def empty_array():
    arr = np.zeros((3, 8, 8), dtype=np.uint64)
    return arr

def set_cell_state(arr, row, col, state):
    arr[state][row][col] = 1

def clear_cell_state(arr, row, col):
    arr[0][row][col] = 0
    arr[1][row][col] = 0
    arr[2][row][col] = 0

def initial_setup(obs_n=5):
    arr = empty_array()

    set_cell_state(arr, 3, 3, PLAYER_BLACK)
    set_cell_state(arr, 4, 4, PLAYER_BLACK)
    set_cell_state(arr, 3, 4, PLAYER_WHITE)
    set_cell_state(arr, 4, 3, PLAYER_WHITE)

    pool = list(range(8 * 8))
    pool.remove(3 * 8 + 4)
    pool.remove(4 * 8 + 3)
    pool.remove(3 * 8 + 3)
    pool.remove(4 * 8 + 4)

    for i in random.sample(pool, obs_n):
        row, col = divmod(i, 8)
        set_cell_state(arr, row, col, OBSTACLE)

    return arr
