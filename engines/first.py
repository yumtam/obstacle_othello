from numba import njit
import random
import game.bitboard as bitop


@njit('i4(u8, u8, u8)')
def run(my, opp, obs):
    my_moves = bitop.generate_moves(my, opp, obs)

    for i in range(8 * 8):
        if my_moves & (1 << i):
            return i
    return 0