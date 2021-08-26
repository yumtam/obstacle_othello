from numba import njit
import random
import game.bitboard as bitop


@njit('i4(u8, u8, u8)')
def run(my, opp, obs):
    my_moves = bitop.generate_moves(my, opp, obs)

    while True:
        i = random.randrange(64)
        if my_moves & (1 << i):
            return i
