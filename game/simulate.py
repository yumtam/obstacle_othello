import numpy as np
from numba import njit
from game.othello import Othello, array_to_bits
import game.bitboard as bitop
import random


@njit
def simulate(o: Othello):
    turn = 0
    while not o.terminated():
        moves = bitop.pack(o.my_moves())

        if moves == 0:
            o = o.make_move_pass()
        else:
            while True:
                index = random.randrange(64)
                if (1 << index) & moves:
                    row, col = divmod(index, 8)
                    o = o.make_move(row, col)
                    break
        turn ^= 1

    if turn:
        o = o.make_move_pass()

    p0 = np.sum(o.array[0])
    p1 = np.sum(o.array[1])

    print(o.to_string())

    if p0 > p1:
        return 1
    elif p0 == p1:
        return 0.5
    else:
        return 0


@njit
def test(N):
    arr = np.zeros((3, 8, 8), dtype=np.uint64)

    arr[0][3][3] = 1
    arr[0][4][4] = 1
    arr[1][3][4] = 1
    arr[1][4][3] = 1
    arr[2][2][2] = 1
    arr[2][5][5] = 1

    my, opp, obs = array_to_bits(arr)

    pts = 0
    for _ in range(N):
        pts += simulate(Othello(my, opp, obs))

    print(pts)


if __name__ == '__main__':
    test(1)
