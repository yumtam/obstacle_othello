import numpy as np
from numba import njit, u8
from numba.experimental import jitclass
import game.bitboard as bitop


@njit
def bits_to_array(bits):
    arr = np.empty((3, 8, 8), dtype=u8)

    arr[0] = bitop.unpack(bits[0])
    arr[1] = bitop.unpack(bits[1])
    arr[2] = bitop.unpack(bits[2])

    return arr


@njit
def array_to_bits(arr):
    bits = np.empty((3,), dtype=u8)

    bits[0] = bitop.pack(arr[0])
    bits[1] = bitop.pack(arr[1])
    bits[2] = bitop.pack(arr[2])

    return bits


@jitclass([('bits', u8[:])])
class Othello:
    def __init__(self, my, opp, obs):
        self.bits = np.array([my, opp, obs], dtype=u8)

    @property
    def array(self):
        return bits_to_array(self.bits)

    @staticmethod
    def from_array(arr):
        my, opp, obs = array_to_bits(arr)
        return Othello(my, opp, obs)

    def terminated(self):
        my, opp, obs = self.bits
        i_cant_move = (bitop.generate_moves(my, opp, obs) == 0)
        opp_cant_move = (bitop.generate_moves(opp, my, obs) == 0)

        return i_cant_move and opp_cant_move

    def my_moves(self):
        my, opp, obs = self.bits
        my_moves = bitop.generate_moves(my, opp, obs)

        return bitop.unpack(my_moves)

    def make_move(self, row, col):
        my, opp, obs = self.bits
        index = row * 8 + col
        my, opp = bitop.resolve_move(my, opp, index)

        return Othello(opp, my, obs)

    def make_move_pass(self):
        my, opp, obs = self.bits

        return Othello(opp, my, obs)

    def to_string(self):
        arr = self.array

        res = ''
        for x in range(8):
            for y in range(8):
                if arr[0][x][y]:
                    res += '.'
                elif arr[1][x][y]:
                    res += 'x'
                elif arr[2][x][y]:
                    res += '#'
                else:
                    res += ' '
            res += '\n'

        return res

    __repr__ = _str__ = to_string
