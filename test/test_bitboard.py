import unittest

import game.bitboard as bitop
import numpy as np


class BitBoardTests(unittest.TestCase):
    def test_pack_one(self):
        arr = np.zeros((8, 8), dtype=np.uint64)
        arr[0, 0] = 1

        self.assertEqual(bitop.pack(arr), np.uint64(1))

    def test_unpack_one(self):
        bits = np.uint64(1)
        arr = np.zeros((8, 8), dtype=np.uint64)
        arr[0, 0] = 1

        self.assertTrue(np.allclose(bitop.unpack(bits), arr))

    def test_pack_max(self):
        arr = np.ones((8, 8), dtype=np.uint64)

        self.assertEqual(bitop.pack(arr), np.uint64(2 ** 64 - 1))

    def test_unpack_max(self):
        bits = np.uint64(2 ** 64 - 1)
        arr = np.ones((8, 8), dtype=np.uint64)

        self.assertTrue(np.allclose(bitop.unpack(bits), arr))

    def test_shift(self):
        res = bitop.shift(1, 4)

        self.assertEqual(res, 2)

    def test_generate_moves(self):
        legal_moves = bitop.generate_moves(1, 2, 8)

        self.assertEqual(legal_moves, 4)

    def test_resolve_move(self):
        my_disks, opp_disks = bitop.resolve_move(1, 2, 2)

        self.assertEqual(my_disks, 7)
        self.assertEqual(opp_disks, 0)
