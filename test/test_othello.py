import unittest

from game.othello import Othello
import numpy as np


class OthelloTests(unittest.TestCase):
    def test_array(self):
        o = Othello(1, 2, 4)
        arr = np.zeros((3, 8, 8), dtype=np.uint64)
        arr[0, 0, 0] = 1
        arr[1, 0, 1] = 1
        arr[2, 0, 2] = 1

        self.assertTrue(np.allclose(o.array, arr))

    def test_from_array(self):
        arr = np.zeros((3, 8, 8), dtype=np.uint64)
        arr[0, 0, 0] = 1
        arr[1, 0, 1] = 1
        arr[2, 0, 2] = 1
        o = Othello.from_array(arr)
        bits = np.array([1, 2, 4], dtype=np.uint64)

        self.assertTrue(np.allclose(o.bits, bits))

    def test_terminated(self):
        self.assertTrue(Othello(0, 0, 0).terminated())
        self.assertFalse(Othello(1, 2, 8).terminated())

    def test_make_move(self):
        o = Othello(1, 2, 8)
        o = o.make_move(0, 2)
        bits = np.array([0, 7, 8], dtype=np.uint64)

        self.assertTrue(np.allclose(o.bits, bits))