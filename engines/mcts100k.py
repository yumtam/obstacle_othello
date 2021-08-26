from numba import njit
from typing import Dict
import random

import game.bitboard as bitop
import game.util as util


c_puct = 0.01
simul_N = 100000


class Node:
    def __init__(self, parent, my, opp, obs, turn):
        self.parent = parent
        self.my = my
        self.opp = opp
        self.obs = obs
        self.turn = turn
        self.N = self.W = self.Q = 0
        self.edges: Dict[int, Node] = {}

        self.leaf = True
        self.my_moves = bitop.generate_moves(my, opp, obs)
        self.opp_moves = bitop.generate_moves(opp, my, obs)
        self.terminal = (self.my_moves == self.opp_moves == 0)

    def is_root(self):
        return self.parent is None

    @staticmethod
    def from_array(arr):
        my, opp, obs = bitop.array_to_bits(arr)

        return Node(None, my, opp, obs, 0)

    def simulate(self):
        node = self
        while not (node.leaf or node.terminal):
            node = node.select()

        node.expand()
        node.backup()

    def best_move_mcts(self):
        if self.terminal:
            return None

        for _ in range(simul_N):
            self.simulate()

        action = max(self.edges, key=lambda a: self.edges[a].N)

        return action

    def best_move_infinite(self, exit):
        if self.terminal:
            return

        while not exit:
            self.simulate()

    def make_move_mcts(self):
        action = self.best_move_mcts()
        next_root = self.edges[action]
        next_root.parent = None

        return next_root

    def select(self):
        best_action = 0
        best_value = float('-inf')

        N_total = sum(child.N for child in self.edges.values())

        for action, child in self.edges.items():
            U = c_puct * N_total / (1 + child.N)
            value = child.Q + U

            if value > best_value:
                best_action = action
                best_value = value

        return self.edges[best_action]

    def evaluate(self):
        my, opp, obs = self.my, self.opp, self.obs
        return Node._fast_evaluate(my, opp, obs)

    @staticmethod
    @njit('i4(u8, u8, u8)')
    def _fast_evaluate(my, opp, obs):
        turn = 0

        while not bitop.is_terminated(my, opp, obs):
            my_moves = bitop.generate_moves(my, opp, obs)

            if my_moves:
                while True:
                    i = random.randrange(64)
                    if my_moves & (1 << i):
                        break
                my, opp = bitop.resolve_move(my, opp, i)

            my, opp = opp, my

            turn ^= 1

        if turn:
            my, opp = opp, my

        score = bitop.evaluate(my, opp, obs)
        if score > 0:
            V = 1
        elif score == 0:
            V = 0.5
        else:
            V = 0

        return V

    def expand(self):
        self.leaf = False
        V = self.evaluate()

        if self.my_moves:
            for action in range(8 * 8):
                if self.my_moves & (1 << action):
                    my, opp = bitop.resolve_move(self.my, self.opp, action)
                    self.edges[action] = Node(self, opp, my, self.obs, self.turn ^ 1)
        elif self.opp_moves:
            self.edges[64] = Node(self, self.opp, self.my, self.obs, self.turn ^ 1)

        self.V = V

    def backup(self):
        node = self
        while True:
            node.N += 1
            node.W += 1 - self.V if (node.turn == self.turn) else self.V
            node.Q = node.W / node.N
            if node.is_root():
                break
            node = node.parent

    def play(self):
        pass

    def __repr__(self):
        my = self.my
        opp = self.opp

        if self.turn == 1:
            my, opp = opp, my

        return bitop.to_string(my, opp, self.obs)


def test(N):
    arr = util.initial_setup()

    for _ in range(N):
        root = Node.from_array(arr)
        while root is not None:
            print(root, root.turn, root.turn * root.Q)
            root = root.make_move_mcts()


def run(my, opp, obs):
    root = Node(None, my, opp, obs, 0)
    action = root.best_move_mcts()
    #print(root, root.edges[action].Q)
    return action

