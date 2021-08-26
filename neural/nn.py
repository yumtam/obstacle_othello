import numpy as np
from typing import Dict

import game.bitboard as bitop
import game.util as util
from neural.model import ResidualCNN


class Node:
    def __init__(self, parent, my, opp, obs, turn=1, P=0):
        self.parent = parent
        self.my = my
        self.opp = opp
        self.obs = obs
        self.turn = turn
        self.N = self.W = self.Q = 0
        self.P = P
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

        return Node(None, my, opp, obs)

    def simulate(self):
        node = self
        while not (node.leaf or node.terminal):
            node = node.select()

        node.expand()
        node.backup()

    def best_move_mcts(self):
        if self.terminal:
            return None

        for _ in range(Node.simul_N):
            self.simulate()

        action = max(self.edges, key=lambda a: self.edges[a].N)

        return action

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
            U = Node.c_puct * child.P * N_total / (1 + child.N)
            value = child.Q + U

            if value > best_value:
                best_action = action
                best_value = value

        return self.edges[best_action]

    def evaluate(self):
        P_a_logit, V = Node.model.eval(self.my, self.opp, self.obs)
        P_a_logit = P_a_logit[0]
        P_a = 1 / (1 + np.exp(-P_a_logit))
        V = V[0]

        if self.terminal:
            my_score = bitop.popcount(self.my)
            opp_score = bitop.popcount(self.opp)

            if my_score > opp_score:
                V = 1
            elif my_score == opp_score:
                V = 0.5
            else:
                V = 0

        return P_a, V

    def expand(self):
        self.leaf = False
        P_a, V = self.evaluate()

        if self.my_moves:
            for action in range(8 * 8):
                if self.my_moves & (1 << action):
                    my, opp = bitop.resolve_move(self.my, self.opp, action)
                    self.edges[action] = Node(self, opp, my, self.obs, -self.turn, P_a[action])
        elif self.opp_moves:
            self.edges[64] = Node(self, self.opp, self.my, self.obs, -self.turn, P_a[64])

        self.V = V

    def backup(self):
        node = self
        while not node.is_root():
            node = node.parent
            node.N += 1
            node.W += self.V if (node.turn == self.turn) else 1-self.V
            node.Q = node.W / node.N

    def play(self):
        pass

    def __repr__(self):
        my = self.
        opp = self.opp

        if self.turn == -1:
            my, opp = opp, my

        return bitop.to_string(my, opp, self.obs)


class MTCS:
    def __init__(self, c_puct, simul_N, model_id):
        Node.c_puct = c_puct
        Node.simul_N = simul_N

        Node.model = ResidualCNN()
        Node.model.load(model_id)

    def run(self, my, opp, obs):
        self.root = Node(None, my, opp, obs)
        action = self.root.best_move_mcts()
        return action

mtcs = MTCS(c_puct=0.05, simul_N=100, model_id=0)

def run(my, opp, obs):
    return mtcs.run(my, opp, obs)

