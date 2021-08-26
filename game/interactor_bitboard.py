import importlib
import game.bitboard as bitop
import game.util as util


def run(module_name_1, module_name_2, N=1000):
    m1 = importlib.import_module(module_name_1)
    m2 = importlib.import_module(module_name_2)

    modules = [m1, m2]
    wins = [0, 0]

    for i in range(N):
        arr = util.initial_setup()
        p1, p2, obs = bitop.array_to_bits(arr)

        first = modules[i % 2]
        second = modules[1 - i % 2]
        turn = 0

        while not bitop.is_terminated(p1, p2, obs):
            if turn == 0:
                if bitop.generate_moves(p1, p2, obs):
                    move_idx = first.run(p1, p2, obs)
                    assert bitop.generate_moves(p1, p2, obs) & (1 << move_idx)
                    p1, p2 = bitop.resolve_move(p1, p2, move_idx)
            else:
                if bitop.generate_moves(p2, p1, obs):
                    move_idx = second.run(p2, p1, obs)
                    assert bitop.generate_moves(p2, p1, obs) & (1 << move_idx)
                    p2, p1 = bitop.resolve_move(p2, p1, move_idx)
            turn ^= 1

        res = bitop.evaluate(p1, p2, obs)
        if res > 0:
            wins[i % 2] += 1
        elif res == 0:
            wins[0] += 0.5
            wins[1] += 0.5
        else:
            wins[1 - i % 2] += 1

        print(wins)

    return wins


if __name__ == '__main__':
    wins = run('engines.mcts', 'neural.nn', 100)
    print(wins)
