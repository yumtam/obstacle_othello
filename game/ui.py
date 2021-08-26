import time
import threading
import game.bitboard as bitop
import game.util as util
import game.consts as consts
from functools import partial
from tkinter import *

colors = {
    'p0': 'lightskyblue',
    'p1': 'lightsalmon',
    'obs': 'gray30',
    'empty': 'gray95',
    'move': 'lightyellow',
    'bestmove': 'yellow',
}

window = Tk()
window.eval('tk::PlaceWindow . center')
window.resizable(False, False)

left_frame = Frame(window)
left_frame.pack(side='left')

right_frame = Frame(window)
right_frame.pack(side='right')

score_frame = Frame(left_frame, relief='solid', bd=1)
score_frame.pack(side='top')

board_frame = Frame(left_frame)
board_frame.pack(side='bottom')

board_state = [0, 0, 0]

p0_bar = Label(score_frame, anchor='w', text='0', width=2, bg=colors['p0'])
p1_bar = Label(score_frame, anchor='e', text='0', bg=colors['p1'])
empty_bar = Label(score_frame, bg=colors['empty'])

p0_bar.pack(side='left')
empty_bar.pack(side='left')
p1_bar.pack(side='left')

buttons = [[None] * 8 for _ in range(8)]
for x in range(8):
    for y in range(8):
        b = Button(board_frame, width=6, height=3)
        b.grid(row=x, column=y)
        buttons[x][y] = b


def board_update(p0, p1, obs):
    board_state[0] = p0
    board_state[1] = p1
    board_state[2] = obs

    str_raw = bitop.to_string(p0, p1, obs)
    str_grid = str_raw.split()

    trans = {
        consts.BOARD_STR_MY: 'p0',
        consts.BOARD_STR_OPP: 'p1',
        consts.BOARD_STR_OBS: 'obs',
        consts.BOARD_STR_EMPTY: 'empty',
    }

    for x in range(8):
        for y in range(8):
            c = str_grid[x][y]
            if c in trans:
                color = colors[trans[c]]
                buttons[x][y].config(bg=color)

    p0_pts = bitop.popcount(p0)
    p1_pts = bitop.popcount(p1)
    top_frame_update(p0_pts, p1_pts)


def top_frame_update(p0_pts, p1_pts):
    total_width = 6 * 8

    p0_width = p0_pts * total_width // (8 * 8 - 5)
    p1_width = p1_pts * total_width // (8 * 8 - 5)
    empty_width = total_width - p0_width - p1_width

    p0_bar.config(text=p0_pts, width=p0_width)
    p1_bar.config(text=p1_pts, width=p1_width)
    empty_bar.config(width=empty_width)


def clear_all_callbacks():
    for b_row in buttons:
        for b in b_row:
            b.config(command=lambda: None)


def self_play():
    turn = 0
    arr = util.initial_setup()
    bits = bitop.array_to_bits(arr)
    my, opp, obs = bits

    while not bitop.is_terminated(my, opp, obs):
        board_update(my, opp, obs)

        if turn:
            my, opp = opp, my

        moves = bitop.generate_moves(my, opp, obs)

        if moves:
            move_index = IntVar()

            def callback(i):
                print(i)
                nonlocal move_index
                move_index.set(i)

            for x in range(8):
                for y in range(8):
                    i = x * 8 + y
                    if moves & (1 << i):
                        buttons[x][y].config(bg=colors['move'], command=partial(callback, i=i))

            window.wait_variable(move_index)
            clear_all_callbacks()

            my, opp = bitop.resolve_move(my, opp, move_index.get())

        if turn:
            my, opp = opp, my

        turn ^= 1

    board_update(my, opp, obs)


def ai_play():
    import engines.mcts100k as ai
    ai_turn = 1
    turn = 0
    arr = util.initial_setup()
    bits = bitop.array_to_bits(arr)
    my, opp, obs = bits

    while not bitop.is_terminated(my, opp, obs):
        board_update(my, opp, obs)

        if turn:
            my, opp = opp, my

        moves = bitop.generate_moves(my, opp, obs)

        if moves:
            move_index = IntVar()

            if turn == ai_turn:
                class AI(threading.Thread):
                    def __init__(self, *args):
                        super().__init__()
                        self.args = args

                    def run(self):
                        i = ai.run(*self.args)
                        move_index.set(i)

                AI(my, opp, obs).start()

            else:
                def callback(i):
                    print(i)
                    nonlocal move_index
                    move_index.set(i)

                for x in range(8):
                    for y in range(8):
                        i = x * 8 + y
                        if moves & (1 << i):
                            buttons[x][y].config(bg=colors['move'], command=partial(callback, i=i))

            window.wait_variable(move_index)
            clear_all_callbacks()

            my, opp = bitop.resolve_move(my, opp, move_index.get())

        if turn:
            my, opp = opp, my

        turn ^= 1

    board_update(my, opp, obs)


def analysis_mcts():
    import engines.mcts100k as ai
    turn = 0
    arr = util.initial_setup()
    bits = bitop.array_to_bits(arr)
    my, opp, obs = bits

    while not bitop.is_terminated(my, opp, obs):
        board_update(my, opp, obs)

        if turn:
            my, opp = opp, my

        moves = bitop.generate_moves(my, opp, obs)

        root = ai.Node(None, my, opp, obs, 0)

        class Event():
            def __init__(self):
                self._v = False

            def __bool__(self):
                return self._v

            def set(self):
                self._v = True

        exitEvent = Event()

        class AI(threading.Thread):
            def __init__(self, exit):
                super().__init__()
                self.exit = exit

            def run(self):
                root.best_move_infinite(self.exit)

        AI(exitEvent).start()

        class Drawer(threading.Thread):
            def __init__(self, exit, done):
                super().__init__()
                self.exit = exit
                self.done = done

            def run(self):
                time.sleep(0.1)
                while not self.exit:
                    edges = root.edges
                    best_i = max((i for i in edges), key=lambda i: edges[i].Q)
                    for i, node in edges.items():
                        x, y = divmod(i, 8)
                        text = f'{node.N // 1000}k\n{round(node.Q * 100, 1)}%'
                        bg = colors['bestmove'] if i == best_i else colors['move']
                        buttons[x][y].config(text=text, bg=bg)
                    time.sleep(0.1)
                for i in root.edges:
                    x, y = divmod(i, 8)
                    buttons[x][y].config(text='')
                drawing_thread_done.set(True)

        drawing_thread_done = BooleanVar(False)
        drawing_thread = Drawer(exitEvent, drawing_thread_done)
        drawing_thread.start()

        if moves:
            move_index = IntVar()

            def callback(i):
                print(i)
                nonlocal move_index
                move_index.set(i)

            for x in range(8):
                for y in range(8):
                    i = x * 8 + y
                    if moves & (1 << i):
                        buttons[x][y].config(bg=colors['move'], command=partial(callback, i=i))

        window.wait_variable(move_index)
        exitEvent.set()
        window.wait_variable(drawing_thread_done)

        clear_all_callbacks()

        my, opp = bitop.resolve_move(my, opp, move_index.get())

        if turn:
            my, opp = opp, my

        turn ^= 1

    board_update(my, opp, obs)


self_play_btn = Button(right_frame, text='Self Play', command=self_play)
self_play_btn.pack(fill='x')

ai_play_btn = Button(right_frame, text='vs Com', command=ai_play)
ai_play_btn.pack(fill='x')

analysis_btn = Button(right_frame, text='Analysis', command=analysis_mcts)
analysis_btn.pack(fill='x')

window.mainloop()
