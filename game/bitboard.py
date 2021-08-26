import numpy as np
from numba import njit
import game.consts as consts


@njit('u8(u8[:, :])')
def pack(arr):
    bits = np.uint64(0)
    for x in range(8):
        for y in range(8):
            bits += arr[x, y] << (x * 8 + y)
    return bits


@njit('u8[:, :](u8)')
def unpack(bits):
    arr = np.empty((8, 8), dtype=np.uint64)
    for x in range(8):
        for y in range(8):
            arr[x, y] = bits & 1
            bits >>= 1
    return arr


@njit('u8[:, :, :](u8, u8, u8)')
def bits_to_array(my, opp, obs):
    arr = np.empty((3, 8, 8), dtype=np.uint64)

    arr[0] = unpack(my)
    arr[1] = unpack(opp)
    arr[2] = unpack(obs)

    return arr


@njit('Tuple((u8, u8, u8))(u8[:, :, :])')
def array_to_bits(arr):
    my = pack(arr[0])
    opp = pack(arr[1])
    obs = pack(arr[2])

    return my, opp, obs


@njit('u8(u8)')
def popcount(x):
    x -= ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + (x >> 2 & 0x3333333333333333)
    return (((x + (x >> 4)) & 0xf0f0f0f0f0f0f0f) * 0x101010101010101 >> 56) & 0xff


def to_string(my, opp, obs):
    pos = 1

    res = ''
    for idx in range(8 * 8):
        if my & pos:
            res += consts.BOARD_STR_MY
        elif opp & pos:
            res += consts.BOARD_STR_OPP
        elif obs & pos:
            res += consts.BOARD_STR_OBS
        else:
            res += consts.BOARD_STR_EMPTY

        if idx % 8 == 7:
            res += '\n'

        pos <<= 1

    return res


def from_string(s):
    s = sum(s.split(), '')

    assert len(s) == 8 * 8

    pos = 1
    my, opp, obs = 0, 0, 0
    for c in s:
        if c == consts.BOARD_STR_MY:
            my |= pos
        elif c == consts.BOARD_STR_OPP:
            opp |= pos
        elif c == consts.BOARD_STR_OBS:
            obs |= pos
        else:
            assert c == consts.BOARD_STR_EMPTY
        pos <<= 1


# Bitboard operations.

MASKS = np.array([
    0x7F7F7F7F7F7F7F7F,  # Right.
    0x007F7F7F7F7F7F7F,  # Down-right.
    0xFFFFFFFFFFFFFFFF,  # Down.
    0x00FEFEFEFEFEFEFE,  # Down-left.
    0xFEFEFEFEFEFEFEFE,  # Left.
    0xFEFEFEFEFEFEFE00,  # Up-left.
    0xFFFFFFFFFFFFFFFF,  # Up.
    0x7F7F7F7F7F7F7F00,  # Up-right.
], dtype=np.uint64)

LSHIFTS = np.array([
    0,  # Right.
    0,  # Down-right.
    0,  # Down.
    0,  # Down-left.
    1,  # Left.
    9,  # Up-left.
    8,  # Up.
    7,  # Up-right.
])

RSHIFTS = np.array([
    1,  # Right.
    9,  # Down-right.
    8,  # Down.
    7,  # Down-left.
    0,  # Left.
    0,  # Up-left.
    0,  # Up.
    0,  # Up-right.
])

assert len(MASKS) == len(LSHIFTS) == len(RSHIFTS)

NUM_DIRS = len(MASKS)


@njit('u8(u8, i4)')
def shift(disks, dir):
    assert 0 <= dir < NUM_DIRS

    if dir < NUM_DIRS // 2:
        assert LSHIFTS[dir] == 0, "Shifting right."
        return (disks >> RSHIFTS[dir]) & MASKS[dir]
    else:
        assert RSHIFTS[dir] == 0, "Shifting left."
        return (disks << LSHIFTS[dir]) & MASKS[dir]


@njit('u8(u8, u8, u8)')
def generate_moves(my_disks, opp_disks, obstacles):
    assert (my_disks & opp_disks) == 0, "Disk sets should be disjoint."

    empty_cells = ~(my_disks | opp_disks | obstacles)
    legal_moves = np.uint64(0)

    for dir in range(NUM_DIRS):
        # Get opponent disks adjacent to my disks in direction dir.
        x = shift(my_disks, dir) & opp_disks

        # Add opponent disks adjacent to those, and so on.
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks

        # Empty cells adjacent to those are valid moves.
        legal_moves |= shift(x, dir) & empty_cells

    return legal_moves


@njit('i4(u8, u8, u8)')
def is_terminated(my_disks, opp_disks, obstacles):
    i_can_move = generate_moves(my_disks, opp_disks, obstacles) > 0
    opp_can_move = generate_moves(opp_disks, my_disks, obstacles) > 0

    return not (i_can_move or opp_can_move)


@njit('i4(u8, u8, u8)')
def evaluate(my_disks, opp_disks, obstacles):
    my_score = popcount(my_disks)
    opp_score = popcount(opp_disks)

    return my_score - opp_score


@njit('Tuple((u8, u8))(u8, u8, i4)')
def resolve_move(my_disks, opp_disks, board_idx):
    new_disk = np.uint64(1 << board_idx)
    captured_disks = np.uint64(0)

    assert 0 <= board_idx < 64, "Move must be within the board."
    assert my_disks & opp_disks == 0, "Disk sets must be disjoint."
    assert (my_disks | opp_disks) & new_disk == 0, "Target not empty!"

    my_disks |= new_disk

    for dir in range(NUM_DIRS):
        # Find opponent disk adjacent to the new disk.
        x = shift(new_disk, dir) & opp_disks

        # Add any adjacent opponent disk to that one, and so on.
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks
        x |= shift(x, dir) & opp_disks

        # Determine whether the disks were captured.
        bounding_disk = shift(x, dir) & my_disks
        captured_disks |= x if bounding_disk else np.uint64(0)

    assert captured_disks, "A valid move must capture disks."

    my_disks ^= captured_disks
    opp_disks ^= captured_disks

    assert my_disks & opp_disks == 0, "The sets must still be disjoint."

    return my_disks, opp_disks
