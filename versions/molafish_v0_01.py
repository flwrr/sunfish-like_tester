#!/usr/bin/env pypy3
from __future__ import print_function

import time, math
from itertools import count
from collections import namedtuple, defaultdict

version = "molafish v0.01"

###############################################################################
# Piece-Square tables. Tune these to change sunfish's behaviour
###############################################################################
# TODO: Compression test
piece = [100, 479, 280, 320, 929, 60000]        # P,R,N,B,Q,K
pst = [
    (    0,   0,   0,   0,   0,   0,   0,   0,  # [0] Pawn
        78,  83,  86,  73, 102,  82,  85,  90,
         7,  29,  21,  44,  40,  31,  44,   7,
       -17,  16,  -2,  15,  14,   0,  15, -13,
       -26,   3,  10,   9,   6,   1,   0, -23,
       -22,   9,   5, -11, -10,  -2,   3, -19,
       -31,   8,  -7, -37, -36, -14,   3, -31,
         0,   0,   0,   0,   0,   0,   0,  0),
    (   35,  29,  33,   4,  37,  33,  56,  50,  # [1] Rook
        55,  29,  56,  67,  55,  62,  34,  60,
        19,  35,  28,  33,  45,  27,  25,  15,
         0,   5,  16,  13,  18,  -4,  -9,  -6,
       -28, -35, -16, -21, -13, -29, -46, -30,
       -42, -28, -42, -25, -25, -35, -26, -46,
       -53, -38, -31, -26, -29, -43, -44, -53,
       -30, -24, -18,   5,  -2, -18, -31, -32),
    (  -66, -53, -75, -75, -10, -55, -58, -70,  # [2] Knight
        -3,  -6, 100, -36,   4,  62,  -4, -14,
        10,  67,   1,  74,  73,  27,  62,  -2,
        24,  24,  45,  37,  33,  41,  25,  17,
        -1,   5,  31,  21,  22,  35,   2,   0,
       -18,  10,  13,  22,  18,  15,  11, -14,
       -23,  -15,   2,   0,   2,   0, -23,-20,
       -74, -23, -26, -24, -19, -35, -22, -69),
    (  -59, -78, -82, -76, -23,-107, -37, -50,  # [3] Bishop
       -11,  20,  35, -42, -39,  31,   2, -22,
        -9,  39, -32,  41,  52, -10,  28, -14,
        25,  17,  20,  34,  26,  25,  15,  10,
        13,  10,  17,  23,  17,  16,   0,   7,
        14,  25,  24,  15,   8,  25,  20,  15,
        19,  20,  11,   6,   7,   6,  20,  16,
        -7,   2, -15, -12, -14, -15, -10, -10),
    (    6,   1,  -8,-104,  69,  24,  88,  26,  # [4] Queen
        14,  32,  60, -10,  20,  76,  57,  24,
        -2,  43,  32,  60,  72,  63,  43,   2,
         1, -16,  22,  17,  25,  20, -13,  -6,
       -14, -15,  -2,  -5,  -1, -10, -20, -22,
       -30,  -6, -13, -11, -16, -11, -16, -27,
       -36, -18,   0, -19, -15, -15, -21, -38,
       -39, -30, -31, -13, -31, -36, -34, -42),
    (    4,  54,  47, -99, -99,  60,  83, -62,   # [5] King
       -32,  10,  55,  56,  56,  55,  10,   3,
       -62,  12, -57,  44, -67,  28,  37, -31,
       -55,  50,  11,  -4, -19,  13,   0, -49,
       -55, -43, -52, -28, -51, -47,  -8, -50,
       -47, -42, -43, -79, -64, -32, -29, -32,
        -4,   3, -14, -50, -57, -18,  13,   4,
        17,  30,  -3, -14,   6,  -1,  40,  18),
]
# Reorder PST to 'Little-Endian File-Rank Mapping' (A1=[0], H8=[63])
pst = [
    tuple(val + piece[i] for rank_start in range(56, -1, -8)
    for val in table[rank_start:rank_start+8]) for i, table in enumerate(pst)
]

###############################################################################
# Global constants
###############################################################################

# Our board is represented as 14 64-bit integer bitboards.
BOARD = 0xffffffffffffffff
A1, H1, A8, H8 = 0x1, 0x80, (0x1 << 8*7), (0x80 << 8*7)
FILE_A, FILE_B = 0x0101010101010101, 0x202020202020202
FILE_G, FILE_H = 0x4040404040404040, 0x8080808080808080
RANK_2, RANK_8 = 0xff00, 0xff00000000000000

initial = (
    0b11111111 << 8,        #  [0] White Pawn
    0b10000001,             #  [1] White Rook
    0b01000010,             #  [2] White Knight
    0b00100100,             #  [3] White Bishop
    0b00001000,             #  [4] White Queen
    0b00010000,             #  [5] White King
    0b11111111 << (8 * 6),  #  [6] Black Pawn
    0b10000001 << (8 * 7),  #  [7] Black Rook
    0b01000010 << (8 * 7),  #  [8] Black Knight
    0b00100100 << (8 * 7),  #  [9] Black Bishop
    0b00001000 << (8 * 7),  # [10] Black Queen
    0b00010000 << (8 * 7),  # [11] Black King
    0xffff,                 # [12] All White Pieces
    0xffff000000000000,     # [13] All Black Pieces
    0xffff00000000ffff      # [14] All Pieces
)


# Lists of possible moves for each piece type.
# Note: nested lamda functions probably have high overhead.
def n(b): return b << 8 & BOARD
def s(b): return b >> 8
def e(b): return (b << 1) & ~FILE_A & BOARD
def w(b): return (b >> 1) & ~FILE_H
def ne(b): return e(n(b))
def nw(b): return w(n(b))
def se(b): return e(s(b))
def sw(b): return w(s(b))


directions = [
    [n, lambda b: n(n(b)), nw, ne],             # [0] Pawn
    [n, e, s, w],                               # [1] Rook
    [lambda b: n(ne(b)), lambda b: e(ne(b)),
     lambda b: e(se(b)), lambda b: s(se(b)),
     lambda b: s(sw(b)), lambda b: w(sw(b)),
     lambda b: w(nw(b)), lambda b: n(nw(b))],   # [2] Knight
    [ne, nw, se, sw],                           # [3] Bishop
    [n, e, s, w, ne, nw, se, sw],               # [4] Queen
    [n, e, s, w, ne, nw, se, sw],               # [5] King
]

# Mate value must be greater than 8*queen + 2*(rook+knight+bishop)
# King value is set to twice this value such that if the opponent is
# 8 queens up, but we got the king, we still exceed MATE_VALUE.
# When a MATE is detected, we'll set the score to MATE_UPPER - plies to get there
# E.g. Mate in 3 will be MATE_UPPER - 6
MATE_LOWER = piece[5] - 10 * piece[4]
MATE_UPPER = piece[5] + 10 * piece[4]

# Constants for tuning search
QS = 40
QS_A = 140
EVAL_ROUGHNESS = 15

# minifier-hide start
opt_ranges = dict(
    QS = (0, 300),
    QS_A = (0, 300),
    EVAL_ROUGHNESS = (0, 50),
)
# minifier-hide end


###############################################################################
# Chess logic
###############################################################################


Move = namedtuple("Move", "i j prom")


class Position(namedtuple("Position", "board score wc bc ep kp")):
    """A state of a chess game
    board -- a 120 char representation of the board
    score -- the board evaluation
    wc -- the castling rights, [west/queen side, east/king side]
    bc -- the opponent castling rights, [west/king side, east/queen side]
    ep - the en passant square (bitboard value)
    kp - the king passant square (bitboard value)
    """
    def gen_moves(self):
        own_pieces, opp_pieces, all_pieces = \
            self.board[12], self.board[13], self.board[14]
        for p, bb in enumerate(self.board[:6]):
            while bb:
                # i,j are bitboard values here (1 set bit for location)
                # Isolate and clear least significant bit
                i = bb & -bb
                bb ^= i
                for d in directions[p]:
                    j = d(i)
                    if not j or j & own_pieces: continue
                    if p == 0:
                        if (j == n(i) or j == n(n(i))) and j & all_pieces: continue
                        if j == n(n(i)) and (i & ~RANK_2 or n(i) & all_pieces): continue
                        if (
                            d in [nw, ne] and j & ~opp_pieces
                            and j & ~(self.ep | self.kp | self.kp << 1 | self.kp >> 1)
                        ):
                            continue
                        # promotion
                        if j & RANK_8:
                            for prom in [2, 3, 1, 4]:   # N,B,R,Q
                                yield Move(i, j, prom)
                            continue
                        yield Move(i, j, 0)
                    # Knight and King
                    elif p in [2, 5] and (j & ~own_pieces):
                        yield Move(i, j, 0)
                    # Rook, Bishop, Queen (generate rays)
                    else:
                        while j & ~own_pieces:
                            yield Move(i, j, 0)
                            if j & opp_pieces: break
                            # Castling, by sliding the rook next to the king
                            if i == A1 and (e(j) & self.board[5]) and self.wc[0]:
                                yield Move(e(j), w(j), 0)
                            if i == H1 and (w(j) & self.board[5]) and self.wc[1]:
                                yield Move(w(j), e(j), 0)
                            j = d(j)

    def rotate(self, nullmove=False):
        # Rotates each bitboard by 180 degrees to switch perspectives
        # Note: very beefy - performed on EVERY bitboard for EVERY move returned
        rotated = []
        for bb in self.board:
            # Reverse the bits in the bitboard
            bb = (bb & 0x5555555555555555) << 1 | (bb & 0xAAAAAAAAAAAAAAAA) >> 1
            bb = (bb & 0x3333333333333333) << 2 | (bb & 0xCCCCCCCCCCCCCCCC) >> 2
            bb = (bb & 0x0F0F0F0F0F0F0F0F) << 4 | (bb & 0xF0F0F0F0F0F0F0F0) >> 4
            bb = (bb & 0x00FF00FF00FF00FF) << 8 | (bb & 0xFF00FF00FF00FF00) >> 8
            bb = (bb & 0x0000FFFF0000FFFF) << 16 | (bb & 0xFFFF0000FFFF0000) >> 16
            bb = (bb & 0x00000000FFFFFFFF) << 32 | (bb & 0xFFFFFFFF00000000) >> 32
            rotated.append(bb)
        # Swap the roles of white and black pieces
        rotated[0:6], rotated[6:12] = rotated[6:12], rotated[0:6]
        rotated[12], rotated[13] = rotated[13], rotated[12]
        ep = 1 << (63 - (self.ep.bit_length()-1)) if self.ep and not nullmove else 0
        kp = 1 << (63 - (self.kp.bit_length()-1)) if self.kp and not nullmove else 0
        return Position(tuple(rotated), -self.score, self.bc, self.wc, ep, kp)

    def move(self, move):
        origin, dest, prom = move
        p, q = None, None
        # Copy variables and reset ep and kp
        board = list(self.board)
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        # Actual move
        for p_type, bb in enumerate(board[:12]):
            if dest & bb:
                for i in [13, p_type]: board[i] ^= dest
                q = p_type
            if origin & bb:
                for i in [12, p_type]: board[i] ^= origin | dest
                p = p_type
        # Castling rights, we move the rook or capture the opponent's
        if origin == A1: wc = (False, wc[1])
        if origin == H1: wc = (wc[0], False)
        if dest == A8: bc = (bc[0], False)      # bc currently unused due to rotate?
        if dest == H8: bc = (False, bc[1])
        # Castling (5=King)
        if p == 5:
            wc = (False, False)
            if w(w(origin)) == dest or e(e(origin)) == dest:
                kp = e(origin) if origin < dest else w(origin)
                rk_orig, rk_dest = (H1, w(dest)) if origin < dest else (A1, e(dest))
                for origin in [1, 12]: board[origin] ^= rk_orig | rk_dest
        # Pawn (promotion, double move and en passant capture. 0=Pawn)
        if p == 0:
            if dest & RANK_8:
                board[0] ^= dest
                board[prom] |= dest
            if n(n(origin)) == dest:
                ep = n(origin)
            if dest == self.ep:
                for origin in [6, 13]: board[origin] ^= s(self.ep)
        # Update all pieces bitboard
        board[14] = board[12] | board[13]
        # We rotate the returned position, so it's ready for the next player
        return Position(tuple(board), score, wc, bc, ep, kp).rotate()

    def value(self, move):
        origin, dest, prom = move
        # convert origin and destination to indices (A1 = 0, H8 = 63)
        i, j = origin.bit_length()-1, dest.bit_length()-1
        p, q = None, None
        for p_type, bb in enumerate(self.board[:12]):
            if origin & bb: p = p_type
            if dest & bb: q = p_type
        # Actual move
        score = pst[p][j] - pst[p][i]
        # Capture
        if q is not None:
            score += pst[q - 6][63 - j]
        # Castling check detection (kp rotates)
        if self.kp and abs(j - (self.kp.bit_length()-1)) < 2:
            score += pst[5][63 - j]
        # Castling (5=King)
        if p == 5 and abs(i - j) == 2:
            score += pst[1][(i + j) // 2]
            score -= pst[1][A1.bit_length()-1 if j < i else H1.bit_length()-1]
        # Special pawn stuff
        if p == 0:
            if dest & RANK_8:
                score += pst[prom][j] - pst[0][j]
            if dest == self.ep:
                score += pst[0][63 - (s(dest).bit_length()-1)]
        return score


###############################################################################
# Search logic
###############################################################################

# lower <= s(pos) <= upper
Entry = namedtuple("Entry", "lower upper")


class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.nodes = 0

    def bound(self, pos, gamma, depth, can_null=True):
        """ Let s* be the "true" score of the sub-tree we are searching.
            The method returns r, where
            if gamma >  s* then s* <= r < gamma  (A better upper bound)
            if gamma <= s* then gamma <= r <= s* (A better lower bound) """
        self.nodes += 1

        # Depth <= 0 is QSearch. Here any position is searched as deeply as is needed for
        # calmness, and from this point on there is no difference in behaviour depending on
        # depth, so so there is no reason to keep different depths in the transposition table.
        depth = max(depth, 0)

        # Sunfish is a king-capture engine, so we should always check if we
        # still have a king. Notice since this is the only termination check,
        # the remaining code has to be comfortable with being mated, stalemated
        # or able to capture the opponent king.
        if pos.score <= -MATE_LOWER:
            return -MATE_UPPER

        # Look in the table if we have already searched this position before.
        # We also need to be sure, that the stored search was over the same
        # nodes as the current search.
        entry = self.tp_score.get((pos, depth, can_null), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma: return entry.lower
        if entry.upper < gamma: return entry.upper

        # Let's not repeat positions. We don't chat
        # - at the root (can_null=False) since it is in history, but not a draw.
        # - at depth=0, since it would be expensive and break "futulity pruning".
        if can_null and depth > 0 and pos in self.history:
            return 0

        # Generator of moves to search in order.
        # This allows us to define the moves, but only calculate them if needed.
        def moves():
            # First try not moving at all. We only do this if there is at least one major
            # piece left on the board, since otherwise zugzwangs are too dangerous.
            # FIXME: We also can't null move if we can capture the opponent king.
            # Since if we do, we won't spot illegal moves that could lead to stalemate.
            # For now we just solve this by not using null-move in very unbalanced positions.
            # TODO: We could actually use null-move in QS as well. Not sure it would be very useful.
            # But still.... We just have to move stand-pat to be before null-move.
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ"):
            #if depth > 2 and can_null and any(c in pos.board for c in "RBNQ") and abs(pos.score) < 500:
            if depth > 2 and can_null and abs(pos.score) < 500:
                yield None, -self.bound(pos.rotate(nullmove=True), 1 - gamma, depth - 3)

            # For QSearch we have a different kind of null-move, namely we can just stop
            # and not capture anything else.
            if depth == 0:
                yield None, pos.score

            # Look for the strongest ove from last time, the hash-move.
            killer = self.tp_move.get(pos)

            # If there isn't one, try to find one with a more shallow search.
            # This is known as Internal Iterative Deepening (IID). We set
            # can_null=True, since we want to make sure we actually find a move.
            if not killer and depth > 2:
                self.bound(pos, gamma, depth - 3, can_null=False)
                killer = self.tp_move.get(pos)

            # If depth == 0 we only try moves with high intrinsic score (captures and
            # promotions). Otherwise we do all moves. This is called quiescent search.
            val_lower = QS - depth * QS_A

            # Only play the move if it would be included at the current val-limit,
            # since otherwise we'd get search instability.
            # We will search it again in the main loop below, but the tp will fix
            # things for us.
            if killer and pos.value(killer) >= val_lower:
                yield killer, -self.bound(pos.move(killer), 1 - gamma, depth - 1)

            # Then all the other moves
            for val, move in sorted(((pos.value(m), m) for m in pos.gen_moves()), reverse=True):
                # Quiescent search
                if val < val_lower:
                    break

                # If the new score is less than gamma, the opponent will for sure just
                # stand pat, since ""pos.score + val < gamma === -(pos.score + val) >= 1-gamma""
                # This is known as futility pruning.
                if depth <= 1 and pos.score + val < gamma:
                    # Need special case for MATE, since it would normally be caught
                    # before standing pat.
                    yield move, pos.score + val if val < MATE_LOWER else MATE_UPPER
                    # We can also break, since we have ordered the moves by value,
                    # so it can't get any better than this.
                    break

                yield move, -self.bound(pos.move(move), 1 - gamma, depth - 1)

        # Run through the moves, shortcutting when possible
        best = -MATE_UPPER
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                # Save the move for pv construction and killer heuristic
                if move is not None:
                    self.tp_move[pos] = move
                break

        # Stalemate checking is a bit tricky: Say we failed low, because
        # we can't (legally) move and so the (real) score is -infty.
        # At the next depth we are allowed to just return r, -infty <= r < gamma,
        # which is normally fine.
        # However, what if gamma = -10 and we don't have any legal moves?
        # Then the score is actaully a draw and we should fail high!
        # Thus, if best < gamma and best < 0 we need to double check what we are doing.

        # We will fix this problem another way: We add the requirement to bound, that
        # it always returns MATE_UPPER if the king is capturable. Even if another move
        # was also sufficient to go above gamma. If we see this value we know we are either
        # mate, or stalemate. It then suffices to check whether we're in check.

        # Note that at low depths, this may not actually be true, since maybe we just pruned
        # all the legal moves. So sunfish may report "mate", but then after more search
        # realize it's not a mate after all. That's fair.

        # This is too expensive to test at depth == 0
        if depth > 2 and best == -MATE_UPPER:
            flipped = pos.rotate(nullmove=True)
            # Hopefully this is already in the TT because of null-move
            in_check = self.bound(flipped, MATE_UPPER, 0) == MATE_UPPER
            best = -MATE_LOWER if in_check else 0

        # Table part 2
        if best >= gamma:
            self.tp_score[pos, depth, can_null] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, can_null] = Entry(entry.lower, best)

        return best

    def search(self, history):
        """Iterative deepening MTD-bi search"""
        self.nodes = 0
        self.history = set(history)
        self.tp_score.clear()

        gamma = 0
        # In finished games, we could potentially go far enough to cause a recursion
        # limit exception. Hence we bound the ply. We also can't start at 0, since
        # that's quiscent search, and we don't always play legal moves there.
        for depth in range(1, 1000):
            # The inner loop is a binary search on the score of the position.
            # Inv: lower <= score <= upper
            # 'while lower != upper' would work, but it's too much effort to spend
            # on what's probably not going to change the move played.
            lower, upper = -MATE_LOWER, MATE_LOWER
            while lower < upper - EVAL_ROUGHNESS:
                score = self.bound(history[-1], gamma, depth, can_null=False)
                if score >= gamma:
                    lower = score
                if score < gamma:
                    upper = score
                yield depth, gamma, score, self.tp_move.get(history[-1])
                gamma = (lower + upper + 1) // 2


###############################################################################
# UCI User interface
###############################################################################


def parse(c):
    # file and rank coordinate to bitboard
    fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
    return 0b1 << fil + rank*8


def render(bb):
    # bitboard coordinate to file and rank
    pos = bb.bit_length() - 1
    rank = pos // 8
    file = pos % 8
    coordinate = chr(file + ord('A')) + str(rank + 1)
    return coordinate


hist = [Position(initial, 0, (True, True), (True, True), 0, 0)]

#input = raw_input

if __name__ == '__main__':
    # minifier-hide start
    import sys, tools.uci
    tools.uci.run(sys.modules[__name__], hist[-1])
    sys.exit()
    # minifier-hide end
