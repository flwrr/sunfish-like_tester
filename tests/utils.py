import chess.pgn
import importlib


def load_engine(version):
    try:
        module_name = f"versions.molafish_{version}"
        module = importlib.import_module(module_name)
        return module
    except ImportError as err:
        print(f"Error: Failed to find engine version '{version}'")
        print(err)
        return None


def debug_print_moves_bb(moves):
    # Prints each origin, its bitboard representation, then all
    # combined moves available from that position as a bitboard
    for origin, move in moves.items():
        combined_moves = 0
        for mov in move:
            combined_moves |= mov
        print(origin)
        debug_print_bb(origin)
        debug_print_bb(combined_moves)


def debug_print_bb(bitboard):
    # Prints out any number input as a 8x8 bitboard
    for row in range(8):
        for col in range(8):
            bit_position = (7 - row) * 8 + col
            if bitboard & (1 << bit_position):
                print('1 ', end=' ')
            else:
                print('. ', end=' ')
        print()
    print()


def bb_to_coord(bb):
    """Converts a Molafish bitboard position to a file-rank coordinate"""
    if bb == 0: return 0
    # More than one set bit (v011+ kp value, which has 3 set bits)
    # Get the middle-most bit (second lsb)
    if bb & (bb - 1):
        bb = (bb & -bb) << 1
    pos = bb.bit_length() - 1
    rank = pos // 8
    file = pos % 8
    coordinate = chr(file + ord('A')) + str(rank + 1)
    return coordinate


def index_to_coord(index):
    """Converts a Sunfish board index to a file-rank coordinate."""
    if index == 0: return 0
    rank = 8 - ((index - 21) // 10)
    file = chr(ord('A') + ((index - 21) % 10))
    return file + str(rank)


def mirror_coord(coord):
    """Mirrors one board coordinates. (e.g., A1=H8, B2=G7)"""
    flip_map = {
        'A': 'H', 'B': 'G', 'C': 'F', 'D': 'E',
        'E': 'D', 'F': 'C', 'G': 'B', 'H': 'A',
        '1': '8', '2': '7', '3': '6', '4': '5',
        '5': '4', '6': '3', '7': '2', '8': '1'
    }
    if coord:
        coord = flip_map[coord[0].upper()] + flip_map[coord[1].upper()]
    return coord


def mirror_coords(coord1, coord2):
    """Mirrors two board coordinates. (e.g., A1=H8, B2=G7)"""
    flip_map = {
        'A': 'H', 'B': 'G', 'C': 'F', 'D': 'E',
        'E': 'D', 'F': 'C', 'G': 'B', 'H': 'A',
        '1': '8', '2': '7', '3': '6', '4': '5',
        '5': '4', '6': '3', '7': '2', '8': '1'
    }
    if coord1:
        coord1 = flip_map[coord1[0].upper()] + flip_map[coord1[1].upper()]
    if coord2:
        coord2 = flip_map[coord2[0].upper()] + flip_map[coord2[1].upper()]
    return coord1, coord2


def format_game_info(game_info):
    """Format chess UCI game information into comma-separated string."""
    info_output = ""
    if game_info is None:
        return "Game data not found."
    for key, value in game_info.items():
        info_output += f"{key}: {value}\n"
    return info_output


def pgn_to_uci(pgn):
    """
    Convert a single PGN game to UCI move format and extract game info.
    Returns as a tuple, a list of UCI moves and a string of game info.
    """
    game = chess.pgn.read_game(pgn)
    if game is None:
        return None
    # Extracting game info
    game_info = {key: game.headers[key] for key in game.headers}
    # Extracting moves in UCI format
    board = game.board()
    uci_moves = [move.uci() for move in game.mainline_moves()]
    return uci_moves, game_info


def bulk_pgn_to_uci(pgn_file):
    """
    Convert all games in a PGN file to a list of moves in UCI format
    :param pgn_file: Path to the PGN file
    :return: A generator yielding lists of UCI moves for each game
    """
    try:
        with open(pgn_file) as pgn:
            while True:
                uci_moves, game_info = pgn_to_uci(pgn)
                if uci_moves is None:
                    break
                yield uci_moves, game_info
    except FileNotFoundError:
        print(f"File not found: {pgn_file}")


# -------------------------- ENGINE SPECIFIC TOOLS -------------------------- #


def get_piece_sunfish(position: object, index):
    return position.board[index].upper()
    # return position.board[index].upper() if index in position.board else None


def get_piece_molafish(position: object, location):
    mf_index_to_piece = ["ERR", "P", "R", "N", "B", "Q", "K"]
    # Molafish v01-02
    if len(position.board) > 3:
        for i, piece_bb in enumerate(position.board):
            if location & piece_bb:
                i = (i - 6) if (i >= 6) else i  # Black pieces
                return mf_index_to_piece[i+1]
    # Molafish v03-...
    else:
        for i, piece_bb in enumerate(position.board[position.player][1:], start=1):
            if location & piece_bb:     # Black pieces
                return mf_index_to_piece[i]


def apply_uci_move_sunfish(uci_move, hist, rotate_board):
    """Apply a UCI-formatted move (e.g., "e2e4") to the Sunfish engine."""
    def parse(c):
        fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
        return 91 + fil - 10 * rank

    i, j = parse(uci_move[:2]), parse(uci_move[2:4])
    prom = uci_move[4:].upper() if len(uci_move) > 4 else None

    # Rotate on black's turn
    if rotate_board:
        if len(hist) % 2 == 0:
            i, j = 119 - i, 119 - j

    for move in hist[-1].gen_moves():
        if move.i == i and move.j == j and (prom is None or move.prom == prom):
            new_position = hist[-1].move(move)
            hist.append(new_position)
            return new_position

    raise ValueError(f"invalid move: {uci_move}")


def apply_uci_move_molafish(uci_move, hist, rotate_board=False):
    """Apply a UCI-formatted move (e.g., "e2e4") to the Molafish engine."""
    def parse(c):
        # Covert file and rank coordinate to bitboard
        fil, rank = ord(c[0]) - ord("a"), int(c[1]) - 1
        return 0b1 << fil + rank * 8

    i, j = parse(uci_move[:2]), parse(uci_move[2:4])

    # For Molafish V03 onward
    board_uses_sublists = len(hist[-1].board) < 10

    # Promotions
    if len(uci_move) > 4:
        prom = uci_move[4:].upper()
        if prom == "Q":
            prom = 4
        elif prom == "B":
            prom = 3
        elif prom == "N":
            prom = 2
        else:
            prom = 1
        if board_uses_sublists:
            prom += 1
    else:
        prom = 0

    # Check for Black's turn and rotate move input if so
    if len(hist) % 2 == 0:
        if rotate_board:
            i, j = 1 << (63 - (i.bit_length() - 1)), 1 << (63 - (j.bit_length() - 1))
            # If black's turn and NOT rotate board, then promotion should be black
        elif not board_uses_sublists:
            prom += 6 if prom else 0

    for move in hist[-1].gen_moves():
        if move.i == i and move.j == j and move.prom == prom:
            new_position = hist[-1].move(move)
            hist.append(new_position)
            return new_position

    raise ValueError(f"invalid move: {uci_move}")


# print(mirror_coordinate("f3"))