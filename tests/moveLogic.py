import pandas as pd
import sunfish
import molafish

from . import utils
import time

DEFAULT_BULK_GAMES = "./tests/pgn/aegon96.pgn"
DEFAULT_UCI = ['e2e4', 'c7c5', 'e4e5', 'd7d5', 'e5d6', 'b7b5', 'b2b4',
               'c5b4', 'd6e7', 'f8e7', 'a2a4', 'b4a3', 'b1c3', 'b8c6',
               'h2h4', 'f7f5', 'h4h5', 'g7g5', 'h5g6', 'h7h5', 'h1h5',
               'g8f6', 'g6g7', 'c6d4', 'g2g4', 'b5b4', 'h5f5', 'd4f5',
               'c3d5', 'h8h1', 'd5f6', 'e8f7', 'g7g8q', 'd8g8', 'f6g8',
               'f7g8', 'd1f3', 'h1g1', 'c1b2', 'c8e6', 'e1c1', 'a3b2',
               'c1b1', 'b4b3', 'f1c4', 'f5e3', 'c4b3', 'e3g4', 'b1a2',
               'b2b1q', 'a2b1', 'g1d1', 'b1b2', 'e6b3', 'c2b3', 'g4e3',
               'b3b4', 'e7h4', 'b4b5', 'h4f2', 'b5b6', 'g8g7', 'b6b7',
               'e3c4', 'b2c2', 'a7a5', 'b7b8q', 'c4d6', 'b8d6', 'a8d8',
               'd6d4', 'f2d4', 'f3e4', 'd1d2', 'c2d2', 'd4b2', 'd2c2',
               'b2e5', 'e4g6', 'g7g6', 'c2b1', 'd8d2', 'b1c1', 'e5f4',
               'c1b1', 'a5a4', 'b1a1', 'd2c2', 'a1b1', 'a4a3', 'b1a1',
               'g6f5', 'a1b1', 'f5e4', 'b1a1', 'e4d3', 'a1b1', 'd3d2',
               'b1a1', 'd2c3', 'a1b1', 'f4c1', 'b1a1', 'a3a2']


def gen_moves_timed(position):
    """
    Generates all moves and executes their resulting board positions from
    a given position, while recording total generation and execution time.
    """
    move_gen = position.gen_moves()

    # Time the generation of all moves
    # Unloads all yields from move_gen
    timer_start = time.perf_counter()
    all_moves = tuple(move_gen)
    timer_end = time.perf_counter()
    gen_time = timer_end - timer_start

    # Time the execution of all moves yielded
    timer_start = time.perf_counter()
    all_positions = tuple(position.move(m) for m in all_moves)
    timer_end = time.perf_counter()
    exec_time = timer_end - timer_start

    return all_moves, all_positions, gen_time, exec_time


def process_moves(engine, position, output_moves_list, output_times_list,
                  convert_coord_fn=utils.index_to_coord,
                  find_piece_fn=utils.get_piece_sunfish,
                  mirror_coords=False):
    """
    Processes moves from a given position and appends them to a provided list.
    Takes a coordinate-conversion and piece-finding function as the last two
    parameters, which default to functions specific to sunfish.
    """
    all_moves, all_positions, gen_time, exec_time = gen_moves_timed(position)
    for move, pos in zip(all_moves, all_positions):

        move_from = convert_coord_fn(move.i)
        move_to = convert_coord_fn(move.j)
        piece = find_piece_fn(position, move.i)
        ep = convert_coord_fn(pos.ep)
        kp = convert_coord_fn(pos.kp)
        if mirror_coords:
            move_from, move_to = utils.mirror_coords(move_from, move_to)

        row_moves = [move_from, engine, move_to, pos.score, pos.wc, pos.bc, ep, kp, piece]
        output_moves_list.append(row_moves)

    row_times = [engine, gen_time, exec_time]
    output_times_list.append(row_times)


def create_moves_and_times_df(hist_mf, hist_sf, move_index,
                              mirror_coords_eng1=False,
                              mirror_coords_eng2=False):
    """
    Creates a Pandas DataFrame from a single position for two engines using
    gen_moves_timed that contains all moves from each piece and all game
    information related to each move including piece moved, score, and flags.
    """
    move_data = []
    time_data = []

    process_moves("SUN", hist_sf[move_index],
                  move_data, time_data,
                  utils.index_to_coord,
                  utils.get_piece_sunfish,
                  mirror_coords_eng1)

    process_moves("MOL", hist_mf[move_index],
                  move_data, time_data,
                  utils.bb_to_coord,
                  utils.get_piece_molafish,
                  mirror_coords_eng2)

    for row in time_data:
        row.insert(0, move_index)

    # Moves dataframe
    columns_moves = ["origin", "engine", "move", "score", "WC", "BC", "EP", "KP", "piece"]
    moves_df = pd.DataFrame(move_data, columns=columns_moves)
    moves_df.sort_values(by=["origin", "move", "score", "engine"], inplace=True)

    # Times dataframe
    times_df = pd.DataFrame(time_data, columns=["move_index", "engine", "gen_time", "exec_time"])
    return moves_df.reset_index(drop=True), times_df


def validate_moves_df(df, eng2_rotates_board, mirrored_ep_kp, mirrored_wc_bc, mode):
    """
    Validates the DataFrame to ensure that each move from each engine has a
    corresponding pair from the other engine, and that respective values between
    pairs match. Returns False if a check is failed, and True if all checks pass.
    """
    seen_moves = []

    for i in range(0, len(df), 2):
        move_one = df.iloc[i]
        move_two = df.iloc[i + 1]

        # Check for duplicates (ignores pawn dupes)
        for j, move in enumerate([move_one, move_two]):
            if move["piece"] == 'P':
                continue
            move_id = (move['origin'], move['engine'], move['move'])
            if move_id in seen_moves:
                dup_i = i+j
                print(f"FAIL: Duplicate move found at row {dup_i}.")
                if mode != "Silent":
                    print(df.to_string())
                print(df.iloc[dup_i:dup_i + 1].to_string())
                return False
            seen_moves.append(move_id)

        # Verify pairs exist
        if move_one['origin'] != move_two['origin'] or move_one['move'] != move_two['move']:
            print(f"FAIL: Pair mismatch at row {i}.")
            # if mode == "Full":
            if mode != "Silent":
                print(df.to_string() + f"\nPair missing at row {i}:")
            print(df.iloc[i:i+1].to_string())
            return False

        # Compare pair values
        if move_one['score'] != move_two['score']:
            print(f"FAIL: Score value mismatch at rows {i}, {i+1}.")
            if mode == "Full":
                print(df.to_string())
            print(df.iloc[i:i+2].to_string())
            return False

        # Compare castle rights
        wc_rights_one, wc_rights_two = move_one['WC'], move_two['WC']
        bc_rights_one, bc_rights_two = move_one['BC'], move_two['BC']
        if mirrored_wc_bc:
            wc_rights_two, bc_rights_two = bc_rights_two, wc_rights_two
        if not eng2_rotates_board:
            bc_rights_two = (bc_rights_two[1], bc_rights_two[0])
        if wc_rights_one != wc_rights_two or bc_rights_one != bc_rights_two:
            print(f"FAIL: Castling rights mismatch at rows {i}, {i + 1}.")
            if mode == "Full":
                print(df.to_string())
            print(df.iloc[i:i + 2].to_string())
            return False

        # Compare enpassant, king-passant
        for key in ['EP', 'KP']:
            val1, val2 = move_one[key], move_two[key]
            if mirrored_ep_kp:
                val2 = utils.mirror_coord(val2)
            if val1 != val2:
                print(f"FAIL: Value mismatch for {key} at rows {i}, {i + 1}.")
                if mode == "Full":
                    print(df.to_string())
                print(df.iloc[i:i + 2].to_string(index=False))
                return False

    return True


def test_single_game_movegen(eng1=sunfish,
                             eng2=molafish,
                             pgn_move_file=None,
                             uci_moves_list=None,
                             mode="Debug",
                             print_game_info=False,
                             print_averages=False,
                             eng1_rotates_board=True,
                             eng2_rotates_board=False):
    """
    Tests both engine's chess logic methods for equivalent results
    across a single game in UCI format, by comparing move origins,
    move destinations, their scores, and special flags.
    Returns None if a check is failed, and the game's dataframe otherwise.
    """
    game_info = None
    if uci_moves_list is None:
        if pgn_move_file is not None:
            with open(pgn_move_file) as pgn:
                uci_moves_list, game_info = utils.pgn_to_uci(pgn)
        else:
            uci_moves_list = DEFAULT_UCI

    # Board position history
    hist_sunfish = list(eng1.hist)    # Deep copy
    hist_molafish = list(eng2.hist)
    all_times_df = pd.DataFrame()
    # all_moves_df = pd.DataFrame()     # for per-piece perf

    if mode == "Full" or print_game_info:
        print("Game info:", utils.format_game_info(game_info))

    for i, uci_move in enumerate(uci_moves_list):
        msg_move_number = f"{i+1}:"
        msg_move_coords = f"{uci_move}"
        msg_test_moves = "TEST gen_move outputs match: "
        width_col1 = 5
        width_col2 = 7
        width_col3 = len(msg_test_moves)

        mirror_eng1_coords = False
        mirror_eng2_coords = False
        allow_mirrored_ep_kp = False
        allow_mirrored_wc_bc = False

        # Check for whether to mirror coordinates
        # Engines that both do not rotate or both rotate
        # will have mirrored ep,kp bc,wc on black's turn
        if eng1_rotates_board and len(hist_sunfish) % 2 == 0:  # Black's move
            mirror_eng1_coords = True if eng1_rotates_board else False
            mirror_eng2_coords = True if eng2_rotates_board else False
        else:   # White's move
            allow_mirrored_ep_kp = True if eng1_rotates_board != eng2_rotates_board else False
            allow_mirrored_wc_bc = True if eng1_rotates_board != eng2_rotates_board else False

        utils.apply_uci_move_sunfish(uci_move, hist_sunfish, eng1_rotates_board)
        utils.apply_uci_move_molafish(uci_move, hist_molafish, eng2_rotates_board)
        # Collect and compare results from applied moves

        moves_df, times_df = create_moves_and_times_df(hist_molafish,
                                                       hist_sunfish, i,
                                                       mirror_eng1_coords,
                                                       mirror_eng2_coords)
        # Print results
        if mode != "Silent":
            print(
                f"{msg_move_number:<{width_col1}}{msg_move_coords:<{width_col2}}"
                f"{msg_test_moves:<{width_col3}}  ", end="")

        # Validate_moves_df will print any failed checks
        if not validate_moves_df(moves_df,
                                 eng2_rotates_board,
                                 allow_mirrored_ep_kp,
                                 allow_mirrored_wc_bc,
                                 mode):
            return None
        if mode != "Silent":
            print("PASS")
        if mode == "Full":
            print(moves_df.to_string() + "\n" + times_df.to_string())

        # Add current game df to aggregate df containing all games' data
        all_times_df = pd.concat([all_times_df, times_df], ignore_index=True)
        # all_moves_df = pd.concat([all_moves_df, moves_df], ignore_index=True)

    # Print averages
    if mode != "Silent" or print_averages:
        # Get time averages
        avg_times_df = all_times_df.groupby('engine')[['gen_time', 'exec_time']].mean()
        # Convert to microseconds
        avg_times_df['gen_time'] *= 1_000_000
        avg_times_df['exec_time'] *= 1_000_000
        avg_times_df.rename(columns={'gen_time': 'gen_time(μs)', 'exec_time': 'exec_time(μs)'}, inplace=True)
        print("\nAverage Generation and Execution Times in Microseconds:\n", avg_times_df, "\n")

    return all_times_df


def test_multiple_games_movegen(pgn_move_file=DEFAULT_BULK_GAMES,
                                engine1=None,
                                engine2=None,
                                eng1_rotates_board=True,
                                eng2_rotates_board=False,
                                print_game_info=False,
                                max_games=100,
                                mode="Debug"):
    """
    Tests both engine's chess logic methods for equivalent results
    across a single game in UCI format, by comparing move origins,
    move destinations, their scores, and special flags.
    Returns True if all tests are passed and False otherwise.
    """
    if engine1 is None:
        engine1 = sunfish
    else:
        engine1 = utils.load_engine(engine1)
        if engine1 is None:
            return False

    if engine2 is None:
        engine2 = molafish
    else:
        engine2 = utils.load_engine(engine2)
        if engine2 is None:
            return False

    eng1_name = engine1.version if hasattr(engine1, 'version') else 'Engine1'
    eng2_name = engine2.version if hasattr(engine2, 'version') else 'Engine2'

    if mode != "Silent":
        print(f"Beginning Movegen comparison between engine1: {eng1_name} and engine2: {eng2_name}")

    # Initialize a DataFrame to collect all average times
    aggregate_times_df = pd.DataFrame()
    print_all_game_times = True if mode == "Full" else False
    per_game_mode = "Silent" if mode != "Full" else "Debug"

    for i, (uci_moves_list, game_info) in enumerate(utils.bulk_pgn_to_uci(pgn_move_file)):
        if max_games is not None and i > max_games:
            break

        if print_game_info or mode == "Full":
            print(f"\nGame {i}:" + "\n" + utils.format_game_info(game_info))
        if mode != "Silent":
            print(f"Processing Game {i:<{4}}...", end="   ")
        if mode == "Full":
            print()

        all_times_df = test_single_game_movegen(eng2=engine2,
                                                mode=per_game_mode,
                                                uci_moves_list=uci_moves_list,
                                                print_averages=print_all_game_times)

        if all_times_df is not None:
            if mode == "Debug":
                print("PASS")
            aggregate_times_df = pd.concat([aggregate_times_df, all_times_df], ignore_index=True)
        else:
            if mode == "Full":
                print("FAIL")
            if mode != "Silent":
                print("UCI moves list:", uci_moves_list)
            return False

    print("All checks PASSED")

    # Print gen and exec averages across all games
    if mode != "Silent":
        # Get time averages
        avg_aggregate_times_df = aggregate_times_df.groupby('engine')[['gen_time', 'exec_time']].mean()
        # Convert to microseconds
        avg_aggregate_times_df['gen_time'] *= 1_000_000
        avg_aggregate_times_df['exec_time'] *= 1_000_000
        avg_aggregate_times_df.rename(
            columns={'gen_time': 'gen_time(μs)', 'exec_time': 'exec_time(μs)'},
            inplace=True)

        print("\n------------------ CHECKS PASSED: FINAL STATS ------------------")
        print("Overall Average Generation and Execution Times in Microseconds:\n", avg_aggregate_times_df)
        print()

    return True


def main():
    # hist_sunfish = [sf.Position(sf.initial, 0, (True, True), (True, True), 0, 0)]
    # hist_molafish = [mf.Position(mf.initial, 0, (True, True), (True, True), 0, 0)]

    # # TEST BETWEEN MOVE GENERATION AND MOVE EXECUTION
    # for position in [hist_sunfish[0], hist_molafish[0]]:
    #     total_gen = 0
    #     total_mov = 0
    #     #
    #     def timed_gen_moves(gen):
    #         while True:
    #             start = time.perf_counter()
    #             try:
    #                 move = next(gen)
    #             except StopIteration:
    #                 break
    #             end = time.perf_counter()
    #             yield move, end - start
    #             #
    #     for move, gen_time in timed_gen_moves(position.gen_moves()):
    #         total_gen += gen_time
    #         #
    #         start = time.perf_counter()
    #         pos = position.move(move)
    #         end = time.perf_counter()
    #         #
    #         move_time = end - start
    #         total_mov += move_time
    #         print(f"generation time: {gen_time * 1_000_000}, execution time: {move_time * 1_000_000}")
    #     print(f"total generation time: {total_gen * 1_000_000}")
    #     print(f"total execution time: {total_mov * 1_000_000}")

    # # TIMER TEST
    # start = time.perf_counter()
    # print("Hello")
    # end = time.perf_counter()
    # time_one = end - start
    # print(time_one * 1_000_000)

    # # TEST POSITION EQUIVALENCIES
    # df = create_moves_dataframe(hist_molafish, hist_sunfish, 0)
    # print(df.to_string()) # prints dataframe for position
    # equivalency = check_equivalency(df)
    # for result in equivalency:
    #     print(result)

    # Molafish v0.2 requires 'player' to be "Black"/"White"
    # Molafish v0.3 requires 'player' to be 0/1

    # # SINGLE UCI GAME TEST
    # # single_game_file = './tools/pgn/pgn_input.pgn'
    # single_game_file = './tools/pgn/pgn_input2.pgn'
    # # # single_game_file = './tools/pgn/pgn_input3.pgn'
    # # # single_game_file = './tools/pgn/pgn_input4.pgn'
    # # single_game_file = './tools/pgn/pgn_input5.pgn'
    # test_single_game_movegen(single_game_file, mode="Debug", eng2_rotates_board=False)
    # # test_single_game_movegen(single_game_file, mode="Debug", eng2_rotates_board=True)

    # TODO: Add rotation parameter to bulk tests
    # TODO: Make pgn file a parameter usable through CLI
    # TODO: Make mol engine a parameter usable through CLI
    # TODO: Print which engines are being tested
    # TODO: Add option to run through a specific game's moves (using UCI output) after fail
    # TODO: Gif of processing games, hit error, ask to run through specific game

    # BULK UCI GAMES TEST
    bulk_games_file = "./tests/pgn/aegon96.pgn"
    test_multiple_games_movegen(bulk_games_file, max_games=300, mode="Debug")

    # # ROTATION TEST
    # positions = [hist_sunfish[0], hist_molafish[0]]
    # for pos in positions:
    #     start = time.perf_counter()
    #     rotated = pos.rotate()
    #     end = time.perf_counter()
    #     print((end - start) * 1_000_000)

    # # TEST DATAFRAME CREATION
    # print(create_moves_and_times_df(mf.hist, sf.hist, 0))

    # # TEST VALIDATE DATAFRAME -- MISSING PAIR
    # data_missing_pair = [
    #     ["A1", "SUN", "A2", 10, True, False, None, None, "P"],
    #     ["A1", "MOL", "A2", 10, True, False, None, None, "P"],
    #     ["B1", "SUN", "B2", 20, False, True, None, None, "P"]
    #     # Missing corresponding MOL entry for origin B1 and move B2
    # ]
    # columns = ["origin", "engine", "move", "score", "WC", "BC", "EP", "KP", "piece"]
    # df_missing_pair = pd.DataFrame(data_missing_pair, columns=columns)
    # # Validate DataFrame
    # print(validate_dataframe(df_missing_pair))

    # # TEST VALIDATE DATAFRAME -- MISMATCHED VALUES
    # data_mismatched_values = [
    #     ["A1", "SUN", "A2", 10, True, False, None, None, "P"],
    #     ["A1", "MOL", "A2", 10, True, False, None, None, "P"],
    #     ["B1", "SUN", "B2", 20, False, True, None, None, "P"],
    #     ["B1", "MOL", "B2", 25, False, True, None, None, "P"]
    # ]
    # columns = ["origin", "engine", "move", "score", "WC", "BC", "EP", "KP", "piece"]
    # df_mismatched_values = pd.DataFrame(data_mismatched_values, columns=columns)
    # # Validate DataFrame
    # print(validate_dataframe(df_mismatched_values))


if __name__ == "__main__":
    main()
# import cProfile
# cProfile.run('main()')
