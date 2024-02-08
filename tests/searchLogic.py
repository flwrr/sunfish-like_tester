from tests import utils
import molafish
import sunfish


def search_data_gen(engine, max_depth=3, position=None, display=False, rotate_pov=False):
    # Generator function that yields search data from the specified engine.
    # Each engine requires a 'version' variable, and two functions specific
    # to that engine found in tools/utils: convert coordinates, find piece
    if position is None:
        if hasattr(engine, 'hist'):
            position = engine.hist[0]
        else:
            print("No 'hist' found in engine: {engine}, and position parameter is empty.")
            return -1

    # eng_rotates_board = False
    convert_coord_fn = None
    find_piece_fn = None

    if hasattr(engine, 'version'):
        if 'sunfish' in engine.version.lower():
            convert_coord_fn = utils.index_to_coord
            find_piece_fn = utils.get_piece_sunfish
            # eng_rotates_board = True

        if 'molafish' in engine.version.lower():
            convert_coord_fn = utils.bb_to_coord
            find_piece_fn = utils.get_piece_molafish
            # Only version 0.01 of molafish rotates
            # if 'v0.01' in engine.version.lower():
                # eng_rotates_board = True

    if convert_coord_fn is None:
        print(f'\nNo conversion function found for {engine}.')
    if find_piece_fn is None:
        print(f'\nNo find piece function found for {engine}.')

    # Both functions are needed
    if not find_piece_fn and convert_coord_fn:
        return -1

    # Search and convert coords

    search_data = []
    searcher = engine.Searcher()
    for depth, gamma, score, move in searcher.search([position]):

        if depth > max_depth:
            break

        # Convert move output to algebraic notation
        origin = convert_coord_fn(move.i)
        dest = convert_coord_fn(move.j)

        # Rotate on odd ply (black's turn)
        # if eng_rotates_board and depth % 2 == 0:
        if rotate_pov and depth % 2 == 0:
            origin, dest = utils.mirror_coords(origin, dest)

        # Get piece
        piece = find_piece_fn(position, move.i)
        result = {"Depth": depth, "Gamma": gamma, "Score": score,
                  "Origin": origin, "Move": dest, "Piece": piece
        }
        if display:
            print(result)
        yield result


def compare_search_data(engine1=None, engine2=None, max_depth=3, start_pos=None, mode="Debug"):

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
        print(f"Beginning Search comparison between engine1: {eng1_name}, and engine2: {eng2_name}")

    # Make generators
    gen1 = search_data_gen(engine1, max_depth, start_pos, display=False)
    gen2 = search_data_gen(engine2, max_depth, start_pos, display=False)

    while True:
        try:
            r1 = next(gen1)
        except StopIteration:
            r1 = None
        try:
            r2 = next(gen2)
        except StopIteration:
            r2 = None

        # Both generators exhausted
        if r1 is None and r2 is None:
            if mode != 'quiet':
                print("All search comparisons complete -- No mismatches found")
            return 0

        # Print formatting
        col_width = 5
        eng_name_col_width = max(len(eng1_name), len(eng2_name)) + 2
        if r2 is not None:
            formatted_r2 = "".join([f"{k:<{col_width}}: {v:<{col_width+3}}" for k, v in r2.items()])
            r2_display = f"{eng2_name:<{eng_name_col_width}}: {formatted_r2} "
        if r1 is not None:
            formatted_r1 = "".join([f"{k:<{col_width}}: {v:<{col_width + 3}}" for k, v in r1.items()])
            r1_display = f"{eng1_name:<{eng_name_col_width}}: {formatted_r1}"

        # Compare and display, halting at mismatches
        if r1 is not None and r2 is not None:
            # "Origin", "Move",
            for key in ("Depth", "Gamma", "Score", "Origin", "Move", "Piece"):
                if r1[key] != r2[key]:
                    if mode != 'Silent':
                        print(f"\nSearch comparison: FAIL -- Mismatch found at depth: {r1['Depth']}, val: {key}\n")
                        print("\n".join([r1_display, r2_display]))

                    return -1
            if mode == 'Full':
                print("\n".join([r1_display, r2_display]))
            if mode == 'Debug':
                print(f"Search comparison: PASS -- {formatted_r1}")

        # Result length mismatch - only gen1 yielded
        elif r1 is not None:
            if mode != 'Silent':
                print(f"\nSearch comparison: FAIL -- No pair yielded from {eng2_name} to match:\n")
                print(r1_display)
            return -1
        elif r2 is not None:
            if mode != 'Silent':
                print(f"\nSearch comparison: FAIL -- No pair yielded from {eng1_name} to match:\n")
                print(f"{eng2_name:<{eng_name_col_width}}: {r2}")
            return -1
