import time
import sunfish
import molafish
from . import utils


def perft_time(limit_seconds, engine, position=None, mode='Debug'):
    pos = engine.hist[0]  # Initialize the position
    searcher = engine.Searcher()
    depth_reached = 0
    start_time = time.time()
    # These could just be single entries for current use, but
    # leaving them as lists in case we want to do more with data
    time_records = []
    node_records = []
    # # Announce engine
    # if mode == 'Full':
    #     engine_name = engine.version if hasattr(engine, 'version') else None
    #     if engine_name is not None:
    #         print(f"{engine_name}:")
    # Increase depth until time limit
    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= limit_seconds:
            break
        depth_reached += 1
        searcher.nodes = 0
        searcher.bound(pos, 0, depth_reached)
        # Update records for each depth
        time_records.append(elapsed_time)
        node_records.append(searcher.nodes)
        if mode == 'Full':
            print(f"Depth: {depth_reached}, Nodes: {searcher.nodes}, Time: {elapsed_time:.2f}s")

    nodes_per_second = node_records[-1] / time_records[-1]
    estimated_total_nodes = nodes_per_second * limit_seconds

    return depth_reached, round(estimated_total_nodes), round(nodes_per_second)


def perft_depth(depth, engine, position=None):
    """Depth-limited perft test."""
    pos = engine.hist[0]
    searcher = engine.Searcher()
    start_time = time.time()
    searcher.bound(pos, 0, depth)
    duration = time.time() - start_time
    return duration

# TODO: perf for endgame positions / specific input positions


def perft(engine=None, mode="Debug", test="Time", limit=10):
    """Handles various perf tests and their outputs"""
    if engine == sunfish or engine == molafish:
        pass
    elif engine is None:
        engine = sunfish
    elif isinstance(engine, str):
        engine = utils.load_engine(engine)
        if engine is None:
            return

    # Initial announcement
    eng_name = engine.version if hasattr(engine, 'version') else '(NO VERSION FOUND)'
    if mode != "Silent":
        print(f"Starting Perft Test: {test} limit: {limit}, Engine: {eng_name}")

    if test == "Time":
        depth_reached, nodes_visited, nodes_per_second = perft_time(limit, engine=engine, mode=mode)
        if mode == "Full":
            print(f"Depth: {depth_reached}", end=", ")
            print(f"Nodes: {nodes_visited}", end=", ")
            print(f"nps: {nodes_per_second}")
        return depth_reached, nodes_visited, nodes_per_second

    if test == "Depth":
        duration, nodes_visited = perft_depth(depth=limit, engine=engine)
        if mode != "Silent":
            print(f"Nodes visited: {nodes_visited}")
        return nodes_visited


def perf_compare(engine1=None, engine2=None, mode="Debug", test="Time", limit=10):
    """Compares the performance of two engines / versions."""
    if engine1 is None:
        engine1 = sunfish
    else:
        engine1 = utils.load_engine(engine1)
        if engine1 is None:
            return

    if engine2 is None:
        engine2 = molafish
    else:
        engine2 = utils.load_engine(engine2)
        if engine2 is None:
            return

    if not engine1 or not engine2:
        return

    # Initial announcement
    eng1_name = engine1.version if hasattr(engine1, 'version') else 'Engine 1'
    eng2_name = engine2.version if hasattr(engine2, 'version') else 'Engine 2'
    if mode != "Silent":
        print(f"\nStarting perft comparison between {eng1_name}, and {eng2_name} - Perft {test} limit: {limit}")

    result1 = perft(test=test, limit=limit, engine=engine1, mode=mode)
    result2 = perft(test=test, limit=limit, engine=engine2, mode=mode)

    # Output comparison
    if mode != "Silent":
        if test == "Time":
            depth_reached1, nodes_visited1, nps1 = result1
            depth_reached2, nodes_visited2, nps2 = result2
            print(f"\n{eng1_name:<15}: Depth: {depth_reached1}, Nodes: {nodes_visited1}, NPS: {nps1}")
            print(f"{eng2_name:<15}: Depth: {depth_reached2}, Nodes: {nodes_visited2}, NPS: {nps2}")
        elif test == "Depth":
            nodes_visited1 = result1
            nodes_visited2 = result2
            print(f"\n{eng1_name}: Nodes visited: {nodes_visited1}")
            print(f"{eng2_name}: Nodes visited: {nodes_visited2}")

    # Return comparison results for further processing if needed
    return result1, result2


# Example usage:
if __name__ == "__main__":

    perf_compare(engine2="v0_01", test='Time', limit=1)
    # perft(test="Time", limit=5)

    # depth, nodes = perft_time(5)  # Run for 5 seconds
    # perft_depth(6)
