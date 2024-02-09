# Sunfish-Like
Sunfish-Like is a CLI tool for debugging and testing chess engine logic of the [Molafish](https://github.com/flwrr/molafish) chess engine against its parent engine, [Sunfish](https://github.com/thomasahle/sunfish). 
It also serves as a detailed performance log, change log, and archive of previous Molafish versions. While the tool currently 

## ♢ Features
Several customizable test groupings accessible through the command line with the `click` python package.<br>
Test methodology and more detailed example usages for each type can be found below.
- `test.py move`   Move generation comparison accross single or bulk input PGN / UCI chess games.
- `test.py search` Search evaluation comparison for consistant output to a specified depth.
- `teast.py perft` Isolated or comparative perft testing by depth limit or time limit.


## Molafish Performance History

<img src="https://i.imgur.com/vvUGHg6.png">
<img src="https://i.imgur.com/wOYbi7f.png">

## 🎲 Usage

The CLI provides several options and commands for testing with the format:
`test.py [OPTIONS] COMMAND [ARGS]...`

### Options:

- `--debug` Normal output - default.
- `--full` Full, detailed output.
- `--silent` Output on error only.

### Commands:

- `move` Compares move generation data across a single game or multiple games.
- `search` Compares search evaluation data from a starting position.
- `perft` Perft tests an engine for performance and accuracy by time or depth.
- `perft-compare` Perft tests two engines by time or depth and compares results.
- `full` Executes move, search, and perft_compare commands with detailed output.
- `quick` Executes move, search, and perft_compare commands with minimal.

# 🏄 Move testing
Move generation and execution are tested with input UCI or PGN formatted games.<br>
PGN inputs will be converted to UCI. For every listed move, the tool will:
1. Execute the move
2. Generate all legal response moves and their corresponding scores and flags
3. Verify each engine has matching move outputs and scores and flags for those outputs 

By default, `move` processes 50 games from a bulk PGN file containing nearly 300 PGN formatted Aegon Tournament games from 1995.

## Example usage: move

`test.py move --games 10`<br>

A successful run will print the average move generation and execution time for each engine.

<img src="https://i.imgur.com/3arALrA.gif">

Any mismatch encountered between engines will cause the tool to display the discrepancy and print the game moves list in UCI format.

<img src="https://i.imgur.com/x7GMNDP.gif">

# 🔎 Search testing
Search evaluation begin from a provided position (defaults to initial board position), and search for the best move to a the provided search depth (default=5).
Similar to move generation testing, each returned move's attributes are compared with sunfish for equivalent results.

## Example usage: search

`test.py search --max_depth 6`<br>

<img src="https://i.imgur.com/rRfgKtz.gif">

# 🌴 Perft testing
Perft testing can be executed in isolation with `python test.py perft`, or compared against sunfish with `python test.py perft-compare`.
Perft tests have two modes:<br>
- **Time-limited**, where the test will run to the specified number of seconds (default=1), and print the depth reached, nodes reached, and an estimated nodes per second (nps).<br>
  `test.py perft --test 'depth'`
- **Depth-limited**, where the test will run to the specified search depth (default=5), and print the time taken to reach it.<br>
  `test.py perft --test 'time'`
  
# Quick and Full
The two main tests used in evaluating Molafish combine `move`, `search`, and `perft-compare` into either a quick test just to ensure the engine is breathing, 
or a full test with enough games to ensure move generation is functioning correctly, and enough time to adequately compare perft tests between engines.
- **quick** tests move generation across 2 games, search evaluation to a depth of 2, and perft comparison to a time limit of 1.
  <br> `test.py peft quick` <br> The equivalent test commands are: <br>`move --games 2` `search --max_depth 2` `perft_compare --test time --limit 1`
- **full** tests move generation across 100 games, search evaluation to a depth of 5, and perft comparison to a time limit of 10.
  <br> `test.py peft full` <br>The equivalent test commands are: <br>`move --games 100` `search --max_depth 5` `perft_compare --test time --limit 10`

## Example usage: quick
<img src="https://i.imgur.com/dwtfZYM.gif">

