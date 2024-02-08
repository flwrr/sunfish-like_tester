from tests import moveLogic, searchLogic
from tests import perft as p
import click


@click.group()
@click.option('--debug', is_flag=True, help="Minimal output")
@click.option('--full', is_flag=True, help="Full output")
@click.option('--silent', is_flag=True, help="Error output only")
@click.pass_context
def cli(ctx, debug, full, silent):
    """Chess Logic Testing CLI."""
    ctx.ensure_object(dict)
    if silent:
        ctx.obj['MODE'] = 'Silent'
    elif full:
        ctx.obj['MODE'] = 'Full'
    else:
        ctx.obj['MODE'] = 'Debug'


@cli.command()
@click.pass_context
@click.option('--games', default=50, help='Maximum number of games to test.', type=int)
@click.option('--version', default=None, help='Version of the engine to test against.', type=str)
def move(ctx, games, version):
    """Compares move generation data across multiple games."""
    mode = ctx.obj['MODE']
    moveLogic.test_multiple_games_movegen(engine2=version, max_games=games, mode=mode)
    # TODO: Single game test with UCI game input


@cli.command()
@click.pass_context
@click.option('--version', default=None, help='Second engine version.')
@click.option('--start-pos', default=None, help='Starting position in FEN format.')
@click.option('--max-depth', default=6, type=int, help='Max search depth.')
def search(ctx, version, start_pos, max_depth):
    """Compares search evaluation data from a starting position."""
    mode = ctx.obj['MODE']
    searchLogic.compare_search_data(engine2=version, start_pos=start_pos, max_depth=max_depth, mode=mode)


@cli.command()
@click.pass_context
@click.option('--version', default=None, help='Engine to test.')
@click.option('--limit', default=1, help='Time limit in seconds or depth limit.')
@click.option('--test', type=click.Choice(['Time', 'Depth'], case_sensitive=False),
              default='Time', help="Test type: 'Time' or 'Depth'")
def perft(ctx, version, test, limit):
    """Perft tests an engine. Tests sunfish by default."""
    mode = ctx.obj['MODE']
    p.perft(engine=version, mode=mode, test=test, limit=limit)


@cli.command()
@click.pass_context
@click.option('--version', default=None, help='Engine to compare.')
@click.option('--limit', default=1, help='Time limit in seconds or depth limit.')
@click.option('--test', type=click.Choice(['Time', 'Depth'], case_sensitive=False),
              default='Time', help="Test type: 'Time' or 'Depth'")
def perft_compare(ctx, version, test, limit):
    """Perft tests two engines and compares results."""
    mode = ctx.obj['MODE']
    p.perf_compare(engine2=version, mode=mode, test=test, limit=limit)
    eng_name = 'Molafish latest' if version is None else version


@cli.command()
@click.pass_context
@click.option('--version', default=None, help='Engine version to test.')
def quick(ctx, version):
    """Executes move, search, and perft_compare commands with 'quick' parameters."""
    ctx.invoke(move, games=2, version=version)
    ctx.invoke(search, version=version, max_depth=2)
    ctx.invoke(perft_compare, version=version, test='Time', limit=1)


@cli.command()
@click.pass_context
@click.option('--version', default=None, help='Engine version to test.')
def full(ctx, version):
    """Executes move, search, and perft_compare commands with 'quick' parameters."""
    ctx.invoke(move, games=100, version=version)
    ctx.invoke(search, version=version, max_depth=5)
    ctx.invoke(perft_compare, version=version, test='Time', limit=10)


if __name__ == '__main__':
    cli()
