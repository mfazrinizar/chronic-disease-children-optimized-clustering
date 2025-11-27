import argparse
from src.main import main as src_main


def _cli():
    parser = argparse.ArgumentParser(description='Run project: dashboard or preprocess')
    parser.add_argument('--mode', choices=['dashboard', 'preprocess'], help='Mode to run')
    args = parser.parse_args()
    if args.mode:
        src_main(['--mode', args.mode])
    else:
        src_main([])


if __name__ == '__main__':
    _cli()
