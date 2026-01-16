import argparse

import tripso


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(metavar='<subcommand>', required=True)
    parse_preprocess = subparsers.add_parser('preprocess')
    parse_preprocess.set_defaults(func=preprocess)
    parse_train = subparsers.add_parser('train')
    parse_train.set_defaults(func=train)
    parse_infer = subparsers.add_parser('infer')
    parse_infer.set_defaults(func=infer)
    args = parser.parse_args()
    args.func(args)


def preprocess(args):
    pass


def train(args):
    pass


def infer(args):
    pass


if __name__ == '__main__':
    main()
