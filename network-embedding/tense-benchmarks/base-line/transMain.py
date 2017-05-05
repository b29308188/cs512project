from __future__ import print_function
import argparse
import transE, transR
def run_transE(args):
    print('running transE models ...')

def run_transD(args):
    print('running transD models ...')

def run_transR(args):
    print('running transR models ...')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='specify the name of model')
    parser.add_argument('-r', '--reg_rate', help='degree of regularization. default is 10')
    parser.add_argument('--regularization', help='enable regularization', action="store_true")
    parser.add_argument('--sense', help='enable sense matix',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model == 'transD' or args.model =='transd':
        run_transD(args)
    elif args.model == 'transE' or args.model == 'transe':
        run_transE(args)
    elif args.model == 'transR' or args.model == 'transr':
        run_transR(args)
