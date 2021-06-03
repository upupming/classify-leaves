import argparse

def getArgs():
    arg_parser=argparse.ArgumentParser()
    arg_parser.add_argument("--data_root",type=str,default='../dataset/classify-leaves',
                            help="the root path of the dataset")
    arg_parser.add_argument("--val_ratio",type=float,default=0.1,help="the ratio of validation set")
    args=arg_parser.parse_args()
    return args