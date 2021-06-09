import argparse


def getArgs():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_root", type=str, default='../dataset/classify-leaves',
                            help="the root path of the dataset")
    arg_parser.add_argument("--val_ratio", type=float,
                            default=0.1, help="the ratio of validation set")
    arg_parser.add_argument("--epoch", type=int, default=10,
                            help="The number of training epoches")
    arg_parser.add_argument("--lr", type=float, default=0.01,
                            help="The learning rate of optimizer")
    arg_parser.add_argument("--batch_size", type=int,
                            default=64, help="The batch size")
    arg_parser.add_argument("--fold", type=int,
                            default=10, help="Number of folds")
    arg_parser.add_argument("--verbose", action="store_true",
                            help="output the verbose information")
    arg_parser.add_argument("--model", type=str, default='seresnext50',
                            help="choose the type of network")
    arg_parser.add_argument("--freeze", action="store_true",help="freeze the former layers or not")
    arg_parser.add_argument(
        "--ckpt_path", type=str, default="model.pth", help="the path of the checkpoint")
    arg_parser.add_argument("--eval_all", action="store_true", help="validate the model after every training epoch")

    arg_parser.add_argument("--resume", action="store_true",
                            help="continue train the model with ckpt file")
    args = arg_parser.parse_args()
    return args
