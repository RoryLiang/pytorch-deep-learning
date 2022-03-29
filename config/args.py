import sys
import os
import time
from dataclasses import dataclass
from utils.typed_args import TypedArgs, add_argument


@dataclass
class Args(TypedArgs):
    model: str = add_argument("--model", default="ae")
    output_dir: str = add_argument("--output_dir", default="../output")
    data_dir: str = add_argument("--data_dir", default="../data")
    gpu: int = add_argument("--gpu", default=0)
    batch_size: int = add_argument("--batch_size", default=128)
    epoch_num: int = add_argument("--epoch_num", default=100)
    learning_rate: float = add_argument("--learning_rate", default=1e-3)


def get_args(argv=sys.argv):
    args, _ = Args.from_known_args(argv)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    args.output_dir = "/".join([args.output_dir, time_str])
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args
