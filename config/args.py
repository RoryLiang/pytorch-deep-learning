import sys
import pathlib
from dataclasses import dataclass
from utils.typed_args import TypedArgs, add_argument


@dataclass
class Args(TypedArgs):
    output_dir: str = add_argument("--output_dir", default="../output")
    data_dir: str = add_argument("--data_dir", default="../data")
    gpu: int = add_argument("--gpu", default=0)
    exp_name: str = add_argument("--exp_name", default="exp_1")
    batch_size: int = add_argument("--batch_size", default=16)
    epoch_num: int = add_argument("--epoch_num", default=100)
    learning_rate: float = add_argument("--learning_rate", default=1e-3)


def get_args(argv=sys.argv):
    args, _ = Args.from_known_args(argv)
    args.output_dir = pathlib.Path(args.output_dir + "/" + args.exp_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    return args
