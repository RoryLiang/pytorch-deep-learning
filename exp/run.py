import sys
from optimizer.Optimizer4AE import AEoptimizer
from config.args import get_args
sys.path.append("/home/liangtianyu/aae/Pytorch-AAE")


def main():
    ae_optimizer = AEoptimizer(get_args())
    ae_optimizer.optimize()


if __name__ == "__main__":
    main()
