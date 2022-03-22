import sys
sys.path.append("/home/liangtianyu/aae/Pytorch-AAE")
from optimizer.Optimizer4AE import AEoptimizer


def main():
    optimizer = AEoptimizer(1)
    optimizer.optimize()


if __name__ == "__main__":
    main()
