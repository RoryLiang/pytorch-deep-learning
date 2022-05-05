import sys
sys.path.append("/home/liangtianyu/aae/Pytorch-AAE")
from optimizer.optimizer4AE import AEoptimizer
from optimizer.optimizer4VAE import VAEoptimizer
from config.args import get_args


def main():
    args = get_args()
    if args.model == "ae":
        optimizer = AEoptimizer(args)
    elif args.model == "vae":
        optimizer = VAEoptimizer(args)
    else:
        raise Exception(f"Invalid model option: {args.model}, try: --model ae")
    optimizer.optimize()


if __name__ == "__main__":
    main()
