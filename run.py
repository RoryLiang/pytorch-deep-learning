from optimizer.optimizer4AE import AEoptimizer
from optimizer.optimizer4VAE import VAEoptimizer
from optimizer.optimizer4AAE import AAEoptimizer
from config.args import get_args


def main():
    args = get_args()
    if args.model == "ae":
        optimizer = AEoptimizer(args)
    elif args.model == "vae":
        optimizer = VAEoptimizer(args)
        optimizer.optimize()
    elif args.model == "aae":
        optimizer = AAEoptimizer(args)
        optimizer.trainAAE()
        generater_path = "./output/2022-05-06-15-48-29"
        optimizer.gen_design(
            generater_path=generater_path,
            exec_num=10
        )
    else:
        raise Exception(f"Invalid model option: {args.model}, try: --model ae")


if __name__ == "__main__":
    main()
