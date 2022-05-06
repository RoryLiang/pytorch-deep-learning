import itertools
import time
import torch
import logging
# import numpy as np
# import functools
from matplotlib import pyplot as plt
from icecream import ic
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import make_grid, save_image
from model.aae import Encoder, Decoder, Discriminator
from utils.logging_ import init_logger


logger = logging.getLogger(__name__)


class AAEoptimizer():
    def __init__(self, args) -> None:
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu}" if torch.cuda.is_available() else "cpu")

    def trainAAE(self):
        init_logger(rank=0, filenmae=self.args.output_dir+"/default.log")
        config_str = ic.format(self.args)
        logger.info(f"config: {config_str}")
        writer = SummaryWriter("/".join([self.args.output_dir, "tb"]))

        train_dataset = datasets.MNIST(
            root=self.args.data_dir,
            train=True,
            transform=ToTensor(),
            download=True
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        logger.info("dataset loading done")

        encoder = Encoder(
            input_height=self.args.rows,
            input_width=self.args.columns,
            input_channel=1
        )
        decoder = Decoder(
            input_height=self.args.rows,
            input_width=self.args.columns,
            input_channel=1
        )
        discriminator = Discriminator()
        encoder.to(self.device)
        decoder.to(self.device)
        discriminator.to(self.device)
        logger.info(f"training on device: {self.device}")

        recon_loss_fn = nn.MSELoss()
        gan_loss_fn = nn.BCELoss()
        recon_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=self.args.learning_rate)
        gan_optimizer = torch.optim.Adam(discriminator.parameters(), lr=self.args.learning_rate)

        for epoch in range(1, self.args.epoch_num+1):

            train_recon_epoch_loss = 0
            train_dis_epoch_loss = 0
            train_iteration_num = 0
            train_raw_desgin = None
            train_recon_design = None
            encoder.train()
            decoder.train()
            discriminator.train()
            for batch_idx, (design, label) in enumerate(train_dataloader):

                # ae reconstruct phase
                design = design.to(self.device)
                z = encoder(design)
                z_label = torch.zeros(design.shape[0], 1, device=self.device)
                recon_tensor = decoder(z)
                recon_loss = recon_loss_fn(recon_tensor, design)
                train_recon_epoch_loss += recon_loss

                recon_optimizer.zero_grad()
                recon_loss.backward()
                recon_optimizer.step()
                train_raw_desgin = design
                train_recon_design = (recon_tensor > 0.5).float()

                # train discriminator
                z_sampled = torch.randn(design.shape[0], z.size(1), device=self.device)
                z_sampled_label = torch.ones(design.shape[0], 1, device=self.device)

                real_pred = discriminator(z_sampled)
                fake_pred = discriminator(z)
                real_dis_loss = gan_loss_fn(real_pred, z_sampled_label)
                fake_dis_loss = gan_loss_fn(fake_pred, z_label)

                dis_loss = (real_dis_loss + fake_dis_loss) / 2
                train_dis_epoch_loss += dis_loss

                gan_optimizer.zero_grad()
                dis_loss.backward
                gan_optimizer.step()

                train_iteration_num = batch_idx + 1
            train_recon_epoch_loss /= train_iteration_num
            train_dis_epoch_loss /= train_iteration_num

            logger.info(",".join([
                "train one epoch"
                f" epoch={epoch:03d}",
                f" reconstruct_loss={train_recon_epoch_loss:.4f}",
                f" discriminator_loss={train_dis_epoch_loss:.4f}"
            ]))

            writer.add_scalar("loss/reconstruct_loss", train_recon_epoch_loss, epoch)
            writer.add_scalar("loss/dis_loss", train_dis_epoch_loss, epoch)
            writer.add_images("raw_designs", train_raw_desgin, epoch)
            writer.add_images("reconstruct_designs", train_recon_design, epoch)

        writer.close()
        torch.save(encoder, self.args.output_dir+"/encoder_model.pth")
        torch.save(decoder, self.args.output_dir+"/decoder_model.pth")
        torch.save(discriminator, self.args.output_dir+"/discriminator_model.pth")

    '''
    def prepare_target_func(self):
        # 加载数据目标信息
        target_conf = self.args.conf.get_config("target")
        target_list = []
        weight = (1, 100)

        for instance_name in target_conf:
            instance = target_conf.get_config(instance_name)
            target = TargetInfo(function_name=instance.get_string("function_name"),
                                frequency_range=instance.get_list("frequency_range"),
                                optimize_type=instance.get_string("optimize_type"),
                                predict_method=instance.get_int("predict_method"))
            target_list.append(target)

        dataset = ICDataset(
            self.args.data_dir,
            target_list,
            general_target_function,
            (self.args.rows, self.args.columns),
            mode=0
        )

        target_compute_helper = TargetComputeHelper(target_list)

        target_func = functools.partial(target_compute_helper.target_function,
                                        weight=weight,
                                        index=dataset.data_index_to_frequency)

        return target_func
    '''

    def gen_design(self, design_num=50, latent_dim=15, generater_path=None, exec_num=1):
        logger.info("using trained encoder as generator to generate new designs")
        decoder = Decoder(
            input_height=self.args.rows,
            input_width=self.args.columns,
            input_channel=1
        )
        decoder.to(self.device)
        model_path = generater_path if generater_path is not None else self.args.output_dir
        logger.info(f"load generator from saved encoder: {model_path}/decoder_model.pth")
        decoder = torch.load(model_path+"/decoder_model.pth")
        decoder.eval()

        with torch.no_grad():
            random_z = torch.randn(design_num, latent_dim)
            random_z = random_z.to(self.device)
            design = decoder(random_z)
            # design_sharp = (design > 0.5).float()

        # logger.info("load pretrained predictor from: %s" % ('/'.join([self.args.data_dir, const.PREDICTOR_PATH])))
        # predictor = build_model(self.args.model_name).to(self.device)
        # net_params = torch.load('/'.join([self.args.data_dir, const.PREDICTOR_PATH]))
        # predictor.load_state_dict(net_params, strict=False)
        # predictor.eval()

        # target_func = self.prepare_target_func()

        # logger.info("predict ic designs\' target value ")
        # with torch.no_grad():
        #     pred_vectors = predictor(design).cpu().detach().numpy()
        #     target_values_list = [target_func(pred_vectors[i]) for i in range(len(pred_vectors))]

        exec_count = exec_num
        while exec_count != 0:
            fig = plt.figure(figsize=(15, 15), dpi=100)
            for i in range(design_num):

                # 1：子图共1行，design_num:子图共design_num列，当前绘制第i+1个子图
                ax = fig.add_subplot(5, 10, i+1, xticks=[], yticks=[])

                # CHW -> HWC
                npimg = design[i].cpu().numpy().transpose(1, 2, 0)

                plt.imshow(npimg)

                title = f"Design: {i}"
                ax.set_title(title, fontsize=13, color=("red"))
            plt.savefig(
                model_path
                + f"/generated_design-{int(time.time())}.png"
            )
            exec_count -= 1
