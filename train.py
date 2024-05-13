# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from monai.losses.dice import DiceLoss

# Own Package
from data.image_dataset import Image_Dataset
from utils.tools import seed_reproducer, save_checkpoint, get_cuda, print_options
from utils.get_logger import open_log
from utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from networks.latent_mapping_model import ResAttnUNet_DS
from networks.models.autoencoder import AutoencoderKL
from networks.models.distributions import DiagonalGaussianDistribution

from tensorboardX import SummaryWriter


def get_multi_loss(criterion, out_dict, label, is_ds=True, key_list=None):
        keys = key_list if key_list is not None else list(out_dict.keys())
        if is_ds:
            multi_loss = sum([criterion(out_dict[key], label) for key in keys])
        else:
            multi_loss = criterion(out_dict['out'], label)
        return multi_loss

def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * mean, logvar

def vae_decode(vae_model, pred_mean, scale_factor):
    z = 1. / scale_factor * pred_mean
    pred_seg = vae_model.decode(z)
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0) # (B, 1, H, W)
    return pred_seg
    

def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train.yaml',
                        type=str, help='load the config file')
    args = parser.parse_args()
    return args

def run_trainer() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['snapshot_path'] = os.path.join(configs['snapshot_path'], time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
    configs['log_path'] = os.path.join(configs['snapshot_path'], 'logs')
    
    # Output folder and save fig folder
    os.makedirs(configs['snapshot_path'], exist_ok=True)
    os.makedirs(configs['log_path'], exist_ok=True)

    # Set GPU ID
    gpus = ','.join([str(i) for i in configs['GPUs']])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    
    # Fix seed (for repeatability)
    seed_reproducer(configs['seed'])

    # Open log file
    open_log(args, configs)
    logging.info(configs)
    print_options(configs)

    # Define summary writer
    writer = SummaryWriter(configs['log_path'])
    ds_list = ['level2', 'level1', 'out']

    # Get data loader
    train_dataset = Image_Dataset(configs['pickle_file_path'], stage='train')
    valid_dataset = Image_Dataset(configs['pickle_file_path'], stage='test')
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], pin_memory=True, drop_last=True, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=configs['batch_size'], pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    mapping_model = get_cuda(ResAttnUNet_DS(
        in_channel=configs['in_channel'], 
        out_channels=configs['out_channels'], 
        num_res_blocks=configs['num_res_blocks'], 
        ch=configs['ch'], 
        ch_mult=configs['ch_mult']
    ))

    # get VAE (first-stage model)
    vae_path = './configs/v2-inference-v-first-stage-VAE.yaml'
    vae_config = OmegaConf.load(f"{vae_path}")
    vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))

    pl_sd = torch.load("SD-VAE-weights/768-v-ema-first-stage-VAE.ckpt", map_location="cpu")
    sd = pl_sd["state_dict"]
    vae_model.load_state_dict(sd, strict=True)

    vae_model.freeze()
    vae_model = get_cuda(vae_model)
    scale_factor = vae_config.first_stage_config.scale_factor

    # Define optimizers
    optimizer = torch.optim.AdamW(mapping_model.parameters(), lr=configs['lr'])
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=configs['epochs'])

    # Define loss functions
    mse_loss  = torch.nn.MSELoss(reduction='mean')
    dice_loss = DiceLoss()
 
    # For Tensorboard Visualization
    iter_num = 0
    best_valid_loss = np.inf
    best_valid_loss_rec = np.inf
    best_valid_dice = 0
    best_valid_dice_epoch = 0
    best_valid_loss_dice = np.inf

    # Network training
    for epoch in range(1, configs['epochs'] + 1):
        epoch_start_time = time.time()
        mapping_model.train()

        T_loss = []
        T_loss_Rec = []
        T_loss_Dice = []
        T_loss_valid = []
        T_loss_Rec_valid = []
        T_loss_Dice_valid = []
        T_Dice_valid = []

        ### Training phase
        for batch_data in tqdm(train_dataloader, desc='Train: '):
            img_rgb = batch_data['img']
            img_rgb = 2. * img_rgb - 1.
            seg_raw = batch_data['seg']
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2. * seg_raw - 1.
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
            name    = batch_data['name']

            img_latent_mean, img_latent_logvar = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(img_rgb)), scale_factor)
            seg_latent_mean, seg_latent_logvar = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(seg_rgb)), scale_factor)
            img_latent_std = torch.exp(0.5 * img_latent_logvar)
            
            if np.random.uniform() > 0.5:
                img_latent_mean_aug = img_latent_mean + img_latent_std * torch.randn_like(img_latent_std)
            else:
                img_latent_mean_aug = img_latent_mean

            # latent matching
            out_latent_mean_dict = mapping_model(img_latent_mean_aug)
            loss_Rec = configs['w_rec'] * get_multi_loss(mse_loss, out_latent_mean_dict, seg_latent_mean, is_ds=True, key_list=ds_list)

            # image matching
            pred_seg_dict = {}
            for level_name in ds_list:
                pred_seg_dict[level_name] = vae_decode(vae_model, out_latent_mean_dict[level_name], scale_factor)
            loss_Dice = configs['w_dice'] * get_multi_loss(dice_loss, pred_seg_dict, get_cuda(seg_img), is_ds=True, key_list=ds_list)

            loss = loss_Rec + loss_Dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            if iter_num % 10 == 0:
                writer.add_scalar('loss/loss', loss, iter_num)
                writer.add_scalar('loss/loss_Rec', loss_Rec, iter_num)
                writer.add_scalar('loss/loss_Dice', loss_Dice, iter_num)

            T_loss.append(loss.item())
            T_loss_Rec.append(loss_Rec.item())
            T_loss_Dice.append(loss_Dice.item())

        scheduler.step()
        writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        T_loss = np.mean(T_loss)
        T_loss_Rec = np.mean(T_loss_Rec)
        T_loss_Dice = np.mean(T_loss_Dice)

        logging.info("Train:")
        logging.info("loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}".format(
            T_loss, T_loss_Rec, T_loss_Dice
        ))

        ### Validation phase
        for batch_data in tqdm(valid_dataloader, desc='Valid: '):
            img_rgb = batch_data['img']
            img_rgb = 2. * img_rgb - 1.
            seg_raw = batch_data['seg']
            seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
            seg_rgb = 2. * seg_raw - 1.
            seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
            name    = batch_data['name']

            mapping_model.eval()

            with torch.no_grad():
                img_latent_mean, img_latent_logvar = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(img_rgb)), scale_factor)
                seg_latent_mean, seg_latent_logvar = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(seg_rgb)), scale_factor)
                
                out_latent_mean_dict = mapping_model(img_latent_mean)
                pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)

                loss_Rec = configs['w_rec'] * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)
                loss_Dice = configs['w_dice'] * dice_loss(pred_seg, get_cuda(seg_img))

                loss = loss_Rec + loss_Dice

                # calc dice
                pred_seg = pred_seg.cpu()
                reduce_axis = list(range(1, len(seg_img.shape)))

                intersection = torch.sum(seg_img * pred_seg, dim=reduce_axis)
                y_o = torch.sum(seg_img, dim=reduce_axis)
                y_pred_o = torch.sum(pred_seg, dim=reduce_axis)
                denominator = y_o + y_pred_o
                dice_raw = (2.0 * intersection) / denominator
                dice_value = dice_raw.mean()

                T_Dice_valid.append(dice_value.item())
                T_loss_valid.append(loss.item())
                T_loss_Rec_valid.append(loss_Rec.item())
                T_loss_Dice_valid.append(loss_Dice.item())

        T_Dice_valid = np.mean(T_Dice_valid)
        T_loss_valid = np.mean(T_loss_valid)
        T_loss_Rec_valid = np.mean(T_loss_Rec_valid)
        T_loss_Dice_valid = np.mean(T_loss_Dice_valid)

        writer.add_scalar('valid/dice', T_Dice_valid, epoch)
        writer.add_scalar('valid/loss', T_loss_valid, epoch)
        writer.add_scalar('valid/loss_Rec', T_loss_Rec_valid, epoch)
        writer.add_scalar('valid/loss_Dice', T_loss_Dice_valid, epoch)

        logging.info("Valid:")
        logging.info("loss: {:.4f}, loss_Rec: {:.4f}, loss_Dice: {:.4f}, Dice: {:.4f}".format(
            T_loss_valid, T_loss_Rec_valid, T_loss_Dice_valid, T_Dice_valid
        ))

        if T_Dice_valid > best_valid_dice:
            save_name = "best_valid_dice.pth"
            save_checkpoint(mapping_model, save_name, configs['snapshot_path'])
            best_valid_dice = T_Dice_valid
            best_valid_dice_epoch = epoch
            logging.info("Save best valid Dice !")

        if T_loss_valid < best_valid_loss:
            save_name = "best_valid_loss.pth"
            save_checkpoint(mapping_model, save_name, configs['snapshot_path'])
            best_valid_loss = T_loss_valid
            logging.info("Save best valid Loss All !")

        if T_loss_Rec_valid < best_valid_loss_rec:
            save_name = "best_valid_loss_rec.pth"
            save_checkpoint(mapping_model, save_name, configs['snapshot_path'])
            best_valid_loss_rec = T_loss_Rec_valid
            logging.info("Save best valid Loss Rec !")

        if T_loss_Dice_valid < best_valid_loss_dice:
            save_name = "best_valid_loss_dice.pth"
            save_checkpoint(mapping_model, save_name, configs['snapshot_path'])
            best_valid_loss_dice = T_loss_Dice_valid
            logging.info("Save best valid Loss Dice !")

        if epoch % configs['save_freq'] == 0:
            save_name = "{}_epoch_{:0>4}.pth".format('latent_mapping_model', epoch)
            save_checkpoint(mapping_model, save_name, configs['snapshot_path'])

        logging.info('Current learning rate: {:.5f}'.format(scheduler.get_last_lr()[0]))
        logging.info('epoch %d / %d \t Time Taken: %d sec' % (epoch, configs['epochs'], time.time() - epoch_start_time))
        logging.info('best valid dice: {:.4f} at epoch: {}'.format(best_valid_dice, best_valid_dice_epoch))
        logging.info('\n')

    writer.close()

if __name__ == '__main__':
    run_trainer()
