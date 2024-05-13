# Basic Package
import torch
import argparse
import numpy as np
import yaml
import logging
import time
import os
import pandas as pd
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# Own Package
from data.image_dataset import Image_Dataset
from utils.tools import seed_reproducer, load_checkpoint, get_cuda, print_options
from utils.get_logger import open_log
from networks.latent_mapping_model import ResAttnUNet_DS
from networks.models.autoencoder import AutoencoderKL
from networks.models.distributions import DiagonalGaussianDistribution

def load_img(path):
    image = Image.open(path).convert("L")
    w, h = 224, 224
    image = image.resize((w, h), resample=Image.NEAREST)
    image = np.array(image).astype(np.float32) / 255.0
    return image

def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * mean, logvar

def vae_decode(vae_model, pred_mean, scale_factor):
    z = 1. / scale_factor * pred_mean
    # z = pred_mean
    pred_seg = vae_model.decode(z)
    pred_seg = torch.mean(pred_seg, dim=1, keepdim=True)
    pred_seg = torch.clamp((pred_seg + 1.0) / 2.0, min=0.0, max=1.0) # (B, 1, H, W)
    return pred_seg

def arg_parse() -> argparse.ArgumentParser.parse_args :
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/valid.yaml',
                        type=str, help='load the config file')
    args = parser.parse_args()
    return args

def run_trainer() -> None:
    args = arg_parse()
    configs = yaml.load(open(args.config), Loader=yaml.FullLoader)
    configs['log_path'] = os.path.join(configs['snapshot_path'], 'logs')
    
    # Output folder and save fig folder
    os.makedirs(configs['snapshot_path'], exist_ok=True)
    os.makedirs(configs['save_seg_img_path'], exist_ok=True)
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

    # Get data loader
    valid_dataset = Image_Dataset(configs['pickle_file_path'], stage='test')
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    # Define networks
    mapping_model = get_cuda(ResAttnUNet_DS(
        in_channel=configs['in_channel'], 
        out_channels=configs['out_channels'], 
        num_res_blocks=configs['num_res_blocks'], 
        ch=configs['ch'], 
        ch_mult=configs['ch_mult']
    ))
    mapping_model = load_checkpoint(mapping_model, configs['model_weight'])
    mapping_model.eval()

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

    # Define loss functions
    mse_loss = torch.nn.MSELoss(reduction='mean')

    epoch_start_time = time.time()

    name_list = []

    T_loss_valid = []

    ### Validation phase
    for batch_data in tqdm(valid_dataloader, desc='Valid: '):
        img_rgb = batch_data['img']
        img_rgb = 2. * img_rgb - 1.
        seg_raw = batch_data['seg']
        seg_raw = seg_raw.permute(0, 3, 1, 2) / 255.0
        seg_rgb = 2. * seg_raw - 1.
        seg_img = torch.mean(seg_raw, dim=1, keepdim=True)
        name = batch_data['name'][0]
        name_list.append(name)

        with torch.no_grad():
            img_latent_mean, _ = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(img_rgb)), scale_factor)
            seg_latent_mean, _ = get_vae_encoding_mu_and_sigma(vae_model.encode(get_cuda(seg_rgb)), scale_factor)
            out_latent_mean_dict = mapping_model(img_latent_mean)

            loss_Rec = configs['w_rec'] * mse_loss(out_latent_mean_dict['out'], seg_latent_mean)

            pred_seg = vae_decode(vae_model, out_latent_mean_dict['out'], scale_factor)
            pred_seg = pred_seg.repeat(1, 3, 1, 1)

            x_sample = rearrange(pred_seg.squeeze().cpu().numpy(), 'c h w -> h w c')
            x_sample = np.where(x_sample > 0.5, 1, 0)
            x_sample = 255. * x_sample
            img = Image.fromarray(x_sample.astype(np.uint8))
            img.save(os.path.join(configs['save_seg_img_path'], name + '.png'))

            T_loss_valid.append(loss_Rec.item())

    T_loss_valid = np.mean(T_loss_valid)

    logging.info("Valid:")
    logging.info("loss: {:.4f}".format(T_loss_valid))

    ### load masks & compute dsc and iou
    csv_path  = os.path.join(configs['snapshot_path'], 'results.csv')
    pred_path = configs['save_seg_img_path']
    true_path = os.path.join(os.path.dirname(configs['pickle_file_path']), 'masks')

    name_list = sorted(os.listdir(pred_path))
    name_list = [x.replace('.png', '') for x in name_list]
    name_list = [x.replace('_segmentation', '') for x in name_list]

    dsc_list = []
    iou_list = []

    for case_name in tqdm(name_list):
        seg_pred = load_img(os.path.join(pred_path, case_name + '.png'))
        seg_true = load_img(os.path.join(true_path, case_name + '.png'))

        preds = np.array(seg_pred).reshape(-1)
        gts = np.array(seg_true).reshape(-1)

        y_pre = np.where(preds>=0.5, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        dsc_list.append(f1_or_dsc)
        iou_list.append(miou)

    # MEAN & Std Value
    name_list.extend(['Avg', 'Std'])
    dsc_list.extend([np.mean(dsc_list), np.std(dsc_list, ddof=1)])
    iou_list.extend([np.mean(iou_list), np.std(iou_list, ddof=1)])

    df = pd.DataFrame({
        'Name': name_list,
        'DSC':  dsc_list,
        'IoU': iou_list
    })
    df.to_csv(csv_path, index=False)

    logging.info("DSC: {:.4f}, IOU: {:.4f}".format(dsc_list[-2], iou_list[-2]))
    logging.info('Time Taken: %d sec' % (time.time() - epoch_start_time))
    logging.info('\n')

if __name__ == '__main__':
    run_trainer()
