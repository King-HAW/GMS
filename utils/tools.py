import torch
import os
import random
import numpy as np
import logging


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def load_checkpoint(model: torch.nn.Module, path: str) -> torch.nn.Module:
    if os.path.isfile(path):
        logging.info("=> loading checkpoint '{}'".format(path))
        
        # remap everthing onto CPU 
        state = torch.load(path, map_location=lambda storage, location: storage)

        # load weights
        model.load_state_dict(state['model'])
        logging.info("Loaded")
    else:
        model = None
        logging.info("=> no checkpoint found at '{}'".format(path))
    return model


def save_checkpoint(model: torch.nn.Module, save_name: str, path: str) -> None:
    model_savepath = os.path.join(path, 'checkpoints')
    if not os.path.exists(model_savepath):
        os.makedirs(model_savepath)
    file_name = os.path.join(model_savepath, save_name)
    torch.save({
        'model': model.state_dict(),
    },file_name)
    logging.info("save model to {}".format(file_name))


def adjust_learning_rate(optimizer, initial_lr, epoch, reduce_epoch, decay=0.5):
    lr = initial_lr * (decay ** (epoch // reduce_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    logging.info('Change Learning Rate to {}'.format(lr))
    return lr


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def print_options(configs):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in configs.items():
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    logging.info(message)

    # save to the disk
    file_name = os.path.join(configs['log_path'], '{}_configs.txt'.format(configs['phase']))
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')