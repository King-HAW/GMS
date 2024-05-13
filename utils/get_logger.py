import os
import logging


def open_log(args, config):
    # open the log file
    log_savepath = config['log_path']
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = args.config.split('/')[-1].split('.')[0]+'-yaml'
    if os.path.isfile(os.path.join(log_savepath, '{}.txt'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.txt'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.txt'.format(log_name)))


def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s-%(levelname)s] %(message)s',
        datefmt='%y-%m-%d %H:%M:%S',
        filename=logFilename,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
