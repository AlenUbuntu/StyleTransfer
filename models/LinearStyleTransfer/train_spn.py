from StyleTransfer.config import cfg 
from StyleTransfer.utils import setup_logger
from torch.utils.collect_env import get_pretty_env_info
from StyleTransfer.models import get_model
from StyleTransfer.data import DatasetNoSeg, IterationBasedBatchSampler
from StyleTransfer.config import get_data
from torch.utils.data import DataLoader, RandomSampler
from StyleTransfer.optimizer import build_optimizer, build_lr_scheduler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
import torch
import os 
import argparse 


def train_spn():
    parser = argparse.ArgumentParser(description='PyTorch Style Transfer -- LinearStyleTransferWithSPN')

    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        help='path to configuration file'
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)

    cfg.freeze()

    # create output dir
    if cfg.OUTPUT_DIR:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # create logger 
    logger = setup_logger(cfg.MODEL.NAME, save_dir=cfg.OUTPUT_DIR, filename=cfg.MODEL.NAME+'.txt')

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    logger.info('Loaded configuration file {}'.format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    # create model 
    model = get_model(cfg.MODEL.NAME, cfg)

    # push model to device
    model.to(cfg.DEVICE)

    logger.info(model)

    # create dataloader
    train_path_content, train_path_style = get_data(cfg, dtype='train')
    content_dataset = DatasetNoSeg(cfg, train_path_content, train=True)

    # content loader
    sampler = torch.utils.data.sampler.RandomSampler(content_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.DATALOADER.BATCH_SIZE, drop_last=False
    )
    content_loader = DataLoader(
        content_dataset,
        batch_sampler=IterationBasedBatchSampler(batch_sampler, cfg.OPTIMIZER.MAX_ITER, start_iter=0),
        num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    logger.info('Content Loader Created!')

    content_loader = iter(content_loader)

    optimizer = build_optimizer(cfg, model.SPN)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    logger.info("Using Optimizer: ")
    logger.info(optimizer)
    logger.info("Using LR Scheduler: {}".format(
        cfg.OPTIMIZER.LR_SCHEDULER.NAME))

    iterator = tqdm(range(cfg.OPTIMIZER.MAX_ITER))

    writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR)
    # start training
    for i in iterator:
        content_img = next(content_loader).to(cfg.DEVICE)

        # reconstruct image - auto encoder only
        reconstructed = model.forward_with_no_trans(content_img)
        # fix distortion with SPN
        propogated = model.forward_spn(reconstructed, content_img)

        loss = model.cal_spn_loss(propogated, content_img)

        # update info
        iterator.set_description(desc='Iteration: {} -- Loss: {:.3f}'.format(
            i+1, loss.item()))
        writer.add_scalar('loss', loss.item(), i+1)

        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update lr
        lr_scheduler.step()

        # save image
        if i % 1000 == 0:
            n = content_img.shape[0]
            all_imgs = torch.cat((reconstructed, content_img), dim=0)
            save_image(all_imgs, os.path.join(cfg.OUTPUT_DIR, '{}.jpg'.format(i)), nrow=n)
    
        if i % 10000 == 0:
            torch.save(model.SPN.state_dict(), os.path.join(cfg.OUTPUT_DIR, '{}_lst_spn.pth'.format(i)))

    torch.save(model.SPN.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'final_lst_spn.pth'))
    writer.close()


if __name__ == '__main__':
    train_spn()
