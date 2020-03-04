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


def train_lst():
    parser = argparse.ArgumentParser(description='PyTorch Style Transfer -- LinearStyleTransfer')

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
    style_dataset = DatasetNoSeg(cfg, train_path_style, train=True)

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

    # style loader
    sampler = torch.utils.data.sampler.RandomSampler(style_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, cfg.DATALOADER.BATCH_SIZE, drop_last=False
    )
    style_loader = DataLoader(
        style_dataset,
        batch_sampler=IterationBasedBatchSampler(batch_sampler, cfg.OPTIMIZER.MAX_ITER, start_iter=0),
        num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    logger.info('Style Loader Created!')
    content_loader = iter(content_loader)
    style_loader = iter(style_loader)

    optimizer = build_optimizer(cfg, model.trans_layer)
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
        style_img = next(style_loader).to(cfg.DEVICE)
        if content_img.shape[0] != style_img.shape[0]:
            continue

        g_t = model.forward_with_trans(content_img, style_img)

        loss, style_loss, content_loss = model.cal_trans_loss(g_t, content_img, style_img)

        # update info
        iterator.set_description(desc='Iteration: {} -- Loss: {:.3f} -- Content Loss: {:.3f} -- Style Loss: {:.3f}'.format(
            i+1, loss.item(), content_loss.item(), style_loss.item()))
        writer.add_scalar('loss_content', content_loss.item(), i+1)
        writer.add_scalar('loss_style', style_loss.item(), i+1)

        # update model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update lr
        lr_scheduler.step()

        # save image
        if i % 1000 == 0:
            n = content_img.shape[0]
            all_imgs = torch.cat((content_img, style_img, g_t), dim=0)
            save_image(all_imgs, os.path.join(cfg.OUTPUT_DIR, '{}.jpg'.format(i)), nrow=n)
    
        if i % 10000 == 0:
            torch.save(model.trans_layer.state_dict(), os.path.join(cfg.OUTPUT_DIR, '{}_lst.pth'.format(i)))

    torch.save(model.trans_layer.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'final_lst.pth'))
    writer.close()


if __name__ == '__main__':
    train_lst()
