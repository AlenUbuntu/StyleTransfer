
import os 
import argparse 
import torch
from StyleTransfer.config import cfg 
from StyleTransfer.utils import setup_logger
from torch.utils.collect_env import get_pretty_env_info
from StyleTransfer.models.FastPhotoStyle.test_fps import prepare_fps_model, fps_style_transfer
from torchvision.utils import save_image, make_grid
from StyleTransfer.data import build_transform, is_img_file, default_loader
from PIL import Image
from tqdm import tqdm
import cv2

if torch.__version__ == '0.4.1':
    from StyleTransfer.models.LinearStyleTransfer.test_lst import prepare_lst_model, lst_style_transfer

    model_factory = {
        'LST': prepare_lst_model,
        'FPS': prepare_fps_model
    }
    method_factory = {
        'LST': lst_style_transfer,
        'FPS': fps_style_transfer
    }
else:
    model_factory = {
        'FPS': prepare_fps_model
    }
    method_factory = {
        'FPS': fps_style_transfer
    }



def memory_limit_image_size(img, MINSIZE, MAXSIZE, logger=None):
    # prevent too small or too big images
    # ensure the shorter side of the image >= MINSIZE, and the longer side of the image <= MAXSIZE
    # if conflicted, we ensure the longest side satisfy the condition to avoid memory overflow
    # use thumbnail which perform resize in place, preserve aspect ratio and does not enlarge image.
    orig_width, orig_height = img.width, img.height 
    if max(img.width, img.height) < MINSIZE:
        # rescale image such that shorter side == min_size
        # however, since max(w, h) < minsize and thumbnail will not enlarge the image
        # this operation seems meaningless to me
        if img.width > img.height:
            img.thumbnail((int(img.width * 1. / img.height * MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            img.thumbnail((MINSIZE, int(img.height * 1. / img.width * MINSIZE)), Image.BICUBIC)
    
    if min(img.width, img.height) > MAXSIZE:
        # rescale image such taht longer side == max_size
        if img.width > img.height:
            img.thumbnail((MAXSIZE, int(img.height * 1. / img.width * MAXSIZE)), Image.BICUBIC)
        else:
            img.thumbnail((int(img.width * 1. / img.height * MAXSIZE), MAXSIZE), Image.BICUBIC)
    if logger:
        logger.info('Resize image: (%d, %d) -> (%d, %d)' % (orig_height, orig_width, img.height, img.width)) 
    else:
        print('Resize image: (%d, %d) -> (%d, %d)' % (orig_height, orig_width, img.height, img.width))
    return img.width, img.height


def infer_image(cfg, name, model, content_img, style_img, logger, output_dir, ch, cw, save_orig=False, content_seg_img=None, style_seg_img=None, orig_content=None, 
test_transform=None):
    """
    content_img, style_img is assumed to be tensor
    content_seg_img, style_seg_img is assumed to be PIL Image
    orig_content is assumed to be PIL Image
    the generated res_img could be either tensor or PIL image, depending on the model
    """
    if content_seg_img is not None and style_seg_img is not None:
        content_img = content_img.to(cfg.DEVICE)
        style_img = style_img.to(cfg.DEVICE)
        model.to(cfg.DEVICE)
        res_img = method_factory[cfg.MODEL.NAME](model, content_img, style_img, ch, cw, content_seg_img, style_seg_img, logger=logger, orig_content=orig_content, 
        test_transform=test_transform)
        # save images
        if save_orig:
            save_image(content_img, os.path.join(output_dir, '{}_content.jpg'.format(name)), nrow=1)
            save_image(style_img, os.path.join(output_dir, '{}_style.jpg'.format(name)), nrow=1)
            content_seg_img.save(os.path.join(output_dir, '{}_content_seg.jpg'.format(name)))
            style_seg_img.save(os.path.join(output_dir, '{}_style_seg.jpg'.format(name)))
        if torch.is_tensor(res_img):
            save_image(res_img, os.path.join(output_dir, '{}_generated.jpg'.format(name)), nrow=1)
        else:
            res_img.save(os.path.join(output_dir, '{}_generated.jpg'.format(name)))
    else:
        content_img, style_img = content_img.to(cfg.DEVICE), style_img.to(cfg.DEVICE)
        model.to(cfg.DEVICE)
        res_img = method_factory[cfg.MODEL.NAME](model, content_img, style_img, ch, cw, logger=logger, orig_content=orig_content, test_transform=test_transform)
        # save images
        if save_orig:
            save_image(content_img, os.path.join(output_dir, '{}_content.jpg'.format(name)), nrow=1)
            save_image(style_img, os.path.join(output_dir, '{}_style.jpg'.format(name)), nrow=1)
        if torch.is_tensor(res_img):
            save_image(res_img, os.path.join(output_dir, '{}_generated.jpg'.format(name)), nrow=1)
        else:
            res_img.save(os.path.join(output_dir, '{}_generated.jpg'.format(name)))

    # convert res_img to PIL image
    if torch.is_tensor(res_img):
        grid = make_grid(res_img, nrow=1, padding=0)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to('cpu', torch.uint8).numpy()
        res_img = Image.fromarray(ndarr)
    return res_img

def prepare_loading(cfg, content_dir, style_dir):
    img_list = [x for x in os.listdir(content_dir) if is_img_file(x)]
    img_list = sorted(img_list)
    names = [x[:x.rindex('.')] for x in img_list]

    content_img_path = [os.path.join(content_dir, each) for each in img_list]
    style_img_path = [os.path.join(style_dir, each) for each in img_list]

    content_img = [default_loader(each) for each in content_img_path]
    style_img = [default_loader(each) for each in style_img_path]
    return content_img, style_img, names

def test():
    parser = argparse.ArgumentParser(description='PyTorch Photo-Realistic Style Transfer Library')

    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        help='path to configuration file'
    )
    parser.add_argument(
        '--outputDir',
        type=str, 
        default='Demo',
        help='name of output folder'
    )
    parser.add_argument(
        '--saveOrig',
        default=False, 
        action='store_true'
    )
    parser.add_argument(
        '--contentDir',
        type=str,
        default='',
        help='path to directory of content images'
    )
    parser.add_argument(
        '--styleDir',
        type=str,
        default='',
        help='path to directory of style images'
    )
    parser.add_argument(
        '--content',
        type=str, 
        default='',
        help='path to content image'
    )
    parser.add_argument(
        '--style',
        type=str,
        default='',
        help='path to style image'
    )
    parser.add_argument(
        '--mode',
        type=int,
        default=0,
        help='Inference mode: 0 - Single Content; 1 - Multiple Content (Stored in a directory)'
    )

    # advanced options
    parser.add_argument(
        '--content-seg',
        default='',
        type=str,
        help='path to content mask image' 
    )
    parser.add_argument(
        '--style-seg',
        default='',
        type=str,
        help='path to style mask image'
    )
    parser.add_argument(
        '--resize',
        default=False,
        action='store_true',
        help='resize original image to accelerate computing'
    )
    args = parser.parse_args()

    # update configuration
    cfg.merge_from_file(args.config_file)

    cfg.freeze()

    test_transform = build_transform(cfg, train=False, interpolation=Image.BICUBIC, normalize=True)
    test_seg_transform = build_transform(cfg, train=False, interpolation=Image.NEAREST, normalize=False)

    if args.content_seg or args.style_seg:
        mask_on = True
    else:
        mask_on = False

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

    # create output dir
    output_dir = os.path.join(cfg.OUTPUT_DIR, args.outputDir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info('Output Dir Created: {}'.format(output_dir))

    # create model 
    model = model_factory[cfg.MODEL.NAME](cfg)
    logger.info(model)

    # inference
    if args.mode == 0:
        if mask_on:
            # 1-content | N-style | 1-mask, process single content image 
            assert args.content, 'Path to the content image should be non-empty'
            assert args.style, 'Paths to the style images should be non-empty'
            assert args.content_seg, 'Path to the content segment image should be non-empty'
            assert args.style_seg, 'Path to the style segment image should be non-empty'

            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_path = os.path.join(cfg.INPUT_DIR, args.style)
            content_seg_path = os.path.join(cfg.INPUT_DIR, args.content_seg) if args.content_seg else args.content_seg
            style_seg_path = os.path.join(cfg.INPUT_DIR, args.style_seg) if args.style_seg else args.style_seg

            name = content_img_path.split('/')[-1]
            name = name[:name.rindex('.')]

            # load image
            content_img = default_loader(content_img_path)
            style_img = default_loader(style_img_path)

            content_copy = content_img.copy()
            cw, ch = content_copy.width, content_copy.height
            sw, sh = style_img.width, style_img.height

            if args.resize:
                # new size after resizing content image
                new_cw, new_ch = memory_limit_image_size(content_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                # new size after resizing style image
                new_sw, new_sh = memory_limit_image_size(style_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
            else:
                new_cw, new_ch = cw, ch 
                new_sw, new_sh = sw, sh

            content_img = test_transform(content_img).unsqueeze(0)
            style_img = test_transform(style_img).unsqueeze(0)
            
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)

            # resize segmentation image the same size as corresponding images
            cont_seg = cont_seg.resize((new_cw, new_ch), Image.NEAREST)
            styl_seg = styl_seg.resize((new_sw, new_sh), Image.NEAREST)
            cont_seg = test_seg_transform(cont_seg)
            styl_seg = test_seg_transform(styl_seg)

            with torch.no_grad():
                infer_image(cfg, name, model, content_img, style_img, logger, output_dir, ch, cw, 
                save_orig=args.saveOrig, content_seg_img=cont_seg, style_seg_img=styl_seg, orig_content=content_copy, 
                test_transform=test_transform) 
            
        elif args.content and args.style:
            # 1-content | 1-style, process single pair of images
            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_path = os.path.join(cfg.INPUT_DIR, args.style)
            name = content_img_path.split('/')[-1]
            name = name[:name.rindex('.')]

            content_img = default_loader(content_img_path)
            style_img = default_loader(style_img_path)
            ch, cw = content_img.width, content_img.height
            content_copy = content_img.copy()

            if args.resize:
                # new size after resizing content image
                new_cw, new_ch = memory_limit_image_size(content_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                # new size after resizing style image
                new_sw, new_sh = memory_limit_image_size(style_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
            else:
                new_cw, new_ch = cw, ch 

            content_img = test_transform(content_img).unsqueeze(0)
            style_img = test_transform(style_img).unsqueeze(0)

            with torch.no_grad():
                infer_image(cfg, name, model, content_img, style_img, logger, output_dir, ch, cw, 
                save_orig=args.saveOrig, orig_content=content_copy, test_transform=test_transform) 
        else:
            raise RuntimeError('Invalid Argument Setting')

    else:
        if args.contentDir and args.styleDir:
            # 1-vs-1, but process multiple images in the directory
            content_img, style_img, names = prepare_loading(
                cfg, 
                os.path.join(cfg.INPUT_DIR, args.contentDir), 
                os.path.join(cfg.INPUT_DIR, args.styleDir), 
            )
            iterator = tqdm(range(len(content_img)))
            for i in iterator:
                c_img, s_img = content_img[i], style_img[i]
                cw, ch = c_img.width, c_img.height
                c_copy = c_img.copy()

                if args.resize:
                    # new size after resizing content image
                    new_cw, new_ch = memory_limit_image_size(c_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                    # new size after resizing style image
                    new_sw, new_sh = memory_limit_image_size(s_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                else:
                    new_cw, new_ch = cw, ch 

                c_img = test_transform(c_img).unsqueeze(0)
                s_img = test_transform(s_img).unsqueeze(0)

                name = names[i]

                with torch.no_grad():
                    infer_image(cfg, name, model, c_img, s_img, logger, output_dir, ch, cw, save_orig=args.saveOrig, orig_content=c_copy, 
                    test_transform=test_transform)
            
                iterator.set_description(desc='Test Case {}'.format(i))
        else:
            raise RuntimeError('Invalid Argument Setting')
    
    logger.info('Done!')
          
    

if __name__ == '__main__':
    test()
