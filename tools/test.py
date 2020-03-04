import os 
import argparse 
import torch 
from StyleTransfer.config import cfg 
from StyleTransfer.utils import setup_logger
from torch.utils.collect_env import get_pretty_env_info
from StyleTransfer.models.AdaIN.test_adain import prepare_adain_model, adain_style_transfer
from StyleTransfer.models.WCT.test_wct import prepare_wct_model, wct_style_transfer
from StyleTransfer.models.FastPhotoStyle.test_fps import prepare_fps_model, fps_style_transfer_non_photo
from torchvision.utils import save_image, make_grid
from StyleTransfer.data import build_transform, is_img_file, default_loader
from tqdm import tqdm
from PIL import Image

if torch.__version__ == '0.4.1':
    from StyleTransfer.models.LinearStyleTransfer.test_lst import prepare_lst_model, lst_style_transfer_non_photo
    model_factory = {
        'AdaIN': prepare_adain_model,
        'WCT': prepare_wct_model,
        'LST': prepare_lst_model,
        'FPS': prepare_fps_model
    }
    method_factory = {
        'AdaIN': adain_style_transfer,
        'WCT': wct_style_transfer,
        'LST': lst_style_transfer_non_photo,
        'FPS': fps_style_transfer_non_photo
    }
else:
    model_factory = {
        'AdaIN': prepare_adain_model,
        'WCT': prepare_wct_model,
        'FPS': prepare_fps_model
    }
    method_factory = {
        'AdaIN': adain_style_transfer,
        'WCT': wct_style_transfer,
        'FPS': fps_style_transfer_non_photo
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


def infer_image(cfg, name, model, content_img, style_img, logger, output_dir, ch, cw, save_orig=False, alpha=1.0, mask_img=None, style_interp_weights=[]):
    """
    content_img, style_img is assumed to be tensor
    mask_img is assumed to be PIL Image
    the generated res_img could be either tensor or PIL image, depending on the model
    """
    if mask_img:
        # 1 content img, N style img, 1 mask img
        content_img, style_img = content_img.to(cfg.DEVICE), [each.to(cfg.DEVICE) for each in style_img]
        res_img = method_factory[cfg.MODEL.NAME](
            logger=logger,
            model=model, 
            content_img=content_img, 
            style_img=style_img, 
            alpha=alpha,
            mask_img=mask_img,
            ch=ch,
            cw=cw
        )
        # save images 
        if save_orig:
            save_image(content_img, os.path.join(output_dir, '{}_content.jpg'.format(name)), nrow=1)
            for i, each_style in enumerate(style_img):
                save_image(each_style, os.path.join(output_dir, '{}_style_{}.jpg'.format(name, i)), nrow=1)
            save_image(mask_img, os.path.join(output_dir, '{}_mask.jpg'.format(name)), nrow=1)
        if torch.is_tensor(res_img):
            save_image(res_img, os.path.join(output_dir, '{}_generated.jpg'.format(name)), nrow=1)
        else:
            res_img.save(os.path.join(output_dir, '{}_generated.jpg'.format(name)))
    elif style_interp_weights:
        content_img, style_img = content_img.to(cfg.DEVICE), [each.to(cfg.DEVICE) for each in style_img]
        res_img = method_factory[cfg.MODEL.NAME](
            logger=logger,
            model=model,
            content_img=content_img,
            style_img=style_img,
            alpha=alpha,
            style_interp_weights=style_interp_weights,
            ch=ch,
            cw=cw
        )
        # save images
        if save_orig:
            save_image(content_img, os.path.join(output_dir, '{}_content.jpg'.format(name)), nrow=1)
            for i, each_style in enumerate(style_img):
                save_image(each_style, os.path.join(output_dir, '{}_style_{}.jpg'.format(name, i)), nrow=1)
        if torch.is_tensor(res_img):
            save_image(res_img, os.path.join(output_dir, '{}_generated.jpg'.format(name)), nrow=1)
        else:
            res_img.save(os.path.join(output_dir, '{}_generated.jpg'.format(name)))
    else:
        content_img, style_img = content_img.to(cfg.DEVICE), style_img.to(cfg.DEVICE)
        res_img = method_factory[cfg.MODEL.NAME](
            logger=logger,
            model=model, 
            content_img=content_img, 
            style_img=style_img, 
            alpha=alpha, 
            ch=ch, 
            cw=cw
        )
        # save images
        if save_orig:
            save_image(content_img, os.path.join(output_dir, '{}_content.jpg'.format(name)), nrow=1)
            save_image(style_img, os.path.join(output_dir, '{}_style.jpg'.format(name)), nrow=1)
        if torch.is_tensor(res_img):
            save_image(res_img, os.path.join(output_dir, '{}_generated.jpg'.format(name)), nrow=1)
        else:
            res_img.save(os.path.join(output_dir, '{}_generated.jpg'.format(name)))
    
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
    parser = argparse.ArgumentParser(description='PyTorch Style Transfer Library')

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
        '--styleInterpWeights',
        default='', 
        type=str,
        help='The weight for blending the style of multiple style images'
    )
    parser.add_argument(
        '--mask',
        default='',
        type=str,
        help='path to mask image' 
    )
    parser.add_argument(
        '--resize',
        default=False,
        action='store_true',
        help='resize image to acclerate computing'
    )
    args = parser.parse_args()
    style_weight_name = args.styleInterpWeights
    if args.styleInterpWeights:
        args.styleInterpWeights = [float(each.strip()) for each in args.styleInterpWeights.split(',')]
        args.styleInterpWeights = [each / sum(args.styleInterpWeights) for each in args.styleInterpWeights]  # normalize weights

    # update configuration
    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    
    test_transform = build_transform(cfg, train=False, normalize=True, interpolation=Image.BICUBIC)
    test_seg_transform = build_transform(cfg, train=False, interpolation=Image.NEAREST, normalize=False)

    mask_on = args.mask != ''
    interpolate_on = args.styleInterpWeights != ''
    assert not (mask_on and interpolate_on), 'Spatial control and Style Interpolation cannot be activated simultaneously.'

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
        if interpolate_on:
            # 1-content | N-style, process single content image
            assert args.content, 'Path to the content image should be non-empty'
            assert args.style, 'Paths to the style images should be non-empty'
            assert args.styleInterpWeights, 'Style interpolation weights must be provided'
            assert cfg.MODEL.NAME != 'LST', 'Interpolation of LinearStyleTransfer is currently not supported!'
            assert cfg.MODEL.NAME != 'FPS', 'Interpolation of FastPhotoTransfer is currently not supported, but should be similar to WCT!'

            style_paths = args.style.split(',')
            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_paths = [os.path.join(cfg.INPUT_DIR, each) for each in style_paths]
            name = content_img_path.split('/')[-1]
            name = name[:name.rindex('.')] + '_' + style_weight_name

            # load image
            content_img = default_loader(content_img_path)
            style_imgs = [default_loader(each) for each in style_img_paths]
            ch, cw = content_img.width, content_img.height

            if args.resize:
                # new size after resizing content image
                new_cw, new_ch = memory_limit_image_size(content_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                # new size after resizing style image
                new_sw, new_sh = list(zip(*[memory_limit_image_size(each, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger) for each in style_imgs]))
            else:
                new_cw, new_ch = cw, ch 

            content_img = test_transform(content_img).unsqueeze(0)
            style_imgs = [test_transform(each).unsqueeze(0) for each in style_imgs]

            infer_image(cfg, name, model, content_img, style_imgs, logger, output_dir, ch, cw, save_orig=args.saveOrig, alpha=cfg.MODEL.ALPHA, 
            style_interp_weights=args.styleInterpWeights)

        elif mask_on:
            # 1-content | N-style | 1-mask, process single content image 
            assert args.content, 'Path to the content image should be non-empty'
            assert args.style, 'Paths to the style images should be non-empty'
            assert args.mask, 'Path to the mask image should be non-empty'
            assert cfg.MODEL.NAME != 'LST', 'Spatial Control of LinearStyleTransfer is currently not supported!'
            assert cfg.MODEL.NAME != 'FPS', 'Spatial Control of FastPhotoTransfer is currently not supported, but should be similar to WCT!'

            style_paths = args.style.split(',')
            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_paths = [os.path.join(cfg.INPUT_DIR, each) for each in style_paths]
            name = content_img_path.split('/')[-1]
            name = name[:name.rindex('.')] + '_mask'

            # read image 
            mask_img = default_loader(os.path.join(cfg.INPUT_DIR, args.mask))
            content_img = default_loader(content_img_path)
            style_imgs = [default_loader(each) for each in style_img_paths]

            cw, ch = content_img.width, content_img.height

            if args.resize:
                # new size after resizing content image
                new_cw, new_ch = memory_limit_image_size(content_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                # new size after resizing style image
                new_sw, new_sh = list(zip(*[memory_limit_image_size(each, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger) for each in style_imgs]))
            else:
                new_cw, new_ch = cw, ch 

            content_img = test_transform(content_img).unsqueeze(0)
            style_imgs = [test_transform(each).unsqueeze(0) for each in style_imgs]
            mask_img = mask_img.resize((new_cw, new_ch), Image.NEAREST)
            mask_img = test_seg_transform(mask_img)

            # read content image 
            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_paths = [os.path.join(cfg.INPUT_DIR, each) for each in style_paths]

            infer_image(cfg, name, model, content_img, style_imgs, logger, output_dir, ch, cw, save_orig=args.saveOrig, alpha=cfg.MODEL.ALPHA,
            mask_img=mask_img)

        elif args.content and args.style:
            # 1-content | 1-style, process single pair of images
            content_img_path = os.path.join(cfg.INPUT_DIR, args.content)
            style_img_path = os.path.join(cfg.INPUT_DIR, args.style)
            name = content_img_path.split('/')[-1]
            name = name[:name.rindex('.')]

            # read images
            content_img = default_loader(content_img_path)
            style_img = default_loader(style_img_path)

            cw, ch = content_img.width, content_img.height

            if args.resize:
                # new size after resizing content image
                new_cw, new_ch = memory_limit_image_size(content_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                # new size after resizing style image
                new_sw, new_sh = memory_limit_image_size(style_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
            else:
                new_cw, new_ch = cw, ch 
            
            content_img = test_transform(content_img).unsqueeze(0)
            style_img = test_transform(style_img).unsqueeze(0)

            infer_image(cfg, name, model, content_img, style_img, logger, output_dir, ch, cw, save_orig=args.saveOrig, alpha=cfg.MODEL.ALPHA) 
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
                name = names[i]
                cw, ch = c_img.width, c_img.height

                if args.resize:
                    # new size after resizing content image
                    new_cw, new_ch = memory_limit_image_size(c_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                    # new size after resizing style image
                    new_sw, new_sh = memory_limit_image_size(s_img, cfg.INPUT.MIN_SIZE, cfg.INPUT.MAX_SIZE, logger=logger)
                else:
                    new_cw, new_ch = cw, ch 

                c_img = test_transform(c_img).unsqueeze(0)
                s_img = test_transform(s_img).unsqueeze(0)

                infer_image(cfg, name, model, c_img, s_img, logger, output_dir, ch, cw, save_orig=args.saveOrig, alpha=cfg.MODEL.ALPHA)
            
                iterator.set_description(desc='Test Case {}'.format(i))
        else:
            raise RuntimeError('Invalid Argument Setting')
    
    logger.info('Done!')
          
    

if __name__ == '__main__':
    test()
