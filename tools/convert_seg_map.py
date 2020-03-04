from PIL import Image 
import numpy as np 
import argparse


def load_img(path):
    img = Image.open(path).convert('RGB')
    img = np.asarray(img)
    return img 

def compare_with_existing_keys(key, dict, pos, arr, eps=10):
    if key in dict:
        return key
    else:
        # it is at boundary
        # check neighbors
        while True:
            if pos-eps >= 0:
                tmp = arr[:, pos-eps].tolist()
                key2 = tuple(tmp)
                if key2 in dict:
                    return key2
            if pos+eps < arr.shape[1]:
                tmp = arr[:, pos+eps].tolist()
                key2 = tuple(tmp)
                if key2 in dict:
                    return key2
            eps *= 2



def convert(content_img, style_img, keep=2):
    label_count = {}
    color_label = {}

    # img (H, W, C)
    ch, cw, cc = content_img.shape
    content_img = content_img.transpose(2, 1, 0)  # (C, H, W)
    content_img_flat = content_img.reshape(cc, -1)  # (C, HW)
    sh, sw, sc = style_img.shape 
    style_img = style_img.transpose(2, 1, 0)
    style_img_flat = style_img.reshape(sc, -1)  # (c, hw)
    content_label = np.zeros((1, content_img_flat.shape[1]))
    style_label = np.zeros((1, style_img_flat.shape[1]))

    for i in range(content_img_flat.shape[1]):
        tmp = content_img_flat[:, i].tolist()
        key = tuple(tmp)
        label_count[key] = label_count.get(key, 0) + 1
    

    for i in range(style_img_flat.shape[1]):
        tmp = style_img_flat[:, i].tolist()
        key = tuple(tmp)
        label_count[key] = label_count.get(key, 0) + 1

    labels = sorted(label_count.items(), key=lambda x: x[1], reverse=True)[:keep]
    label_mapping = dict([(k, v) for v, (k, c) in enumerate(labels)])

    for i in range(content_img_flat.shape[1]):
        tmp = content_img_flat[:, i].tolist()
        key = tuple(tmp)
        key = compare_with_existing_keys(key, label_mapping, i, content_img_flat)
        content_label[0, i] = label_mapping[key]
    

    for i in range(style_img_flat.shape[1]):
        tmp = style_img_flat[:, i].tolist()
        key = tuple(tmp)
        key = compare_with_existing_keys(key, label_mapping, i, style_img_flat)
        style_label[0, i] = label_mapping[key]


    content_label = content_label.reshape(1, ch, cw).transpose(1, 2, 0)  # (ch, cw, 1)
    style_label = style_label.reshape(1, sh, sw).transpose(1, 2, 0)  # (sh, sw, 1)
    content_label = content_label.astype(np.uint8)
    style_label = style_label.astype(np.uint8)

    print("content_label: ", np.unique(content_label), type(content_label))
    print("style_label: ", np.unique(style_label), type(style_label))

    content_label = np.tile(content_label, (1, 1, 3))
    style_label = np.tile(style_label, (1, 1, 3))
    content_label = Image.fromarray(content_label)
    style_label = Image.fromarray(style_label)

    return content_label, style_label




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Segmentation Color Map to Label Maps')

    parser.add_argument(
        '--content',
        type=str,
        default='',
        help='path to content color map'
    )
    parser.add_argument(
        '--style',
        type=str, 
        default='',
        help='path to style color map'
    )
    parser.add_argument(
        '--keep',
        type=int,
        default=2,
        help='Number of colors in the map'
    )
    args = parser.parse_args()

    content_img = load_img(args.content)
    style_img = load_img(args.style)
    content_label, style_label = convert(content_img, style_img, keep=args.keep)

    content_path = args.content[:args.content.rindex('.')] + '_label.jpg'
    style_path = args.style[:args.style.rindex('.')] + '_label.jpg'
    content_label.save(content_path)
    style_label.save(style_path)