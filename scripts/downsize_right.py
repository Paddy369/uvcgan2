import argparse
import os
import tqdm
from PIL import Image, ImageOps

def resize_image(image, out_shape, interpolation):
    if interpolation == 'bilinear':
        resample = Image.BILINEAR
    elif interpolation == 'bicubic':
        resample = Image.BICUBIC
    elif interpolation == 'lanczos':
        resample = Image.LANCZOS
    else:
        raise ValueError(f'Unknown interpolation method: {interpolation}')

    return image.resize(out_shape, resample = resample)

def parse_cmdargs():
    parser = argparse.ArgumentParser(description = 'Resize Images with PIL')

    parser.add_argument(
        'source',
        help    = 'source directory',
        metavar = 'SOURCE',
        type    = str,
    )

    parser.add_argument(
        'target',
        help    = 'target directory',
        metavar = 'TARGET',
        type    = str,
    )

    parser.add_argument(
        '-s', '--shape',
        dest     = 'shape',
        help     = 'output shape',
        type     = int,
        required = True,
        nargs    = 2,
    )

    parser.add_argument(
        '-i',
        choices  = [ 'bilinear', 'bicubic', 'lanczos' ],
        default  = 'lanczos',
        dest     = 'interpolation',
        help     = 'interpolation',
        type     = str,
    )

    return parser.parse_args()

def collect_images(root):
    result = []

    for curr_root, _dirs, files in os.walk(root):
        rel_path = os.path.relpath(curr_root, root)

        for fname in files:
            result.append((rel_path, fname))

    return result

def load_image(path):
    result = Image.open(path)
    result = ImageOps.exif_transpose(result)  # Handle EXIF orientation
    return result

def save_image(image, path):
    if not path.endswith('.png'):
        path = path + '.png'

    image.save(path)

def resize_images(source, images, target, shape, interpolation):
    for (subdir, fname) in tqdm.tqdm(images, total = len(images)):
        path_src = os.path.join(source, subdir, fname)

        root_dst = os.path.join(target, subdir)
        path_dst = os.path.join(root_dst, fname)

        os.makedirs(root_dst, exist_ok = True)

        image_src = load_image(path_src)

        # Check dimensions
        width, height = image_src.size
        if width < shape[0] or height < shape[1]:
            # Upsize if the image is smaller than the target shape
            new_shape = (max(width, shape[0]), max(height, shape[1]))
        else:
            # Downsize if the image is larger than or equal to the target shape
            new_shape = shape

        image_dst = resize_image(image_src, new_shape, interpolation)
        save_image(image_dst, path_dst)

def main():
    cmdargs = parse_cmdargs()
    images = collect_images(cmdargs.source)

    resize_images(
        cmdargs.source, images, cmdargs.target, cmdargs.shape,
        cmdargs.interpolation
    )

if __name__ == '__main__':
    main()
