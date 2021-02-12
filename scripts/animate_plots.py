import argparse
import imageio
import os
import glob
import re
from tqdm import tqdm

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def animate_plots(input_path, output_path='./animation.mp4', fps=10):
    with imageio.get_writer(output_path, fps=fps) as writer:
        for filename in tqdm(sorted(glob.iglob(os.path.join(input_path, '*.png')), key=natural_keys)):
            image = imageio.imread(filename)
            writer.append_data(image)

def generate_animations(parent_directory, image_folder, output_folder, fps=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for exp_dir in glob.iglob(os.path.join(parent_directory, '*')):
        # Images are contained in <parent_directory>/<dir>/<image_folder>/*
        image_dir = os.path.join(exp_dir, image_folder)

        # Output file name is rewards_<id>.mp4
        output_filename = f'rewards_{exp_dir.split("/")[-1].split("-")[0]}.mp4'

        # Output path is <output_folder>/<output_filename>
        output_path = os.path.join(output_folder, output_filename)

        # Animate images and save in output path
        animate_plots(image_dir, output_path, fps=fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a GIF or video from a series of images.')
    parser.add_argument('--input_path', type=str, help='Directory containing the images')
    parser.add_argument('--output_path', type=str, help='File to save the animation to')
    parser.add_argument('--fps', type=int, nargs='?', default=10, help='Frames per second for the animation. Default: 10')
    parser.add_argument('--multiple_folders', action='store_true', default=False, help='If True, will treat input_path as the parent directory containing many folders')
    args = parser.parse_args()

    if args.multiple_folders:
        generate_animations(args.input_path, 'comparisons', args.output_path, fps=args.fps)
    else:
        animate_plots(args.input_path, args.output_path, args.fps)