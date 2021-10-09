import argparse
import glob
import os
import sys # for Command Line Arguments
import time
from jeanCV import skinDetector


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')

    config = parser.parse_args()

    subj_dirs = sorted(glob.glob(os.path.join(config.dataset, '*')))

    for subj_idx, subj_dir in enumerate(subj_dirs):
        
        print(f'Progressing {subj_idx} / {len(subj_dirs)} ...')

        # Make dir
        skin_dir = os.path.join(subj_dir, 'skin')
        if not os.path.exists(skin_dir):
            os.makedirs(skin_dir, exist_ok=True)

        image_path_list = sorted(glob.glob(os.path.join(subj_dir, 'images', '*.png')))

        s = time.time()
        
        for image_path in image_path_list:
            image_name = image_path.split('/')[-1]
            save_path = os.path.join(skin_dir, image_name[:-4]+'_skin.png')
            detector = skinDetector(image_path, save_path)
            detector.find_skin()

        print('elp:', time.time() - s)


