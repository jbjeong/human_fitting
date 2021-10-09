import argparse
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--root')
parser.add_argument('--path')
config = parser.parse_args()

if config.root is not None:
    mesh_dirs = sorted(glob.glob(config.root + '/*'))
elif config.path is not None:
    mesh_dirs = [config.path]
else:
    raise ValueError('No root or path.')

for idx, path in enumerate(mesh_dirs):
    print('Progressing %d ...' % idx)
    image_dir = os.path.join(path, 'images')
    write_json = os.path.join(path, 'keypts')
    #os.system("./build/examples/openpose/openpose.bin -render_pose 0 -display 0 --hand --face --number_people_max 1 --output_resolution -1x-1 --image_dir %s --write_json %s" % (image_dir, write_json))
    os.system("./build/examples/openpose/openpose.bin --hand --hand_scale_number 6 --hand_scale_range 0.4 --face --number_people_max 1 --output_resolution -1x-1 --image_dir %s --write_json %s" % (image_dir, write_json))


