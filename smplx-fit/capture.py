import argparse
import copy
import glob
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def capture_image(out_path,
                  mesh,
                  w=1024,
                  h=1024
                  ):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='open3d', width=w, height=h, left=0, top=0, visible=False)

    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on = True 

    vis.poll_events()
    vis.update_renderer()
    #image = vis.capture_screen_float_buffer(False)
    vis.capture_screen_image(out_path)
    vis.destroy_window()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root')
    parser.add_argument('--save_root')
    parser.add_argument('--resolution', type=int, default=1024)
    config = parser.parse_args()

    if config.root is None:
        raise ValueError('config.root is None')

    if config.save_root is None:
        raise ValueError('config.out_dir is None')

    save_root = os.path.join(config.save_root)
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)


    subj_dirs_list = sorted(glob.glob(config.root + '/*'))

    for subj_idx, subj_dir in enumerate(subj_dirs_list):

        subj_name = os.path.basename(subj_dir)
        save_dir = os.path.join(save_root, subj_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        obj_list = sorted(glob.glob(subj_dir + '/*.obj'))
        for obj_idx, obj_path in enumerate(obj_list):

            mesh = o3d.io.read_triangle_mesh(obj_path)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([1, 0.7, 0])

            ### Snapshot
            obj_name = os.path.basename(obj_path)
            image_path = os.path.join(save_dir, obj_name.replace('.obj', '.png'))
            capture_image(image_path, mesh, w=config.resolution, h=config.resolution)


