import argparse
import copy
import glob
import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from data_info import (not_standing, not_upright_plus, not_upright_minus, no_texture, mesh_list_needed_to_fix)

from util import preprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)



def capture_image(out_path,
                  depth_path,
                  mesh,
                  ctr_x_rot,
                  ctr_y_rot,
                  w=1024,
                  h=1024
                  ):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='open3d', width=w, height=h, left=0, top=0, visible=False)

    vis.add_geometry(mesh)
    vis.get_render_option().mesh_show_back_face = True
    vis.get_render_option().light_on = False 

    ctr = vis.get_view_control()
    ctr.rotate(ctr_x_rot, ctr_y_rot)
    param = ctr.convert_to_pinhole_camera_parameters()

    filename = out_path[:-3] + 'json'
    o3d.io.write_pinhole_camera_parameters(filename, param)

    s_cap = time.time()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_path)
    depth_image = vis.capture_depth_float_buffer(False)
    depth_image = np.array(depth_image, dtype=np.float32)
    np.save(depth_path, depth_image)
    logger.debug('Elapsed cap: {}'.format(time.time() - s_cap))

    vis.destroy_window()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir')
    parser.add_argument('--out_dir')
    parser.add_argument('--resolution', type=int, default=1024)
    config = parser.parse_args()

    if config.mesh_dir is None:
        raise ValueError('config.mesh_dir is None')

    if config.out_dir is None:
        raise ValueError('config.out_dir is None')

    ### Make dir
    save_root = config.out_dir
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    #o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    total_mesh_path_list = sorted(glob.glob(f'{config.mesh_dir}/*/*_textured.obj'))

    mesh_path_list = []
    for path in total_mesh_path_list:
        subj_name = path.split('/')[-2]
        subj_name = subj_name[:-4]
        if not subj_name in not_standing:
            mesh_path_list.append(path)

    logger.info(len(total_mesh_path_list))
    logger.info(len(not_standing))
    logger.info(len(mesh_path_list))

    start_iter = time.time()
    for mesh_i, mesh_path in enumerate(mesh_path_list):

        subj_name = mesh_path.split('/')[-2]
        subj_name = subj_name[:-4]

        ### Make dir
        image_dir = os.path.join(save_root, subj_name, 'images')
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)
        depth_dir = os.path.join(save_root, subj_name, 'depth')
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir, exist_ok=True)
        subj_dir = os.path.join(save_root, subj_name)

        logger.info('Progressing [{} / {}] ...'.format(mesh_i+1, len(mesh_path_list)))
        logger.info(mesh_path)
        start = time.time()
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        # Move the mesh into unit sphere
        mesh = preprocess(mesh)

        ### ** Make mesh to stand upright 
        if subj_name in not_upright_plus:
            r_x = np.pi / 2
            rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                                [0, np.sin(r_x), np.cos(r_x)]])
            mesh.rotate(rot_x, False)
        elif subj_name in not_upright_minus:
            r_x = -np.pi / 2
            rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                                [0, np.sin(r_x), np.cos(r_x)]])
            mesh.rotate(rot_x, False)

        max_theta = 360
        rot_list = []

        if subj_name in mesh_list_needed_to_fix:
            r_y_value = 0 
        else:
            r_y_value = -90 
        rot_y = np.asarray([[np.cos(r_y_value), 0, np.sin(r_y_value)], 
                            [0, 1, 0],
                            [-np.sin(r_y_value), 0, np.cos(r_y_value)]])
        mesh.rotate(rot_y, False)

        # Save pcd
        sampled_vs = np.array(mesh.vertices)[::5]
        sampled_ns = np.array(mesh.vertex_normals)[::5]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_vs)
        colors = np.zeros(sampled_vs.shape)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        out_complete_path = os.path.join(subj_dir, f'pcd.ply')
        o3d.io.write_point_cloud(out_complete_path, pcd)

        normal_path = os.path.join(subj_dir, f'pcd_normal.npy')
        np.save(normal_path, sampled_ns)

        init_r_x = np.pi
        for x_value in range(-2, 3):
            r_x = init_r_x + 0.5*x_value
            for r_value in range(-20, 20):
                r_y = (5*r_value) * (np.pi * 2 / max_theta) 
                rot_list.append((r_x, r_y))

        rot_list = []
        for x_rot in range(0, 2100, 50):
            for y_rot in range(-300, 301, 100):
                rot_list.append((x_rot, y_rot))

        original_mesh = mesh 
        check_start_time = time.time()
        for view_idx, rot in enumerate(rot_list):
            mesh = copy.deepcopy(original_mesh)
            rot_x, rot_y = rot
        
            ### Snapshot
            image_path = os.path.join(image_dir, f'view_{view_idx:03}.png')
            depth_image_path = os.path.join(depth_dir, f'view_{view_idx:03}_depth.npy')
            capture_image(image_path, depth_image_path, mesh, rot_x, rot_y, w=config.resolution, h=config.resolution)

