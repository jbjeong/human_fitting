import argparse
import glob
import json
import os
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def back_projection(point, camera_matrix):
    intrinsic = camera_matrix['intrinsic']
    extrinsic = camera_matrix['extrinsic']

    vt = np.matmul(extrinsic, np.array([point[0], point[1], point[2], 1]))
    u = float(vt[0] * intrinsic[0][0]) / vt[2] + intrinsic[0][2]
    v = float(vt[1] * intrinsic[1][1]) / vt[2] + intrinsic[1][2]
    d =  float(vt[2])

    u = int(round(u))
    v = int(round(v))

    return u, v, d


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    config = parser.parse_args()

    depth_diff_threshold = 0.034
    maximum_allowable_depth = 5.1 

    subj_dirs = sorted(glob.glob(os.path.join(config.dataset, '*')))
    for subj_idx, subj_dir in enumerate(subj_dirs):
        print(f'Processing [{subj_idx} / {len(subj_dirs)}] ...')
        pcd_path = os.path.join(subj_dir, 'pcd.ply')
        save_path = os.path.join(subj_dir, 'skin_label.npy')
        pcd_save_path = os.path.join(subj_dir, 'pcd_body.ply')
        if os.path.exists(pcd_save_path):
            print('Results have already existed. Skip.')
            continue

        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd_points = np.array(pcd.points)
    
        image_path_list = sorted(glob.glob(os.path.join(subj_dir, 'images/*.png')))
        depth_path_list = sorted(glob.glob(os.path.join(subj_dir, 'depth/*.npy')))
        skin_path_list = sorted(glob.glob(os.path.join(subj_dir, 'skin/*.png')))
        images = [cv2.imread(x) for x in image_path_list]
        depth_maps = [np.load(x) for x in depth_path_list]
        skin_maps = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in skin_path_list]

        camera_path_list = sorted(glob.glob(os.path.join(subj_dir, 'images/*.json')))
        cameras = []
        for camera_path in camera_path_list:
            with open(camera_path) as fp:
                data = json.load(fp)
            cameras.append({
                'extrinsic': np.array(data['extrinsic']).reshape(4,4).T[:3],
                'intrinsic': np.array(data['intrinsic']['intrinsic_matrix']).reshape([3,3]).T})

        visibility_vertex_to_imageview = [[] for _ in range(len(pcd_points))]

        assert len(images) == len(cameras) and \
            len(cameras) == len(depth_maps) and \
            len(depth_maps) == len(skin_maps)

    
        s_time = time()
        for vertex_id, point in enumerate(pcd_points):
            for view_id in range(len(images)):
                image = images[view_id]
                camera_matrix = cameras[view_id]
                depth_map = depth_maps[view_id]
                u, v, d = back_projection(point, camera_matrix)

                if d < 0.0: continue
                if not ((0 < v < image.shape[0]) and (0 < u < image.shape[1])): continue

                d_sensor = depth_map[v, u]
                if d_sensor > maximum_allowable_depth: continue
                if (image[v, u] == 255).all(): continue

                depth_diff = np.abs(d - d_sensor)
                if depth_diff >= depth_diff_threshold: continue

                visibility_vertex_to_imageview[vertex_id].append( (view_id, u, v) )

        e_time = time()
        print(f'elp: {e_time - s_time} s')

        vertex_skin_label = []
        for vertex_id in range(len(pcd_points)):
            visible_views = visibility_vertex_to_imageview[vertex_id]
            if len(visible_views) == 0:
                vertex_skin_label.append(0)
                continue

            count_skin = 0
            for view_id, _u, _v in visible_views:
                skin_map = skin_maps[view_id]
                skin_label = skin_map[_v, _u] / 255
                count_skin += skin_label

            skin_ratio = float(count_skin) / len(visible_views)
            if skin_ratio > 0.7:
                vertex_skin_label.append(1)
            else:
                vertex_skin_label.append(0)

        vertex_skin_label = np.array(vertex_skin_label, dtype=np.int64)
        np.save(save_path, vertex_skin_label)


        ### Make 'pcd_body.ply'
        points = np.array(pcd.points)
        normal_path = os.path.join(subj_dir, 'pcd_normal.npy')
        normals = np.load(normal_path)

        new_points = []
        for idx, point in enumerate(points):
            if vertex_skin_label[idx] != 1:
                new_point = point - 0.015*normals[idx]
            else:
                new_point = point
            new_points.append(new_point)

        new_points = np.array(new_points)
        new_colors = np.zeros(new_points.shape)

        _pcd = o3d.geometry.PointCloud()
        _pcd.points = o3d.utility.Vector3dVector(new_points)
        _pcd.colors = o3d.utility.Vector3dVector(new_colors)

        o3d.io.write_point_cloud(pcd_save_path, _pcd)




