import argparse
import glob
import json
import os
import time
import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import cv2

import open3d as o3d

from util import preprocess


def get_Pmat(P_path):
    with open(P_path) as fp:
        data = json.load(fp)

        extrinsic = data['extrinsic']
        extrinsic = np.array(extrinsic).reshape(4, 4).transpose()
        extrinsic = extrinsic[:3, :]

        intrinsic = data['intrinsic']['intrinsic_matrix']
        intrinsic = np.array(intrinsic).reshape(3, 3).transpose()

        P = np.matmul(intrinsic, extrinsic)

    return P

def get_keypts(path, key):
    with open(path) as fp:
        data = json.load(fp)
    if len(data['people']) == 0:
        return None
    else:
        keypoints = np.array(data['people'][0][key]).reshape(-1, 3)
        return keypoints


def triangulatePoints(projMatrs, projPoints):

    if len(projPoints) == 2:
        a = np.zeros([4, 4])

        i = 0
        for j in range(2):
            #x = projPoints[j][i][0]
            #y = projPoints[j][i][1]
            x = projPoints[j][0]
            y = projPoints[j][1]
            for k in range(4):
                a[j*2+0, k] = x * projMatrs[j][2, k] - projMatrs[j][0, k]
                a[j*2+1, k] = y * projMatrs[j][2, k] - projMatrs[j][1, k]

    elif len(projPoints) == 3:
        a = np.zeros([6, 4])

        i = 0
        for j in range(3):
            #x = projPoints[j][i][0]
            #y = projPoints[j][i][1]
            x = projPoints[j][0]
            y = projPoints[j][1]
            for k in range(4):
                a[j*2+0, k] = x * projMatrs[j][2, k] - projMatrs[j][0, k]
                a[j*2+1, k] = y * projMatrs[j][2, k] - projMatrs[j][1, k]
                #a[j*3+2, k] = x * projMatrs[j][1, k] - y * projMatrs[j][0, k]

    else:
        a = np.zeros([8, 4])

        i = 0
        for j in range(4):
            x = projPoints[j][0]
            y = projPoints[j][1]
            for k in range(4):
                a[j*2+0, k] = x * projMatrs[j][2, k] - projMatrs[j][0, k]
                a[j*2+1, k] = y * projMatrs[j][2, k] - projMatrs[j][1, k]
                #a[j*3+2, k] = x * projMatrs[j][1, k] - y * projMatrs[j][0, k]
    
    for row in range(a.shape[0]):
        a[row] = a[row] / np.linalg.norm(a[row])

    u, s, vh = np.linalg.svd(a)
    points4d = vh[3]

    return points4d


def func_triangulate(args):
    view_tuple, k_idx, keypts_list, P_list = args
    
    N = len(keypts_list)

    inlier_dist_thresh = 20
    inlier_ratio_thresh = 0.45

    pose_3d = np.zeros(keypts_list[0].shape)
    min_errors = np.zeros(pose_3d.shape[0])
    min_errors[:] = np.inf

    i, j = view_tuple

    keypts1 = keypts_list[i]
    keypts2 = keypts_list[j]
    P1 = P_list[i]
    P2 = P_list[j]

    projMatrs = [P1, P2]
    projPoints = [keypts1[k_idx][:-1], keypts2[k_idx][:-1]]
    point3d = triangulatePoints(projMatrs, projPoints)

    inliers = []
    errors = 0
    for tgt_idx in range(N):
        
        keypts_tgt = keypts_list[tgt_idx]
        P_tgt = P_list[tgt_idx]

        proj_xy = P_tgt @ point3d
        proj_xy = proj_xy.reshape(-1)
        if proj_xy[2] == 0: continue
        pr_x = proj_xy[0] / proj_xy[2]
        pr_y = proj_xy[1] / proj_xy[2]
        error = np.sqrt((pr_x - keypts_tgt[k_idx][0])**2 + (pr_y - keypts_tgt[k_idx][1])**2)
        if error < inlier_dist_thresh:
            inliers.append(tgt_idx)
            errors += error

    if len(inliers) / float(N) > inlier_ratio_thresh:

        proj_xy = P1 @ point3d
        proj_xy = proj_xy.reshape(-1)
        pr_x = proj_xy[0] / proj_xy[2]
        pr_y = proj_xy[1] / proj_xy[2]

        error = np.sqrt((pr_x - keypts1[k_idx][0])**2 + (pr_y - keypts1[k_idx][1])**2)
        errors += error

        proj_xy = P2 @ point3d
        proj_xy = proj_xy.reshape(-1)
        pr_x = proj_xy[0] / proj_xy[2]
        pr_y = proj_xy[1] / proj_xy[2]

        error = np.sqrt((pr_x - keypts2[k_idx][0])**2 + (pr_y - keypts2[k_idx][1])**2)
        errors += error
        
        mean_error = errors / (len(inliers) + 2)

        if min_errors[k_idx] > mean_error:
            point3d_xyz = np.array([
                point3d[0] / point3d[3],
                point3d[1] / point3d[3],
                point3d[2] / point3d[3]]).reshape(3)
            min_errors[k_idx] = mean_error
            pose_3d[k_idx] = point3d_xyz

    return min_errors, pose_3d


def get_pose_3d_two_view_ransac(orig_keypts_list, orig_P_list, wrist_point=None,
        conf_thresh1=0.5, conf_thresh2=0.6):

    total_cpu = mp.cpu_count()
    num_cpu = min(40, total_cpu)
    pool = Pool(processes=num_cpu)

    wrist_dist_thresh = 90

    inlier_dist_thresh = 20
    inlier_ratio_thresh = 0.45
    
    pose_3d = np.zeros(orig_keypts_list[0].shape)
    min_errors = np.zeros(pose_3d.shape[0])
    min_errors[:] = np.inf

    for k_idx in range(len(orig_keypts_list[0])):

        # Select keypts that is highly confident.
        keypts_list = []
        P_list = []
        for _keypts, _P in zip(orig_keypts_list, orig_P_list):
            if _keypts[k_idx][2] > conf_thresh1:
                if wrist_point is not None:
                    hg_test_point3d = np.array([
                        wrist_point[0], wrist_point[1], wrist_point[2], 1])
                    hg_test_point2d = _P @ hg_test_point3d
                    test_point = np.array([
                        hg_test_point2d[0] / hg_test_point2d[2],
                        hg_test_point2d[1] / hg_test_point2d[2]])
                    dist = np.sqrt(((_keypts[k_idx][:2] - test_point) ** 2).sum()) 
                    if dist < wrist_dist_thresh:
                        keypts_list.append(_keypts)
                        P_list.append(_P)
                else:
                    keypts_list.append(_keypts)
                    P_list.append(_P)


        N = len(keypts_list)
        #print(N)

        if N <= 3:
            min_errors[k_idx] = np.inf 
            continue

        total_view_tuple_list = []
        for i in range(N):
            for j in range(i+1, N):
                total_view_tuple_list.append( (i, j) )

        randinds = np.arange(len(total_view_tuple_list))
        
        nom = N * (N-1)
        denom = 2
        num_samples = min(int(nom / denom), 10000)
        view_tuple_idxs = np.random.choice(randinds, size=num_samples, replace=False)
        view_tuple_list = [(total_view_tuple_list[idx], k_idx, keypts_list, P_list) for idx in view_tuple_idxs] 

        result = pool.map(func_triangulate, view_tuple_list)
        min_errors_result = [x[0][k_idx] for x in result]
        min_errors_result = np.stack(min_errors_result)

        min_errors[k_idx] = min_errors_result.min()
        min_view_idx = min_errors_result.argmin()
        pose_3d[k_idx] = result[min_view_idx][1][k_idx]

    mask = min_errors > 40
    pose_3d[mask] = np.inf

    pool.close()
    pool.join()

    return pose_3d, min_errors


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')

    config = parser.parse_args()

    subj_dir_list = sorted(glob.glob(config.dataset + '/*'))

    for subj_idx, subj_dir in enumerate(subj_dir_list):

        print(f'Progressing {subj_idx+1} / {len(subj_dir_list)} ...')
        print(subj_dir)
        time_start = time.time()

        image_path_list = sorted(glob.glob(f'{subj_dir}/images/*.png'))
        P_path_list = sorted(glob.glob(f'{subj_dir}/images/*.json'))
        keypts_path_list = sorted(glob.glob(f'{subj_dir}/keypts/*'))

        pass_list = []
        for idx, path in enumerate(keypts_path_list):
            keypts = get_keypts(path, key='pose_keypoints_2d')
            if keypts is None: 
                pass_list.append(idx) 

        P_list = [] 
        for idx, path in enumerate(P_path_list):
            if idx in pass_list:
                continue
            P_list.append(get_Pmat(path))

        body_keypts_list = []
        for idx, path in enumerate(keypts_path_list):
            if idx in pass_list:
                continue
            keypts = get_keypts(path, key='pose_keypoints_2d')
            body_keypts_list.append(keypts)

        face_keypts_list = []
        for idx, path in enumerate(keypts_path_list):
            if idx in pass_list:
                continue
            face_keypts_list.append(get_keypts(path, key='face_keypoints_2d'))

        hand_left_keypts_list = []
        for idx, path in enumerate(keypts_path_list):
            if idx in pass_list:
                continue
            hand_left_keypts_list.append(get_keypts(path, key='hand_left_keypoints_2d'))

        hand_right_keypts_list = []
        for idx, path in enumerate(keypts_path_list):
            if idx in pass_list:
                continue
            hand_right_keypts_list.append(get_keypts(path, key='hand_right_keypoints_2d'))

        s = time.time()
        body_pose_3d, min_errors_body  = get_pose_3d_two_view_ransac(
            body_keypts_list, P_list)
        e = time.time()
        print(f'elp body: {e - s:3f}')
        face_pose_3d, min_errors_face = get_pose_3d_two_view_ransac(
            face_keypts_list, P_list)
        e2 = time.time()
        print(f'elp face: {e2 - e:3f}')


        hand_left_pose_3d, min_errors_left = get_pose_3d_two_view_ransac(
            hand_left_keypts_list, P_list, wrist_point=body_pose_3d[7], conf_thresh2=0.55)
        e3 = time.time()
        print(f'elp hand left: {e3 - e2:3f}')
        hand_right_pose_3d, min_errors_right = get_pose_3d_two_view_ransac(
            hand_right_keypts_list, P_list, wrist_point=body_pose_3d[4], conf_thresh2=0.55)

        e4 = time.time()
        print(f'elp hand right: {e4 - e3:3f}')

        time_end = time.time()
        print(f'elp: {time_end - time_start:3f}')

        # Write
        save_path = os.path.join(subj_dir, 'pose3d.npz')
        np.savez(save_path, 
            body_pose_3d=body_pose_3d, 
            face_pose_3d=face_pose_3d,
            hand_left_pose_3d=hand_left_pose_3d,
            hand_right_pose_3d=hand_right_pose_3d)


