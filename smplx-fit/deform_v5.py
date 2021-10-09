import argparse
import glob
import os
from time import time
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans

import open3d as o3d

import smplx
from data_info import gender_dict
from visualize import preprocess, rotate_mesh


def get_signed_volume(a, b, c, d):
    return (1.0 / 6.0) * np.dot(np.cross(b-a, c-a), d-a)


def get_transform(v1, v2, w):
    center1 = np.sum((w * v1), axis=0) / w.sum()
    center2 = np.sum((w * v2), axis=0) / w.sum()

    points1 = v1 - center1
    points2 = v2 - center2

    X = points1.T
    Y = points2.T
    W = np.diag(w.reshape(-1))
    S = np.matmul(np.matmul(X,W), Y.T)
    U, sigma, Vh = np.linalg.svd(S)
    #out = u @ np.diag(sigma) @ vh
    rect = np.diag(np.ones(U.shape[0]))
    rect[-1,-1] = np.linalg.det(Vh.T @ U.T)
    R = Vh.T @ rect @ U.T
    t = center2 - R @ center1 

    return R, t


def get_triangle_area(x1, x2, x3):
    a = x2 - x1
    b = x3 - x1
    area = (a[1]*b[2]-b[1]*a[2])**2 + (b[0]*a[2]-a[0]*b[2])**2 + (a[0]*b[1]-b[0]*a[1])**2
    area = np.sqrt(area) / 2
    return area


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--mesh_data", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--smpl_part_label", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--model_path_male", 
        default='./models/smplx/SMPLX_MALE.npz')
    parser.add_argument("--model_path_female", 
        default='./models/smplx/SMPLX_FEMALE.npz')
    config = parser.parse_args()

    #device = torch.device('cuda')
    device = torch.device('cpu')
    dtype = torch.float64

    # TODO:Test
#    x1 = np.array([0,0,1])
#    x2 = np.array([1,0,0])
#    x3 = np.array([0,1,0])
#    area = get_traingle_area(x1, x2, x3)

#    v1 = np.array([[1,2,3],[4,5,6],[1,1,1]])
#    v2 = np.array([[1,2,4],[4,5,7],[1,1,2]])
#    w = np.ones(3)
#    get_transform(v1, v2, w)

    mocap_dir = sorted(glob.glob('./AMASS_Dataset/CMU/01/*.npz'))

    subj_dirs_list = sorted(glob.glob(config.root + '/*'))
    #mesh_dirs_list = sorted(glob.glob(config.mesh_data + '/*'))

    save_root = './result'
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    for subj_idx, subj_dir in enumerate(subj_dirs_list):
        if subj_idx != 0: continue

        print(f'Progressing subj [{subj_idx} / {len(subj_dirs_list)}] ...')
        print(f'subj_name: {subj_dir}')

        subj_name = os.path.basename(subj_dir)
        save_dir = os.path.join(save_root, subj_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        for npz_idx, npz_path in enumerate(mocap_dir):
            if npz_idx != 0: continue 

            filename = os.path.basename(npz_path)[:-4]

            bdata = np.load(npz_path)
            num_frames = bdata['poses'].shape[0]

            for fidx, fid in enumerate(range(num_frames)):
                #if fidx != 1 and fidx != 2 and fidx != 3: continue 
                if fidx != 0: continue
                print(f'Progressing fid [{fidx} / {num_frames}] ...')

                s_time = time()
    
                gender = gender_dict[subj_name]
                gender = 'male' if gender == 'M' else 'female'

                #orig_part_label = np.load(os.path.join(subj_dir, 'part_label.npy'))
                smpl_part_label = np.load(config.smpl_part_label)

                # Color map
                cmap1 = plt.get_cmap('Set1').colors
                cmap2 = plt.get_cmap('Set2').colors
                cmap3 = plt.get_cmap('Set3').colors
                cmap = list(cmap1) + list(cmap2) + list(cmap3)

                # Load mesh
                orig_mesh = glob.glob(os.path.join(config.mesh_data, subj_name+'_OBJ', '*.obj'))[0]
                orig_mesh = o3d.io.read_triangle_mesh(orig_mesh)
                orig_mesh = preprocess(orig_mesh)
                orig_mesh = rotate_mesh(orig_mesh, subj_name)

                # Test connected graph - remove object
#                indices = np.where(np.array(orig_mesh.cluster_connected_triangles()[0]) == 1)[0]
#                uv_mask = np.ones([np.array(orig_mesh.triangle_uvs).shape[0]])
#                uv_mask[indices*3] = 0
#                uv_mask[indices*3+1] = 0
#                uv_mask[indices*3+2] = 0
#                uv_mask = uv_mask.astype('bool')
#                orig_triangle_uvs = np.array(orig_mesh.triangle_uvs)
#                orig_triangle_uvs = orig_triangle_uvs[uv_mask]
#                orig_mesh.remove_triangles_by_index(indices)
#                orig_mesh.remove_unreferenced_vertices()
#                orig_mesh.compute_vertex_normals()
#                orig_mesh.triangle_uvs = o3d.utility.Vector2dVector(orig_triangle_uvs)
#
                
                save_mesh = o3d.geometry.TriangleMesh()
                save_mesh.vertices = orig_mesh.vertices
                save_mesh.triangles = orig_mesh.triangles
                save_mesh.triangle_uvs = orig_mesh.triangle_uvs
                save_mesh.textures = orig_mesh.textures
                
                save_path = os.path.join(save_dir, f'{filename}_{fid}_orig.obj')
                o3d.io.write_triangle_mesh(save_path, save_mesh)
        
                orig_vs = np.array(orig_mesh.vertices)
                orig_triangles = np.array(orig_mesh.triangles)
                orig_normals = np.array(orig_mesh.vertex_normals) 
                orig_mesh.compute_adjacency_list()
                orig_adj_list = orig_mesh.adjacency_list
        
                # Move body
                model_path = config.model_path_male if gender=='male' else config.model_path_female
                model_params = dict(model_path=model_path,
                                    model_type='smplx',
                                    #joint_mapper=joint_mapper,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    use_pca=False,
                                    dtype=dtype,
                                    )
        
                model = smplx.create(gender=gender, **model_params)
                if gender == 'male':
                    # left elbow
                    model.J_regressor[18][4284] += 0.6
                    model.J_regressor[18][4347] += 0.6
                    model.J_regressor[18] = model.J_regressor[18] / model.J_regressor[18].sum()
                elif gender == 'female':
                    # left elbow
                    model.J_regressor[18][4288] += 0.2
                    model.J_regressor[18][4301] += 0.2
                    model.J_regressor[18] = model.J_regressor[18] / model.J_regressor[18].sum()
                else:
                    raise ValueError('gender is not found!')

                model.to(device)
                
                aux_path = os.path.join(subj_dir, 'output_aux.npz')
                aux_data = np.load(aux_path)
                body_pose = torch.tensor(aux_data['body_pose'], dtype=dtype, device=device)
                betas = torch.tensor(aux_data['betas'], dtype=dtype, device=device)
                global_orient = torch.tensor(aux_data['global_orient'], dtype=dtype, device=device)
                transl = torch.tensor(aux_data['transl'], dtype=dtype, device=device)
                scale = torch.tensor(aux_data['scale'], dtype=dtype, device=device)
                left_hand_pose = torch.tensor(aux_data['left_hand_pose'], dtype=dtype, device=device)
                right_hand_pose = torch.tensor(aux_data['right_hand_pose'], dtype=dtype, device=device)
        
                # body
                output = model(body_pose=body_pose,
                               betas=betas,
                               global_orient=global_orient,
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose)
        
                body_mesh = o3d.geometry.TriangleMesh()
                tmp_vs = output.vertices.detach().cpu().numpy()[0]
                tmp_vs = tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
                body_mesh.vertices = o3d.utility.Vector3dVector(tmp_vs)
                body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
                body_mesh.paint_uniform_color([1, 0.7, 0])
    
                save_path = os.path.join(save_dir, f'{filename}_{fid}_body.obj')
                o3d.io.write_triangle_mesh(save_path, body_mesh)
        
                body_mesh.compute_vertex_normals()
        
                body_vs = np.array(body_mesh.vertices)
                body_normals = np.array(body_mesh.vertex_normals)
                body_mesh.compute_adjacency_list()
                body_adj_list = body_mesh.adjacency_list

                # Move body
                b_root_orient = torch.tensor(bdata['poses'][fid:fid+1, :3], dtype=dtype).to(device)
                b_body_pose = torch.tensor(bdata['poses'][fid:fid+1, 3:66], dtype=dtype).to(device)
                b_left_hand_pose = torch.tensor(bdata['poses'][fid:fid+1, 66:111], dtype=dtype).to(device)
                b_right_hand_pose = torch.tensor(bdata['poses'][fid:fid+1, 111:], dtype=dtype).to(device)


                #b_body_pose = 0.5*b_body_pose + 0.5*body_pose

                # TODO:Test
                #b_body_pose = torch.clone(body_pose)
                #b_body_pose[:,48:51] -= 0.40
                b_body_pose[:] = 0
                b_left_hand_pose[:] = 0
                b_right_hand_pose[:] = 0
        
                # moved body
                output = model(body_pose=b_body_pose,
                               betas=betas,
                               global_orient=global_orient,
                               left_hand_pose=b_left_hand_pose,
                               right_hand_pose=b_right_hand_pose)
        
                moved_body_mesh = o3d.geometry.TriangleMesh()
                tmp_vs = output.vertices.detach().cpu().numpy()[0]
                tmp_vs = tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
                moved_body_mesh.vertices = o3d.utility.Vector3dVector(tmp_vs)
                moved_body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)

                moved_body_mesh.compute_vertex_normals()
                moved_body_normals = np.array(moved_body_mesh.vertex_normals)
        
                moved_body_vs = np.array(moved_body_mesh.vertices)

#                # Visualization
#                vis_list = []
#
#                test_mesh1 = o3d.geometry.TriangleMesh()
#                test_mesh1.vertices = orig_mesh.vertices
#                test_mesh1.triangles = orig_mesh.triangles
#                #test_mesh1.vertex_colors = o3d.utility.Vector3dVector(np.array(body_vertex_colors))
#                vis_list.append(test_mesh1)
#
#                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
#                sphere.translate(orig_vs[2468])
#                sphere.paint_uniform_color([1, 0, 0.7])
#                vis_list.append(sphere)
#                o3d.visualization.draw_geometries(vis_list)

                # 3D renderpeople part segmentation
                # Test for seg_before
#                part_seg_colors = []
#                for i in range(orig_vs.shape[0]):
#                    color = cmap[orig_part_label[i]]
#                    part_seg_colors.append(color)
#
#                seg_mesh = o3d.geometry.TriangleMesh()
#                seg_mesh.vertices = orig_mesh.vertices
#                seg_mesh.triangles = orig_mesh.triangles
#                seg_mesh.vertex_colors = o3d.utility.Vector3dVector(part_seg_colors)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_seg_before.obj')
#                o3d.io.write_triangle_mesh(save_path, seg_mesh)

#                label_clusters = [[] for _ in range(15)]
#                for vidx in range(orig_vs.shape[0]):
#                    label_clusters[orig_part_label[vidx]].append(vidx)
#
#                for i in range(15):
#                    label_vertices = label_clusters[i]
#                    
#                    groups = []
#                    while len(label_vertices) > 0:
#                        root = label_vertices[0]
#                        
#                        visited = []
#                        queue = deque([root])
#                        while queue:
#                            node = queue.popleft()
#                            if node not in visited:
#                                visited.append(node)
#                                cur_adjs = orig_adj_list[node] - set(visited)
#                                cluster_adjs = []
#                                for tmp_adj_vidx in cur_adjs:
#                                    if tmp_adj_vidx in label_vertices:
#                                        cluster_adjs.append(tmp_adj_vidx)
#                                queue += cluster_adjs
#                        
#                        groups.append(visited)
#
#                        for tmp_vidx in visited:
#                            label_vertices.remove(tmp_vidx)
#                    
#                    max_group_idx = None
#                    max_num_group_vtx = 0
#                    for g_idx, group in enumerate(groups):
#                        if max_num_group_vtx < len(group):
#                            max_group_idx = g_idx
#                            max_num_group_vtx = len(group)
#
#                    for g_idx, group in enumerate(groups):
#                        if g_idx == max_group_idx:
#                            continue
#                        for vidx in group:
#                            orig_part_label[vidx] = 0 # background
#                
#                # Assign background_label to part_label
#                print(f'Assigning part_label to background_label')
#                for vidx in range(orig_vs.shape[0]):
#                    #print(f'vidx: {vidx}')
#                    if orig_part_label[vidx] == 0:
#                        root = vidx
#                        
#                        visited = []
#                        this_label = None
#                        queue = deque([root])
#                        is_finished = False
#                        while queue and not is_finished:
#                            node = queue.popleft()
#                            if node not in visited:
#                                visited.append(node)
#                                cur_adjs = orig_adj_list[node] - set(visited)
#                                for tmp_adj_vidx in cur_adjs:
#                                    if orig_part_label[tmp_adj_vidx] != 0:
#                                        this_label = orig_part_label[tmp_adj_vidx]
#                                        is_finished = True
#                                        break
#                                queue += cur_adjs 
#
#                        if this_label is None:
#                            this_label = 0
#
#                        orig_part_label[vidx] = this_label
#
#                # 3D renderpeople part segmentation
#                part_seg_colors = []
#                for i in range(orig_vs.shape[0]):
#                    color = cmap[orig_part_label[i]]
##                    if orig_part_label[i] == 10:
##                        color = np.array([0, 0, 0])
#                        
#                    part_seg_colors.append(color)
#
#                arm_labels = [3,4,5,6]
#                contact_labels = np.zeros(orig_vs.shape[0])
#                for vidx in range(orig_vs.shape[0]):
#                    cur_adjs = list(orig_adj_list[vidx])
#                    for adj_vidx in cur_adjs:
#                        if (orig_part_label[adj_vidx] == 2 and \
#                                orig_part_label[vidx] in arm_labels) or \
#                                (orig_part_label[adj_vidx] in arm_labels and \
#                                orig_part_label[vidx] == 2):
#                            contact_labels[vidx] = 1
#
#                new_contact_labels = np.zeros(orig_vs.shape[0])
#                for vidx in range(orig_vs.shape[0]):
#                    if contact_labels[vidx] == 1:
#                        vert = orig_vs[vidx]
#                        dist = np.sqrt(((vert - orig_vs)**2).sum(1))
#                        mask = dist < 0.005
#                        new_contact_labels[mask] = 1
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_contact_labels.npy')
#                np.save(save_path, np.array(new_contact_labels))
#                import pdb; pdb.set_trace()
#
#                for vidx, v_contact_label in enumerate(new_contact_labels):
#                
#                    if v_contact_label == 1:
#                        part_seg_colors[vidx] = np.array([0, 0, 0])
#
#                seg_mesh = o3d.geometry.TriangleMesh()
#                seg_mesh.vertices = orig_mesh.vertices
#                seg_mesh.triangles = orig_mesh.triangles
#                seg_mesh.vertex_colors = o3d.utility.Vector3dVector(part_seg_colors)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_seg.obj')
#                o3d.io.write_triangle_mesh(save_path, seg_mesh)


                # 3D SMPL part segmentation
#                print(f'Assigning smpl part_label to background_label')
#                label_clusters = [[] for _ in range(15)]
#                for vidx in range(body_vs.shape[0]):
#                    label_clusters[smpl_part_label[vidx]].append(vidx)
#
#                for i in range(15):
#                    label_vertices = label_clusters[i]
#                    
#                    groups = []
#                    while len(label_vertices) > 0:
#                        root = label_vertices[0]
#                        
#                        visited = []
#                        queue = deque([root])
#                        while queue:
#                            node = queue.popleft()
#                            if node not in visited:
#                                visited.append(node)
#                                cur_adjs = body_adj_list[node] - set(visited)
#                                cluster_adjs = []
#                                for tmp_adj_vidx in cur_adjs:
#                                    if tmp_adj_vidx in label_vertices:
#                                        cluster_adjs.append(tmp_adj_vidx)
#                                queue += cluster_adjs
#                        
#                        groups.append(visited)
#
#                        for tmp_vidx in visited:
#                            label_vertices.remove(tmp_vidx)
#                    
#                    max_group_idx = None
#                    max_num_group_vtx = 0
#                    for g_idx, group in enumerate(groups):
#                        if max_num_group_vtx < len(group):
#                            max_group_idx = g_idx
#                            max_num_group_vtx = len(group)
#
#                    for g_idx, group in enumerate(groups):
#                        if g_idx == max_group_idx:
#                            continue
#                        for vidx in group:
#                            smpl_part_label[vidx] = 0 # background
                
#                # Assign background_label to part_label
#                print(f'Propagate smpl part_label. ')
#                for vidx in range(body_vs.shape[0]):
#                    #print(f'vidx: {vidx}')
#                    if smpl_part_label[vidx] == 0:
#                        root = vidx
#                        
#                        visited = []
#                        this_label = None
#                        queue = deque([root])
#                        is_finished = False
#                        while queue and not is_finished:
#                            node = queue.popleft()
#                            if node not in visited:
#                                visited.append(node)
#                                cur_adjs = body_adj_list[node] - set(visited)
#                                for tmp_adj_vidx in cur_adjs:
#                                    if smpl_part_label[tmp_adj_vidx] != 0:
#                                        this_label = smpl_part_label[tmp_adj_vidx]
#                                        is_finished = True
#                                        break
#                                queue += cur_adjs 
#
#                        if this_label is None:
#                            this_label = 0
#
#                        smpl_part_label[vidx] = this_label
#
#
#                part_seg_colors = []
#                for i in range(body_vs.shape[0]):
#                    color = cmap[smpl_part_label[i]]
#                    part_seg_colors.append(color)
#
#                smpl_seg_mesh = o3d.geometry.TriangleMesh()
#                smpl_seg_mesh.vertices = body_mesh.vertices
#                smpl_seg_mesh.triangles = body_mesh.triangles
#                smpl_seg_mesh.vertex_colors = o3d.utility.Vector3dVector(part_seg_colors)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_seg_smpl.obj')
#                o3d.io.write_triangle_mesh(save_path, smpl_seg_mesh)


#                # Find transformation 
#                print('Finding smpl transformation ...')
#                K = 3
#                Rs = []
#                ts = []
#                for vidx in range(body_vs.shape[0]):
#                    vert1 = body_vs[vidx]
#                    dist1 = np.sqrt(((vert1 - body_vs)**2).sum(1))
#                    if smpl_part_label[vidx] == 0:
#                        part_mask = np.zeros(*smpl_part_label.shape).astype(np.bool)
#                    else:
#                        part_mask = (smpl_part_label != smpl_part_label[vidx])
#                    dist1[part_mask] = np.inf
#                    self_kidxs = dist1.argsort()[:K]
#                    v1s = body_vs[self_kidxs]
#                    v2s = moved_body_vs[self_kidxs]
#        
#                    w = np.ones((v1s.shape[0], 1))
#                    R, t = get_transform(v1s, v2s, w)
#                    Rs.append(R)
#                    ts.append(t)
#        
#                Rs = np.array(Rs)
#                ts = np.array(ts)
        
                # Find NN body_v correspondences of each orig_v 
                print('Finding NN body_v correspondences of each orig_v ...')
                body_to_orig_map = [[] for _ in range(body_vs.shape[0])]

                NN_K = 5
                SPHERE_RADIUS = 0.002
                STDDEV = 0.01
                DOT_THRESH = 0.2
                #moved_rp_verts = np.zeros([orig_vs.shape[0], 3])
                moved_rp_verts = []
                orig_to_body_list = []

                d_field_list = [] # [N]

                orig_vtx_to_tri_map = [[] for _ in range(orig_vs.shape[0])]
                for triangle in orig_triangles:
                    orig_vtx_to_tri_map[triangle[0]].append(triangle)
                    orig_vtx_to_tri_map[triangle[1]].append(triangle)
                    orig_vtx_to_tri_map[triangle[2]].append(triangle)

                TRI_DIST_THRESH = 0.02
                for vidx in range(body_vs.shape[0]):
                    if vidx % 100 == 0:
                        print(f'vidx: [{vidx} / {body_vs.shape[0]}] ...')
                    vert = body_vs[vidx]

                    st = time()
                    search_triangle_list = []
                    dist = np.sqrt(((vert - orig_vs)**2).sum(1))
                    in_orig_vidxs = np.where(dist < TRI_DIST_THRESH)[0]
                    
                    for orig_vidx in in_orig_vidxs:
                        for triangle in orig_vtx_to_tri_map[orig_vidx]:
                            search_triangle_list.append(triangle)
#                    for triangle in orig_triangles:
#                        p1 = orig_vs[triangle[0]]
#                        p2 = orig_vs[triangle[1]]
#                        p3 = orig_vs[triangle[2]]
#                        dist1 = np.sqrt(((vert - p1)**2).sum())
#                        dist2 = np.sqrt(((vert - p2)**2).sum())
#                        dist3 = np.sqrt(((vert - p3)**2).sum())
#
#                        if dist1 < TRI_DIST_THRESH or dist2 < TRI_DIST_THRESH or dist3 < TRI_DIST_THRESH:
#                            search_triangle_list.append(triangle)
                    et = time()
                    #print(f'search elp: {et - st} s')

                    st = time()
                    v_normal = body_normals[vidx]
                    line_a = vert
                    line_b = vert + v_normal
                    intersected_tris = []
                    for triangle in search_triangle_list:
                        tri_a = orig_vs[triangle[0]]
                        tri_b = orig_vs[triangle[1]]
                        tri_c = orig_vs[triangle[2]]
                        if ((get_signed_volume(line_a, tri_a, tri_b, tri_c) * \
                             get_signed_volume(line_b, tri_a, tri_b, tri_c)) < 0):

                            sign1 = get_signed_volume(line_a, line_b, tri_a, tri_b) > 0
                            sign2 = get_signed_volume(line_a, line_b, tri_b, tri_c) > 0
                            sign3 = get_signed_volume(line_a, line_b, tri_c, tri_a) > 0
                            if (sign1 == sign2) and (sign2 == sign3):
                                intersected_tris.append(triangle)

                    et = time()
                    #print(f'intersected elp: {et - st} s')

                    st = time()
                    nearest_point = None
                    min_dist = np.inf
                    for triangle in intersected_tris:
                        q1 = line_a
                        q2 = line_b
                        p1 = orig_vs[triangle[0]]
                        p2 = orig_vs[triangle[1]]
                        p3 = orig_vs[triangle[2]]
                        _N = np.cross(p2-p1, p3-p1)
                        _t = -np.dot(q1-p1, _N) / np.dot(q2-q1, _N)
                        cur_point = q1 + _t*(q2-q1)
                        cur_dist = np.sqrt(((vert - cur_point)**2).sum())
                        if cur_dist < min_dist:
                            min_dist = cur_dist
                            nearest_point = cur_point

                    et = time()
                    #print(f'nearest elp: {et - st} s')

                    if nearest_point is None:
                        dist = np.sqrt(((vert - orig_vs)**2).sum(1))
                        nn_orig_vidx = dist.argmin()
                        d_field = orig_vs[nn_orig_vidx] - vert
                    else:
                        d_field = nearest_point - vert
                    d_field_list.append( d_field )
                    
                hand_labels = [7, 8]
                moved_rp_vertices = []
                for vidx in range(moved_body_vs.shape[0]):
                    vtx = moved_body_vs[vidx]
                    if smpl_part_label[vidx] in hand_labels:
                        d_field = 0
                    else:
                        d_field = d_field_list[vidx]
                    moved_rp_vertices.append( vtx + d_field )

                tmp_mesh = o3d.geometry.TriangleMesh()
                tmp_mesh.vertices = o3d.utility.Vector3dVector(np.array(moved_rp_vertices))
                tmp_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
                tmp_mesh.paint_uniform_color([1, 0.7, 0])

                save_path = os.path.join(save_dir, f'{filename}_{fid}_zero_rp.obj')
                o3d.io.write_triangle_mesh(save_path, tmp_mesh)

                #o3d.visualization.draw_geometries([tmp_mesh])


#                # Write
#                print('Writing ...')
#                moved_rp_verts = np.stack(moved_rp_verts)
#                moved_rp_mesh = o3d.geometry.TriangleMesh()
#                moved_rp_mesh.vertices = o3d.utility.Vector3dVector(moved_rp_verts)
#                moved_rp_mesh.triangles = o3d.utility.Vector3iVector(np.array(orig_mesh.triangles))
#                moved_rp_mesh.triangle_uvs = orig_mesh.triangle_uvs
#                moved_rp_mesh.textures = orig_mesh.textures
#        
#                #moved_rp_mesh.paint_uniform_color([0,0,0.8])
#                moved_body_mesh.paint_uniform_color([0, 1, 0.7])
#                #o3d.visualization.draw_geometries([orig_mesh, body_mesh, moved_body_mesh, moved_rp_mesh])
#        
#        #        orig_mesh = glob.glob(os.path.join(mesh_dirs_list[idx], '*.obj'))[0]
#        #        orig_mesh = o3d.io.read_triangle_mesh(orig_mesh)
#        #        orig_mesh = preprocess(orig_mesh)
#        #        orig_mesh = rotate_mesh(orig_mesh, subj_name)
#        #        orig_vs = np.array(orig_mesh.vertices)
#        #        o3d.io.write_triangle_mesh('test_orig.obj', orig_mesh)
#        
#                #o3d.io.write_triangle_mesh('test_body.obj', body_mesh)
#        
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_de_body.obj')
#                o3d.io.write_triangle_mesh(save_path, moved_body_mesh)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_de_rp.obj')
#                o3d.io.write_triangle_mesh(save_path, moved_rp_mesh)
#
#
#                # Correspondence visualization
#                default_body_output = model()
#                default_body_mesh = o3d.geometry.TriangleMesh()
#                default_tmp_vs = default_body_output.vertices.detach().cpu().numpy()[0]
#                default_tmp_vs = default_tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
#                default_body_mesh.vertices = o3d.utility.Vector3dVector(default_tmp_vs)
#                default_body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
#                default_body_vs = np.array(default_body_mesh.vertices)
#
#                kmeans = KMeans(n_clusters=len(cmap)).fit(default_body_vs)
#                body_vertex_colors = []
#                cidx = 0
#                for i in range(body_vs.shape[0]):
#                    body_vertex_colors.append(cmap[kmeans.labels_[i]])
#
#                test_mesh1 = o3d.geometry.TriangleMesh()
#                test_mesh1.vertices = body_mesh.vertices
#                test_mesh1.triangles = body_mesh.triangles
#                test_mesh1.vertex_colors = o3d.utility.Vector3dVector(np.array(body_vertex_colors))
#
#                orig_vertex_colors = np.zeros([orig_vs.shape[0], 3])
#                for i in range(body_vs.shape[0]):
#                    
#                    origs = body_to_orig_map[i]
#                    for j in origs:
#                        orig_vertex_colors[j] = body_vertex_colors[i]
#
#                test_mesh2 = o3d.geometry.TriangleMesh()
#                test_mesh2.vertices = orig_mesh.vertices
#                test_mesh2.triangles = orig_mesh.triangles
#                test_mesh2.vertex_colors = o3d.utility.Vector3dVector(orig_vertex_colors)
#
#                de_rp_vertex_colors = np.zeros([orig_vs.shape[0], 3])
#                for i in range(body_vs.shape[0]):
#                    
#                    origs = body_to_orig_map[i]
#                    for j in origs:
#                        de_rp_vertex_colors[j] = body_vertex_colors[i]
#
#                test_mesh3 = o3d.geometry.TriangleMesh()
#                test_mesh3.vertices = moved_rp_mesh.vertices
#                test_mesh3.triangles = moved_rp_mesh.triangles
#                test_mesh3.vertex_colors = o3d.utility.Vector3dVector(de_rp_vertex_colors)
#
#                test_mesh4 = o3d.geometry.TriangleMesh()
#                test_mesh4.vertices = moved_rp_mesh.vertices
#                remove_list = []
#                for tri_idx, tri_vtx_idxs in enumerate(moved_rp_mesh.triangles):
#                    orig_tri_vtx_idxs = orig_mesh.triangles[tri_idx]
#                    orig_v0, orig_v1, orig_v2 = orig_tri_vtx_idxs
#                    defo_v0, defo_v1, defo_v2 = tri_vtx_idxs
#                    orig_line0 = np.sqrt(((orig_vs[orig_v0] - orig_vs[orig_v1]) ** 2).sum())
#                    defo_line0 = np.sqrt(((moved_rp_verts[defo_v0] - moved_rp_verts[defo_v1]) ** 2).sum())
#
#                    orig_line1 = np.sqrt(((orig_vs[orig_v0] - orig_vs[orig_v2]) ** 2).sum())
#                    defo_line1 = np.sqrt(((moved_rp_verts[defo_v0] - moved_rp_verts[defo_v2]) ** 2).sum())
#
#                    orig_line2 = np.sqrt(((orig_vs[orig_v1] - orig_vs[orig_v2]) ** 2).sum())
#                    defo_line2 = np.sqrt(((moved_rp_verts[defo_v1] - moved_rp_verts[defo_v2]) ** 2).sum())
#
#                    if 5*orig_line0 < defo_line0 or 5*orig_line1 < defo_line1 or 5*orig_line2 < defo_line2:
#                        remove_list.append(tri_idx)
#                
#                
#                new_triangles = []
#                for tri_idx, tri_vtx_idxs in enumerate(moved_rp_mesh.triangles):
#                    if tri_idx in remove_list: continue
#
#                    new_triangles.append(tri_vtx_idxs)
#
#                test_mesh4.triangles = o3d.utility.Vector3iVector(new_triangles)
#                test_mesh4.vertex_colors = o3d.utility.Vector3dVector(de_rp_vertex_colors)
#
#                final_mesh = o3d.geometry.TriangleMesh()
#                final_mesh.vertices = moved_rp_mesh.vertices
#                
#                new_triangles = []
#                for tri_idx, tri_vtx_idxs in enumerate(moved_rp_mesh.triangles):
#                    #if tri_idx in remove_list: continue
#                    is_continue = False
#                    for cur_vidx in tri_vtx_idxs:
#                        if new_contact_labels[cur_vidx] == 1:
#                            is_continue = True
#                            break
#                    if is_continue: continue
#                    new_triangles.append(tri_vtx_idxs)
#
#                new_uvs = []
#                for uv_idx, uv in enumerate(moved_rp_mesh.triangle_uvs):
#                    #if int(uv_idx / 3) in remove_list: continue
#                    is_continue = False
#                    for cur_vidx in orig_triangles[int(uv_idx / 3)]:
#                        if new_contact_labels[cur_vidx] == 1:
#                            is_continue = True
#                            break
#                    if is_continue: continue
#                    new_uvs.append(uv)
#
#                final_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
#                final_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.stack(new_uvs))
#                final_mesh.textures = orig_mesh.textures
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_final_mesh.obj')
#                o3d.io.write_triangle_mesh(save_path, final_mesh)
#
#                new_contact_labels
#
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh1.obj')
#                o3d.io.write_triangle_mesh(save_path, test_mesh1)
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh2.obj')
#                o3d.io.write_triangle_mesh(save_path, test_mesh2)
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh3.obj')
#                o3d.io.write_triangle_mesh(save_path, test_mesh3)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh4.obj')
#                o3d.io.write_triangle_mesh(save_path, test_mesh4)
#
#                #o3d.visualization.draw_geometries([test_mesh1, test_mesh2])

                e_time = time()
                print(f'elapsed time: {e_time - s_time}s')
        



        



