import argparse
import glob
import os
import copy
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans

import open3d as o3d

import smplx
from data_info import gender_dict
from visualize import preprocess, rotate_mesh


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
#    import pdb; pdb.set_trace()

#    v1 = np.array([[1,2,3],[4,5,6],[1,1,1]])
#    v2 = np.array([[1,2,4],[4,5,7],[1,1,2]])
#    w = np.ones(3)
#    get_transform(v1, v2, w)

    mocap_dir = sorted(glob.glob('./AMASS_Dataset/CMU/01/*.npz'))

    subj_dirs_list = sorted(glob.glob(config.root + '/*'))
    mesh_dirs_list = sorted(glob.glob(config.mesh_data + '/*'))

    for idx, subj_dir in enumerate(subj_dirs_list):
        if idx != 0: continue

        for npz_idx, npz_path in enumerate(mocap_dir):
            if npz_idx != 0: continue 

            filename = os.path.basename(npz_path)[:-4]
            save_dir = './result_CMU'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            bdata = np.load(npz_path)
            num_frames = bdata['poses'].shape[0]

            for fidx, fid in enumerate(range(num_frames)):
                #if fidx != 1 and fidx != 2 and fidx != 3: continue 
                if fidx != 0: continue
                print(f'Progressing fid [{fidx} / {num_frames}] ...')

                s_time = time()
    
                subj_name = os.path.basename(subj_dir)
                gender = gender_dict[subj_name]
                gender = 'male' if gender == 'M' else 'female'
        
                orig_mesh = glob.glob(os.path.join(mesh_dirs_list[idx], '*.obj'))[0]
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
#                o3d.visualization.draw_geometries([orig_mesh])
#                import pdb; pdb.set_trace()
                
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
        
        #        test_mesh = o3d.geometry.TriangleMesh()
        #        test_mesh.vertices = o3d.utility.Vector3dVector(np.array(orig_mesh.vertices))
        #        test_mesh.triangles = o3d.utility.Vector3iVector(np.array(orig_mesh.triangles))
        #        test_mesh.triangle_uvs = orig_mesh.triangle_uvs
        #        test_mesh.texture = orig_mesh.texture
        #        import pdb; pdb.set_trace()
        #        o3d.visualization.draw_geometries([orig_mesh, test_mesh])
        
        #        body_mesh = os.path.join(subj_dir, 'output.obj')
        #        body_mesh = o3d.io.read_triangle_mesh(body_mesh)
        #        body_vs = np.array(body_mesh.vertices)
        
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
                
        #        body_pose = torch.zeros([1,63], dtype=dtype,
        #                                device=device,
        #                                requires_grad=True)
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

                # TODO:Test
                b_body_pose = 0.5*b_body_pose + 0.5*body_pose
#                b_body_pose = torch.clone(body_pose)
#                b_body_pose[:,48:51] -= 0.40
        
                # move body
                output = model(body_pose=b_body_pose,
                               betas=betas,
                               global_orient=global_orient,
                               left_hand_pose=left_hand_pose,
                               right_hand_pose=right_hand_pose)
        
                moved_body_mesh = o3d.geometry.TriangleMesh()
                tmp_vs = output.vertices.detach().cpu().numpy()[0]
                tmp_vs = tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
                moved_body_mesh.vertices = o3d.utility.Vector3dVector(tmp_vs)
                moved_body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
        
                moved_body_vs = np.array(moved_body_mesh.vertices)
        
                ### Calculate transform matrix of body.
                print('Calculate transform matrix of body.')
                K = 5
                Rs = []
                ts = []
                for vidx in range(body_vs.shape[0]):
                    vert1 = body_vs[vidx]
                    dist1 = np.sqrt(((vert1 - body_vs)**2).sum(1))
                    self_kidxs = dist1.argsort()[:K]
                    v1s = body_vs[self_kidxs]
                    v2s = moved_body_vs[self_kidxs]
        
                    w = np.ones((v1s.shape[0], 1))
                    R, t = get_transform(v1s, v2s, w)
                    Rs.append(R)
                    ts.append(t)
        
                Rs = np.array(Rs)
                ts = np.array(ts)
        
                ### Find NN body_v correspondences of each orig_v 
                print('Find NN body_v correspondences of each orig_v.')
                STDDEV = 0.001
                orig_to_body = [None for _ in range(orig_vs.shape[0])]
                moved_rp_verts = [None for _ in range(orig_vs.shape[0])]

                body_to_orig_map = [[] for _ in range(body_vs.shape[0])]

                for vidx in range(body_vs.shape[0]):
                    body_v = body_vs[vidx]
                    dist = np.sqrt(((body_v - orig_vs)**2).sum(1))
                    orig_vidx = dist.argmin()
                    orig_to_body[orig_vidx] = vidx

                    body_to_orig_map[vidx].append(orig_vidx)

                    knns = list(body_adj_list[vidx])
                    num_knns = len(knns)

                    knn_dist = np.sqrt(((body_v - body_vs[knns])**2).sum(1))
                    knn_dist_inv = 1. / knn_dist.reshape(num_knns, 1)
                    knn_weight = np.exp(-knn_dist.reshape(num_knns, 1) / STDDEV)
                    knn_weight = knn_weight / knn_weight.sum()

                    new_vert_cands = []
                    for cur_vidx in body_adj_list[vidx]:
                        cur_Rot = Rs[cur_vidx]
                        cur_transl  = ts[cur_vidx]
                        cur_vert = cur_Rot @ body_v + cur_transl
                        new_vert_cands.append(cur_vert)
                    
                    #new_vert = (np.stack(new_vert_cands) * knn_dist_inv).sum(0) / knn_dist_inv.sum()
                    new_vert = (np.stack(new_vert_cands) * knn_weight).sum(0)
                    moved_rp_verts[orig_vidx] = new_vert

                print('Process "None".')
                for vidx in range(orig_vs.shape[0]):
                    if vidx % 100 == 0:
                        print(f'vidx: [{vidx} / {orig_vs.shape[0]}] ..')

                    if orig_to_body[vidx] is None:
                        orig_v = orig_vs[vidx]
                        adjs_lv1 = list(orig_adj_list[vidx])
                        adjs = list(orig_adj_list[vidx])
                        for adj_idx1 in adjs_lv1:
                            adjs_lv2 = orig_adj_list[adj_idx1]
                            for adj_idx2 in adjs_lv2:
                                if not adj_idx2 in adjs:
                                    adjs.append(adj_idx2)

                        knns = [orig_to_body[i] for i in adjs if orig_to_body[i] is not None]
                        while len(knns) == 0:
                            _adjs = copy.deepcopy(adjs)
                            for adj_idx1 in _adjs:
                                adjs_lv2 = orig_adj_list[adj_idx1]
                                for adj_idx2 in adjs_lv2:
                                    if not adj_idx2 in adjs:
                                        adjs.append(adj_idx2)
                            knns = [orig_to_body[i] for i in adjs if orig_to_body[i] is not None]

                        body_to_orig_map[knns[0]].append(vidx)

                        num_knns = len(knns)
                        knn_dist = np.sqrt(((orig_v - orig_vs[knns])**2).sum(1))
                        knn_dist_inv = 1. / knn_dist.reshape(num_knns, 1)
                        knn_weight = np.exp(-knn_dist.reshape(num_knns, 1) / STDDEV)
                        knn_weight = knn_weight / knn_weight.sum()

                        new_vert_cands = [moved_rp_verts[i] for i in adjs if orig_to_body[i] is not None]

                        #new_vert = (np.stack(new_vert_cands) * knn_dist_inv).sum(0) / knn_dist_inv.sum()
                        new_vert = (np.stack(new_vert_cands) * knn_weight).sum(0)
                        moved_rp_verts[vidx] = new_vert


#                body_to_orig_map = [[] for _ in range(body_vs.shape[0])]
#
#                NN_K = 7
#                SPHERE_RADIUS = 0.002
#                STDDEV = 0.01
#                DOT_THRESH = 0.2
#                #moved_rp_verts = np.zeros([orig_vs.shape[0], 3])
#                moved_rp_verts = []
#                orig_to_body_list = []
#
#                #for tri_idx in range(orig_triangles.shape[0]):
#                for vidx in range(orig_vs.shape[0]):
#            
#                    vert = orig_vs[vidx]
#                    dist = np.sqrt(((vert - body_vs)**2).sum(1))
#                    knns_sorted = dist.argsort()
#        
#                    knns = []
#                    for i in range(body_vs.shape[0]):
#                        cur_body_idx = knns_sorted[i]
#                        #cur_center = body_vs[cur_body_idx]
#                        #cur_dist = np.sqrt(((cur_center - body_vs)**2).sum(1))
#                        #cur_center_knn = cur_dist.argsort()[:5]
#                        #if (np.dot(body_normals[cur_center_knn], orig_normals[vidx]) > DOT_THRESH).all():
#                        if np.dot(body_normals[cur_body_idx], orig_normals[vidx]) > DOT_THRESH:
#                            knns.append(cur_body_idx)
#                            if len(knns) == NN_K:
#                                break
#                    
#                    knns = np.array(knns)
#                    knn_dist = dist[knns]
#
#                    body_to_orig_map[knns[0]].append(vidx)
#                    orig_to_body_list.append(knns[0])
#                    
#                    if -1 in knns:
#                        vis_list = []
#                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
#                        sphere.translate(orig_vs[vidx])
#                        sphere.paint_uniform_color([1, 0, 0])
#                        vis_list.append(sphere)
#                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
#                        sphere.translate(body_vs[knns[0]])
#                        sphere.paint_uniform_color([0, 1, 0])
#                        vis_list.append(sphere)
#        #                    for point in inrange_orig_vs:
#        #                        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
#        #                        sphere.translate(point)
#        #                        sphere.paint_uniform_color([0.7, 0, 1])
#        #                        vis_list.append(sphere)
#                        o3d.visualization.draw_geometries([orig_mesh, body_mesh] + vis_list)
#                        import pdb; pdb.set_trace()
#        
#                    knn_dist_inv = 1. / knn_dist.reshape(knns.shape[0], 1)
#                    knn_weight = np.exp(-knn_dist.reshape(knns.shape[0], 1) / STDDEV)
#                    knn_weight = knn_weight / knn_weight.sum()
#        
#                    new_vert_cands= []
#                    for body_vidx in knns:
#                        #cur_Rot = Rs[body_mask][body_vidx]
#                        #cur_transl  = ts[body_mask][body_vidx]
#                        cur_Rot = Rs[body_vidx]
#                        cur_transl  = ts[body_vidx]
#                        cur_vert = cur_Rot @ vert + cur_transl
#                        new_vert_cands.append(cur_vert)
#                    
#                    #new_vert = (np.stack(new_vert_cands) * knn_dist_inv).sum(0) / knn_dist_inv.sum()
#                    new_vert = (np.stack(new_vert_cands) * knn_weight).sum(0)
#                    moved_rp_verts.append(new_vert)
#
                # Write
                moved_rp_verts = np.stack(moved_rp_verts)
                moved_rp_mesh = o3d.geometry.TriangleMesh()
                moved_rp_mesh.vertices = o3d.utility.Vector3dVector(moved_rp_verts)
                moved_rp_mesh.triangles = o3d.utility.Vector3iVector(np.array(orig_mesh.triangles))
                moved_rp_mesh.triangle_uvs = orig_mesh.triangle_uvs
                moved_rp_mesh.textures = orig_mesh.textures
        
                #moved_rp_mesh.paint_uniform_color([0,0,0.8])
                moved_body_mesh.paint_uniform_color([0, 1, 0.7])
                #o3d.visualization.draw_geometries([orig_mesh, body_mesh, moved_body_mesh, moved_rp_mesh])
        
        #        orig_mesh = glob.glob(os.path.join(mesh_dirs_list[idx], '*.obj'))[0]
        #        orig_mesh = o3d.io.read_triangle_mesh(orig_mesh)
        #        orig_mesh = preprocess(orig_mesh)
        #        orig_mesh = rotate_mesh(orig_mesh, subj_name)
        #        orig_vs = np.array(orig_mesh.vertices)
        #        o3d.io.write_triangle_mesh('test_orig.obj', orig_mesh)
        
                #o3d.io.write_triangle_mesh('test_body.obj', body_mesh)
        
                save_path = os.path.join(save_dir, f'{filename}_{fid}_de_body.obj')
                o3d.io.write_triangle_mesh(save_path, moved_body_mesh)

                save_path = os.path.join(save_dir, f'{filename}_{fid}_de_rp.obj')
                o3d.io.write_triangle_mesh(save_path, moved_rp_mesh)

                # Correspondence visualization
                cmap1 = plt.get_cmap('Set1').colors
                cmap2 = plt.get_cmap('Set2').colors
                cmap3 = plt.get_cmap('Set3').colors
                cmap = list(cmap1) + list(cmap2) + list(cmap3)

                default_body_output = model()
                default_body_mesh = o3d.geometry.TriangleMesh()
                default_tmp_vs = default_body_output.vertices.detach().cpu().numpy()[0]
                default_tmp_vs = default_tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
                default_body_mesh.vertices = o3d.utility.Vector3dVector(default_tmp_vs)
                default_body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
                default_body_vs = np.array(default_body_mesh.vertices)

                kmeans = KMeans(n_clusters=len(cmap)).fit(default_body_vs)
                body_vertex_colors = []
                cidx = 0
                for i in range(body_vs.shape[0]):
                    body_vertex_colors.append(cmap[kmeans.labels_[i]])

                test_mesh1 = o3d.geometry.TriangleMesh()
                test_mesh1.vertices = body_mesh.vertices
                test_mesh1.triangles = body_mesh.triangles
                test_mesh1.vertex_colors = o3d.utility.Vector3dVector(np.array(body_vertex_colors))

                orig_vertex_colors = np.zeros([orig_vs.shape[0], 3])
                for i in range(body_vs.shape[0]):
                    
                    origs = body_to_orig_map[i]
                    for j in origs:
                        orig_vertex_colors[j] = body_vertex_colors[i]

                test_mesh2 = o3d.geometry.TriangleMesh()
                test_mesh2.vertices = orig_mesh.vertices
                test_mesh2.triangles = orig_mesh.triangles
                test_mesh2.vertex_colors = o3d.utility.Vector3dVector(orig_vertex_colors)

                de_rp_vertex_colors = np.zeros([orig_vs.shape[0], 3])
                for i in range(body_vs.shape[0]):
                    
                    origs = body_to_orig_map[i]
                    for j in origs:
                        de_rp_vertex_colors[j] = body_vertex_colors[i]

                test_mesh3 = o3d.geometry.TriangleMesh()
                test_mesh3.vertices = moved_rp_mesh.vertices
                test_mesh3.triangles = moved_rp_mesh.triangles
                test_mesh3.vertex_colors = o3d.utility.Vector3dVector(de_rp_vertex_colors)

                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh1.obj')
                o3d.io.write_triangle_mesh(save_path, test_mesh1)
                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh2.obj')
                o3d.io.write_triangle_mesh(save_path, test_mesh2)
                save_path = os.path.join(save_dir, f'{filename}_{fid}_test_mesh3.obj')
                o3d.io.write_triangle_mesh(save_path, test_mesh3)

                #o3d.visualization.draw_geometries([test_mesh1, test_mesh2])

                e_time = time()
                print(f'elapsed time: {e_time - s_time}s')
        



        



