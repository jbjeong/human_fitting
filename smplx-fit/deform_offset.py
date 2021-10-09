import argparse
import glob
import os
from time import time

import numpy as np
import matplotlib.pyplot as plt
import torch

import open3d as o3d

import smplx
from data_info import gender_dict
from util_write import write_simple_obj 
from visualize import preprocess, rotate_mesh


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--save_root")
#    parser.add_argument("--mesh_root", 
#        default='/root/code/human_recon_make_data/images_for_test')
#    parser.add_argument("--smpl_part_label", 
#        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--model_path_male", 
        default='./models/smplx/SMPLX_MALE.npz')
    parser.add_argument("--model_path_female", 
        default='./models/smplx/SMPLX_FEMALE.npz')
    config = parser.parse_args()

    if not os.path.exists(config.save_root):
        os.makedirs(config.save_root, exist_ok=True)

    mocap_dir = sorted(glob.glob('./AMASS_Dataset/CMU/01/*.npz'))

    subj_dirs_list = sorted(glob.glob(config.root + '/*'))

    device = torch.device('cuda')

    for subj_idx, subj_dir in enumerate(subj_dirs_list):

        print(f'Progressing subj [{subj_idx} / {len(subj_dirs_list)}] ...')
        print(f'subj_name: {subj_dir}')

        disp_path = os.path.join(subj_dir, 'output_smpld_offset.npy')
        if not os.path.exists(disp_path):
            print("no 'output_smpld_offset.npy'. skipped.") 
            continue

        aux_path = os.path.join(subj_dir, 'output_aux.npz')
        aux_data = np.load(aux_path)

        subj_name = os.path.basename(subj_dir)
        save_dir = os.path.join(config.save_root, subj_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        gender = gender_dict[subj_name]
        gender = 'male' if gender == 'M' else 'female'

        dtype = torch.float32

#        rp_mesh_path = glob.glob(os.path.join(config.mesh_root, subj_name + '_OBJ') + '/*.obj')[0]
#        rp_mesh = o3d.io.read_triangle_mesh(rp_mesh_path)
#        rp_mesh = preprocess(rp_mesh)
#        rp_mesh = rotate_mesh(rp_mesh, subj_name)
#
#        th_rp_mesh = kaolin.rep.TriangleMesh.from_tensors(
#            vertices=torch.from_numpy(np.array(rp_mesh.vertices, dtype=np.float32)),
#            faces=torch.from_numpy(np.array(rp_mesh.triangles, dtype=np.int64)),
#            )
#        th_rp_mesh.cuda()


        model_path = config.model_path_male if gender=='male' else config.model_path_female

        model_params = dict(model_path=model_path,
                            model_type='smplxd',
                            #joint_mapper=joint_mapper,
                            create_global_orient=False,
                            create_body_pose=False,
                            create_betas=False,
                            create_left_hand_pose=False,
                            create_right_hand_pose=False,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
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

        body_pose = torch.tensor(aux_data['body_pose'], dtype=dtype, device=device)
        betas = torch.tensor(aux_data['betas'], dtype=dtype, device=device)
        global_orient = torch.tensor(aux_data['global_orient'], dtype=dtype, device=device)
        transl = torch.tensor(aux_data['transl'], dtype=dtype, device=device)
        scale = torch.tensor(aux_data['scale'], dtype=dtype, device=device)
        left_hand_pose = torch.tensor(aux_data['left_hand_pose'], dtype=dtype, device=device)
        right_hand_pose = torch.tensor(aux_data['right_hand_pose'], dtype=dtype, device=device)

        disp_path = os.path.join(subj_dir, 'output_smpld_offset.npy')
        disp_data = np.load(disp_path)
        input_disp = torch.tensor(disp_data,
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=True)

        for npz_idx, npz_path in enumerate(mocap_dir):
            #if npz_idx != 0: continue 

            filename = os.path.basename(npz_path)[:-4]

            bdata = np.load(npz_path)
            #bdata_keys = list(bdata.keys())
            #import pdb; pdb.set_trace()
            num_frames = bdata['poses'].shape[0]

            for fidx, fid in enumerate(range(num_frames)):
                #if fidx != 0: continue
                if fidx % 500 != 0: continue
                #if fidx == 100: break

                print(f'Progressing fid [{fidx} / {num_frames}] ...')

                b_root_orient = torch.tensor(bdata['poses'][fid, :3], dtype=dtype).to(device)
                b_body_pose = torch.tensor(bdata['poses'][fid, 3:66], dtype=dtype).to(device)
                b_left_hand_pose = torch.tensor(bdata['poses'][fid, 66:111], dtype=dtype).to(device)
                b_right_hand_pose = torch.tensor(bdata['poses'][fid, 111:], dtype=dtype).to(device)

                input_body_pose = 0.5*b_body_pose + 0.5*body_pose
                input_left_hand_pose = 0.5*b_left_hand_pose + 0.5*left_hand_pose
                input_right_hand_pose = 0.5*b_right_hand_pose + 0.5*right_hand_pose

                # Init th_init_smpl_mesh
                body_model_output = model(body_pose=input_body_pose,
                                          betas=betas,
                                          global_orient=global_orient,
                                          left_hand_pose=input_left_hand_pose,
                                          right_hand_pose=input_right_hand_pose,
                                          return_verts=True,
                                          displacements=input_disp)

                v = body_model_output.vertices.detach().cpu().numpy()
                v_tr = v * scale.cpu().numpy() + transl.cpu().numpy()
                faces = model.faces

                obj_save_path = os.path.join(save_dir, f'{filename}_{fid:0004}.obj')
                write_simple_obj(v_tr[0], faces, obj_save_path)

#                moved_body_mesh = o3d.geometry.TriangleMesh()
#                tmp_vs = output.vertices.detach().cpu().numpy()[0]
#                tmp_vs = tmp_vs * scale.detach().cpu().numpy() + transl.detach().cpu().numpy()
#                moved_body_mesh.vertices = o3d.utility.Vector3dVector(tmp_vs)
#                moved_body_mesh.triangles = o3d.utility.Vector3iVector(model.faces)
#
#                save_path = os.path.join(save_dir, f'{filename}_{fid}.obj')
#                o3d.io.write_triangle_mesh(save_path, smpl_seg_mesh)



