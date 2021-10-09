import argparse
import glob
import os
from time import time

import torch

import numpy as np
import open3d as o3d

import smplx
from utils_smplify import JointMapper, smpl_to_openpose, rel_change, GMoF
from util_write import save_obj 
from data_info import gender_dict
from fitting import create_optimizer, create_fitting_closure, run_fitting


def main(model, pose_3d, pcd, save_dir, device, dtype,
         input_params=None):

    #print(male_model)
#    orig_body_pose = torch.clone(male_model.body_pose.detach())
#    body_pose[:,54:57] += 0.11
#    body_pose[:,48:51] += 0.61
#    body_pose[:,57:60] += 0.11
#    orig_body_pose[:,51:54] += 0.61
#    output = male_model(body_pose=orig_body_pose)
#    output = male_model()
#
#    orig_mesh = o3d.geometry.TriangleMesh()
#    vertices = output.vertices.detach().cpu().numpy()
#    faces = male_model.faces
#    orig_mesh.vertices = o3d.utility.Vector3dVector(vertices[0])
#    orig_mesh.triangles = o3d.utility.Vector3iVector(faces)
#    orig_mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([orig_mesh])

#    rest_output = male_model()

    gt_body_pose_3d = pose_3d['body_pose_3d']
    gt_body_pose_3d = torch.tensor(gt_body_pose_3d, device=device).unsqueeze(0)

    gt_face_pose_3d = pose_3d['face_pose_3d']
    gt_face_pose_3d = torch.tensor(gt_face_pose_3d, device=device).unsqueeze(0)

    gt_left_hand_pose_3d = pose_3d['hand_left_pose_3d']
    gt_left_hand_pose_3d = torch.tensor(gt_left_hand_pose_3d, device=device).unsqueeze(0)

    gt_right_hand_pose_3d = pose_3d['hand_right_pose_3d']
    gt_right_hand_pose_3d = torch.tensor(gt_right_hand_pose_3d, device=device).unsqueeze(0)

    gt_list = [
        gt_body_pose_3d,
        gt_face_pose_3d,
        gt_left_hand_pose_3d,
        gt_right_hand_pose_3d,
    ]

    # Params
    if input_params is None:
        input_body_pose = torch.zeros([1, 63],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_left_hand_pose = torch.zeros([1, 45],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_right_hand_pose = torch.zeros([1, 45],
                                      dtype=dtype,
                                      device=device,
                                      requires_grad=True)
        input_betas = torch.zeros([1, 10],
                                  dtype=dtype,
                                  device=device,
                                  requires_grad=True)
        input_global_orient = torch.zeros([1, 3],
                                          dtype=dtype,
                                          device=device,
                                          requires_grad=True)
        input_transl = torch.zeros([1, 3],
                                   dtype=dtype,
                                   device=device,
                                   requires_grad=True)
        input_scale = torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        input_params = [
            input_body_pose,
            input_betas,
            input_global_orient,
            input_transl,
            input_scale,
            input_left_hand_pose,
            input_right_hand_pose
        ]
    else:
        input_betas = input_params[1]

    closure_max_iters = 30
    ftol = 1e-9
    gtol = 1e-9
    #robustifier = GMoF(rho=100)

    s_time = time()

    print('step0: start rigid fitting ...')
    optimizer, params = create_optimizer(input_params, step='orient+transl')
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step='orient+transl')

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step0.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step0: finsihed ...\n')

    print('step1: start rigid fitting with scale ...')

    with torch.no_grad():
        input_betas[0][0] = 0

    optimizer, params = create_optimizer(input_params, step='orient+transl+scale')
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step='orient+transl+scale')

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step1.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step1: finished ...\n')


    print('step2: start fitting non-rigid shoulder ...')
    step = 'shoulder'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step2.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step2: finished ...\n')

    #print('body_pose:', input_body_pose[0][36:42])
    #print('betas:', input_betas)

    print('step3: start fitting non-rigid elbow ...')
    step = 'elbow-L'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    step = 'elbow-R'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step3.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step3: finished ...\n')


    #print('body_pose:', input_body_pose[0][45:51])
    #print('betas:', input_betas)

    print('step4: start fitting non-rigid wrist ...')
    step = 'wrist'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
    save_path = os.path.join(save_dir, './result_step_wrist.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step4: finished ...\n')

    #print('body_pose:', input_body_pose[0][51:57])
    #print('betas:', input_betas)

    print('step-fingers: start fitting non-rigid finger ...')
    step = 'fingers'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step, pcd=pcd)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step_fingers.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step-fingers: finished ...\n')


#    print('step_local_arm_pose: start fitting non-rigid local arm pose...')
#    step = 'local_arm_pose'
#    optimizer, params = create_optimizer(input_params, step=step)
#    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
#                                     step=step, pcd=pcd)
#
#    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
#    save_path = os.path.join(save_dir, './result_step_local_arm_pose.obj')
#    with torch.no_grad():
#        save_obj(save_path, model, input_params)
#    print('step_local_arm_pose: finished ...\n')

    #print('input_betas:', input_betas)

    print('step5: start fitting non-rigid knee ...')
    step = 'knee'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step5.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step5: finished ...\n')


    print('step6: start fitting non-rigid ankle ...')
    step = 'ankle'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step6.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step6: finished ...\n')

    print('step7: start fitting non-rigid toe ...')
    step = 'toe'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step7.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step7: finished ...\n')


#    print('step_local_leg_pose: start fitting non-rigid local leg pose...')
#    step = 'local_leg_pose'
#    optimizer, params = create_optimizer(input_params, step=step)
#    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
#                                     step=step, pcd=pcd)
#
#    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
#    save_path = os.path.join(save_dir, './result_step_local_leg_pose.obj')
#    with torch.no_grad():
#        save_obj(save_path, model, input_params)
#    print('step_local_leg_pose: finished ...\n')


    print('step_face: start fitting non-rigid face...')
    step = 'face'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step_face.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step_face: finished ...\n')

    print('step_global_pose: start fitting global pose ...')
    step = 'global_pose'
    optimizer, params = create_optimizer(input_params, step=step)
    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
                                     step=step, pcd=pcd)

    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)

    save_path = os.path.join(save_dir, './result_step_global_pose.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)
    print('step_global_pose: finished ...\n')


#    print('step8-1: start fitting body betas ...')
#    step = 'betas'
#    optimizer, params = create_optimizer(input_params, step=step)
#    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
#                                     step=step, pcd=pcd)
#
#    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
#
#    save_path = os.path.join(save_dir, './result_step8-1.obj')
#    with torch.no_grad():
#        save_obj(save_path, model, input_params)
#    print('step8-1: finished ...\n')


#    print('step11: start fitting non-rigid finger with betas ...')
#
#    step = 'fingers+betas'
#    optimizer, params = create_optimizer(input_params, step=step)
#    closure = create_fitting_closure(optimizer, model, input_params, gt_list, 
#                                     step=step, pcd=pcd)
#
#    loss = run_fitting(optimizer, closure, params, model, closure_max_iters, ftol, gtol)
#
#    save_path = os.path.join(save_dir, './result_step11.obj')
#    with torch.no_grad():
#        save_obj(save_path, model, input_params)
#    print('step11: finished ...\n')

    e_time = time()
    print(f'elp: {e_time - s_time:3f}s')

    save_path = os.path.join(save_dir, 'output.obj')
    with torch.no_grad():
        save_obj(save_path, model, input_params)

    aux_path = os.path.join(subj_dir, 'output_aux.npz')
    np.savez(aux_path, 
             body_pose=input_body_pose.detach().cpu().numpy(),
             left_hand_pose=input_left_hand_pose.detach().cpu().numpy(),
             right_hand_pose=input_right_hand_pose.detach().cpu().numpy(),
             betas=input_betas.detach().cpu().numpy(),
             global_orient=input_global_orient.detach().cpu().numpy(),
             transl=input_transl.detach().cpu().numpy(),
             scale=input_scale.detach().cpu().numpy())

    return input_params
    

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--model_path_male", 
        default='./models/smplx/SMPLX_MALE.npz')
    parser.add_argument("--model_path_female", 
        default='./models/smplx/SMPLX_FEMALE.npz')
    config = parser.parse_args()
    

    #device = torch.device('cpu')
    device = torch.device('cuda')

    subj_dirs_list = sorted(glob.glob(config.root + '/*'))

    for idx, subj_dir in enumerate(subj_dirs_list):

        print(f'Progressing {idx+1} / {len(subj_dirs_list)} ...')
        print(subj_dir)

        save_path = os.path.join(subj_dir, 'output.obj')
        if os.path.exists(save_path):
            print('skipped.')
            continue

        subj_name = os.path.basename(subj_dir)
        gender = gender_dict[subj_name]
        gender = 'male' if gender == 'M' else 'female'

        pose_3d = np.load(os.path.join(subj_dir, 'pose3d.npz'))

        dtype = torch.float32

        #pcd_path = os.path.join(subj_dir, 'images', 'view_000.ply')
        pcd_path = os.path.join(subj_dir, 'pcd_body.ply')
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd = np.array(pcd.points, dtype=np.float32)
        pcd = torch.from_numpy(pcd)
        pcd = pcd.unsqueeze(0)
        pcd = pcd.to(device)


        model_path = config.model_path_male if gender=='male' else config.model_path_female
#        tmp_model = smplx.create(gender=gender, model_path=model_path, model_type='smplx')
#        import pdb; pdb.set_trace()

        model_params = dict(model_path=model_path,
                            model_type='smplx',
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

        timer_start = time()
        input_params = main(model, pose_3d, pcd, subj_dir, device, dtype)

#        print('Again')
#        input_params = main(model, pose_3d, pcd, output_path, device, dtype, input_params)
        timer_end = time()
        print(f'Total elapsed time: {timer_end - timer_start}s')

