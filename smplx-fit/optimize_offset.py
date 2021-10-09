import argparse
import glob
import os
from time import time

from tqdm import tqdm
import torch

import numpy as np
import open3d as o3d

import kaolin
from kaolin.metrics.mesh import point_to_surface, laplacian_loss

import smplx
from utils_smplify import JointMapper, smpl_to_openpose, rel_change, GMoF
from util_write import write_simple_obj 
from data_info import gender_dict
from fitting import create_optimizer, create_fitting_closure, run_fitting

from visualize import preprocess, rotate_mesh


def get_loss_weights():
    """Set loss weights"""

    loss_weight = {'s2m': lambda cst, it: 10. ** 2 * cst * (1 + it),
                   'm2s': lambda cst, it: 10. ** 2 * cst, #/ (1 + it),
                   'lap': lambda cst, it: 10. ** 4 * cst / (1 + it),
                   'offsets': lambda cst, it: 10. ** 1 * cst / (1 + it)}
    return loss_weight


def forward_step(th_rp_mesh, th_init_smpl_mesh, body_model, input_params):
    """
    Performs a forward step, given smpl and scan meshes.
    Then computes the losses.
    """
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose,
     input_disp) = input_params
    
    # forward
    body_model_output = body_model(body_pose=input_body_pose,
                                   betas=input_betas,
                                   global_orient=input_global_orient,
                                   left_hand_pose=input_left_hand_pose,
                                   right_hand_pose=input_right_hand_pose,
                                   return_verts=True,
                                   displacements=input_disp)

    vv = body_model_output.vertices * input_scale + input_transl

    th_smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(
        vertices=vv[0],
        faces=torch.from_numpy(model.faces.astype(np.int64))
        )
    th_smpl_mesh.cuda()

    # losses
    loss = dict()
    loss['s2m'] = point_to_surface(th_rp_mesh.vertices, th_smpl_mesh)
    loss['m2s'] = point_to_surface(th_smpl_mesh.vertices, th_rp_mesh)
    loss['lap'] = laplacian_loss(th_init_smpl_mesh, th_smpl_mesh)
    loss['offsets'] = torch.mean(torch.mean(input_disp**2, axis=1))
    return loss


def backward_step(loss_dict, weight_dict, it):
    w_loss = dict()
    for k in loss_dict:
        w_loss[k] = weight_dict[k](loss_dict[k], it)

    tot_loss = list(w_loss.values())
    tot_loss = torch.stack(tot_loss).sum()
    return tot_loss


def optimize_adam(th_rp_mesh, th_init_smpl_mesh, body_model, input_params, iterations, steps_per_iter):

    optimizer = torch.optim.Adam([input_params[-1]], 0.005, betas=(0.9, 0.999))

    weight_dict = get_loss_weights()

    for it in range(iterations):
        loop = tqdm(range(steps_per_iter))
        loop.set_description('Optimizing SMPL+D')
        for i in loop:
            optimizer.zero_grad()
            # Get losses for a forward pass
            loss_dict = forward_step(th_rp_mesh, th_init_smpl_mesh, body_model, input_params)
            # Get total loss for backward pass
            tot_loss = backward_step(loss_dict, weight_dict, it)
            tot_loss.backward()
            optimizer.step()

            l_str = 'Lx100. Iter: {}'.format(i)
            for k in loss_dict:
                l_str += ', {}: {:0.4f}'.format(k, loss_dict[k].mean().item()*100)
            loop.set_description(l_str)


def create_fitting_closure(optimizer, body_model, input_params, th_rp_mesh, th_init_smpl_mesh): 

    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose,
     input_disp) = input_params

    def fitting_func(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = body_model(body_pose=input_body_pose,
                                       betas=input_betas,
                                       global_orient=input_global_orient,
                                       left_hand_pose=input_left_hand_pose,
                                       right_hand_pose=input_right_hand_pose,
                                       return_verts=True,
                                       displacements=input_disp)

        vv = body_model_output.vertices * input_scale + input_transl

        th_smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(
            vertices=vv[0],
            faces=torch.from_numpy(model.faces.astype(np.int64))
            )
        th_smpl_mesh.cuda()

        rp_to_smpl_point2surf_loss = point_to_surface(th_rp_mesh.vertices, th_smpl_mesh) 
        smpl_to_rp_point2surf_loss = point_to_surface(th_smpl_mesh.vertices, th_rp_mesh)
        #rp_to_smpl_point2surf_loss = 0
        #smpl_to_rp_point2surf_loss = 0
        lap_loss = laplacian_loss(th_init_smpl_mesh, th_smpl_mesh)
        l2_reg_loss = torch.mean(input_disp**2)

        # best
        total_loss = 10.**3 * rp_to_smpl_point2surf_loss + \
            10.**3 * smpl_to_rp_point2surf_loss + \
            10.**2 * lap_loss + \
            10.**2 * l2_reg_loss

#        total_loss = 10.**3 * rp_to_smpl_point2surf_loss + \
#            10.**3 * smpl_to_rp_point2surf_loss + \
#            10.**2.5 * lap_loss + \
#            10.**2 * l2_reg_loss

#        total_loss = 10.**2 * rp_to_smpl_point2surf_loss 
#        total_loss = 10.**2 * smpl_to_rp_point2surf_loss 
#        total_loss = 10.**2 * smpl_to_rp_point2surf_loss 

        if backward:
            total_loss.backward(create_graph=False)

        #import pdb; pdb.set_trace()

        return total_loss

    return fitting_func

def run_fitting(optimizer, closure, params, body_model, max_iters, ftol, gtol):

    prev_loss = None

    for n in range(max_iters):
        loss = optimizer.step(closure)

        if torch.isnan(loss).sum() > 0:
            print('NaN loss value, stopping!')
            break

        if torch.isinf(loss).sum() > 0:
            print('Infinite loss value, stopping!')
            break

        if n > 0 and prev_loss is not None and ftol > 0:
            loss_rel_change = rel_change(prev_loss, loss.item())
            if loss_rel_change <= ftol:
                print('loss converge')
                break
        
        if all([torch.abs(var.grad.view(-1).max()).item() < gtol
                for var in params if var.grad is not None]):
#        all_true = []
#        for var in params:
#            if var.grad is not None:
#                all_true.append(torch.abs(var.grad.view(-1).max()).item() < gtol)
#        all_true = [torch.abs(var.grad.view(-1).max()).item() < gtol
#                    for var in params if var.grad is not None]
#        if np.array(all_true).all():
            print('grad converge')
            break

        prev_loss = loss.item() 
        print(prev_loss)

    return prev_loss 


def main(model, th_rp_mesh, aux_data, save_dir, device, dtype):
        
    body_pose = torch.tensor(aux_data['body_pose'], dtype=dtype, device=device)
    betas = torch.tensor(aux_data['betas'], dtype=dtype, device=device)
    global_orient = torch.tensor(aux_data['global_orient'], dtype=dtype, device=device)
    transl = torch.tensor(aux_data['transl'], dtype=dtype, device=device)
    scale = torch.tensor(aux_data['scale'], dtype=dtype, device=device)
    left_hand_pose = torch.tensor(aux_data['left_hand_pose'], dtype=dtype, device=device)
    right_hand_pose = torch.tensor(aux_data['right_hand_pose'], dtype=dtype, device=device)

    input_disp = torch.zeros([1, 10475, 3],
                             dtype=dtype,
                             device=device,
                             requires_grad=True)

    # Init th_init_smpl_mesh
    body_model_output = model(body_pose=body_pose,
                              betas=betas,
                              global_orient=global_orient,
                              left_hand_pose=left_hand_pose,
                              right_hand_pose=right_hand_pose,
                              return_verts=True,
                              displacements=input_disp.clone().detach())

    vv = body_model_output.vertices * scale + transl

    th_init_smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(
        vertices=vv[0].clone().detach(),
        faces=torch.from_numpy(model.faces.astype(np.int64))
        )
    th_init_smpl_mesh.cuda()

    input_params = [body_pose, betas, global_orient, transl, scale, left_hand_pose, right_hand_pose]
    input_params.append(input_disp)

#    closure_max_iters = 30
#    ftol = 1e-9
#    gtol = 1e-9
#
#    # Create optimizer
#    optim_params = [input_disp]
#    optimizer = torch.optim.LBFGS(
#        optim_params,
#        lr=1.0,
#        max_iter=30,
#        line_search_fn='strong_wolfe')
#
#    # Create create_fitting_closure
#    closure = create_fitting_closure(optimizer, model, input_params, th_rp_mesh, th_init_smpl_mesh)
#
#    # Run fitting
#    loss = run_fitting(optimizer, closure, optim_params, model, closure_max_iters, ftol, gtol)

    # Adam
    iterations = 5
    steps_per_iter = 10 
    optimize_adam(th_rp_mesh, th_init_smpl_mesh, model, input_params, iterations, steps_per_iter)

    # Write
    obj_save_path = os.path.join(save_dir, 'output_smpld.obj')
    aux_save_path = os.path.join(save_dir, 'output_smpld_offset.npy')

    output = model(
        return_verts=True, 
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        left_hand_pose=left_hand_pose, 
        right_hand_pose=right_hand_pose,
        displacements=input_disp)
    v = output.vertices.detach().cpu().numpy()
    v_tr = v * scale.cpu().numpy() + transl.cpu().numpy()
    faces = model.faces

    write_simple_obj(v_tr[0], faces, obj_save_path)

    np.save(aux_save_path, input_disp.detach().cpu().numpy())

    #print(f'loss: {loss}')
    print(f'input_disp mean: {input_disp.detach().cpu().mean()}')
    print(f'input_disp max: {input_disp.detach().cpu().max()}')

if __name__=='__main__':

#    ### Test
#    model_path = './models/smplx/SMPLX_MALE.npz'
#
#    dtype = torch.float32
#    model_params = dict(model_path=model_path,
#                        model_type='smplxd',
#                        use_pca=False,
#                        dtype=dtype,
#                        )
#    
#    model = smplx.create(gender='male', **model_params)
#
#    device = torch.device('cuda')
#    model.to(device)
#    pose = torch.zeros(1,63).to(device)
#    displacements = torch.rand(1, 10475, 3).to(device) * 0
#    output = model(body_pose=pose, displacements=displacements)
#    np.save('test_smplx/output.npy', output.vertices.cpu().detach().numpy())

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", 
        default='/root/code/human_recon_make_data/images_for_test')
    parser.add_argument("--model_path_male", 
        default='./models/smplx/SMPLX_MALE.npz')
    parser.add_argument("--model_path_female", 
        default='./models/smplx/SMPLX_FEMALE.npz')
    parser.add_argument("--mesh_root", 
        default='/root/code/human_recon_make_data/mesh_data')
    config = parser.parse_args()

    #device = torch.device('cpu')
    device = torch.device('cuda')

    subj_dirs_list = sorted(glob.glob(config.root + '/*'))

    for idx, subj_dir in enumerate(subj_dirs_list):

        aux_path = os.path.join(subj_dir, 'output_aux.npz')
        if not os.path.exists(aux_path):
            print("no 'output_aux.npz'. skipped.") 
            continue

        print(f'Progressing {idx+1} / {len(subj_dirs_list)} ...')
        print(subj_dir)

        obj_save_path = os.path.join(subj_dir, 'output_smpld.obj')
        aux_save_path = os.path.join(subj_dir, 'output_smpld_offset.npy')
        if os.path.exists(obj_save_path):
            print('skipped.')
            continue

        subj_name = os.path.basename(subj_dir)
        gender = gender_dict[subj_name]
        gender = 'male' if gender == 'M' else 'female'

        dtype = torch.float32

        rp_mesh_path = glob.glob(os.path.join(config.mesh_root, subj_name + '_OBJ') + '/*.obj')[0]
        rp_mesh = o3d.io.read_triangle_mesh(rp_mesh_path)
        rp_mesh = preprocess(rp_mesh)
        rp_mesh = rotate_mesh(rp_mesh, subj_name)

        th_rp_mesh = kaolin.rep.TriangleMesh.from_tensors(
            vertices=torch.from_numpy(np.array(rp_mesh.vertices, dtype=np.float32)),
            faces=torch.from_numpy(np.array(rp_mesh.triangles, dtype=np.int64)),
            )
        th_rp_mesh.cuda()

        aux_path = os.path.join(subj_dir, 'output_aux.npz')
        aux_data = np.load(aux_path)

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

        timer_start = time()
        input_params = main(model, th_rp_mesh, aux_data, subj_dir, device, dtype)
        timer_end = time()

        print(f'Total elapsed time: {timer_end - timer_start}s')
