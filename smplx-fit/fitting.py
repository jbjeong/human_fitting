import torch

from utils_smplify import GMoF, rel_change

#joint_correspondence_idx_map_openpose_to_smpl

j_map_o2s = {
    3: 19,
    6: 18,
    10: 5,
    13: 4,
    4: 21,
    7: 20,
    2: 17,
    5: 16,
    9: 2,
    12: 1,
    16: 23,
    15: 24,
}

open_joint = {
    'head': 0,
    'neck': 1,
    'right_shoulder': 2,
    'right_elbow': 3,
    'right_wrist': 4,
    'left_shoulder': 5,
    'left_elbow': 6,
    'left_wrist': 7,
    #'pelvis': 8,
    'right_hip': 9,
    'right_knee': 10,
    'right_ankle': 11,
    'left_hip': 12,
    'left_knee': 13,
    'left_ankle': 14,
    'right_eye': 15,
    'left_eye': 16,
}


# Left, Right
id_mapper_smpl_finger_thumb = [37,38,39,66]+[52,53,54,71]
id_mapper_smpl_finger_index = [25,26,27,67]+[40,41,42,72]
id_mapper_smpl_finger_middle = [28,29,30,68]+[43,44,45,73]
#id_mapper_smpl_finger_pinky = [31,32,33,69]+[49,50,51,74]
#id_mapper_smpl_finger_ring = [34,35,36,70]+[46,47,48,75]
# Left pinky idx 1, 2 are needed to be replace with ring
id_mapper_smpl_finger_pinky = [31,35,36,69]+[49,50,51,74] 
id_mapper_smpl_finger_ring = [34,32,33,70]+[46,47,48,75]

id_mapper_openpose_finger_thumb = [1,2,3,4,1+21,2+21,3+21,4+21]
id_mapper_openpose_finger_index = [5,6,7,8,5+21,6+21,7+21,8+21]
id_mapper_openpose_finger_middle = [9,10,11,12,9+21,10+21,11+21,12+21]
id_mapper_openpose_finger_pinky = [13,14,15,16,13+21,14+21,15+21,16+21]
id_mapper_openpose_finger_ring = [17,18,19,20,17+21,18+21,19+21,20+21]


def create_optimizer(input_params, step):

    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    params = []
    if step == 'orient+transl':
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_transl)
    elif step == 'orient+transl+scale':
        params.append(input_global_orient)
        params.append(input_transl)
        params.append(input_scale)
    elif step == 'shoulder':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'elbow-L':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'elbow-R':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'wrist':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'fingers':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'local_arm_pose':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'knee':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'ankle':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'toe':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'local_leg_pose':
        params.append(input_body_pose)
        params.append(input_betas)
    elif step == 'face':
        params.append(input_body_pose)
    elif step == 'global_pose':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_global_orient)
        params.append(input_transl)
        params.append(input_scale)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    elif step == 'betas':
        params.append(input_betas)
    elif step == 'fingers+betas':
        params.append(input_body_pose)
        params.append(input_betas)
        params.append(input_left_hand_pose)
        params.append(input_right_hand_pose)
    else:
        raise ValueError(f'step {step} is not found !')

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=30,
        line_search_fn='strong_wolfe')

    return optimizer, params


def create_fitting_closure(optimizer,
                           body_model, 
                           input_params,
                           gt_list, 
                           step,
                           pcd=None):

    robustifier = GMoF(rho=100)

    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    prev_input_body_pose = torch.clone(input_body_pose.detach())
    prev_input_betas = torch.clone(input_betas.detach())
    prev_input_global_orient = torch.clone(input_global_orient.detach())
    prev_input_transl = torch.clone(input_transl.detach())
    prev_input_scale = torch.clone(input_scale.detach())
    prev_input_left_hand_pose = torch.clone(input_left_hand_pose.detach())
    prev_input_right_hand_pose = torch.clone(input_right_hand_pose.detach())

    (gt_body_joint, gt_face_joint, 
     gt_left_hand_joint, gt_right_hand_joint) = gt_list

    def fitting_func(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = body_model(body_pose=input_body_pose,
                                       betas=input_betas,
                                       global_orient=input_global_orient,
                                       left_hand_pose=input_left_hand_pose,
                                       right_hand_pose=input_right_hand_pose,
                                       return_verts=True)

        j_tr = body_model_output.joints * input_scale + input_transl

        betas_reg_prior = torch.sum(input_betas ** 2)
        body_pose_reg_prior = torch.sum(input_body_pose ** 2)
        hand_pose_reg_prior = torch.sum(input_left_hand_pose**2 + input_right_hand_pose**2)

        body_pose_preserve_prior = ((prev_input_body_pose - input_body_pose) ** 2).sum()
        betas_preserve_prior = ((prev_input_betas - input_betas) ** 2).sum()
        global_orient_preserve_prior = ((prev_input_global_orient - input_global_orient) ** 2).sum()
        transl_preserve_prior = ((prev_input_transl - input_transl) ** 2).sum()
        scale_preserve_prior = ((prev_input_scale - input_scale) ** 2).sum()
        left_hand_pose_preserve_prior = ((prev_input_left_hand_pose - input_left_hand_pose) ** 2).sum()
        right_hand_pose_preserve_prior = ((prev_input_right_hand_pose - input_right_hand_pose) ** 2).sum()

        if step == 'orient+transl':

            id_mapper_smpl=[12, 17, 16, 2, 1]
            id_mapper_openpose=[1, 2, 5, 9, 12]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.001 * global_orient_preserve_prior + \
                0.001 * transl_preserve_prior + \
                0.001 * betas_preserve_prior

        elif step == 'orient+transl+scale':

            id_mapper_smpl=[17, 16, 2, 1]
            id_mapper_openpose=[2, 5, 9, 12]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.001 * global_orient_preserve_prior + \
                0.001 * transl_preserve_prior + \
                0.001 * betas_preserve_prior

        elif step == 'shoulder':

            id_mapper_smpl=[16, 17]
            id_mapper_openpose=[5, 2]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = 2*joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.00001*body_pose_reg_prior + \
                0.00001*betas_reg_prior

        elif step == 'elbow-L':

            #id_mapper_smpl=[18, 19]
            #id_mapper_openpose=[6, 3]
            id_mapper_smpl=[18]
            id_mapper_openpose=[6]

#            _gt_body_joint_left_elbow = gt_body_joint[0,id_mapper_smpl]
#            _gt_body_joint_left_elbow[0][1] -= 0.03
#            _gt_body_joint_left_elbow[0][2] -= 0.01
#            j_tr[0,id_mapper_smpl][0,1] += 0.03*input_scale.detach().item()
#            j_tr[0,id_mapper_smpl][0,2] += 0.01*input_scale.detach().item()
            smpl_elbow_error = torch.tensor([[+0.02, +0.01, +0.002]], dtype=body_model.dtype, 
                                            device=j_tr.device)
            #smpl_elbow_error *= input_scale.detach()
            #import pdb; pdb.set_trace()

            #diff = gt_body_joint[0,id_mapper_openpose] - (j_tr[0,id_mapper_smpl] + smpl_elbow_error)
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.00001*body_pose_reg_prior + \
                0.00001*betas_reg_prior

        elif step == 'elbow-R':

            id_mapper_smpl=[19]
            id_mapper_openpose=[3]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.00001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.00001*body_pose_reg_prior + \
                0.00001*betas_reg_prior

        elif step == 'wrist':
            id_mapper_smpl=[20, 21]
            id_mapper_openpose=[7, 4]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.00001*body_pose_reg_prior + \
                0.00001*betas_reg_prior

        elif step == 'fingers':

            id_mapper_smpl = \
                (id_mapper_smpl_finger_thumb +
                id_mapper_smpl_finger_index +
                id_mapper_smpl_finger_middle +
                id_mapper_smpl_finger_pinky +
                id_mapper_smpl_finger_ring)
            id_mapper_openpose = \
                (id_mapper_openpose_finger_thumb +
                id_mapper_openpose_finger_index +
                id_mapper_openpose_finger_middle +
                id_mapper_openpose_finger_pinky +
                id_mapper_openpose_finger_ring)

            gt_both_hand_joint = torch.cat([gt_left_hand_joint, 
                                            gt_right_hand_joint], dim=1)

            skip_idx_of_open_mapper = []
            for index, joint_idx in enumerate(id_mapper_openpose): 
                if torch.isinf(gt_both_hand_joint[0,joint_idx]).any():
                    skip_idx_of_open_mapper.append(index)

            new_id_mapper_smpl = []
            new_id_mapper_openpose = []
            for index in range(len(id_mapper_openpose)):
                if not index in skip_idx_of_open_mapper:
                    new_id_mapper_smpl.append(id_mapper_smpl[index])
                    new_id_mapper_openpose.append(id_mapper_openpose[index])
            id_mapper_smpl = new_id_mapper_smpl
            id_mapper_openpose = new_id_mapper_openpose


            diff = gt_both_hand_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)

            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*body_pose_reg_prior + \
                0.00001*betas_preserve_prior + \
                0.00001*betas_reg_prior + \
                0.000001*left_hand_pose_preserve_prior + \
                0.000001*right_hand_pose_preserve_prior + \
                0.000001*hand_pose_reg_prior 
                
        elif step == 'local_arm_pose':

            id_mapper_openpose = [
                open_joint['left_elbow'], open_joint['right_elbow'],
                open_joint['left_wrist'], open_joint['right_wrist']]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]
            arm_diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            arm_joint_diff = robustifier(arm_diff)
            arm_joint_loss = torch.sum(arm_joint_diff)

            id_mapper_smpl = \
                (id_mapper_smpl_finger_thumb +
                id_mapper_smpl_finger_index +
                id_mapper_smpl_finger_middle +
                id_mapper_smpl_finger_pinky +
                id_mapper_smpl_finger_ring)
            id_mapper_openpose = \
                (id_mapper_openpose_finger_thumb +
                id_mapper_openpose_finger_index +
                id_mapper_openpose_finger_middle +
                id_mapper_openpose_finger_pinky +
                id_mapper_openpose_finger_ring)

            gt_both_hand_joint = torch.cat([gt_left_hand_joint, 
                                            gt_right_hand_joint], dim=1)
            diff = gt_both_hand_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            finger_joint_loss = torch.sum(joint_diff)
            
            total_loss = 2*arm_joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.0001*global_orient_preserve_prior + \
                0.0001*transl_preserve_prior + \
                0.0001*scale_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior + \
                2*finger_joint_loss + \
                0.0001*left_hand_pose_preserve_prior + \
                0.0001*right_hand_pose_preserve_prior + \
                0.0001*hand_pose_reg_prior 

        elif step == 'knee':
            id_mapper_smpl=[4, 5]
            id_mapper_openpose=[13, 10]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.0001*betas_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior

        elif step == 'ankle':
            id_mapper_smpl=[8, 7]
            id_mapper_openpose=[11, 14]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.0001*betas_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior

        elif step == 'toe':
            id_mapper_smpl=[10, 11]
            id_mapper_openpose=[20, 23]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.001*betas_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior

        elif step == 'local_leg_pose':

            id_mapper_smpl = [8, 7, 10, 11]
            id_mapper_openpose = [11, 14, 20, 23]
            foot_diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            foot_joint_diff = robustifier(foot_diff)
            foot_joint_loss = torch.sum(foot_joint_diff)

            vv = body_model_output.vertices * input_scale + input_transl
            dist, _ = ((pcd.unsqueeze(1) - vv.unsqueeze(2))**2).sum(3).min(2)
            

            total_loss = 20*foot_joint_loss + \
                0.1*dist.sum() + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.0001*global_orient_preserve_prior + \
                0.0001*transl_preserve_prior + \
                0.0001*scale_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior

        elif step == 'face':
            id_mapper_smpl=[23, 24, 58, 59]
            id_mapper_openpose=[16, 15, 17, 18]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)
            
            total_loss = joint_loss + \
                0.0001*body_pose_preserve_prior + \
                0.0001*body_pose_reg_prior

        elif step == 'global_pose':

            id_mapper_openpose = [
                open_joint['left_hip'], open_joint['right_hip'],
                open_joint['left_knee'], open_joint['right_knee'],
                open_joint['left_shoulder'], open_joint['right_shoulder'],
            ]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            body_joint_loss = torch.sum(joint_diff)

            # Face
            id_mapper_smpl=[23, 24, 58, 59]
            id_mapper_openpose=[16, 15, 17, 18]
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            face_joint_loss = torch.sum(joint_diff)

            # Elbow
            id_mapper_smpl=[18]
            id_mapper_openpose=[6]
            smpl_elbow_error = torch.tensor([[+0.02, +0.01, +0.002]], dtype=body_model.dtype, 
                                            device=j_tr.device)
            #diff = gt_body_joint[0,id_mapper_openpose] - (j_tr[0,id_mapper_smpl] + smpl_elbow_error)
            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            left_elbow_joint_loss = torch.sum(joint_diff)

            id_mapper_smpl=[19]
            id_mapper_openpose=[3]

            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            right_elbow_joint_loss = torch.sum(joint_diff)

            elbow_joint_loss = left_elbow_joint_loss + right_elbow_joint_loss
            
            # Wrist
            id_mapper_openpose = [
                open_joint['left_wrist'], open_joint['right_wrist']]
            id_mapper_smpl = [j_map_o2s[jidx] for jidx in id_mapper_openpose]
            wrist_diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            wrist_joint_diff = robustifier(wrist_diff)
            wrist_joint_loss = torch.sum(wrist_joint_diff)

            id_mapper_smpl = \
                (id_mapper_smpl_finger_thumb +
                id_mapper_smpl_finger_index +
                id_mapper_smpl_finger_middle +
                id_mapper_smpl_finger_pinky +
                id_mapper_smpl_finger_ring)
            id_mapper_openpose = \
                (id_mapper_openpose_finger_thumb +
                id_mapper_openpose_finger_index +
                id_mapper_openpose_finger_middle +
                id_mapper_openpose_finger_pinky +
                id_mapper_openpose_finger_ring)

            gt_both_hand_joint = torch.cat([gt_left_hand_joint, 
                                            gt_right_hand_joint], dim=1)

            skip_idx_of_open_mapper = []
            for index, joint_idx in enumerate(id_mapper_openpose): 
                if torch.isinf(gt_both_hand_joint[0,joint_idx]).any():
                    skip_idx_of_open_mapper.append(index)

            new_id_mapper_smpl = []
            new_id_mapper_openpose = []
            for index in range(len(id_mapper_openpose)):
                if not index in skip_idx_of_open_mapper:
                    new_id_mapper_smpl.append(id_mapper_smpl[index])
                    new_id_mapper_openpose.append(id_mapper_openpose[index])
            id_mapper_smpl = new_id_mapper_smpl
            id_mapper_openpose = new_id_mapper_openpose

            diff = gt_both_hand_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            finger_joint_loss = torch.sum(joint_diff)

            id_mapper_smpl = [8, 7, 10, 11]
            id_mapper_openpose = [11, 14, 20, 23]
            foot_diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            foot_joint_diff = robustifier(foot_diff)
            foot_joint_loss = torch.sum(foot_joint_diff)


            vv = body_model_output.vertices * input_scale + input_transl
            dist, _ = ((pcd.unsqueeze(1) - vv.unsqueeze(2))**2).sum(3).min(2)
            dist = dist.sum()
            
            total_loss = body_joint_loss + \
                elbow_joint_loss + \
                wrist_joint_loss + \
                finger_joint_loss + \
                foot_joint_loss + \
                face_joint_loss + \
                0.02*dist + \
                0.0001*body_pose_preserve_prior + \
                0.00001*betas_preserve_prior + \
                0.0001*global_orient_preserve_prior + \
                0.0001*transl_preserve_prior + \
                0.0001*scale_preserve_prior + \
                0.0001*body_pose_reg_prior + \
                0.0001*betas_reg_prior

        elif step == 'betas':
#            id_mapper_smpl =     [12, 17, 19, 21, 16, 18, 20, 2,  5,  8,  1,  4,  7]
#            id_mapper_openpose = [ 1,  2,  3,  4,  5,  6,  7, 9, 10, 11, 12, 13, 14]
#            diff = gt_body_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
#            joint_diff = robustifier(diff)
#            joint_loss = torch.sum(joint_diff)

            vv = body_model_output.vertices * input_scale + input_transl
            dist, _ = ((pcd.unsqueeze(1) - vv.unsqueeze(2))**2).sum(3).min(2)
            dist = dist.sum()
            
            total_loss = 0.1*dist + 0.0001*betas_preserve_prior


        elif step == 'fingers+betas':

            id_mapper_smpl = \
                (id_mapper_smpl_finger_thumb +
                id_mapper_smpl_finger_index +
                id_mapper_smpl_finger_middle +
                id_mapper_smpl_finger_pinky +
                id_mapper_smpl_finger_ring)
            id_mapper_openpose = \
                (id_mapper_openpose_finger_thumb +
                id_mapper_openpose_finger_index +
                id_mapper_openpose_finger_middle +
                id_mapper_openpose_finger_pinky +
                id_mapper_openpose_finger_ring)

            gt_both_hand_joint = torch.cat([gt_left_hand_joint, 
                                            gt_right_hand_joint], dim=1)
            diff = gt_both_hand_joint[0,id_mapper_openpose] - j_tr[0,id_mapper_smpl]
            joint_diff = robustifier(diff)
            joint_loss = torch.sum(joint_diff)

            total_loss = joint_loss + \
                0.01*body_pose_preserve_prior + \
                0.001*left_hand_pose_preserve_prior + \
                0.001*right_hand_pose_preserve_prior + \
                0.001*betas_preserve_prior + \
                0.001*hand_pose_reg_prior + \
                0.001*betas_reg_prior

        else:
            raise ValueError(f'step {step} is not found in clolsure!')

#            if step == '8':
#                #betas_prior = torch.sum(torch.clone(input_betas.detach()) ** 2) 
#                betas_prior = torch.sum(input_betas ** 2)
#                joint_prior = torch.sum(input_body_pose ** 2)
#                #total_loss = joint_loss + betas_prior + joint_prior
#                total_loss = joint_loss + 0.01*betas_prior + 0.01*joint_prior
#            elif step == '9':
#                betas_prior = torch.sum(torch.clone(input_betas.detach()) ** 2) 
#                total_loss = joint_loss + betas_prior
#            elif step == '10':
#                hand_prior = torch.sum(input_left_hand_pose**2 + input_right_hand_pose**2)
#                betas_prior = torch.sum(torch.clone(input_betas.detach()) ** 2) 
#                total_loss = joint_loss + 0.01*hand_prior + 0.1*betas_prior
#            elif step == '11':
#                hand_prior = torch.sum(input_left_hand_pose**2 + input_right_hand_pose**2)
#                betas_prior = torch.sum(torch.clone(input_betas.detach()) ** 2) 
#                total_loss = joint_loss + 0.0001*hand_prior + 0.01*betas_prior
#            else:
#                joint_prior = torch.sum(input_body_pose ** 2)
#                total_loss = joint_loss + 0.001*joint_prior

        if backward:
            total_loss.backward(create_graph=False)

            if step == 'orient+transl':
                input_betas.grad[0][1:] = 0
            elif step == 'orient+transl+scale':
                pass
            elif step == 'shoulder':
                input_body_pose.grad[0][:36] = 0
                input_body_pose.grad[0][42:] = 0
            elif step == 'elbow-L':
                #input_body_pose.grad[0][:36] = 0
                #input_body_pose.grad[0][42:45] = 0
                input_body_pose.grad[0][:45] = 0
                input_body_pose.grad[0][48:] = 0
            elif step == 'elbow-R':
                input_body_pose.grad[0][:48] = 0
                input_body_pose.grad[0][51:] = 0
            elif step == 'wrist': # wrist
                input_body_pose.grad[0][:51] = 0
                input_body_pose.grad[0][57:] = 0
            elif step == 'fingers': # finger-3 
                input_body_pose.grad[0][:57] = 0
                input_body_pose.grad[0][63:] = 0
            elif step == 'local_arm_pose':
                input_body_pose.grad[0][:45] = 0
                #input_body_pose.grad[0][57:] = 0
                #input_body_pose.grad[0][:57] = 0
                input_body_pose.grad[0][63:] = 0
#            elif step == '4R': # wrist
#                input_body_pose.grad[0][:54] = 0
#                input_body_pose.grad[0][57:] = 0
            elif step == 'knee': # knee
                input_body_pose.grad[0][6:] = 0
            elif step == 'ankle': # ankle
                input_body_pose.grad[0][:9] = 0
                input_body_pose.grad[0][15:] = 0
            elif step == 'toe': # toe
                input_body_pose.grad[0][:18] = 0
                input_body_pose.grad[0][24:] = 0
            elif step == 'local_leg_pose':
                #input_body_pose.grad[0][6:9] = 0
                input_body_pose.grad[0][:9] = 0
                input_body_pose.grad[0][15:18] = 0
                input_body_pose.grad[0][24:] = 0
            elif step == 'face':
                input_body_pose.grad[0][:33] = 0
                input_body_pose.grad[0][36:42] = 0
                input_body_pose.grad[0][45:] = 0
            elif step == 'global_pose':
                input_body_pose.grad[0][6:9] = 0
                input_body_pose.grad[0][15:45] = 0
                input_body_pose.grad[0][57:] = 0
            elif step == 'betas': 
                pass
            elif step == 'fingers+betas': # finger-3 
                input_body_pose.grad[0][:57] = 0
                input_body_pose.grad[0][63:] = 0
#            elif step == '9': 
#                pass
##                input_body_pose.grad[0][:57] = 0
##                input_body_pose.grad[0][63:] = 0
##                input_body_pose.grad[0][:51] = 0
##                input_body_pose.grad[0][63:] = 0
##            elif step == 9: # finger-2 
##                pass
##                input_right_hand_pose.grad[0][3:9] = 0
##                input_right_hand_pose.grad[0][12:18] = 0
##                input_right_hand_pose.grad[0][21:27] = 0
##                input_right_hand_pose.grad[0][30:36] = 0
##                input_right_hand_pose.grad[0][40:46] = 0
#
##                input_right_hand_pose.grad[0][0:6] = 0
##                input_right_hand_pose.grad[0][9:15] = 0
##                input_right_hand_pose.grad[0][18:24] = 0
##                input_right_hand_pose.grad[0][27:33] = 0
##                input_right_hand_pose.grad[0][36:45] = 0
#
##                input_right_hand_pose.grad[0][0:3] = 0
##                input_right_hand_pose.grad[0][6:9] = 0
##
##                input_right_hand_pose.grad[0][9:12] = 0
##                input_right_hand_pose.grad[0][15:18] = 0
##
##                input_right_hand_pose.grad[0][18:21] = 0
##                input_right_hand_pose.grad[0][24:27] = 0
##
##                input_right_hand_pose.grad[0][27:30] = 0
##                input_right_hand_pose.grad[0][33:36] = 0
##
##                input_right_hand_pose.grad[0][36:39] = 0
##                input_right_hand_pose.grad[0][42:45] = 0
#            elif step == '10': # finger-3 
#                input_body_pose.grad[0][:57] = 0
#                input_body_pose.grad[0][63:] = 0
#            elif step == '11': # finger-3 
#                input_body_pose.grad[0][:57] = 0
#                input_body_pose.grad[0][63:] = 0
            else:
                raise ValueError('No step in backward')

        return total_loss

    return fitting_func

def run_fitting(optimizer, closure, params, body_model, max_iters, ftol, gtol):

    prev_loss = None

    for n in range(max_iters):
        #prev_pose = optimizer._params[0].detach()
        #prev_input_body_pose = input_body_pose.detach()
        loss = optimizer.step(closure)
        #post_input_body_pose = input_body_pose.detach()
        #cur_pose = optimizer._params[0].detach()

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


