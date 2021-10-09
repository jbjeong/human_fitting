import torch

def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f:
            fp.write('f %d %d %d\n' % (f[0]+1, f[1]+1, f[2]+1))
    if verbose:
            print(f'mesh saved to: {filepath}')


def save_obj(save_path, body_model, input_params):
    (input_body_pose, input_betas, input_global_orient,
     input_transl, input_scale, 
     input_left_hand_pose, input_right_hand_pose) = input_params

    body_pose = torch.clone(input_body_pose.detach())
    betas = torch.clone(input_betas.detach())
    global_orient = torch.clone(input_global_orient.detach())
    transl = torch.clone(input_transl.detach())
    scale = torch.clone(input_scale.detach())
    left_hand_pose = torch.clone(input_left_hand_pose.detach())
    right_hand_pose = torch.clone(input_right_hand_pose.detach())

    output = body_model(
        return_verts=True, 
        body_pose=body_pose,
        betas=betas,
        global_orient=global_orient,
        left_hand_pose=left_hand_pose, 
        right_hand_pose=right_hand_pose)
    v = output.vertices.detach().cpu().numpy()
    v_tr = v * scale.cpu().numpy() + transl.cpu().numpy()
    faces = body_model.faces
    write_simple_obj(v_tr[0], faces, save_path)

