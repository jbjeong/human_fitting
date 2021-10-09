import torch
import numpy as np
import open3d as o3d
import smplx


#model_path = '/home/jbjeong/projects/smpl_dir/models/smplx/SMPLX_NEUTRAL.npz'
#model_path = '/home/jbjeong/projects/smpl_dir/models/smplx/SMPLX_MALE.npz'
model_path = '/home/jbjeong/projects/smpl_dir/models/smplx/SMPLX_FEMALE.npz'
dtype=torch.float32
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

model = smplx.create(gender='female', **model_params)
#import pdb; pdb.set_trace()
#model.J_regressor[18][4284] += 0.6
#model.J_regressor[18][4347] += 0.6
#model.J_regressor[18] = model.J_regressor[18] / model.J_regressor[18].sum()

#model.J_regressor[18][4288] += 0.2
#model.J_regressor[18][4301] += 0.2
#model.J_regressor[18] = model.J_regressor[18] / model.J_regressor[18].sum()


output = model()
ori_mesh = o3d.geometry.TriangleMesh()
vertices = output.vertices.detach().cpu().numpy()
vertices = vertices.squeeze(0)
faces = model.faces
ori_mesh.vertices = o3d.utility.Vector3dVector(vertices)
ori_mesh.triangles = o3d.utility.Vector3iVector(faces)
ori_mesh.compute_vertex_normals()
ori_mesh.paint_uniform_color([0.7, 0.7, 0])

body_pose = torch.zeros(model.body_pose.shape,
                        dtype=dtype,
                        requires_grad=True)
#body_pose[0][45:48] += 0.4
#betas = torch.zeros(model.betas.shape,
#                    dtype=dtype,
#                    requires_grad=True)
#betas[0][0] = 0
#right_hand_pose = torch.zeros(model.right_hand_pose.shape,
#                        dtype=dtype,
#                        requires_grad=True)
#right_hand_pose[0][0:3] += 0.5 
#right_hand_pose[0][6:9] += 0.5 
#right_hand_pose[0][9:12] += 0.5 
#right_hand_pose[0][-3:] += 0.5 
#scale=0.5
#input_global_orient = torch.tensor([[0.5, 0.2, 0.4]], dtype=dtype, requires_grad=True)
scale = 1.0
input_global_orient = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)
output = model(global_orient=input_global_orient,
               body_pose=body_pose)

vertices = output.vertices.detach().cpu().numpy()
vertices = vertices.squeeze(0) * scale
faces = model.faces

joints = output.joints.detach().cpu().numpy()
joints = joints.squeeze(0) * scale


vis_list = []
for j_idx, xyz in enumerate(joints):
    if (xyz == np.inf).any(): continue
    #sphere.translate([0, 0, 0.5])
    if j_idx == 19:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.042)
        sphere.paint_uniform_color([1, 0.2, 0])
        sphere.translate(xyz)
        vis_list.append(sphere)
    elif j_idx == 18:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.042)
        sphere.paint_uniform_color([0, 0.2, 1])
        #xyz[0] += 0.01
        #xyz[1] += 0.03
        #xyz[2] += 0.01
        sphere.translate(xyz)
        vis_list.append(sphere)
    else:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.033)
        sphere.paint_uniform_color([0, 1, 0.7])

#    sphere.translate(xyz)
#    vis_list.append(sphere)

#import pdb; pdb.set_trace()
elbow_idxs = torch.where(model.J_regressor[18] > 0)[0]
elbow_vertices = output.vertices[0, elbow_idxs]
elbow_vertices = elbow_vertices.detach().cpu().numpy()

#sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
#sphere.paint_uniform_color([0, 1, 0])
#sphere.translate(elbow_vertices.mean(0))
#vis_list.append(sphere)

#for idx, xyz in enumerate(elbow_vertices):
#    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
#    if idx == 4:
#        sphere.paint_uniform_color([1, 1, 0])
#    else:
#        sphere.paint_uniform_color([0, 1, 0])
#
#    print('idx:', idx)
#    print('vidx:', elbow_idxs[idx], '| J:', model.J_regressor[18][elbow_idxs[idx]])
#
#    sphere.translate(xyz)
#    vis_list.append(sphere)

#import pdb; pdb.set_trace()
left_elbow = joints[18]

vs = output.vertices.squeeze(0).detach().cpu().numpy()
dist = np.sqrt(((vs - left_elbow) ** 2).sum(1))

#for idx, xyz in enumerate(vs):
#
#    if dist[idx] < 0.05:
#        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
#        sphere.paint_uniform_color([0, 1, 0])
#        sphere.translate(xyz)
#        vis_list.append(sphere)

#sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
#sphere.paint_uniform_color([0, 1, 0])
#sphere.translate(vs[dist < 0.05].mean(0))
#vis_list.append(sphere)




mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.compute_vertex_normals()
#mesh.paint_uniform_color([0, 0, 0])

#o3d.visualization.draw_geometries(vis_list + [mesh, ori_mesh])
#o3d.visualization.draw_geometries([mesh, ori_mesh])
o3d.visualization.draw_geometries(vis_list + [mesh])
