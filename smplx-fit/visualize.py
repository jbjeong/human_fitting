import argparse
import glob
import os
import numpy as np
import open3d as o3d

not_upright_plus = [
    'rp_aneko_posed_011',
    'rp_bianca_posed_020',
    'rp_jessica_posed_040',
    'rp_shawn_posed_024',
    'rp_shawn_posed_031',
    'rp_shawn_posed_033',
    'rp_sydney_posed_032',
]

not_upright_minus = [
    'rp_luisa_posed_024',
    'rp_sophia_posed_041',
]

mesh_list_needed_to_fix = [
    'rp_aaron_posed_014',
    'rp_amber_posed_007',
    'rp_aneko_posed_011',
    'rp_antonia_posed_004',
    'rp_beatrice_posed_019',
    'rp_bianca_posed_001',
    'rp_bianca_posed_020',
    'rp_cindy_posed_007',
    'rp_claudia_posed_019',
    'rp_corey_posed_010',
    'rp_corey_posed_014',
    'rp_corey_posed_016',
    'rp_cornell_posed_004',
    'rp_dennis_posed_037',
    'rp_dennis_posed_040',
    'rp_elizabeth_posed_006',
    'rp_ellie_posed_012',
    'rp_ellie_posed_013',
    'rp_ellie_posed_014',
    'rp_fabienne_posed_011',
    'rp_fernanda_posed_010',
    'rp_fernanda_posed_012',
    'rp_fernanda_posed_034',
    'rp_hannah_posed_007',
    'rp_holly_posed_002',
    'rp_holly_posed_006',
    'rp_holly_posed_009',
    'rp_janett_posed_007',
    'rp_janett_posed_029',
    'rp_jessica_posed_040',
    'rp_joel_posed_017',
    'rp_joko_posed_015',
    'rp_julia_posed_047',
    'rp_kal_posed_020',
    'rp_kal_posed_022',
    'rp_kal_posed_024',
    'rp_maya_posed_010',
    'rp_maya_posed_014',
    'rp_maya_posed_016',
    'rp_maya_posed_019',
    'rp_naomi_posed_001',
    'rp_philip_posed_006',
    'rp_philip_posed_028',
    'rp_ricarda_posed_028',
    'rp_saki_posed_027',
    'rp_seiko_posed_020',
    'rp_shawn_posed_024',
    'rp_shawn_posed_031',
    'rp_shawn_posed_033',
    'rp_sophia_posed_010',
    'rp_sophia_posed_016',
    'rp_sophia_posed_019',
    'rp_sophia_posed_027',
    'rp_sophia_posed_033',
    'rp_sydney_posed_032',
    'rp_toshiro_posed_036',
]


def preprocess(model):
    min_bound = model.get_min_bound()
    max_bound = model.get_max_bound()
    center = min_bound + (max_bound - min_bound) / 2.0
    #scale = np.linalg.norm(max_bound - min_bound) / 2.0
    scale = np.linalg.norm(max_bound - min_bound) / 1.0
    vertices = np.asarray(model.vertices)
    vertices -= center 
    #vertices = vertices - np.array([0, 0, 2])
    model.vertices = o3d.utility.Vector3dVector(vertices / scale)
    return model

def rotate_mesh(mesh, subj_name):
    if subj_name in not_upright_plus:
        r_x = np.pi / 2
        rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                            [0, np.sin(r_x), np.cos(r_x)]])
        mesh.rotate(rot_x, np.zeros(3))
    elif subj_name in not_upright_minus:
        r_x = -np.pi / 2
        rot_x = np.asarray([[1, 0, 0], [0, np.cos(r_x), -np.sin(r_x)],
                            [0, np.sin(r_x), np.cos(r_x)]])
        mesh.rotate(rot_x, np.zeros(3))
    
    if not subj_name in mesh_list_needed_to_fix:
        r_y = -90
        rot_y = np.asarray([[np.cos(r_y), 0, np.sin(r_y)], 
                            [0, 1, 0],
                            [-np.sin(r_y), 0, np.cos(r_y)]])
        mesh.rotate(rot_y, np.zeros(3))
    return mesh


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default='/home/jbjeong/projects/human_recon_make_data/images_for_test')
    parser.add_argument("--idx", type=int)
    parser.add_argument("--path")
    parser.add_argument("--gt", action='store_true')
    config = parser.parse_args()

    if config.path is not None:
        mesh_path = config.path
        mesh_list = [mesh_path]
    else:
        mesh_list = sorted(glob.glob(config.root + '/*/output.obj'))
        if config.idx is not None:
            mesh_list = [mesh_list[config.idx]]

    for mesh_idx, mesh_path in enumerate(mesh_list):

        print(f'Progressing [{mesh_idx} / {len(mesh_list)}] ...')

        subj_name = mesh_path.split('/')[-2]
        subj_dir = '/'.join(mesh_path.split('/')[:-1])

        ori_mesh_path = glob.glob(f'/home/jbjeong/projects/human_recon_make_data/mesh_data/{subj_name}_OBJ/*.obj')
        ori_mesh_path = ori_mesh_path[0]
        ori_mesh = o3d.io.read_triangle_mesh(ori_mesh_path)
        ori_mesh = preprocess(ori_mesh)

        ### Make mesh to stand upright 
        ori_mesh = rotate_mesh(ori_mesh, subj_name)

#        cluster_result = ori_mesh.cluster_connected_triangles()
#        import pdb; pdb.set_trace()
#        ori_vs = np.array(ori_mesh.vertices)
#        ori_ts = np.array(ori_mesh.triangles)
#        test_vis_list = []
#        for tidx, cluster_label in enumerate(cluster_result[0]):
#            if cluster_label == 0:
#                pass
#            else:
#                tri = ori_ts[tidx]
#                for vidx in tri:
#                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
#                    sphere.translate(ori_vs[vidx])
#                    sphere.paint_uniform_color([1, 0, 0])
#                    test_vis_list.append(sphere)



        pcd_path = os.path.join(subj_dir, 'pcd.ply')
        pcd = o3d.io.read_point_cloud(pcd_path)

        joint_path = os.path.join(subj_dir, 'pose3d.npz')
        body_pose_3d = np.load(joint_path)['body_pose_3d']
        hand_left_pose_3d = np.load(joint_path)['hand_left_pose_3d']
        hand_right_pose_3d = np.load(joint_path)['hand_right_pose_3d']

        color_5 = [
            np.array((1, 0, 0)),
            np.array((0.1, 1, 0)),
            np.array((1, 1, 0)),
            np.array((0, 0, 1)),
            np.array((1, 0, 1))
        ]
        hand_color = [(1, 0, 0)]
        for cur_color in color_5:
            for alpha in [0.8, 0.9, 0.9, 1.0]:
                hand_color.append(cur_color*alpha)
    
        vis_list = []
        both_hand_pose_3d = np.concatenate([hand_left_pose_3d, hand_right_pose_3d])
        both_hand_color = hand_color + hand_color

        if config.gt:
            for xyz in body_pose_3d:  
                if (xyz == np.inf).any(): continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(xyz)
                sphere.paint_uniform_color([0, 1, 0.7])
                vis_list.append(sphere)

            for kidx, xyz in enumerate(both_hand_pose_3d):  
                if (xyz == np.inf).any(): continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                sphere.translate(xyz)
                cur_color = both_hand_color[kidx]
                sphere.paint_uniform_color(cur_color)
                vis_list.append(sphere)

        else:
            for xyz in body_pose_3d:  
                if (xyz == np.inf).any(): continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                sphere.translate(xyz)
                sphere.paint_uniform_color([0, 1, 0.7])
                vis_list.append(sphere)

            for kidx, xyz in enumerate(both_hand_pose_3d):
                if (xyz == np.inf).any(): continue
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
                sphere.translate(xyz)
                #sphere.translate([-0.5, 0, 0])
                cur_color = both_hand_color[kidx]
                sphere.paint_uniform_color(cur_color)
                vis_list.append(sphere)
        
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0.7, 0])

        w = 800 
        h = 800 
        left = 1800 
        top = 500
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='open3d', width=w, height=h, left=left, top=top, visible=True)

        
        if config.gt:
            vis.add_geometry(ori_mesh)
#            for sphere in test_vis_list:
#                vis.add_geometry(sphere)


        else:
            vis.add_geometry(mesh)
            vis.add_geometry(pcd)

        for sphere in vis_list:
            vis.add_geometry(sphere)

        #frame = mesh.create_coordinate_frame(size=0.5, origin=np.array([0,0,0]))
        #vis.add_geometry(frame)

        vis.run()
        vis.destroy_window()
    

