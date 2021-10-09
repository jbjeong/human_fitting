import argparse
import glob
import os
import numpy as np
import open3d as o3d


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--idx", type=int)
    config = parser.parse_args()

    w = 800 
    h = 800 
    left = 1800 
    top = 500
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='open3d', width=w, height=h, left=left, top=top, visible=True)

    if config.path:
        mesh = o3d.io.read_triangle_mesh(config.path)
        vis.add_geometry(mesh)
    else:  
        if config.idx == 0:
            orig = o3d.io.read_triangle_mesh('test_orig.obj')
            vis.add_geometry(orig)
        elif config.idx == 1:
            de_rp = o3d.io.read_triangle_mesh('test_de_rp.obj')
            #de_rp.paint_uniform_color([1, 0.7, 0])
            vis.add_geometry(de_rp)
        elif config.idx == 2:
            body = o3d.io.read_triangle_mesh('test_body.obj')
            body.paint_uniform_color([1, 0.7, 0])
            body.compute_vertex_normals()
            vis.add_geometry(body)
        else:
            de_body = o3d.io.read_triangle_mesh('test_de_body.obj')
            vis.add_geometry(de_body)


    vis.run()
    vis.destroy_window()

