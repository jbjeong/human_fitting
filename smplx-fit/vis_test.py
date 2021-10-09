
import argparse
import glob
import os
import numpy as np
import open3d as o3d


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    config = parser.parse_args()

    w = 800 
    h = 800 
    left = 1800 
    top = 500
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='open3d', width=w, height=h, left=left, top=top, visible=True)

    mesh1 = o3d.io.read_triangle_mesh('result_CMU/01_01_poses_0_body.obj')
    mesh2 = o3d.io.read_triangle_mesh('result_CMU/01_01_poses_0_orig.obj')
    vis.add_geometry(mesh1)
    vis.add_geometry(mesh2)


    vis.run()
    vis.destroy_window()
