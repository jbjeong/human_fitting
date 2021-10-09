import numpy as np
import open3d as o3d

from data_info import not_upright_plus, not_upright_minus, mesh_list_needed_to_fix

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

