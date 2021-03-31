"""
things needed 

predictions for image 
ground thruth for that image 
3d model loaded 

compare the poses.

"""
import os
import numpy as np 
import math 
from scipy import spatial
import copy 
from pyquaternion import Quaternion
import visii 

def create_obj(
    name = 'name',
    path_obj = "",
    scale = 0.01, 
    rot_base = None, #visii quat
    pos_base = None, # visii vec3
    ):

    
    # This is for YCB like dataset
    obj_mesh = visii.mesh.create_from_obj(name, path_obj)
    
    obj_entity = visii.entity.create(
        name = name,
        # mesh = visii.mesh.create_sphere("mesh1", 1, 128, 128),
        mesh = obj_mesh,
        transform = visii.transform.create(name),
        material = visii.material.create(name)
    )

    # should randomize
    obj_entity.get_material().set_metallic(0)  # should 0 or 1      
    obj_entity.get_material().set_transmission(0)  # should 0 or 1      
    obj_entity.get_material().set_roughness(1) # default is 1  

    obj_entity.get_transform().set_scale(visii.vec3(scale))

    if not rot_base is None:
        obj_entity.get_transform().set_rotation(rot_base)
    if not pos_base is None:
        obj_entity.get_transform().set_position(pos_base)

    return obj_entity


def get_add(obj1, obj2, pos1, pos2, quat1, quat2):
    obj1.get_transform().set_position(visii.vec3(pos1[0],pos1[1],pos1[2]))
    obj1.get_transform().set_rotation(visii.quat(quat1[0],quat1[1],quat1[2],quat1[3]))
    obj2.get_transform().set_position(visii.vec3(pos2[0],pos2[1],pos2[2]))
    obj2.get_transform().set_rotation(visii.quat(quat2[0],quat2[1],quat2[2],quat2[3]))

    dist = []
    vertices = obj1.get_mesh().get_vertices()
    for i in range(len(vertices)):
        v = visii.vec4(vertices[i][0],vertices[i][1],vertices[i][2],1)
        p0 = obj1.get_transform().get_local_to_world_matrix() * v
        p1 = obj2.get_transform().get_local_to_world_matrix() * v
        dist.append(visii.distance(p0, p1))

    dist = np.mean(dist)

    return dist