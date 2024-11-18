
import numpy as np
from numpy.random import choice
from scipy import interpolate
from isaacgym import terrain_utils


def gap_parkour(terrain, platform_length=1., lava_width=0.5, lava_depth=-1.0, gap_length=0.5, gap_platform_length_min=1.25, gap_platform_length_max=1.5, gap_platform_height=0.05):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    gap_platform_length_min = int(gap_platform_length_min / terrain.horizontal_scale)
    gap_platform_length_max = int(gap_platform_length_max / terrain.horizontal_scale)

    gap_length = int(gap_length / terrain.horizontal_scale)
    gap_platform_height = int(gap_platform_height / terrain.vertical_scale)

    # add gap
    start_gap = platform_length
    while start_gap + gap_length <= terrain.length - platform_length//2:
        gap_platform_length = np.random.randint(gap_platform_length_min, gap_platform_length_max)
        terrain.height_field_raw[ start_gap:start_gap+gap_length,:] = lava_depth
        # randomize gap platform height
        if start_gap + gap_length + gap_platform_length <= terrain.length - platform_length //2:
            #terrain.height_field_raw[:, start_gap+gap_length:start_gap+gap_length+gap_platform_length] = np.random.randint(-gap_platform_height, gap_platform_height)
            terrain.height_field_raw[start_gap+gap_length:start_gap+gap_length+gap_platform_length:] = -gap_platform_height

        start_gap += gap_length + gap_platform_length

    # the floor is lava
    terrain.height_field_raw[ 0:terrain.length,0:lava_width] = lava_depth
    terrain.height_field_raw[0:terrain.length,-lava_width: ] = lava_depth

def jump_parkour(terrain, platform_length=1.25, lava_width=0.5, lava_depth=-1.0, height=0.5, height_platform_length=1.5):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height_platform_length = int(height_platform_length / terrain.horizontal_scale)
    height = int(height / terrain.vertical_scale)

    # Version with 2 jumps
    #terrain.height_field_raw[:, platform_length:platform_length+3*height_platform_length] = height
    #terrain.height_field_raw[:, platform_length+height_platform_length:platform_length+2*height_platform_length] = 2*height

    # Version with 3 jumps
    terrain.height_field_raw[ 1*platform_length:6*platform_length,:] = 1*height
    terrain.height_field_raw[ 2*platform_length:5*platform_length,:] = 2*height
    terrain.height_field_raw[3*platform_length:4*platform_length,:] = 3*height
 
    # the floor is lava
    terrain.height_field_raw[0:terrain.length,0:lava_width] = lava_depth
    terrain.height_field_raw[ 0:terrain.length,-lava_width:] = lava_depth

def stairs_parkour(terrain, platform_length=1., lava_width=0.5, lava_depth=-1.0, height=0.18, width=0.3, stairs_platform_length=1.25):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    stairs_platform_length = int(stairs_platform_length / terrain.horizontal_scale)
    height = int(height / terrain.vertical_scale)
    width = int(width / terrain.horizontal_scale)

    start = platform_length
    stop = terrain.length - platform_length//2
    curr_height = height
    while stop - start > platform_length:
        terrain.height_field_raw[ start:stop,:] = curr_height
        curr_height += height
        start += width
        stop -= width
    
    # the floor is lava
    terrain.height_field_raw[ 0:terrain.length,0:lava_width] = lava_depth
    terrain.height_field_raw[ 0:terrain.length,-lava_width:] = lava_depth


def hurdle_parkour(terrain, platform_length=1.5, lava_width=0.5, lava_depth=-1.0, height=0.2, width_min=0.3, width_max=0.5):
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height = int(height / terrain.vertical_scale)
    width_min = int(width_min / terrain.horizontal_scale)
    width_max = int(width_max / terrain.horizontal_scale)

    start = platform_length
    width = np.random.randint(width_min, width_max)
    while start + platform_length + width <= terrain.length - platform_length//2:
        terrain.height_field_raw[start:start+width,:] = height
        start += platform_length + width
        width = np.random.randint(width_min, width_max)

    
    # the floor is lava
    terrain.height_field_raw[ 0:terrain.length,0:lava_width] = lava_depth
    terrain.height_field_raw[0:terrain.length,-lava_width:] = lava_depth

def crawl_parkour(terrain, platform_length=2.0, lava_width=0.5, lava_depth=-1.0, height=0.2, depth=1.0, width=3.0, height_step=0.15):
    # First put the barriers
    boxes = []
    boxes += box_trimesh(np.array([depth, width, 0.5]), np.array([2.5, 0.0, height+0.25])),
    boxes += box_trimesh(np.array([depth, width, 0.5]), np.array([6.5, 0.0, height+0.25+height_step])),

    # Then create the heightmap
    platform_length = int(platform_length / terrain.horizontal_scale)
    lava_width = int(lava_width / terrain.horizontal_scale)
    lava_depth = int(lava_depth / terrain.vertical_scale)

    height = int(height / terrain.vertical_scale)
    height_step = int(height_step / terrain.vertical_scale)
    depth = int(depth / terrain.horizontal_scale)
    
    terrain.height_field_raw[:, int(6.0/terrain.horizontal_scale):int(7.0/terrain.horizontal_scale)] = 1*height_step

    # the floor is lava
    terrain.height_field_raw[0:lava_width, 0:terrain.length] = lava_depth
    terrain.height_field_raw[-lava_width:, 0:terrain.length] = lava_depth
    
    return boxes




def box_trimesh(
        size, # float [3] for x, y, z axis length (in meter) under box frame
        center_position, # float [3] position (in meter) in world frame
    ):

    vertices = np.empty((8, 3), dtype= np.float32)
    vertices[:] = center_position
    vertices[[0, 4, 2, 6], 0] -= size[0] / 2
    vertices[[1, 5, 3, 7], 0] += size[0] / 2
    vertices[[0, 1, 2, 3], 1] -= size[1] / 2
    vertices[[4, 5, 6, 7], 1] += size[1] / 2
    vertices[[2, 3, 6, 7], 2] -= size[2] / 2
    vertices[[0, 1, 4, 5], 2] += size[2] / 2

    triangles = -np.ones((12, 3), dtype= np.uint32)
    triangles[0] = [0, 2, 1] #
    triangles[1] = [1, 2, 3]
    triangles[2] = [0, 4, 2] #
    triangles[3] = [2, 4, 6]
    triangles[4] = [4, 5, 6] #
    triangles[5] = [5, 7, 6]
    triangles[6] = [1, 3, 5] #
    triangles[7] = [3, 7, 5]
    triangles[8] = [0, 1, 4] #
    triangles[9] = [1, 5, 4]
    triangles[10]= [2, 6, 3] #
    triangles[11]= [3, 6, 7]

    return vertices, triangles

def combine_trimeshes(*trimeshes):
    if len(trimeshes) > 2:
        return combine_trimeshes(
            trimeshes[0],
            combine_trimeshes(trimeshes[1:])
        )

    # only two trimesh to combine
    trimesh_0, trimesh_1 = trimeshes
    if trimesh_0[1].shape[0] < trimesh_1[1].shape[0]:
        trimesh_0, trimesh_1 = trimesh_1, trimesh_0
    
    trimesh_1 = (trimesh_1[0], trimesh_1[1] + trimesh_0[0].shape[0])
    vertices = np.concatenate((trimesh_0[0], trimesh_1[0]), axis= 0)
    triangles = np.concatenate((trimesh_0[1], trimesh_1[1]), axis= 0)

    return vertices, triangles

def move_trimesh(trimesh, move: np.ndarray):
    trimesh = list(trimesh)
    trimesh[0] += move
    return tuple(trimesh)


