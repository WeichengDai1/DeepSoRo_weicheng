import numpy as np
import pymesh
import os
from pathlib import Path
# color_file = '../data/color_list.npy'
# color_list = np.load(color_file)
# print(color_list.shape)
# print(color_list)
obj_file_path='./data/scene/scene_2/original_obj/0.obj'
# mesh = pymesh.load_mesh(obj_file_path)
# SIZE = 6
# STEP = 10
# npSeed = 0
# faces = mesh.faces # (2614, 3)
# num_faces = len(faces) # 2614
# vertices = mesh.vertices # (1319*3d)
# num_vertices = len(mesh.vertices) # 1319

colorized_vertices = set() # seeds that are already Colorized
red_vertices = set()
green_vertices = set()
blue_vertices = set()
half_R_vertices = set()
half_G_vertices = set()
half_B_vertices = set()
RG_vertices = set()
GB_vertices = set()
RB_vertices = set()
half_RB_vertices = set()
half_RG_vertices = set()
half_GB_vertices = set()

# neighbours = [] 

def colorize_whole(obj_file_path, target_mesh_folder_path, npSeed = 0, size=96, step=10):
    mesh = pymesh.load_mesh(obj_file_path)
    mesh = pymesh.separate_mesh(mesh)[5]
    # f = open(obj_file_path, 'w')
    faces = mesh.faces
    num_faces = len(faces)
    vertices = mesh.vertices
    num_vertices = len(vertices)
    # the numpy seed to control the random number generator
    np.random.seed(npSeed)
    seed = np.random.randint(low = 0, high = num_faces, size = size) # maybe [1, 2, 3, 4, 5, 6]
    red_neighbours = set()
    green_neighbours = set()
    blue_neighbours = set()
    half_R_neighbours = set()
    half_G_neighbours = set()
    half_B_neighbours = set()
    RG_neighbours = set()
    GB_neighbours = set()
    RB_neighbours = set()
    half_RB_neighbours = set()
    half_RG_neighbours = set()
    half_GB_neighbours = set()

    mesh.add_attribute('red')
    mesh.add_attribute('green')
    mesh.add_attribute('blue')

    # initialize the first vertices
    for j in range(len(seed)): # each j denotes a face, with 3 vertices
        for m in range(3): # 3 vertices
            if(faces[seed[j]][m] in colorized_vertices):
                continue                   
            # TODO: color the vertices (faces[seed[j]][m])
            
            # TODO: color the vertices
            if(j % 12 == 0):
                red_vertices.add(faces[seed[j]][m])
                red_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 1):
                green_vertices.add(faces[seed[j]][m])
                green_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 2):
                blue_vertices.add(faces[seed[j]][m])
                blue_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 3):
                RG_vertices.add(faces[seed[j]][m])
                RG_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 4):
                GB_vertices.add(faces[seed[j]][m])
                GB_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 5):
                RB_vertices.add(faces[seed[j]][m])
                RB_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 6):
                half_RB_vertices.add(faces[seed[j]][m])
                half_RB_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 7):
                half_RG_vertices.add(faces[seed[j]][m])
                half_RG_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 8):
                half_GB_vertices.add(faces[seed[j]][m])
                half_GB_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 9):
                half_R_vertices.add(faces[seed[j]][m])
                half_R_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 10):
                half_B_vertices.add(faces[seed[j]][m])
                half_B_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))
            elif(j % 12 == 11):
                half_G_vertices.add(faces[seed[j]][m])
                half_G_neighbours.update(neighbour_vertice(faces[seed[j]][m], faces))

            colorized_vertices.add(faces[seed[j]][m])

    for i in range(step): # the step to propagate
        # propagate the red ones
        cur_red_neighbours = red_neighbours.copy()
        for red_neighbour in cur_red_neighbours:
            if red_neighbour not in red_vertices:
                # 
                if red_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices red_neighbour

                    # TODO: color the vertices
                    red_vertices.add(red_neighbour)
                    colorized_vertices.add(red_neighbour)
                    red_neighbours.update(neighbour_vertice(red_neighbour, faces))
            red_neighbours.remove(red_neighbour)
        # propagate the green ones
        cur_green_neighbours = green_neighbours.copy()
        for green_neighbour in cur_green_neighbours:
            if green_neighbour not in green_vertices:
                if green_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices green_neighbour

                    # TODO: color the vertices
                    green_vertices.add(green_neighbour)
                    green_neighbours.update(neighbour_vertice(green_neighbour, faces))
                    colorized_vertices.add(green_neighbour)
            green_neighbours.remove(green_neighbour)
        # propagate the blue ones
        cur_blue_neighbours = blue_neighbours.copy()
        for blue_neighbour in cur_blue_neighbours:
            if blue_neighbour not in blue_vertices:
                if blue_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    blue_vertices.add(blue_neighbour)
                    blue_neighbours.update(neighbour_vertice(blue_neighbour, faces))
                    colorized_vertices.add(blue_neighbour)
            blue_neighbours.remove(blue_neighbour)

        cur_half_R_neighbours = half_R_neighbours.copy()
        for half_R_neighbour in cur_half_R_neighbours:
            if half_R_neighbour not in half_R_vertices:
                if half_R_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    half_R_vertices.add(half_R_neighbour)
                    half_R_neighbours.update(neighbour_vertice(half_R_neighbour, faces))
                    colorized_vertices.add(half_R_neighbour)
            half_R_neighbours.remove(half_R_neighbour)

        cur_half_G_neighbours = half_G_neighbours.copy()
        for half_G_neighbour in cur_half_G_neighbours:
            if half_G_neighbour not in half_G_vertices:
                if half_G_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    half_G_vertices.add(half_G_neighbour)
                    half_G_neighbours.update(neighbour_vertice(half_G_neighbour, faces))
                    colorized_vertices.add(half_G_neighbour)
            half_G_neighbours.remove(half_G_neighbour)
        
        cur_half_B_neighbours = half_B_neighbours.copy()
        for half_B_neighbour in cur_half_B_neighbours:
            if half_B_neighbour not in half_B_vertices:
                if half_B_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    half_B_vertices.add(half_B_neighbour)
                    half_B_neighbours.update(neighbour_vertice(half_B_neighbour, faces))
                    colorized_vertices.add(half_B_neighbour)
            half_B_neighbours.remove(half_B_neighbour)

        cur_half_RG_neighbours = half_RG_neighbours.copy()
        for half_RG_neighbour in cur_half_RG_neighbours:
            if half_RG_neighbour not in half_RG_vertices:
                if half_RG_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    half_RG_vertices.add(half_RG_neighbour)
                    half_RG_neighbours.update(neighbour_vertice(half_RG_neighbour, faces))
                    colorized_vertices.add(half_RG_neighbour)
            half_RG_neighbours.remove(half_RG_neighbour)

        cur_half_RB_neighbours = half_RB_neighbours.copy()
        for half_RB_neighbour in cur_half_RB_neighbours:
            if half_RB_neighbour not in half_RB_vertices:
                if half_RB_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    half_RB_vertices.add(half_RB_neighbour)
                    half_RB_neighbours.update(neighbour_vertice(half_RB_neighbour, faces))
                    colorized_vertices.add(half_RB_neighbour)
            half_RB_neighbours.remove(half_RB_neighbour)

        cur_RG_neighbours = RG_neighbours.copy()
        for RG_neighbour in cur_RG_neighbours:
            if RG_neighbour not in RG_vertices:
                if RG_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    RG_vertices.add(RG_neighbour)
                    RG_neighbours.update(neighbour_vertice(RG_neighbour, faces))
                    colorized_vertices.add(RG_neighbour)
            RG_neighbours.remove(RG_neighbour)

        cur_RB_neighbours = RB_neighbours.copy()
        for RB_neighbour in cur_RB_neighbours:
            if RB_neighbour not in RB_vertices:
                if RB_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    RB_vertices.add(RB_neighbour)
                    RB_neighbours.update(neighbour_vertice(RB_neighbour, faces))
                    colorized_vertices.add(RB_neighbour)
            RB_neighbours.remove(RB_neighbour)

        cur_GB_neighbours = GB_neighbours.copy()
        for GB_neighbour in cur_GB_neighbours:
            if GB_neighbour not in GB_vertices:
                if GB_neighbour not in colorized_vertices: # Maybe it's colored, but in a different color
                    # TODO: color the vertices blue_neighbour

                    # TODO: color the vertices
                    GB_vertices.add(GB_neighbour)
                    GB_neighbours.update(neighbour_vertice(GB_neighbour, faces))
                    colorized_vertices.add(GB_neighbour)
            GB_neighbours.remove(GB_neighbour)

    # color the rest of the vertices
    # print(colorized_vertices)
    
    # for vertice in vertices:
    #     # print(vertice)
    #     row = np.where(vertice == vertices)
    #     print(row)
    #     if row not in colorized_vertices:
    #         # TODO: color the vertices
    #         yellow_vertices.add(row)
    #         # TODO: color the vertices
    #         colorized_vertices.add(row)


    # we have rgb & yellow vertices and now we can colorize the whole 
    color_list = np.ones((num_vertices, 3), dtype = np.int8)*255
    for i in range(num_vertices):
        if(i in red_vertices):
            color_list[i][1] -= 255
            color_list[i][2] -= 255
        elif(i in green_vertices):
            color_list[i][0] -= 255
            color_list[i][2] -= 255
        elif(i in blue_vertices):
            color_list[i][0] -= 255
            color_list[i][1] -= 255
        elif(i in half_R_vertices):
            color_list[i][0] -= 127
            color_list[i][1] -= 255
            color_list[i][2] -= 255
        elif(i in half_G_vertices):
            color_list[i][0] -= 255
            color_list[i][1] -= 127
            color_list[i][2] -= 255
        elif(i in half_B_vertices):
            color_list[i][0] -= 255
            color_list[i][1] -= 255
            color_list[i][2] -= 127
        elif(i in half_RG_vertices):
            color_list[i][0] -= 127
            color_list[i][1] -= 127
            color_list[i][2] -= 255
        elif(i in half_RB_vertices):
            color_list[i][0] -= 127
            color_list[i][1] -= 255
            color_list[i][2] -= 127
        elif(i in half_GB_vertices):
            color_list[i][0] -= 255
            color_list[i][1] -= 127
            color_list[i][2] -= 127
        elif(i in RG_vertices):
            color_list[i][2] -= 255
        elif(i in RB_vertices):
            color_list[i][1] -= 255
        elif(i in GB_vertices):
            color_list[i][0] -= 255

    # mesh.add_attribute(u'red')
    # mesh.add_attribute(u'green')
    # mesh.add_attribute(u'blue')
    # print(color_list)
    mesh.set_attribute('red', np.array([int(x) for x in color_list[:num_vertices, 0]]))
    mesh.set_attribute('green', np.array([int(x) for x in color_list[:num_vertices, 1]]))
    mesh.set_attribute('blue', np.array([int(x) for x in color_list[:num_vertices, 2]]))

    obj_name = os.path.splitext(Path(obj_file_path).name)[0]
    output_path = os.path.abspath(os.path.join(target_mesh_folder_path, obj_name + ".ply"))
    pymesh.save_mesh(output_path, mesh, *mesh.get_attribute_names(), ascii=True)


def neighbour_vertice(vertice, faces):
    ans = []
    vertice_exist = np.argwhere(faces==vertice)
    for vertice_index in vertice_exist:
        row = vertice_index[0]
        col = vertice_index[1]
        if(col==0):
            ans.append(faces[row][1])
            ans.append(faces[row][2])
        elif(col==1):
            ans.append(faces[row][0])
            ans.append(faces[row][2])
        elif(col==2):
            ans.append(faces[row][0])
            ans.append(faces[row][1])

    return(set(ans))
        
    # print(set(ans))



if __name__ == '__main__':
    obj_file_path='./data/scene/scene_2/original_obj/0.obj'
    mesh = pymesh.load_mesh(obj_file_path)
    faces = mesh.faces # (2614, 3)
    # print(faces[0][1])
    colorize_whole(obj_file_path, './')
    # neighbour_vertice(1, faces)
