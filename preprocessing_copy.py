import os, sys
import pymesh
import numpy as np
import random
import math
import multiprocessing as mp
from pathlib import Path
from pytransform3d.rotations import axis_angle_from_matrix, matrix_from_two_vectors, extrinsic_euler_xzy_from_active_matrix
from pytransform3d.transformations import transform, transform_from

NUM_THREADS = 1

class Preprocessing():
    def __init__(self, obj_folder_path, color_file, target_mesh_folder_path, render_folder_path):
        self.obj_folder_path = obj_folder_path
        self.color_file = color_file
        self.color_list = np.load(self.color_file)
        self.target_mesh_folder_path = target_mesh_folder_path
        self.render_folder_path = render_folder_path   
        self.colorized_vertices = set() # seeds that are already Colorized
        self.red_vertices = set()
        self.green_vertices = set()
        self.blue_vertices = set()
        self.half_R_vertices = set()
        self.half_G_vertices = set()
        self.half_B_vertices = set()
        self.RG_vertices = set()
        self.GB_vertices = set()
        self.RB_vertices = set()
        self.half_RB_vertices = set()
        self.half_RG_vertices = set()
        self.half_GB_vertices = set()

    def parallel_worker(self):
        # the main preprocessing functions
        # this allow the script to perform multi-processing on dataset, speed up!
        # remember to change num_of_threads for your computer

        self.pool = mp.Pool(NUM_THREADS)

        # Colorized Original OBJ Files
        original_obj_file_path = self.find_files(self.obj_folder_path, '.obj')
        self.pool.map(self.process_mesh, original_obj_file_path)

        # Perform Rendering
        target_mesh_path = self.find_files(self.target_mesh_folder_path, '.ply')
        self.pool.map(self.image_rendering, target_mesh_path)

    def process_mesh(self, obj_file_path):
        print(f'##### PROCRESS MESH FILE ##### Progress: {obj_file_path}')
        # load mesh file
        # mesh = pymesh.load_mesh(obj_file_path)
        # separate the mesh
        # mesh = pymesh.separate_mesh(mesh)[5]
        # get length of the mesh vertices
        # num_vertices = len(mesh.vertices)
        # add vertex color
        # mesh.add_attribute(u'red')
        # mesh.set_attribute(u'red', np.array([int(255*x) for x in self.color_list[:num_vertices, 0]]))
        # mesh.add_attribute(u'green')
        # mesh.set_attribute(u'green', np.array([int(255*x) for x in self.color_list[:num_vertices, 1]]))
        # mesh.add_attribute(u'blue')
        # mesh.set_attribute(u'blue', np.array([int(255*x) for x in self.color_list[:num_vertices, 2]]))
        # # generate output path
        # obj_name = os.path.splitext(Path(obj_file_path).name)[0]
        # output_path = os.path.abspath(os.path.join(self.target_mesh_folder_path, obj_name + ".ply"))
        # # save the mesh file
        # pymesh.save_mesh(output_path, mesh, *mesh.get_attribute_names(), ascii=True)
        self.colorize_whole(obj_file_path, self.target_mesh_folder_path)
        

    def image_rendering(self, target_mesh_path):
        print(f'##### RENDERING ##### Progress: {target_mesh_path}')
        mesh = pymesh.load_mesh(target_mesh_path)
        vertices_index = [0, 19, 5] #mesh.faces[0]
        vertices_coordinates = mesh.vertices[vertices_index]
        #print(vertices_coordinates)
        # the camera location is mean of three selected vertices
        camera_coordinates = np.mean(vertices_coordinates, axis=0)
        print(f'Camera: {camera_coordinates}')
        # first vector OA connects the first vertex to the center(origin)
        # second vector OB connects the second vertex to the center(origin)
        # third vector OC connects the third vertex to the center(origin)
        OA = vertices_coordinates[0] - camera_coordinates
        OB = vertices_coordinates[1] - camera_coordinates
        OC = vertices_coordinates[2] - camera_coordinates
        # calculate the normal vector to this surface
        # ref: https://www.khronos.org/opengl/wiki/Calculating_a_Surface_Normal
        U = vertices_coordinates[1] - vertices_coordinates[0]
        V = vertices_coordinates[2] - vertices_coordinates[1]
        Nx = U[1]*V[2] - U[2]*V[1]
        Ny = U[2]*V[0] - U[0]*V[2]
        Nz = U[0]*V[1] - U[1]*V[0]
        normal_vec = self.get_unit_vec([Nx, Ny, Nz])
        #print(f'Normal: {normal_vec}')
        # three unit vectors for the new coordinate
        OY = self.get_unit_vec(OC)
        OZ = self.get_unit_vec(normal_vec)
        OX = self.get_unit_vec((np.cross(OY, OZ)))

        A = np.empty((3, 3))
        A[:, 0] = OX; A[:, 1] = OY; A[:, 2] = OZ

        # calculate the transformation matrix
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        R = A@np.linalg.inv(B)
        x=math.atan2(R[2,1],R[2,2])
        y=math.atan2(-R[2,0],(R[2,1]**2+R[2,2]**2)**0.5)
        z=math.atan2(R[1,0],R[0,0])
        
        euler_angle = [x, y ,z]
        print(f'Euler Angle: {euler_angle}')

        # generate command for blender rendering
        ply_name = os.path.splitext(Path(target_mesh_path).name)[0]
        output_path = os.path.abspath(os.path.join(self.render_folder_path, ply_name + ".png"))
        bash_command = "blender blender_project.blend -b --python blender_render.py -- " + target_mesh_path + " " + output_path + " " + \
            str(camera_coordinates[0]) + " " + str(camera_coordinates[1]) + " " + str(camera_coordinates[2]) + " " + \
                str(euler_angle[0]) + " " + str(euler_angle[1]) + " " + str(euler_angle[2])
        os.system(bash_command)

    def generate_color_file(self, num_sample):
        # call this function to generate color_list file if not exist
        color_list = np.empty([num_sample, 3])
        for i in range(num_sample):
            color_list[i, :] = [random.random(), random.random(), random.random()]

        np.save('./data/color_list.npy', color_list)
    
    def get_unit_vec(self, vector):
        return vector/np.linalg.norm(vector)
        
    def find_files(self, folder_path, extension):
        print(f'\n##### LOCATING {extension} FILES #####')
        target_file_path = []
        # loop through the folder
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith(extension):
                    target_file_path.append(os.path.abspath(os.path.join(root, filename)))

        print(f'Found {len(target_file_path)} {extension} files in folder {folder_path}.')

        return target_file_path

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def colorize_whole(self, obj_file_path, target_mesh_folder_path, npSeed = 0, size=96, step=10):
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
                if(faces[seed[j]][m] in self.colorized_vertices):
                    continue                   
                # TODO: color the vertices (faces[seed[j]][m])
                
                # TODO: color the vertices
                if(j % 12 == 0):
                    self.red_vertices.add(faces[seed[j]][m])
                    red_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 1):
                    self.green_vertices.add(faces[seed[j]][m])
                    green_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 2):
                    self.blue_vertices.add(faces[seed[j]][m])
                    blue_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 3):
                    self.RG_vertices.add(faces[seed[j]][m])
                    RG_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 4):
                    self.GB_vertices.add(faces[seed[j]][m])
                    GB_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 5):
                    self.RB_vertices.add(faces[seed[j]][m])
                    RB_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 6):
                    self.half_RB_vertices.add(faces[seed[j]][m])
                    half_RB_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 7):
                    self.half_RG_vertices.add(faces[seed[j]][m])
                    half_RG_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 8):
                    self.half_GB_vertices.add(faces[seed[j]][m])
                    half_GB_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 9):
                    self.half_R_vertices.add(faces[seed[j]][m])
                    half_R_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 10):
                    self.half_B_vertices.add(faces[seed[j]][m])
                    half_B_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))
                elif(j % 12 == 11):
                    self.half_G_vertices.add(faces[seed[j]][m])
                    half_G_neighbours.update(self.neighbour_vertice(faces[seed[j]][m], faces))

                self.colorized_vertices.add(faces[seed[j]][m])

        for i in range(step): # the step to propagate
            # propagate the red ones
            cur_red_neighbours = red_neighbours.copy()
            for red_neighbour in cur_red_neighbours:
                if red_neighbour not in self.red_vertices:
                    # 
                    if red_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices red_neighbour

                        # TODO: color the vertices
                        self.red_vertices.add(red_neighbour)
                        self.colorized_vertices.add(red_neighbour)
                        red_neighbours.update(self.neighbour_vertice(red_neighbour, faces))
                red_neighbours.remove(red_neighbour)
            # propagate the green ones
            cur_green_neighbours = green_neighbours.copy()
            for green_neighbour in cur_green_neighbours:
                if green_neighbour not in self.green_vertices:
                    if green_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices green_neighbour

                        # TODO: color the vertices
                        self.green_vertices.add(green_neighbour)
                        green_neighbours.update(self.neighbour_vertice(green_neighbour, faces))
                        self.colorized_vertices.add(green_neighbour)
                green_neighbours.remove(green_neighbour)
            # propagate the blue ones
            cur_blue_neighbours = blue_neighbours.copy()
            for blue_neighbour in cur_blue_neighbours:
                if blue_neighbour not in self.blue_vertices:
                    if blue_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.blue_vertices.add(blue_neighbour)
                        blue_neighbours.update(self.neighbour_vertice(blue_neighbour, faces))
                        self.colorized_vertices.add(blue_neighbour)
                blue_neighbours.remove(blue_neighbour)

            cur_half_R_neighbours = half_R_neighbours.copy()
            for half_R_neighbour in cur_half_R_neighbours:
                if half_R_neighbour not in self.half_R_vertices:
                    if half_R_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.half_R_vertices.add(half_R_neighbour)
                        half_R_neighbours.update(self.neighbour_vertice(half_R_neighbour, faces))
                        self.colorized_vertices.add(half_R_neighbour)
                half_R_neighbours.remove(half_R_neighbour)

            cur_half_G_neighbours = half_G_neighbours.copy()
            for half_G_neighbour in cur_half_G_neighbours:
                if half_G_neighbour not in self.half_G_vertices:
                    if half_G_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.half_G_vertices.add(half_G_neighbour)
                        half_G_neighbours.update(self.neighbour_vertice(half_G_neighbour, faces))
                        self.colorized_vertices.add(half_G_neighbour)
                half_G_neighbours.remove(half_G_neighbour)
            
            cur_half_B_neighbours = half_B_neighbours.copy()
            for half_B_neighbour in cur_half_B_neighbours:
                if half_B_neighbour not in self.half_B_vertices:
                    if half_B_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.half_B_vertices.add(half_B_neighbour)
                        half_B_neighbours.update(self.neighbour_vertice(half_B_neighbour, faces))
                        self.colorized_vertices.add(half_B_neighbour)
                half_B_neighbours.remove(half_B_neighbour)

            cur_half_RG_neighbours = half_RG_neighbours.copy()
            for half_RG_neighbour in cur_half_RG_neighbours:
                if half_RG_neighbour not in self.half_RG_vertices:
                    if half_RG_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.half_RG_vertices.add(half_RG_neighbour)
                        half_RG_neighbours.update(self.neighbour_vertice(half_RG_neighbour, faces))
                        self.colorized_vertices.add(half_RG_neighbour)
                half_RG_neighbours.remove(half_RG_neighbour)

            cur_half_RB_neighbours = half_RB_neighbours.copy()
            for half_RB_neighbour in cur_half_RB_neighbours:
                if half_RB_neighbour not in self.half_RB_vertices:
                    if half_RB_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.half_RB_vertices.add(half_RB_neighbour)
                        half_RB_neighbours.update(self.neighbour_vertice(half_RB_neighbour, faces))
                        self.colorized_vertices.add(half_RB_neighbour)
                half_RB_neighbours.remove(half_RB_neighbour)

            cur_RG_neighbours = RG_neighbours.copy()
            for RG_neighbour in cur_RG_neighbours:
                if RG_neighbour not in self.RG_vertices:
                    if RG_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.RG_vertices.add(RG_neighbour)
                        RG_neighbours.update(self.neighbour_vertice(RG_neighbour, faces))
                        self.colorized_vertices.add(RG_neighbour)
                RG_neighbours.remove(RG_neighbour)

            cur_RB_neighbours = RB_neighbours.copy()
            for RB_neighbour in cur_RB_neighbours:
                if RB_neighbour not in self.RB_vertices:
                    if RB_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.RB_vertices.add(RB_neighbour)
                        RB_neighbours.update(self.neighbour_vertice(RB_neighbour, faces))
                        self.colorized_vertices.add(RB_neighbour)
                RB_neighbours.remove(RB_neighbour)

            cur_GB_neighbours = GB_neighbours.copy()
            for GB_neighbour in cur_GB_neighbours:
                if GB_neighbour not in self.GB_vertices:
                    if GB_neighbour not in self.colorized_vertices: # Maybe it's colored, but in a different color
                        # TODO: color the vertices blue_neighbour

                        # TODO: color the vertices
                        self.GB_vertices.add(GB_neighbour)
                        GB_neighbours.update(self.neighbour_vertice(GB_neighbour, faces))
                        self.colorized_vertices.add(GB_neighbour)
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
            if(i in self.red_vertices):
                color_list[i][1] -= 255
                color_list[i][2] -= 255
            elif(i in self.green_vertices):
                color_list[i][0] -= 255
                color_list[i][2] -= 255
            elif(i in self.blue_vertices):
                color_list[i][0] -= 255
                color_list[i][1] -= 255
            elif(i in self.half_R_vertices):
                color_list[i][0] -= 127
                color_list[i][1] -= 255
                color_list[i][2] -= 255
            elif(i in self.half_G_vertices):
                color_list[i][0] -= 255
                color_list[i][1] -= 127
                color_list[i][2] -= 255
            elif(i in self.half_B_vertices):
                color_list[i][0] -= 255
                color_list[i][1] -= 255
                color_list[i][2] -= 127
            elif(i in self.half_RG_vertices):
                color_list[i][0] -= 127
                color_list[i][1] -= 127
                color_list[i][2] -= 255
            elif(i in self.half_RB_vertices):
                color_list[i][0] -= 127
                color_list[i][1] -= 255
                color_list[i][2] -= 127
            elif(i in self.half_GB_vertices):
                color_list[i][0] -= 255
                color_list[i][1] -= 127
                color_list[i][2] -= 127
            elif(i in self.RG_vertices):
                color_list[i][2] -= 255
            elif(i in self.RB_vertices):
                color_list[i][1] -= 255
            elif(i in self.GB_vertices):
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
            
        # print(set(ans))

if __name__ == '__main__':
    # change working directory
    os.chdir(sys.path[0])
    # create class instance
    for i in range(1, 2):
        print(f'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ {i} $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        PP = Preprocessing(obj_folder_path=f'../data/scene/scene_{i}/original_obj',
                            color_file = '../data/color_list.npy',
                            target_mesh_folder_path=f'../data/scene/scene_{i}/target_mesh',
                            render_folder_path=f'../data/scene/scene_{i}/render')

        PP.parallel_worker()
