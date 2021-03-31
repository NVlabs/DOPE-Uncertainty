"""
This file starts a ROS node to run DOPE, 
listening to an image topic and publishing poses.
"""

from __future__ import print_function

import cv2
import numpy as np
import visii
from PIL import Image
from PIL import ImageDraw

import sys 
sys.path.append("../src/dope/inference")
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from detector import ModelData, ObjectDetector

import simplejson as json
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from scipy import stats

from add_simple import get_add, create_obj
from fitting import GP, LR

quat_trans_dope_to_visii = Quaternion(w=0.707388, x=-0.000000, y=-0.000000, z=-0.706825) #WXYZ
'''
def draw_uncertainty(score, uncertainty, path):
    my_dpi = 100
    if len(uncertainty) > 0:
        fig, axs = plt.subplots(len(uncertainty), 1, figsize=(200/my_dpi, 400/my_dpi), dpi=my_dpi)
        labels = ('c', 'x', 'y', 'z')
        for index in range(len(uncertainty)):
            value = np.array([score[index], uncertainty[index][0], uncertainty[index][1], uncertainty[index][2]])
            if len(uncertainty) > 1:
                ax = axs[index]
            else:
                ax = axs
            ax.barh(np.arange(1,4), value[1:], align='center')
            ax.set_yticks(np.arange(4))
            ax.set_yticklabels(labels)
            ax2 = ax.twiny()
            ax2.barh(0, value[0], align='center', color='red')
            ax.set_xlim([0, 50])
            ax2.set_xlim([0, 0.5])
            ax.invert_yaxis()
            ax.set_xlabel('#'+str(index+1))
    else:
        plt.figure(figsize=(200/my_dpi, 400/my_dpi), dpi=my_dpi)    
    plt.tight_layout() 
    plt.savefig(path, dpi=my_dpi)
    plt.close()
'''

def draw_uncertainty_multi(id, Metric, path, colors):
    my_dpi = 100
    label = []
    L = len(Metric)
    score = []
    uncertainty = []
    color = []
    for name in Metric:
        label.append(name[:3]) # for a short name
        score.append(Metric[name]['centroid score'])
        uncertainty.append(sum(Metric[name]['uncertainty'][:3]))
        color.append(tuple(t/255 for t in colors[name]))

    fig, axs = plt.subplots(2, 1, figsize=(256/my_dpi, 512/my_dpi), dpi=my_dpi)
    axs[0].barh(np.arange(L), score, color=color, align='center')
    axs[0].set_yticks(np.arange(L))
    axs[0].set_yticklabels(label)
    axs[0].set_xlim([0, 0.5])
    axs[0].set_xlabel('c score')
    axs[0].set_title('#'+id)
    axs[1].barh(np.arange(L), uncertainty, color=color, align='center')
    axs[1].set_yticks(np.arange(L))
    axs[1].set_yticklabels(label)
    axs[1].set_xlim([0, 0.6])
    axs[1].set_xlabel('pos std (m)')
    plt.tight_layout() 
    plt.savefig(path, dpi=my_dpi)
    plt.close()

class Draw(object):
    """Drawing helper class to visualize the neural network output"""

    def __init__(self, im):
        """
        :param im: The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """Draws line on image"""
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius=2, text=None):
        """Draws dot (filled circle) on image"""
        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )
        if not text is None:
            self.draw.text((point[0]+4, point[1]), text)

    def draw_cube(self, points, color=(255, 0, 0), text='1'):
        """
        Draws cube with a thick solid line across
        the front top edge and an X on the top face.
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)

        x_max_index = 0
        for i in range(8):
            if points[i][0] > points[x_max_index][0]:
                x_max_index = i

        #self.draw.text((points[x_max_index][0]+4, points[x_max_index][1]), text)

class DopeNode(object):
    """ROS node that listens to image topic, runs DOPE, and publishes DOPE results"""
    def __init__(self,
            config, # config yaml loaded eg dict
        ):
        self.pubs = {}
        self.models = {}
        self.pnp_solvers = {}
        self.pub_dimension = {}
        self.draw_colors = {}
        self.dimensions = {}
        self.class_ids = {}
        self.model_transforms = {}
        self.meshes = {}
        self.mesh_scales = {}

        self.input_is_rectified = config['input_is_rectified']
        self.downscale_height = config['downscale_height']

        try:
            self.padding_width = config['padding_added_width']
            self.padding_height = config['padding_added_height']
        except:
            self.padding_width = 0
            self.padding_height = 0


        self.config_detect = lambda: None
        self.config_detect.mask_edges = 1
        self.config_detect.mask_faces = 1
        self.config_detect.vertex = 1
        self.config_detect.threshold = 0.5
        self.config_detect.softmax = 1000
        self.config_detect.thresh_angle = config['thresh_angle']
        self.config_detect.thresh_map = config['thresh_map']
        self.config_detect.sigma = config['sigma']
        self.config_detect.thresh_points = config["thresh_points"]
        self.config_detect.run_sampling = config["run_sampling"]
        
        # For each object to detect, load network model, create PNP solver, and start ROS publishers
        print(config['weights'])
        for model in config['weights']:
            print(model)
            self.models[model] = \
                ModelData(
                    model,
                    config['weights'][model]
                )
            self.models[model].load_net_model()
            print('loaded')

            try:
                self.draw_colors[model] = tuple(config["draw_colors"][model])
            except:
                self.draw_colors[model] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            self.dimensions[model] = tuple(config["dimensions"][opt.object])
            self.class_ids[model] = config["class_ids"][opt.object]

            self.pnp_solvers[model] = \
                CuboidPNPSolver(
                    model,
                    cuboid3d=Cuboid3d(config['dimensions'][opt.object])
                )

        # Ensemble the keypoint level
        if config['keypoints_ensemble']:
            self.draw_colors[opt.object+'_ensemble'] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
            self.m1 = config['models'][0]
            self.m2 = config['models'][1]
            self.keypoints_ensemble_1 = self.models[self.m1]
            self.keypoints_ensemble_2 = self.models[self.m2]
            self.models[opt.object+'_ensemble'] = 1

        # print("Running DOPE...  (Listening to camera topic: '{}')".format(config['~topic_camera')))
        print("Ctrl-C to stop")

    def image_callback(self, 
        img, 
        camera_info, 
        P_matrix,
        img_name = "00000.png", # this is the name of the img file to save, it needs the .png at the end
        output_folder = 'out_inference', # folder where to put the output
        showbelief = False,
        gt_keypoints = None,
        ):
        # img_name = str(img_name).zfill(5)+'.png'
        """Image callback"""

        # img = self.cv_bridge.imgmsg_to_cv2(image_msg, "rgb8")

        # cv2.imwrite('img.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # for debugging

        # Update camera matrix and distortion coefficients
        if self.input_is_rectified:
            '''
            P = np.matrix(camera_info['camera_matrix']['data'], dtype='float64').copy()
            # print(P)
            P.resize((3, 3))
            '''
            P = np.matrix(P_matrix, dtype='float64').copy()
            P.resize((3, 3))
            camera_matrix = P
            dist_coeffs = np.zeros((4, 1))
            # print(camera_matrix)
            # raise()
        else:
            # TODO
            camera_matrix = np.matrix(camera_info.K, dtype='float64')
            camera_matrix.resize((3, 3))
            dist_coeffs = np.matrix(camera_info.D, dtype='float64')
            dist_coeffs.resize((len(camera_info.D), 1))

        # add padding to the image 
        img = cv2.copyMakeBorder( img, 0, self.padding_height, 0, self.padding_width, cv2.BORDER_CONSTANT,(0,0,0))
        # Downscale image if necessary
        height, width, _ = img.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0:
            camera_matrix[:2] *= scaling_factor
            img = cv2.resize(img, (int(scaling_factor * width), int(scaling_factor * height)))

        for m in self.models:
            if 'ensemble' not in m:
                self.pnp_solvers[m].set_camera_intrinsic_matrix(camera_matrix)
                self.pnp_solvers[m].set_dist_coeffs(dist_coeffs)
        
        # Copy and draw image
        img_copy = img.copy()
        im = Image.fromarray(img_copy)
        draw = Draw(im)


        # dictionary for the final output
        dict_out = {"camera_data":{},"objects":[]}

        beliefs_outputs = []

        Metric = {}

        for m in self.models:
            # Detect object
            if 'ensemble' in m:
                if 'full' in self.keypoints_ensemble_1.net_path:
                    full = True
                else:
                    full = False
            else:
                if 'full' in self.models[m].net_path:
                    full = True
                else:
                    full = False
            
            Score = []
            Location = []
            Orientation = []
            Uncertainty = []
            Reprojection_error = []

            if 'visii' in m:
                new = False
                visii_flag = True
            else:
                new = False
                visii_flag = False

            if 'ensemble' in m:
                results, beliefs = ObjectDetector.detect_object_in_image(
                    self.keypoints_ensemble_1.net,
                    self.pnp_solvers[self.m1],
                    img,
                    self.config_detect,
                    full = full,
                    grid_belief_debug = showbelief,
                    run_sampling = self.config_detect.run_sampling,
                    new = new,
                    second_model = self.keypoints_ensemble_2.net,
                    visii = visii_flag
                )
            else:
                results, beliefs = ObjectDetector.detect_object_in_image(
                    self.models[m].net,
                    self.pnp_solvers[m],
                    img,
                    self.config_detect,
                    full = full,
                    grid_belief_debug = showbelief,
                    run_sampling = self.config_detect.run_sampling,
                    new = new,
                    visii = visii_flag
                )                
            beliefs_outputs.append(beliefs)
            # print(results)
            # print('---')
            # continue
            # Publish pose and overlay cube on image

            for i_r, result in enumerate(results):
                if result["location"] is None:
                    continue
                loc = result["location"]
                ori = result["quaternion"]

                # print(loc)
                Score.append(result['score'])
                Location.append(loc)
                Orientation.append(ori)
                Uncertainty.append(result['uncertainty'])

                # Compute the reprojection error
                projected_cuboid = result['projected_points']
                raw_cuboid = result['raw_points']
                reprojection_error = 0
                for i in range(9):
                    temp1 = np.array(projected_cuboid[i])
                    temp2 = np.array(raw_cuboid[i])
                    if raw_cuboid[i] is not None:
                        reprojection_error += np.linalg.norm(temp1 - temp2)

                Reprojection_error.append(reprojection_error)

                dict_out['objects'].append({
                    'class':m,
                    'location':np.array(loc).tolist(),
                    'quaternion_xyzw':np.array(ori).tolist(),
                    'projected_cuboid':np.array(result['projected_points']).tolist(),
                    'confidence':np.array(result['confidence']).tolist(),
                    'raw_cuboid':np.array(result['raw_points']).tolist(),
                    'reprojection_error':reprojection_error.tolist()
                })
                # print( dict_out )

                # transform orientation
                # TODO 
                # transformed_ori = tf.transformations.quaternion_multiply(ori, self.model_transforms[m])

                # rotate bbox dimensions if necessary
                # (this only works properly if model_transform is in 90 degree angles)
                # dims = rotate_vector(vector=self.dimensions[m], quaternion=self.model_transforms[m])
                # dims = np.absolute(dims)
                # dims = tuple(dims)

                # Draw the cube
                if None not in result['projected_points']:
                    points2d = []
                    for pair in result['projected_points']:
                        points2d.append(tuple(pair))
                    draw.draw_cube(points2d, self.draw_colors[m], text=str(i_r+1))
                # draw the raw prediction points 
                text = 0
                for p in result['raw_points']:
                    # draw.draw_dot(p,self.draw_colors[m], text=str(text)) #debug
                    draw.draw_dot(p,self.draw_colors[m])
                    text += 1
                # draw the ground truth keypoints
                if not gt_keypoints is None:
                    text = 0
                    for p in gt_keypoints:
                        draw.draw_dot(p,'black', text=str(text))
                        text += 1

            if len(Score) > 0:
                best_index = 0
                for index in range(len(Score)):
                    # Using score to select the best candidate
                    # if Score[index] > Score[best_index]:
                    # Using reprojection_error to select the best candidate
                    if Reprojection_error[index] < Reprojection_error[best_index]:
                        best_index = index

            Metric[m] = {}
            if len(Score) > 0:
                Metric[m]['centroid score'] = Score[best_index]
                Metric[m]['uncertainty'] = np.array(Uncertainty[best_index])
                Metric[m]['uncertainty'][:3] = 0.01*Metric[m]['uncertainty'][:3] # cm -> m           
                Metric[m]['location'] = 0.01*np.array(Location[best_index]) # cm -> m
                Metric[m]['orientation'] = np.array(Orientation[best_index])
            else:
                Metric[m]['centroid score'] = 0
                Metric[m]['uncertainty'] = np.array([1000,1000,1000,1000,1000,1000,1000])
                Metric[m]['location'] = None
                Metric[m]['orientation'] = None

            # draw the single object uncertainty
            # draw_uncertainty(Score, uncertainty, f"{output_folder}/uncertainty_{img_name}")
        
        draw_uncertainty_multi(img_name.replace('.png',''), Metric, f"{output_folder}/{img_name.replace('.png','.jpg')}", colors=self.draw_colors)

                # print (result)

        # save the output of the image. 
        if not opt.json_only:
            im.save(f"{output_folder}/{img_name}")

        # save the json files 
        path_json = f"{output_folder}/{img_name.replace('png','')}json"

        if os.path.exists(path_json):
            # with open(path_json) as f:
            #     data = json.load(f)
            # data['objects'] = data['objects'] + dict_out['objects']
            with open(path_json, 'w') as fp:
                json.dump(dict_out, fp, indent=4)
        else:
            with open(path_json, 'w') as fp:
                json.dump(dict_out, fp, indent=4)

        return im, beliefs_outputs, Metric

def rotate_vector(vector, quaternion):
    q_conj = tf.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf.transformations.quaternion_multiply(q_conj, vector)
    vector = tf.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]

if __name__ == "__main__":

    import argparse
    import yaml 
    import glob 
    import os 

    parser = argparse.ArgumentParser()
    parser.add_argument("--pause",
        default=0,
        help='pause between images')
    parser.add_argument("--showbelief",
        action="store_true",
        help='show the belief maps')
    parser.add_argument("--headless",
        action="store_true",
        help='headless mode')
    parser.add_argument("--json_only",
        action="store_true",
        help='only store the json files')
    parser.add_argument("--outf",
        default="out_experiment",
        help='where to store the output')
    parser.add_argument("--resultf",
        default=None,
        help='where to store the output')
    parser.add_argument("--data",
        default=None,
        help='folder for data images to load, *.png (default), see --suffix')
    parser.add_argument("--suffix",
        default='*.png',
        help='default *.png, note that * is needed to search')
    parser.add_argument("--config",
        default="../config/config_pose_webcam.yaml",
        help='folder for the inference configs')
    parser.add_argument("--camera",
        default="../config/camera_info_webcam.yaml",
        help='camera info file')
    parser.add_argument("--model",
        default=None,
        nargs='+',        
        help='model to load, this overwrites the config yaml file, although it has to be defined there')
    parser.add_argument('--gt_keypoints',
        action='store_true',
        default=False,
        help = "display the ground truth keypoints")
    parser.add_argument('--ensemble',
        action='store_true',
        default=False,
        help = "use ensemble method")
    parser.add_argument("--transform_visii",
        action='store_true',
        default=False,
        help='transform between dope and visii')
    parser.add_argument("--transform_hope",
        action='store_true',
        default=False,
        help='transform between dope and hope')
    parser.add_argument("--camera_from_yaml",
        action='store_true',
        default=False,
        help='camera info from yaml')
    parser.add_argument("--object",
        default="Ketchup",
        help='which object for ensemble method')
    parser.add_argument("--keypoints_ensemble",
        action='store_true',
        default=False,
        help='ensemble the keypoint level')
    parser.add_argument("--real_world",
        action='store_true',
        default=False,
        help='real_world setting does not need ground truth')

    opt = parser.parse_args()

    # load the configs
    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    with open(opt.camera) as f:
        camera_info = yaml.load(f, Loader=yaml.FullLoader)
    
    # create the output folder
    print (f"output is located in {opt.outf}")
    try:
        shutil.rmtree(f"{opt.outf}")
    except:
        pass

    try:
        os.makedirs(f"{opt.outf}")
    except OSError:
        pass


    # load the images if there are some
    imgs = []
    imgsname = []


    if not opt.data is None:
        videopath = opt.data

        imgs = []
        imgsname = []
        imgsjson = []

        filter_one_object = opt.transform_hope and opt.ensemble

        def add_images(path):
            for j in sorted(glob.glob(path+"/"+opt.suffix)):
                count = 0
                if filter_one_object:
                    with open(j.replace('png','json')) as f:
                        gt_json = json.load(f)['objects']
                        for obj in gt_json:
                            if obj['class'] == opt.object:
                                count += 1
                if (count == 1 and filter_one_object) or (not filter_one_object):
                    imgs.append(j)
                    imgsname.append(j.replace(path,"").replace("/",""))
                    #imgsjson.append(path+'/camera.json')
                    # Add json files
                    imgsjson.append(j.replace('.png','.json'))


        def explore(path):
            if not os.path.isdir(path):
                return
            folders = [os.path.join(path, o) for o in os.listdir(path) 
                            if os.path.isdir(os.path.join(path,o))]
            if len(folders)>0:
                for path_entry in folders:                
                    explore(path_entry)
            else:   
                add_images(path)

        explore(opt.data)

    else:
        # if not opt.realsense:
        cap = cv2.VideoCapture(0)
    if opt.model is not None:
        new_weights = {}
        # for i, w in enumerate(config['weights']):
        #     if w in opt.model:
        #         new_weights[w] = config['weights'][w]
        
        for m in opt.model:
            if 'full' in m:
                new_weights[m] = "../content/dope_network_full/"+m+".pth"
            elif 'visii' in m:
                new_weights[m] = "../content/dope_network_visii/"+m+".pth"
            else:
                new_weights[m] = "../content/dope_network/"+m+".pth"
        config['weights'] = new_weights

    config['models'] = opt.model
    config['keypoints_ensemble'] = opt.keypoints_ensemble

    # An object to run dope node
    dope_node = DopeNode(config)

    # starting the loop here
    i_image = -1 
    Metrics = []
    Names = []

    while True:
        i_image+=1
        
        # Capture frame-by-frame
        
        folder_substructure = "" 

        if not opt.data:
            ret, frame = cap.read()
            img_name = i_image

        else:
            if i_image >= len(imgs):
                i_image = 0
                # If you want to loop around your images, please comment the break 
                break
                
            frame = cv2.imread(imgs[i_image])
            img_name = imgsname[i_image]
            with open(imgsjson[i_image]) as f:
                visii_camera = json.load(f)

            # find the substructure
            folder_substructure = imgs[i_image].replace(opt.data,"").replace(imgsname[i_image],"")
            print(folder_substructure + img_name)
        frame = frame[...,::-1].copy()
        
        # check if the subfolder exist and create it 
        try:
            os.makedirs(f"{opt.outf}/{folder_substructure}")
        except OSError:
            pass

        if opt.camera_from_yaml:
            P_matrix = np.matrix(camera_info['camera_matrix']['data'], dtype='float64').copy() 
        else:
            fx = visii_camera['camera_data']['intrinsics']['fx']
            fy = visii_camera['camera_data']['intrinsics']['fy']
            x0 = visii_camera['camera_data']['intrinsics']['cx']
            y0 = visii_camera['camera_data']['intrinsics']['cy']
            P_matrix = [fx, 0, x0, 0, fy, y0, 0, 0, 1]

        gt_keypoints = None
        # only for debug and only works for one object
        if opt.gt_keypoints and (not opt.real_world):
            gt_keypoints = visii_camera['objects'][0]['projected_cuboid']

        # call the inference node
        out_img, out_beliefs, Metric = dope_node.image_callback(
            frame, 
            camera_info,
            img_name = img_name,
            output_folder = f"{opt.outf}/{folder_substructure}" ,
            showbelief = opt.showbelief,
            P_matrix = P_matrix,
            gt_keypoints = gt_keypoints)

        # transfer from dope to visii frame
        for name in Metric:
            temp = Metric[name]['orientation']
            if not temp is None:
                '''
                q_dope = Quaternion(w=temp[3],x=temp[0],y=temp[1],z=temp[2])
                q_dope_trans = q_dope * quat_trans_dope_to_visii
                Metric[name]['orientation'] = q_dope_trans
                '''
                rot = visii.quat(temp[3],temp[0],temp[1],temp[2])
                if opt.transform_visii:
                    rot = rot * visii.angleAxis(-1.57, visii.vec3(0,0,1)) * visii.angleAxis(-1.57, visii.vec3(0,1,0))
                    rot.w,rot.x,rot.y,rot.z = rot.w,rot.x,-rot.y,-rot.z
                if opt.transform_hope:
                    rot = rot * visii.angleAxis(1.57, visii.vec3(1,0,0)) * visii.angleAxis(1.57, visii.vec3(0,0,1))
                Metric[name]['orientation'] = Quaternion(w=rot.w,x=rot.x,y=rot.y,z=rot.z)

        # add ground truth
        if not opt.real_world:
            objects_gt = visii_camera['objects']
            for object_gt in objects_gt:
                if object_gt['class'] in Metric:
                    location_gt = np.array(object_gt['location'])
                    if opt.transform_visii:
                        location_gt[1] = -location_gt[1]
                        location_gt[2] = -location_gt[2]
                    if opt.transform_hope:
                        location_gt = location_gt * 0.01
                    Metric[object_gt['class']]['location gt'] = location_gt
                    temp = object_gt['quaternion_xyzw']
                    q_visii = Quaternion(w=temp[3],x=temp[0],y=temp[1],z=temp[2])
                    Metric[object_gt['class']]['orientation gt'] = q_visii

        # support running several models for a same object
        if not opt.real_world:
            for name in Metric:
                if opt.object in name:
                    Metric[name]['location gt'] = Metric[opt.object]['location gt']
                    Metric[name]['orientation gt'] = Metric[opt.object]['orientation gt']

        # evaluation
        if not opt.real_world:
            for name in Metric:
                if Metric[name]['location'] is None:
                    Metric[name]['location error'] = 1000
                    Metric[name]['orientation error'] = 1000
                else:
                    Metric[name]['location error'] = np.linalg.norm(Metric[name]['location'] - Metric[name]['location gt'])
                    q1 = Quaternion(Metric[name]['orientation'])
                    q2 = Quaternion(Metric[name]['orientation gt'])
                    Metric[name]['orientation error'] = Quaternion.absolute_distance(q1, q2)

        Names.append(img_name)
        Metrics.append(Metric)
        print(Metric)

        if not opt.headless:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('DOPE',np.array(out_img)[...,::-1])
            if opt.showbelief:
                # TODO remove the [0] for a for loop when multiple objects. 
                cv2.imshow('DOPE BELIEFS',np.array(out_beliefs[0])[...,::-1])
                # print(img_name)
                cv2.imwrite(opt.outf+'DOPE_BELIEFS_'+img_name,np.array(out_beliefs[0])[...,::-1])                

    all_location_error = {}
    all_orientation_error = {}
    all_score = {}
    all_pose_std = {}
    all_quat_std = {}
    all_location = {}
    all_orientation = {}
    all_location_gt = {}
    all_orientation_gt = {}
    all_add = {}
    for m in Metric:
        all_location_error[m] = []
        all_orientation_error[m] = []
        all_score[m] = []
        all_pose_std[m] = []
        all_quat_std[m] = []
        all_location[m] = []
        all_orientation[m] = []
        all_location_gt[m] = []
        all_orientation_gt[m] = []
        all_add[m] = []

    for Metric in Metrics:
        for m in Metric:
            if not opt.real_world:
                all_location_error[m].append(Metric[m]['location error'])
                all_orientation_error[m].append(Metric[m]['orientation error'])
                all_location_gt[m].append(Metric[m]['location gt'])
                all_orientation_gt[m].append(Metric[m]['orientation gt'])

            all_score[m].append(Metric[m]['centroid score'])
            all_pose_std[m].append(sum(Metric[m]['uncertainty'][:3]))
            all_quat_std[m].append(sum(Metric[m]['uncertainty'][3:]))
            all_location[m].append(Metric[m]['location'])
            all_orientation[m].append(Metric[m]['orientation'])

    for m in Metric:
        if not opt.real_world:
            all_location_error[m] = np.array(all_location_error[m])
            all_orientation_error[m] = np.array(all_orientation_error[m])
        all_score[m] = np.array(all_score[m])
        all_pose_std[m] = np.array(all_pose_std[m])
        all_quat_std[m] = np.array(all_quat_std[m])


    # Compute the ADD
    if not opt.real_world:
        visii.initialize_headless()

        folder = '../content/models/grocery/'
        obj1 = create_obj(
            name = opt.object + '_1',
            path_obj = folder + opt.object + "/google_16k/textured_simple.obj"
        )
        obj2 = create_obj(
            name = opt.object + '_2',
            path_obj = folder + opt.object + "/google_16k/textured_simple.obj"
        )

        for m in Metric:
            for i in range(len(all_location[m])):
                pos1 = all_location[m][i]
                quat1 = all_orientation[m][i]
                pos2 = all_location_gt[m][i]
                quat2 = all_orientation_gt[m][i]
                try:
                    all_add[m].append(get_add(obj1, obj2, pos1, pos2, quat1, quat2))
                except:
                    all_add[m].append(1.0)
            all_add[m] = np.array(all_add[m])

        visii.cleanup()

    all_data = {}
    
    index = 0
    for n in Names:
        all_data[n] = {}
        for m in opt.model:
            all_data[n][m] = {}
            if not opt.real_world:
                all_data[n][m]['loc gt'] = all_location_gt[m][index].tolist()
                temp_q = all_orientation_gt[m][index]
                all_data[n][m]['quat gt'] = [temp_q[0], temp_q[1], temp_q[2], temp_q[3]]
                all_data[n][m]['add error'] = all_add[m][index].tolist()

            if all_location[m][index] is None:
                all_data[n][m]['loc'] = None
            else:
                all_data[n][m]['loc'] = all_location[m][index].tolist()
            
            temp_q = all_orientation[m][index]
            if temp_q is None:
                all_data[n][m]['quat'] = None
            else:
                all_data[n][m]['quat'] = [temp_q[0], temp_q[1], temp_q[2], temp_q[3]]
            
            all_data[n][m]['score'] = all_score[m][index].tolist()
            all_data[n][m]['GUAPO std'] = all_pose_std[m][index].tolist()
        index += 1

    if opt.resultf is None:
        opt.resultf = opt.outf
    else:
        if os.path.isdir(opt.resultf):
            pass
        else:
            os.makedirs(opt.resultf)

    # Save raw data
    with open(opt.resultf+'/raw_data.json', 'w') as fp:
        json.dump(all_data, fp, indent=4)


if not opt.headless:
    cv2.destroyAllWindows()
