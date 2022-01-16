from torchvision import datasets, transforms
import torch.utils.data
import torch
import sys
import argparse
import matplotlib.pyplot as plt
from utils import * 
from utils import *
from models import *
import os, shutil
from collections import OrderedDict
from tqdm import tqdm, trange
import pandas as pd
import transforms3d
import open3d as o3d
import copy
import gc

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,   default='runs/test',    help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=1024,            help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')
# args = parser.parse_args(args=['atlas_baseline=0, autoencoder=1,panos_baseline=0'])

VOXEL_SZ = 0.2
DEBUG=False
# IMAGE_HEIGHT=29


# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()
model = VAE(args).cuda()
MODEL_USED_DATA_PARALLEL = False
model_file = 'trying_path_3/models/gen_15.pth'
# model_file = '/home/sabyasachi/Projects/ati/ati_motors/adversarial_based/some_runs/new_runs/unet_more_layers_restarted_with_more_data_ctd/models/gen_200.pth'
print("Loading model from {}".format(model_file))

network=torch.load(model_file)

if MODEL_USED_DATA_PARALLEL:
    # original saved file with DataParallel
    state_dict = network
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    # load params
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(network)
model.eval()

# network=torch.load(model_file)
# model.load_state_dict(network)
# model.eval()

def getint(name):
    return int(name.split('.')[0])

#STATIC_TRAIN_FOLDER_PATH  = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/static_out_npy/'
#DYNAMIC_TRAIN_FOLDER_PATH = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/dynamic_out_npy/'
#STATIC_PREPROCESS_DATA_PATH  =  "../prashVAE/static_prepreprocess_dir"
DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map_dynamic_high_only/data"
DYNAMIC_PREPROCESS_GT_PATH = "../training_data/small_map_dynamic_high_only/label"

#train_file  = sorted(os.listdir( STATIC_TRAIN_FOLDER_PATH), key=getint)[0]

# val_file  = sorted(os.listdir(DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[1]
# gt_file = sorted(os.listdir(DYNAMIC_PREPROCESS_GT_PATH), key=getint)[1]

val_file = "3.npy"
gt_file = "3_gt.csv"

print("Loading dynamic val file {}".format(val_file))

dataset_val_dict = {}
dataset_val_dict['data'] = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)
dataset_val_dict['label'] = pd.read_csv(os.path.join(DYNAMIC_PREPROCESS_GT_PATH, gt_file))


#dataset_val = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)
#print("Loading static file")
#static_dataset_val = np.load(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)

# kitti='kitti_data/'
# data=np.load(kitti+'lidar_test.npz')
# # print(data.shape)  		#(832,60,512,4)


# x=onePC=data[0:1,:,:,:]
# x=onePC.reshape([60,512,4])

# dataset_val   = np.load('../lidar_generation/kitti_data/lidar_val.npz') 

# dataset_val   = np.load('/home/prashant/P/carla-0.8.4/PythonClient/_out_npy/carlaData/0_2/val.npy') 
# test_folder = "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/dynamic_out_npy"
# static_test_folder = "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/static_out_npy"

# test_file = os.path.join(test_folder, sorted(os.listdir(test_folder), key=getint)[-1])
# static_test_file = os.path.join(static_test_folder, sorted(os.listdir(static_test_folder), key=getint)[-1])
# dataset_val   = np.load(test_file)

# static_dataset_val = np.load(static_test_file)

#class SupervisedConsecutivePairdata(torch.utils.data.Dataset):
#    """
#    Dataset of numbers in [a,b] inclusive
#    """
#
#    def __init__(self, dataset):
#        super(SupervisedConsecutivePairdata, self).__init__()
#        self.dataset_data  = dataset['data']
#        self.dataset_label = dataset['label']
#        
#        assert len(self.dataset_data) == len(self.dataset_label)
#
#    def __len__(self):
#        return self.dataset_data.shape[0] - 1   # We don't want a pair for last frame
#
#    def __getitem__(self, index):
#        index1 = index
#        index2 = index+1 if index+1 < self.dataset_data.shape[0] else index  # The pair for last lidar frame is itself
#        return index, self.dataset_data[index1], self.dataset_data[index2]#, self.dataset_label.iloc[index1].to_dict(), self.dataset_label.iloc[index2].to_dict()

def pose2matrix(translation_list, rotation_angle_list, zoom_list=[1,1,1]):
    # trans_vec = np.array(translation_list)
    rot_ang = [np.deg2rad(ang) for ang in rotation_angle_list ]
    rot_mat = transforms3d.euler.euler2mat(rot_ang[0], rot_ang[1], rot_ang[2])
    zoom_vec = np.array(zoom_list)
    # transform_mat = transforms3d.affines.compose(trans_vec, rot_mat, zoom_vec)
    transform_mat = transforms3d.affines.compose(translation_list, rot_mat, zoom_list)
    return transform_mat


def get_gt_pose(prev_gt, next_gt):
    # get_gt_pose execution starts here
    prev_inv_mat = np.linalg.inv(pose2matrix([prev_gt['location_x'],
                                              prev_gt['location_y'],
                                              prev_gt['location_z']],
                                             [prev_gt['rotation_roll'],
                                              prev_gt['rotation_pitch'],
                                              prev_gt['rotation_yaw']]))
    next_mat = pose2matrix([next_gt['location_x'],
                            next_gt['location_y'],
                            next_gt['location_z']],
                           [next_gt['rotation_roll'],
                            next_gt['rotation_pitch'],
                            next_gt['rotation_yaw']])
    transformation_gt = np.matmul(prev_inv_mat, next_mat)
    transformation_gt = np.linalg.inv(transformation_gt) # Open 3d assumes transform is applied on source and not target
    return transformation_gt


def dict_idx(that_dict, idx):
    return {k:v[idx] for k, v in that_dict.items()}

class SupervisedConsecutivePairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset):
        super(SupervisedConsecutivePairdata, self).__init__()
        self.dataset_data  = dataset['data']
        self.dataset_label = dataset['label']
        
        assert len(self.dataset_data) == len(self.dataset_label)

    def __len__(self):
        return self.dataset_data.shape[0] - 1   # We don't want a pair for last frame

    def __getitem__(self, index):
        index1 = index
        index2 = index+1 if index+1 < self.dataset_data.shape[0] else index  # The pair for last lidar frame is itself
        label_transform = get_gt_pose(self.dataset_label.iloc[index1].to_dict(), self.dataset_label.iloc[index2].to_dict())
        return index, self.dataset_data[index1], self.dataset_data[index2], label_transform


val_data = SupervisedConsecutivePairdata(dataset_val_dict)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=False, num_workers=4, drop_last=False)

   
#class Pairdata(torch.utils.data.Dataset):
#    """
#    Dataset of numbers in [a,b] inclusive
#    """
#
#    def __init__(self, dataset1,dataset2):
#        super(Pairdata, self).__init__()
#        self.dataset1 = dataset1
#        self.dataset2 = dataset2
#
#    def __len__(self):
#        return self.dataset1.shape[0]
#
#    def __getitem__(self, index):
#        
#        return index, self.dataset1[index],self.dataset2[index]



# print(dataset_val.shape)    #(154,60,512,4)
# exit(1)
# print(dataset_val[0:1,:,:,:].shape)  #[1,60,512,4] 

# print("Here")
# print(type(dataset_val))
# exit(1)
# dataset_val1   = preprocess(dataset_val[0:1,:,:,:]).astype('float32')
# print(dataset_val1.shape)      #(1,2,40,256)

# print(dataset_val.shape)          #(152,2,40,256)
# exit(1)

#val_data = Pairdata(dataset_val, static_dataset_val)
#val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
#                    shuffle=False, num_workers=4, drop_last=False)


#val_loader    = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
process_input = from_polar if args.no_polar else lambda x : x
# print 

def pose2matrix(translation_list, rotation_angle_list, zoom_list=[1,1,1]):
    # trans_vec = np.array(translation_list)
    rot_ang = [np.deg2rad(ang) for ang in rotation_angle_list ]
    rot_mat = transforms3d.euler.euler2mat(rot_ang[0], rot_ang[1], rot_ang[2])
    zoom_vec = np.array(zoom_list)
    # transform_mat = transforms3d.affines.compose(trans_vec, rot_mat, zoom_vec)
    transform_mat = transforms3d.affines.compose(translation_list, rot_mat, zoom_list)
    return transform_mat

# Function to transform given lidar pcd to ground truth to get it upright
def transform_lidar_to_gt_frame(pcd):
    new_pcd = copy.deepcopy(pcd)
    transformation_lidar2gt = pose2matrix([0,0,0], [0,0,90],[1,1,-1])
    new_pcd.transform(transformation_lidar2gt)
    return new_pcd

# Function to get pcd for given range image in torch.cuda
def get_pcd_from_img(img):
    img = img * 100
    frame = from_polar(img).detach().cpu().numpy()[0]
    # frame_actual = np.array([frame_image[:29] for frame_image in frame])
    frame_flat = frame.reshape((3,-1))
    some_pcd = o3d.PointCloud()
    some_arr = frame_flat.T
    some_pcd.points = o3d.utility.Vector3dVector(some_arr)

    new_some_pcd = transform_lidar_to_gt_frame(some_pcd)
    return new_some_pcd

# Function to get ICP pose for given src pcd and dst pcd
def get_icp_pose(src, dst, voxel_size=VOXEL_SZ):
    def crop_pcd(old_pcd, crop_min_arr=np.array([-100,-100,-2]), crop_max_arr=np.array([100,100,100])):
        np.random.seed(0)
        pcd = copy.deepcopy(old_pcd)

        cropped_pcd = o3d.geometry.crop_point_cloud(pcd, crop_min_arr, crop_max_arr)
        pcd = cropped_pcd
        return pcd

    def prepare_dataset(source, target, voxel_size):
        def preprocess_point_cloud(pcd, voxel_size):
            pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
            radius_normal = voxel_size * 2
            o3d.geometry.estimate_normals(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            radius_feature = voxel_size * 5
            pcd_fpfh = o3d.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            return pcd_down, pcd_fpfh
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        return source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        if DEBUG:
            print("start execute global reg")
        result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
        if DEBUG:
            print("finish execute global reg")
        return result

    def refine_registration(source, target, voxel_size, trans_init):
        distance_threshold = voxel_size * 0.4
        result = o3d.registration.registration_icp(
                    source, target, distance_threshold, trans_init,
                    o3d.registration.TransformationEstimationPointToPlane())
                    # o3d.registration.TransformationEstimationPointToPlane())
        return result
    # get_icp_pose execution starts here
    if DEBUG:
        print("start icp pose")
    # source = crop_pcd(src)
    source = src
    if DEBUG:
        print("cropped src")
    # target = crop_pcd(dst)
    target = dst
    if DEBUG:
        print("cropped dst")

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    if DEBUG:
        print("prepared dataset")
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    if DEBUG:
        print("executed global reg")
    result_icp = refine_registration(source_down, target_down, voxel_size, result_ransac.transformation)
    if DEBUG:
        print("refined reg")

    evaluation = o3d.registration.evaluate_registration(source_down, target_down, voxel_size * 5, result_icp.transformation)
    if DEBUG:
        print("evaluated")

    # print("Before ICP")
    # draw_registration_result(source_down, target_down, pose2matrix([0,0,0], [0,0,0],[1,1,1]))

    # print("After ICP")
    # draw_registration_result(source_down, target_down, result_icp.transformation)

    return result_icp.transformation, evaluation

# Function to give slam pose for given two consecutive range images in torch.cuda
def get_slam_pose_transform(recon_curr_img, recon_next_img):
    dynamic_pcd_curr = get_pcd_from_img(recon_curr_img)
    if DEBUG:
        print("got pcd curr")

    dynamic_pcd_next = get_pcd_from_img(recon_next_img)
    if DEBUG:
        print("got pcd next")

    slam_pose_transform, slam_pose_err = get_icp_pose(dynamic_pcd_curr, dynamic_pcd_next)
    if DEBUG:
        print("got slam pose")

    # print("Before ICP")
    # draw_registration_result(dynamic_pcd_curr, dynamic_pcd_next,
    #                             pose2matrix([0,0,0], [0,0,0],[1,1,1]))

    # print("After ICP")
    # draw_registration_result(dynamic_pcd_curr, dynamic_pcd_next, slam_pose_transform)

    gc.collect()
    return slam_pose_transform, slam_pose_err 


#if not os.path.exists("samples"):
#    os.makedirs("samples/reconstructed")
#    os.makedirs("samples/dynamic")
#    os.makedirs("samples/static")
#else:
#    shutil.rmtree("samples")
#    os.makedirs("samples/reconstructed")
#    os.makedirs("samples/dynamic")
#    os.makedirs("samples/static")

recons=[]
original=[]
#staticc=[]

pose_list = []
old_pt = np.array([0,0,0,1])
pose_list.append(old_pt)

for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
#     if i != len(val_loader) - 1:
#         continue
#     print(i)
    # if(i!=10):
    # 	continue

    # print(type(img))								
    # exit(1)
    dynamic_img_curr = img_data[1].cuda()
    dynamic_img_next = img_data[2].cuda()
#     print(type(dynamic_img))								#torch.Tensor
    # print(img.shape)       						#[64,2,40,256]

    # print(((process_input(img))).shape)  # 			[64,3,40,256] 

#     print(((process_input(dynamic_img))[0:1,:,:,:]).shape)  #[1,3,40,256] 
    #print(show_pc_lite((process_input(img))[0:1,:,:,:]))
    # exit(1)
    recon_curr, kl_cost,hidden_z = model(process_input(dynamic_img_curr))
    recon_next, kl_cost,hidden_z = model(process_input(dynamic_img_next))

    for frame_num in range(dynamic_img_curr.shape[0]):
        # recon_curr_frame = recon_curr[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :]
        # recon_next_frame = recon_next[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :]
        recon_curr_frame = recon_curr[frame_num:frame_num+1, :, :, :]
        recon_next_frame = recon_next[frame_num:frame_num+1, :, :, :]

        # Get SLAM Pose as blackbox
        pose_transform, pose_err = get_slam_pose_transform(recon_curr_frame, recon_next_frame)
        old_pt = pose_list[-1]
        new_pt = np.matmul(np.linalg.inv(pose_transform), old_pt)
        pose_list.append(new_pt)

#     print(type(recon))
#    recons=recon
#    original=dynamic_img
#    staticc=static_img
    # print(recon.detach().shape)  					[64,2,40,256]
    # print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
    # print((recon[0:1,:,:,:]).shape)				([1, 2, 40, 256]
    #print(show_pc_lite(from_polar((recon[0:1,:,:,:]).detach())))

#     break

#    recons_temp=np.array(recons.detach().cpu())    			#(64, 2, 40, 256)
#    original_temp=np.array(original.detach().cpu())    	    #(64, 2, 40, 256)
#    staticc_temp=np.array(staticc.detach().cpu())
    # recons_temp=np.array(recons)    			#(64, 2, 40, 256)
    # original_temp=np.array(original)    			#(64, 2, 40, 256)

    # print((recons_temp).shape)
    # print((original_temp).shape)



#    for frame_num in range(recons_temp.shape[0]):
#        frame=from_polar(recons[frame_num:frame_num+1,:,:,:]).detach().cpu()
#        plt.figure()
#        plt.title("Reconstructed")
#        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='g')
#        plt.savefig('samples/reconstructed/'+str(i*args.batch_size+frame_num)+'.jpg') 
#        # plt.show()
#        plt.close()
#
#    for frame_num in range(original_temp.shape[0]):
#        frame=from_polar(original[frame_num:frame_num+1,:,:,:]).detach().cpu()
#        plt.figure()
#        plt.title("Dynamic")
#        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='b')
#        plt.savefig('samples/dynamic/'+str(i*args.batch_size+frame_num)+'.jpg') 
#        # plt.show()
#        plt.close()

#    for frame_num in range(staticc_temp.shape[0]):
#        frame=from_polar(staticc[frame_num:frame_num+1,:,:,:]).detach().cpu()
#        plt.figure()
#        plt.title("Static")
#        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='r')
#        plt.savefig('samples/static/'+str(i*args.batch_size+frame_num)+'.jpg') 
        # plt.show()
#        plt.close()

np.save("samples/pose_est.npy", np.array(pose_list))
print('final exit from my_eval')

#ANIMATE

#shape must be [1,60,512,4]  accorsing to from where it is called , 1 because we are trying to train one by one
def getHiddenRep(img):
	img   = preprocess(img).astype('float32')		#[1,2,40,256]
	show_pc_lite(from_polar((img).detach()))
	img = img.cuda()
	# print(img.shape)       						[64,2,40,256]
	# print(((process_input(img))).shape)  # 			[64,3,40,256] 
	# print(((process_input(img))[0:1,:,:,:]).shape)  [1,3,40,256] 
	# print(show_pc_lite((process_input(img))[0:1,:,:,:]))
	recon, kl_cost, hidden_z = model(process_input(img))
	# print(recon.detach().shape)  					[64,2,40,256]
	# print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
	# print((recon[0:1,:,:,:]).shape)				([1, 2, 40, 256]
	print(show_pc_lite(from_polar((recon[0:1,:,:,:]).detach())))
	return hidden_z




