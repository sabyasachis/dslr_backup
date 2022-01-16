import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from pydoc import locate
import tensorboardX
from tqdm import tqdm, trange
import os, shutil
from torchsummary import summary

from utils import * 
from models import * 

import pandas as pd
import numpy as np
import gc
import copy

import transforms3d
import open3d as o3d

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

DEBUG = False
VOXEL_SZ=0.2

parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,                           help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=1024,             help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')

# ------------------------ARGS stuff----------------------------------------------
args = parser.parse_args()
maybe_create_dir(args.base_dir)
print_and_save_args(args, args.base_dir)

# the baselines are very memory heavy --> we split minibatches into mini-minibatches
if args.atlas_baseline or args.panos_baseline: 
    """ Tested on 12 Gb GPU for z_dim in [128, 256, 512] """ 
    bs = [4, 8 if args.atlas_baseline else 6][min(1, 511 // args.z_dim)]
    factor = args.batch_size // bs
    args.batch_size = bs
    is_baseline = True
    args.no_polar = 1
    print('using batch size of %d, ran %d times' % (bs, factor))
else:
    factor, is_baseline = 1, False


# --------------------- Setting same seeds for reproducibility ------------------------------
# reproducibility is good
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# -------------------- MODEL BUILDING -------------------------------------------------------
# construct model and ship to GPU
model = VAE(args).cuda()
print(model)
# print(model.state_dict())
# assert False
print(summary(model, input_size=(2, 16, 1024)))

# assert False

model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr) 

# MODEL_PATH = '../static_reconstruction_method/small_map_unet_restarted/models/gen_90.pth'
# if(os.path.exists(MODEL_PATH)):
#     network_state_dict = torch.load(MODEL_PATH)
#     model.load_state_dict(network_state_dict)
#     print("Previous weights loaded from {}".format(MODEL_PATH))
# else:
print("Starting from scratch")
#assert False

# -------------------- TENSORBOARD SETUP FOR LOGGING -------------------------------------------
# Logging
maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
writer = tensorboardX.SummaryWriter(log_dir=os.path.join(args.base_dir, 'TB'))
writes = 0
ns     = 16

# -------------------- DATASET SETUP ----------------------------------------------------------
def getint(name):
    try:
        return int(name.split('.')[0])
    except Exception as e:
        print("Error occured while trying to read {}".format(name))
    return None

def getgtint(name):
    try:
        return int(name.split('.')[0][:-3])
    except Exception as e:
        print("Error occured while trying to read {}".format(name))
    return None

PREPROCESS_FRESHLY = False
# IMAGE_HEIGHT = -3

DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map_dynamic_high_only/data"
DYNAMIC_PREPROCESS_GT_PATH = "../training_data/small_map_dynamic_high_only/label"


if PREPROCESS_FRESHLY:
    DYNAMIC_TRAIN_FOLDER_PATH = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/16beam-Data/small_map/dynamic_training_data/npy_data'
    DYNAMIC_TRAIN_GT_FOLDER_PATH = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/16beam-Data/small_map/dynamic_training_data/gt'
    
    val_file  = sorted(os.listdir( DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[-1]
    npyList = sorted(os.listdir(DYNAMIC_TRAIN_FOLDER_PATH), key=getint)[:-1]

    val_gt_file  = sorted(os.listdir( DYNAMIC_TRAIN_GT_FOLDER_PATH), key=getgtint)[-1]
    npy_gt_List = sorted(os.listdir(DYNAMIC_TRAIN_GT_FOLDER_PATH), key=getgtint)[:-1]

    print("Dynamic preprocessing of:")
    if not os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH):
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)
    else:
        shutil.rmtree(DYNAMIC_PREPROCESS_DATA_PATH)
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)

    if not os.path.exists(DYNAMIC_PREPROCESS_GT_PATH):
        os.makedirs(DYNAMIC_PREPROCESS_GT_PATH)
    else:
        shutil.rmtree(DYNAMIC_PREPROCESS_GT_PATH)
        os.makedirs(DYNAMIC_PREPROCESS_GT_PATH)

    print("training dataset:")
    for file in tqdm(npyList):
        dynamic_dataset_train = np.load(os.path.join(DYNAMIC_TRAIN_FOLDER_PATH, file))
        dynamic_dataset_train = preprocess(dynamic_dataset_train).astype('float32')
        dynamic_dataset_train.dump(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file))

    print("validation dataset:")    
    dynamic_dataset_val   = np.load(os.path.join(DYNAMIC_TRAIN_FOLDER_PATH, val_file))
    dynamic_dataset_val   = preprocess(dynamic_dataset_val).astype('float32')
    dynamic_dataset_val.dump(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file))

    print("labels:")
    for file in tqdm(os.listdir(DYNAMIC_TRAIN_GT_FOLDER_PATH)):
        src_path = os.path.join(DYNAMIC_TRAIN_GT_FOLDER_PATH, file)
        dst_path = os.path.join(DYNAMIC_PREPROCESS_GT_PATH, file)
        shutil.copy(src_path, dst_path)

    print("Freshly processed datasets successfully! Exiting!")
    assert False
else:

    val_file  = sorted(os.listdir( DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[-1]
    npyList = sorted(os.listdir(DYNAMIC_PREPROCESS_DATA_PATH), key=getint)[:-1]

    val_gt_file  = sorted(os.listdir( DYNAMIC_PREPROCESS_GT_PATH), key=getgtint)[-1]
    npy_gt_List = sorted(os.listdir(DYNAMIC_PREPROCESS_GT_PATH), key=getgtint)[:-1]

    print("Have already preprocessed datasets at {}".format(DYNAMIC_PREPROCESS_DATA_PATH))    
    print("Loading dynamic training data and labels")
    dynamic_dataset_train_dict = {}
    for data_file, gt_file in tqdm(zip(npyList, npy_gt_List), total=len(npyList)):
        if getint(data_file) != 3:
            continue
        some_dynamic_dataset = {}
        some_dynamic_dataset['data'] = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, data_file), allow_pickle=True)
        some_dynamic_dataset['label'] = pd.read_csv(os.path.join(DYNAMIC_PREPROCESS_GT_PATH, gt_file))
        dynamic_dataset_train_dict[getint(data_file)] = some_dynamic_dataset
    print("Loading dynamic validation data and labels")
    dynamic_dataset_val = {}
    dynamic_dataset_val['data'] = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)
    dynamic_dataset_val['label'] = pd.read_csv(os.path.join(DYNAMIC_PREPROCESS_GT_PATH, val_gt_file))
    print("done")

# assert False

def draw_registration_result(src, dst, transformation):
    source = copy.deepcopy(src)
    target = copy.deepcopy(dst)
    
    source.paint_uniform_color([1, 0, 0]) # red
    target.paint_uniform_color([0, 0, 1]) # blue
    target.transform(transformation)
    o3d.visualization.draw_geometries([source, target], width=1280, height=800)

# Function to get transformation matrix for a given pose
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

# def get_rpe(transform1, transform2, angle_err_wt=1):
#     transformation_rpe =  np.matmul(np.linalg.inv(transform1), transform2)
#     trans_arr, rot_mat, scale_mat, shear_mat = transforms3d.affines.decompose44(transformation_rpe)
#     rot_list = transforms3d.euler.mat2euler(rot_mat, axes='sxyz')
#     rot_arr = np.array(rot_list)
#     rpe_total = np.linalg.norm(trans_arr) + (np.linalg.norm(rot_arr)*angle_err_wt)
#     return rpe_total

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

# def slam_loss_fn(prev_pcd, next_pcd, prev_gt, next_gt):
#     # icp_loss_fn starts here
#     if DEBUG:
#         print("start slam loss function")
#     prev_pcd, next_pcd = transform_lidar_to_gt_frame(prev_pcd, next_pcd)
#     if DEBUG:
#         print("transformed lidar to gt")
#     transformation_gt = get_gt_pose(prev_gt, next_gt)
#     if DEBUG:
#         print("got gt pose")
#     # draw_registration_result(prev_pcd, next_pcd, transformation_gt)
    
#     transformation_icp, evaluation_icp = get_icp_pose(prev_pcd, next_pcd)
#     if DEBUG:
#         print("got icp pose")
#     # draw_registration_result(prev_pcd, next_pcd, transformation_icp)
    
#     rpe_loss = get_rpe(transformation_gt, transformation_icp)
#     if DEBUG:
#         print("got rpe loss")
    
#     return rpe_loss

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

val_data = SupervisedConsecutivePairdata(dynamic_dataset_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=False, num_workers=20, drop_last=False)

# assert False

    # VAE training
# ----------------------- AE TRAINING -------------------------------------------------------
# rangee=150 if args.autoencoder else 300
# print("Begin training:")

#LOSS_FACTOR = 1e5
# LOSS_FACTOR = 1

for epoch in range(500):
    print("\n*************************************\n")
    print('Epoch #%s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x
    loss_fn = get_chamfer_dist()

    # FOR EVERY SMALL FILE
    print("Training: ")
    for key, dynamic_dataset_train in dynamic_dataset_train_dict.items():
        print("Path #{} / {}".format(key, len(dynamic_dataset_train_dict)))
        train_data = SupervisedConsecutivePairdata(dynamic_dataset_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=False, num_workers=20, drop_last=False)
        gc.collect()

        # TRAIN HERE
        for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            optim.zero_grad()

            # if DEBUG:
            print("Batch #{} / {}".format(i+1, len(train_loader)))
            dynamic_img_curr = img_data[1].cuda()
            dynamic_img_next = img_data[2].cuda()
            dynamic_gt_transform = img_data[3].float().cuda()

            gc.collect()

            recon_curr, kl_cost_curr, hidden_z_curr = model(process_input(dynamic_img_curr))
            recon_next, kl_cost_next, hidden_z_next = model(process_input(dynamic_img_next))
            
            slam_loss_list = []
            # For every pair in batch
            for frame_num in trange(dynamic_img_curr.shape[0]):
                if DEBUG:
                    print(frame_num)

                # recon_curr_frame = recon_curr[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :]
                # recon_next_frame = recon_next[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :]
                recon_curr_frame = recon_curr[frame_num:frame_num+1, :, :, :]
                recon_next_frame = recon_next[frame_num:frame_num+1, :, :, :]
                frame_gt_transform = dynamic_gt_transform[frame_num]

                # Get SLAM Pose as blackbox
                pose_transform, pose_err = get_slam_pose_transform(recon_curr_frame, recon_next_frame)
                pose_transform = torch.from_numpy(pose_transform).float().cuda()

                # Transform reconstructed frames to xyz
                recon_curr_frame_3d = from_polar(recon_curr_frame)[0]
                recon_next_frame_3d = from_polar(recon_next_frame)

                recon_curr_frame_3d_flat  = torch.cat([recon_curr_frame_3d[0].flatten().unsqueeze(0),
                                                    recon_curr_frame_3d[1].flatten().unsqueeze(0),
                                                    recon_curr_frame_3d[2].flatten().unsqueeze(0),
                                                    torch.zeros_like(recon_curr_frame_3d[0].flatten().unsqueeze(0))],
                                                 0)


                # Apply SLAM transformation on src frame
                recon_curr_frame_3d_flat_icp_transformed  = torch.mm(pose_transform, recon_curr_frame_3d_flat)
                recon_curr_frame_3d_flat_gt_transformed   = torch.mm(frame_gt_transform, recon_curr_frame_3d_flat)

                recon_curr_frame_3d_icp_transformed = recon_curr_frame_3d_flat_icp_transformed[:3].view(recon_next_frame_3d.shape)
                recon_curr_frame_3d_gt_transformed  = recon_curr_frame_3d_flat_gt_transformed[:3].view(recon_next_frame_3d.shape)

                # Apply SLAM loss (like Chamfer Distance, etc) on the pair
                slam_frame_loss = loss_fn(recon_curr_frame_3d_icp_transformed, recon_next_frame_3d)
                gt_frame_loss = loss_fn(recon_curr_frame_3d_gt_transformed, recon_next_frame_3d)
                frame_loss = torch.abs(gt_frame_loss - slam_frame_loss)
                # frame_loss_squared = frame_loss * frame_loss

                # Append SLAM loss to the list
                slam_loss_list.append(frame_loss)
                gc.collect()

            # Create SLAM loss tensor
            loss_recon = torch.cat(slam_loss_list)

            if args.autoencoder:
                kl_obj, kl_cost_curr = [torch.zeros_like(loss_recon)] * 2
            else:
                kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                                torch.clamp(kl_cost_curr, min=5)

            loss = (kl_obj  + loss_recon).mean(dim=0)
            # loss = torch.sqrt(loss)
            elbo = (kl_cost_curr + loss_recon).mean(dim=0)

            loss_    += [loss.item()]
            elbo_    += [elbo.item()]
            kl_cost_ += [kl_cost_curr.mean(dim=0).item()]
            kl_obj_  += [kl_obj.mean(dim=0).item()]
            recon_   += [loss_recon.mean(dim=0).item()]

            # baseline loss is very memory heavy 
            # we accumulate gradient to simulate a bigger minibatch
            # if (i+1) % factor == 0 or not is_baseline: 
            
            loss.backward()

            # params = list(model.parameters())
            # print(params[0].grad)
            # print([param.grad for param in params])

            # if (i+1) % factor == 0 or not is_baseline: 
            # prev_weight = model.state_dict()['encoder_conv2d_6.0.weight']
            # a = list(model.parameters())[0].clone()
            
            optim.step()

            # next_weight = model.state_dict()['encoder_conv2d_6.0.weight']
            # b = list(model.parameters())[0].clone()
            # print(torch.equal(a.data, b.data))
            # print(a.data)
            # print("*************")
            # print(b.data)


            # print(next_weight - prev_weight)
            # print(list(model.parameters())[0].grad)
            gc.collect()
    #####

            writes += 1
            mn = lambda x : np.mean(x)
            print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
            print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
            print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
            print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
            print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

            torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
        
    
         
    
    # Testing loop
    # --------------------------------------------------------------------------

    print("Validating: ")
    continue
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 10 == 0:
            # print('test set evaluation')
            for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                dynamic_img_curr = img_data[1].cuda()
                dynamic_img_next = img_data[2].cuda()
                dynamic_label_curr = img_data[3]
                dynamic_label_next = img_data[4]

                recon_curr, kl_cost_curr, hidden_z_curr = model(process_input(dynamic_img_curr))
                recon_next, kl_cost_next, hidden_z_next = model(process_input(dynamic_img_next))
           
                loss_recon = 0
                for frame_num in range(args.batch_size):
                    # dynamic_pcd_curr = get_pcd_from_img(recon_curr[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :])
                    # dynamic_pcd_next = get_pcd_from_img(recon_next[frame_num:frame_num+1, :, :IMAGE_HEIGHT, :])
                    dynamic_pcd_curr = get_pcd_from_img(recon_curr[frame_num:frame_num+1, :, :, :])
                    dynamic_pcd_next = get_pcd_from_img(recon_next[frame_num:frame_num+1, :, :, :])
                    loss_recon      += slam_loss_fn(dynamic_pcd_curr, dynamic_pcd_next, dict_idx(dynamic_label_curr, frame_num), dict_idx(dynamic_label_next, frame_num))
#                 loss_recon *= LOSS_FACTOR
                loss_recon = np.array(loss_recon)
                loss_recon = torch.from_numpy(loss_recon)

                if args.autoencoder:
                    kl_obj, kl_cost_curr = [torch.zeros_like(loss_recon)] * 2
                else:
                    kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                                    torch.clamp(kl_cost_curr, min=5)
                
                loss = (kl_obj  + loss_recon).mean(dim=0)
                elbo = (kl_cost_curr + loss_recon).mean(dim=0)

                loss_    += [loss.item()]
                elbo_    += [elbo.item()]
                kl_cost_ += [kl_cost_curr.mean(dim=0).item()]
                kl_obj_  += [kl_obj.mean(dim=0).item()]
                recon_   += [loss_recon.mean(dim=0).item()]

                print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)
                print_and_log_scalar(writer, 'valid/elbo', mn(elbo_), writes)
                print_and_log_scalar(writer, 'valid/kl_cost_', mn(kl_cost_), writes)
                print_and_log_scalar(writer, 'valid/kl_obj_', mn(kl_obj_), writes)
                print_and_log_scalar(writer, 'valid/recon', mn(recon_), writes)
                loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

            # if epoch % 10 == 0:
            #     with open(os.path.join(args.base_dir, 'samples/test_{}.npz'.format(epoch)), 'wb') as f: 
            #         recon = recon[:ns].cpu().data.numpy()
            #         np.save(f, recon)
            #         # print('saved test recons')
               
            #     sample = model.sample()
            #     with open(os.path.join(args.base_dir, 'samples/sample_{}.npz'.format(epoch)), 'wb') as f: 
            #         sample = sample.cpu().data.numpy()
            #         np.save(f, recon)
                
            #     # print('saved model samples')
                
            # if epoch == 0: 
            #     with open(os.path.join(args.base_dir, 'samples/real.npz'), 'wb') as f: 
            #         static_img = static_img.cpu().data.numpy()
            #         np.save(f, static_img)
                
                # print('saved real LiDAR')

    
    torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
