import torch
import torch.utils.data

import pandas as pd
import numpy as np
import gc
import copy

import transforms3d
import open3d as o3d

DEBUG = False
VOXEL_SZ=0.2

def getint(name):
    try:
        return int(name.split('.')[0])
    except Exception as e:
        print("Error occured while trying to read {}".format(name))
    return None


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
    img = img * 25
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

def parallel_slam(parallel_arg):
    recon_curr_frame, recon_next_frame = parallel_arg[0], parallel_arg[1]

    pose_transform, pose_err = get_slam_pose_transform(recon_curr_frame, recon_next_frame)
    return pose_transform