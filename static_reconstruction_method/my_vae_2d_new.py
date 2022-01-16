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
import pickle
import gc
import pandas as pd
import transforms3d

from utils import * 
from models import * 

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
gc.collect()

# --------------------- Setting same seeds for reproducibility ------------------------------
# reproducibility is good
#np.random.seed(0)
#torch.manual_seed(0)
#torch.cuda.manual_seed_all(0)

# -------------------- MODEL BUILDING -------------------------------------------------------
# construct model and ship to GPU

# model = VAE_filtered(args, n_filters=32).cuda()
model = Unet_filtered(args, n_filters=64).cuda()
print(model)
print(summary(model, input_size=(2, 16, 1024)))
gc.collect()
# assert False

model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr)#, amsgrad=True) 

MODEL_PATH = './trying_new_unet_correctly_64f_continued/models/gen_80.pth'
if(os.path.exists(MODEL_PATH)):
    network_state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(network_state_dict)
    print("Previous weights loaded from {}".format(MODEL_PATH))
else:
    print("Starting with new model")
#assert False
OPTIM_PATH = './trying_new_unet_correctly_64f_continued/optims/gen_80.pth'
if(os.path.exists(OPTIM_PATH)):
    network_optim_dict = torch.load(OPTIM_PATH)
    optim.load_state_dict(network_optim_dict['optimizer'])
    print("Previous optimizer loaded from {}".format(OPTIM_PATH))
    epoch_start = network_optim_dict['epoch']
    mask_factor = network_optim_dict['mask_factor']
    slam_factor = network_optim_dict['slam_factor']
else:
    print("Starting with new optim")
    mask_factor = 0.0
    slam_factor = 0.0
    epoch_start = 0
   
gc.collect()

# -------------------- TENSORBOARD SETUP FOR LOGGING -------------------------------------------
# Logging
#maybe_create_dir(os.path.join(args.base_dir, 'samples'))
maybe_create_dir(os.path.join(args.base_dir, 'models'))
maybe_create_dir(os.path.join(args.base_dir, 'optims'))

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

def dict_idx(that_dict, idx):
    return {k:v[idx] for k, v in that_dict.items()}

# Function to get transformation matrix for a given pose
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


class SemPairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset):
        super(SemPairdata, self).__init__()
        self.dataset_static_data  = dataset['static_data']
        self.dataset_dynamic_data = dataset['dynamic_data']
        self.dataset_label        = dataset['label']

        assert len(self.dataset_static_data) == len(self.dataset_dynamic_data) == len(self.dataset_label)

    def __len__(self):
        return self.dataset_static_data.shape[0] - 1 # We don't want a pair for last frame

    def __getitem__(self, index):
        index1 = index
        index2 = index+1 if index+1 < self.dataset_static_data.shape[0] else index  # The pair for last lidar frame is itself
        label_transform = get_gt_pose(self.dataset_label.iloc[index1].to_dict(), self.dataset_label.iloc[index2].to_dict())
        return index, self.dataset_static_data[index1], self.dataset_static_data[index2], self.dataset_dynamic_data[index1], self.dataset_dynamic_data[index2], label_transform
        # return index, self.dataset1[index],self.dataset2[index]


PREPROCESS_FRESHLY = False
# IMAGE_HEIGHT = -3

# STATIC_PREPROCESS_DATA_PATH  =  "../training_data/small_map/dynamic_high/static_prepreprocess_dir"
# DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map/dynamic_high/dynamic_prepreprocess_dir"
STATIC_PREPROCESS_DATA_PATH  =  "/home/saby/Projects/ati/ati_motors/adversarial_based/training_data/small_map/SEM_8_24_48_cropped/static_prepreprocess_dir"
DYNAMIC_PREPROCESS_DATA_PATH = "/home/saby/Projects/ati/ati_motors/adversarial_based/training_data/small_map/SEM_8_24_48_cropped/dynamic_prepreprocess_dir"
GT_PREPROCESS_DATA_PATH = "/home/saby/Projects/ati/ati_motors/adversarial_based/training_data/small_map/SEM_8_24_48_cropped/gt_prepreprocess_dir"


if PREPROCESS_FRESHLY:
    STATIC_TRAIN_FOLDER_PATH  = '/home/saby/Projects/ati/data/data/datasets/Carla/16beam-Data/small_map/testing/SEM_data/static_npy_data'
    DYNAMIC_TRAIN_FOLDER_PATH = '/home/saby/Projects/ati/data/data/datasets/Carla/16beam-Data/small_map/testing/SEM_data/dynamic_npy_data'
    GT_TRAIN_FOLDER_PATH      = '/home/saby/Projects/ati/data/data/datasets/Carla/16beam-Data/small_map/testing/SEM_data/gt_data'

    STATIC_VALID_FOLDER_PATH  =  STATIC_TRAIN_FOLDER_PATH
    DYNAMIC_VALID_FOLDER_PATH = DYNAMIC_TRAIN_FOLDER_PATH
    GT_VALID_FOLDER_PATH      =      GT_TRAIN_FOLDER_PATH
    LIDAR_RANGE = 100

    # train_file  = sorted(os.listdir( STATIC_TRAIN_FOLDER_PATH), key=getint)[0]
    val_file  = sorted(os.listdir( STATIC_TRAIN_FOLDER_PATH), key=getint)[-1]
    npyList = sorted(os.listdir(STATIC_TRAIN_FOLDER_PATH), key=getint)[:-1]

    # val_gt_file = sorted(os.listdir(GT_TRAIN_FOLDER_PATH), key=getgtint)[-1]
    # npy_gt_List = sorted(os.listdir(GT_TRAIN_FOLDER_PATH), key=getgtint)[:-1]


    print("Static preprocessing of:")
    if not os.path.exists(STATIC_PREPROCESS_DATA_PATH):
        os.makedirs(STATIC_PREPROCESS_DATA_PATH)
    else:
        shutil.rmtree(STATIC_PREPROCESS_DATA_PATH)
        os.makedirs(STATIC_PREPROCESS_DATA_PATH)

    print("validation dataset:")
    static_dataset_val = np.load(os.path.join(STATIC_VALID_FOLDER_PATH, val_file))
    static_dataset_val = preprocess(static_dataset_val, LIDAR_RANGE).astype('float32')
    with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), 'wb') as pfile:
        pickle.dump(static_dataset_val, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
        
    print("training dataset:")
    for file in tqdm(npyList):
        static_dataset_train = np.load(os.path.join(STATIC_TRAIN_FOLDER_PATH, file))
        static_dataset_train = preprocess(static_dataset_train, LIDAR_RANGE).astype('float32')
        with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, file), 'wb') as pfile:
            pickle.dump(static_dataset_train, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()

    print("Dynamic preprocessing of:")
    if not os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH):
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)
    else:
        shutil.rmtree(DYNAMIC_PREPROCESS_DATA_PATH)
        os.makedirs(DYNAMIC_PREPROCESS_DATA_PATH)

    print("validation dataset:")    
    dynamic_dataset_val   = np.load(os.path.join(DYNAMIC_VALID_FOLDER_PATH, val_file))
    dynamic_dataset_val   = preprocess(dynamic_dataset_val, LIDAR_RANGE).astype('float32')
    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), 'wb') as pfile:
        pickle.dump(dynamic_dataset_val, pfile, protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()

    print("training dataset:")
    for file in tqdm(npyList):
        dynamic_dataset_train = np.load(os.path.join(DYNAMIC_TRAIN_FOLDER_PATH, file))
        dynamic_dataset_train = preprocess(dynamic_dataset_train, LIDAR_RANGE).astype('float32')
        with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file), 'wb') as pfile:
            pickle.dump(dynamic_dataset_train, pfile, protocol=pickle.HIGHEST_PROTOCOL)
        gc.collect()

    print("Preprocessing of labels:")   
    if not os.path.exists(GT_PREPROCESS_DATA_PATH):
        os.makedirs(GT_PREPROCESS_DATA_PATH)
    else:
        shutil.rmtree(GT_PREPROCESS_DATA_PATH)
        os.makedirs(GT_PREPROCESS_DATA_PATH)

    # Assuming same train and valid folder
    for file in tqdm(os.listdir(GT_TRAIN_FOLDER_PATH)):
        src_path = os.path.join(GT_TRAIN_FOLDER_PATH, file)
        dst_path = os.path.join(GT_PREPROCESS_DATA_PATH, file)
        shutil.copy(src_path, dst_path)

    print("Freshly processed datasets successfully! Exiting!")

    assert False

else:

    # train_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[0]
    val_file  = sorted(os.listdir(STATIC_PREPROCESS_DATA_PATH), key=getint)[-1]
    npyList =   sorted(os.listdir(STATIC_PREPROCESS_DATA_PATH), key=getint)[:-1]
    val_gt_file = sorted(os.listdir(GT_PREPROCESS_DATA_PATH), key=getgtint)[-1]
    npy_gt_List = sorted(os.listdir(GT_PREPROCESS_DATA_PATH), key=getgtint)[:-1]

    if os.path.exists(STATIC_PREPROCESS_DATA_PATH) and os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH) and os.path.exists(GT_PREPROCESS_DATA_PATH):
        print("Have already preprocessed datasets at {} and {} and {}".format(STATIC_PREPROCESS_DATA_PATH, DYNAMIC_PREPROCESS_DATA_PATH, GT_PREPROCESS_DATA_PATH))
    else:
        print("No preprocessed datasets at {} and {} and ".format(STATIC_PREPROCESS_DATA_PATH, DYNAMIC_PREPROCESS_DATA_PATH, GT_PREPROCESS_DATA_PATH))
        assert False
    print("Load and create validation dataloaders")
    dataset_val = {}
    with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), 'rb') as pkl_file:
        dataset_val['static_data'] = pickle.load(pkl_file)
    gc.collect()
    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), 'rb') as pkl_file:
        dataset_val['dynamic_data'] = pickle.load(pkl_file)
    gc.collect()
    dataset_val['label'] = pd.read_csv(os.path.join(GT_PREPROCESS_DATA_PATH, val_gt_file))
    gc.collect()
    val_data = SemPairdata(dataset_val)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=True, num_workers=8, drop_last=False)
    gc.collect()


    print("Load and create training dataloaders")
    dataset_train_dict = {}
    for file, gt_file in tqdm(zip(npyList, npy_gt_List), total=len(npyList)):
        dataset_train = {}
        with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, file), 'rb') as pkl_file:
            dataset_train['static_data'] = pickle.load(pkl_file)
        gc.collect()
        with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file), 'rb') as pkl_file:
            dataset_train['dynamic_data'] = pickle.load(pkl_file)
        gc.collect()
        dataset_train['label'] = pd.read_csv(os.path.join(GT_PREPROCESS_DATA_PATH, gt_file))
        gc.collect()

        train_data = SemPairdata(dataset_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                            shuffle=True, num_workers=8, drop_last=False)
        gc.collect()
        dataset_train_dict[getint(file)] = train_loader


    print("Loaded and created dataloaders successfully!")

# assert False

# -------------------------------------------------------------------------------------------
#class to load own dataset

# static_val_loader = torch.utils.data.DataLoader(static_dataset_val, batch_size=args.batch_size,
#                     shuffle=False, num_workers=20, drop_last=False)
# dynamic_val_loader    = torch.utils.data.DataLoader(dynamic_dataset_val, batch_size=args.batch_size,
#                     shuffle=False, num_workers=20, drop_last=False)

# train_data = SemPairdata(dynamic_dataset_train, static_dataset_train)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
#                     shuffle=True, num_workers=8, drop_last=False)
# gc.collect()

# mask_factor = 0.0

    # VAE training
# ----------------------- AE TRAINING -------------------------------------------------------
rangee=150 if args.autoencoder else 300

def recon_loss_fn(recon, static, mask):
    Z_THRESHOLD = 2
    WT_FACTOR = 10
    MASK_WT_FACTOR = 5

    LIDAR_RANGE = 100
    Z_THRESHOLD /= LIDAR_RANGE

    shape_tuple = (static.shape[0],1,static.shape[2],static.shape[3])
    z_val = static[:,1].view(shape_tuple)
    z_mask = z_val > Z_THRESHOLD
    # z_mask = torch.cat([z_mask, z_mask], axis=1)
    z_mask = torch.cat([z_mask, torch.ones(shape_tuple, dtype=torch.bool).cuda()], axis=1)  # give less weightening to only R channel of ground points

    loss_wt = torch.ones(static.shape)
    loss_wt[z_mask] = WT_FACTOR
    loss_wt = loss_wt.cuda()
    ground_wt_loss = ((recon - static) * loss_wt).abs().sum(-1).sum(-1).sum(-1)

    #bin_mask = mask[:,1].round().view((mask.shape[0], 1, mask.shape[2], mask.shape[3]))
    #bin_mask = torch.cat([bin_mask, bin_mask], axis=1)
    #bin_mask = bin_mask * MASK_WT_FACTOR
    #inpainting_wt_loss = ((recon - static) * bin_mask).abs().sum(-1).sum(-1).sum(-1)
    inpainting_wt_loss = torch.zeros_like(ground_wt_loss)

    # total_loss = ground_wt_loss + inpainting_wt_loss
    actual_loss = (recon - static).abs().sum(-1).sum(-1).sum(-1)
    return actual_loss, ground_wt_loss, inpainting_wt_loss

def mask_loss_fn(dynamic, recon, mask, static):
    # Assuming channel 1 to be dynamic 
    # if channel 1 rounds to 0 (static) then take points from dynamic (because these are static points in dynamic frame)
    # else if channel 1 rounds to 1 (dynamic) then take points from reconstructed static (because these are dynamic points in dynamic frame)
    shape_tuple = (mask.shape[0], 1, mask.shape[2], mask.shape[3])
    bin_mask = mask[:,1].round().view(shape_tuple)
    bin_mask = torch.cat([bin_mask, bin_mask], axis=1)

    new_recon = (dynamic * (1-bin_mask)) + (bin_mask * recon)
    this_loss = new_recon - static
    return this_loss.abs().sum(-1).sum(-1).sum(-1)

chamfer_loss_fn = get_chamfer_dist()

def slam_loss_fn(recon_curr, recon_next, gt_transform):
    slam_loss_list = []
    # For every pair in batch
    for frame_num in range(recon_curr.shape[0]):
        recon_curr_frame = recon_curr[frame_num:frame_num+1, :, :, :]
        recon_next_frame = recon_next[frame_num:frame_num+1, :, :, :]
        frame_gt_transform = gt_transform[frame_num]

        # Get SLAM Pose as blackbox
#         pose_transform, pose_err = get_slam_pose_transform(recon_curr_frame, recon_next_frame)
#         pose_transform = torch.from_numpy(pose_transform).float().cuda()

        # Transform reconstructed frames to xyz
        recon_curr_frame_3d = from_polar(recon_curr_frame)[0]
        recon_next_frame_3d = from_polar(recon_next_frame)

        recon_curr_frame_3d_flat  = torch.cat([recon_curr_frame_3d[0].flatten().unsqueeze(0),
                                            recon_curr_frame_3d[1].flatten().unsqueeze(0),
                                            recon_curr_frame_3d[2].flatten().unsqueeze(0),
                                            torch.zeros_like(recon_curr_frame_3d[0].flatten().unsqueeze(0))],
                                         0)


        # Apply SLAM transformation on src frame
#         recon_curr_frame_3d_flat_icp_transformed  = torch.mm(pose_transform, recon_curr_frame_3d_flat)
        recon_curr_frame_3d_flat_gt_transformed   = torch.mm(frame_gt_transform, recon_curr_frame_3d_flat)

#         recon_curr_frame_3d_icp_transformed = recon_curr_frame_3d_flat_icp_transformed[:3].view(recon_next_frame_3d.shape)
        recon_curr_frame_3d_gt_transformed  = recon_curr_frame_3d_flat_gt_transformed[:3].view(recon_next_frame_3d.shape)

        # Apply SLAM loss (like Chamfer Distance, etc) on the pair
#         slam_frame_loss = chamfer_loss_fn(recon_curr_frame_3d_icp_transformed, recon_next_frame_3d)
        gt_frame_loss = chamfer_loss_fn(recon_curr_frame_3d_gt_transformed, recon_next_frame_3d)
        frame_loss = torch.abs(gt_frame_loss)
#         frame_loss = torch.abs(gt_frame_loss - slam_frame_loss)
        # frame_loss_squared = frame_loss * frame_loss

        # Append SLAM loss to the list
        slam_loss_list.append(frame_loss)
        gc.collect()

    # Create SLAM loss tensor
    loss_recon = torch.cat(slam_loss_list)
    return loss_recon

# print("Begin training:")
for epoch in range(epoch_start, 1000):
    print('Epoch #%s' % epoch)
    model.train()
    #loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    recon_act_, recon_gwt_, recon_inwt_, mask_, slam_, total_ = [[] for _ in range(6)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x

    # FOR EVERY SMALL FILE
    print("Training: ")
    for key, train_loader in dataset_train_dict.items():
        print("Path #{} out of {} runs".format(key, len(dataset_train_dict)))
#         assert False
        
        # TRAIN HERE
        for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # if i == 100:
            #    break
            static_img_curr  = img_data[1].cuda()
            static_img_next  = img_data[2].cuda()
            dynamic_img_curr = img_data[3].cuda()
            dynamic_img_next = img_data[4].cuda()
            gt_transform     = img_data[5].float().cuda()

            # if i == 0:
            #     print(static_img.shape)
            #     print(dynamic_img.shape)

            recon_curr, mask_curr = model(process_input(dynamic_img_curr))
            recon_next, mask_next = model(process_input(dynamic_img_next))

            static_img_curr = process_input(static_img_curr)
            static_img_next = process_input(static_img_next)
            # if i == 0:
                # print(recon.shape)

            # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
            loss_recon_act_curr, loss_recon_gwt_curr, loss_recon_inwt_curr = recon_loss_fn(recon_curr, static_img_curr, mask_curr)
            loss_recon_act_next, loss_recon_gwt_next, loss_recon_inwt_next = recon_loss_fn(recon_next, static_img_next, mask_next)
            loss_recon_act  = loss_recon_act_curr  + loss_recon_act_next
            loss_recon_gwt  = loss_recon_gwt_curr  + loss_recon_gwt_next
            loss_recon_inwt = loss_recon_inwt_curr + loss_recon_inwt_next
            loss_recon_tot = loss_recon_gwt + (mask_factor * loss_recon_inwt)      # because inpainting wt depends on mask generated

            loss_mask = mask_loss_fn(dynamic_img_curr, recon_curr, mask_curr, static_img_curr) + mask_loss_fn(dynamic_img_next, recon_next, mask_next, static_img_next)
            loss_slam = slam_loss_fn(recon_curr, recon_next, gt_transform)
            # print("Recon loss: {} | Mask loss: {} | SLAM loss: {}".format(loss_recon, loss_mask, loss_slam))
            # assert False
            loss_recon = loss_recon_tot + (mask_factor * loss_mask) + (slam_factor * loss_slam)
            # if i == 0:
            #     print(loss_recon)
            #     assert False

            #if args.autoencoder:
            #    kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
            #else:
            #    kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
            #                    torch.clamp(kl_cost, min=5)

            #loss = (kl_obj  + loss_recon).mean(dim=0)
            loss = loss_recon.mean(dim=0)
            #elbo = (kl_cost + loss_recon).mean(dim=0)

            #loss_    += [loss.item()]
            #elbo_    += [elbo.item()]
            #kl_cost_ += [kl_cost.mean(dim=0).item()]
            #kl_obj_  += [kl_obj.mean(dim=0).item()]
            #recon_   += [loss_recon.mean(dim=0).item()]

            recon_act_  += [loss_recon_act.mean(dim=0).item()]
            recon_gwt_  += [loss_recon_gwt.mean(dim=0).item()]
            recon_inwt_ += [loss_recon_inwt.mean(dim=0).item()]
            mask_      += [loss_mask.mean(dim=0).item()]
            slam_      += [loss_slam.mean(dim=0).item()]
            total_     += [loss_recon.mean(dim=0).item()]


            # baseline loss is very memory heavy 
            # we accumulate gradient to simulate a bigger minibatch
            # if (i+1) % factor == 0 or not is_baseline: 
            optim.zero_grad()

            loss.backward()
            # if (i+1) % factor == 0 or not is_baseline: 
            optim.step()
        
    #####

    writes += 1
    mn = lambda x : np.mean(x)
    #print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    #print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
    #print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
    #print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
    #print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)

    print_and_log_scalar(writer, 'train/recon_act_', mn(recon_act_), writes)
    print_and_log_scalar(writer, 'train/recon_gwt_', mn(recon_gwt_), writes)
    print_and_log_scalar(writer, 'train/recon_inwt_',mn(recon_inwt_), writes)
    print_and_log_scalar(writer, 'train/mask_',      mn(mask_),      writes)
    print_and_log_scalar(writer, 'train/slam_',      mn(slam_),      writes)
    print_and_log_scalar(writer, 'train/total_',     mn(total_),     writes)


    #loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    recon_act_, recon_gwt_, recon_inwt_, mask_, slam_, total_ = [[] for _ in range(6)]
    gc.collect()
        
    # save some training reconstructions
    # if epoch % 5 == 0:
    #      recon = recon[:ns].cpu().data.numpy()
    #      with open(os.path.join(args.base_dir, 'samples/train_{}.npz'.format(epoch)), 'wb') as f: 
    #          np.save(f, recon)

         # print('saved training reconstructions')
         
    
    # Testing loop
    # --------------------------------------------------------------------------

    print("Validating: ")
    #loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    recon_act_, recon_gwt_, recon_inwt_, mask_, slam_, total_ = [[] for _ in range(6)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            # print('test set evaluation')
            for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):

                static_img_curr  = img_data[1].cuda()
                static_img_next  = img_data[2].cuda()
                dynamic_img_curr = img_data[3].cuda()
                dynamic_img_next = img_data[4].cuda()
                gt_transform     = img_data[5].float().cuda()

                # if i == 0:
                #     print(static_img.shape)
                #     print(dynamic_img.shape)

                recon_curr, mask_curr = model(process_input(dynamic_img_curr))
                recon_next, mask_next = model(process_input(dynamic_img_next))

                static_img_curr = process_input(static_img_curr)
                static_img_next = process_input(static_img_next)
                # if i == 0:
                    # print(recon.shape)

                # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
                loss_recon_act_curr, loss_recon_gwt_curr, loss_recon_inwt_curr = recon_loss_fn(recon_curr, static_img_curr, mask_curr)
                loss_recon_act_next, loss_recon_gwt_next, loss_recon_inwt_next = recon_loss_fn(recon_next, static_img_next, mask_next)
                loss_recon_act  = loss_recon_act_curr  + loss_recon_act_next
                loss_recon_gwt  = loss_recon_gwt_curr  + loss_recon_gwt_next
                loss_recon_inwt = loss_recon_inwt_curr + loss_recon_inwt_next
                loss_recon_tot = loss_recon_gwt + (mask_factor * loss_recon_inwt)      # because inpainting wt depends on mask generated

                loss_mask = mask_loss_fn(dynamic_img_curr, recon_curr, mask_curr, static_img_curr) + mask_loss_fn(dynamic_img_next, recon_next, mask_next, static_img_next)
                loss_slam = slam_loss_fn(recon_curr, recon_next, gt_transform)
                loss_recon = loss_recon_tot + (mask_factor * loss_mask) + (slam_factor * loss_slam)

                #if args.autoencoder:
                #    kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                #else:
                #    kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                #                    torch.clamp(kl_cost, min=5)
                
                #loss = (kl_obj  + loss_recon).mean(dim=0)
                loss = loss_recon.mean(dim=0)
                #elbo = (kl_cost + loss_recon).mean(dim=0)

                #loss_    += [loss.item()]
                #elbo_    += [elbo.item()]
                #kl_cost_ += [kl_cost.mean(dim=0).item()]
                #kl_obj_  += [kl_obj.mean(dim=0).item()]
                #recon_   += [loss_recon.mean(dim=0).item()]
                
                recon_act_  += [loss_recon_act.mean(dim=0).item()]
                recon_gwt_  += [loss_recon_gwt.mean(dim=0).item()]
                recon_inwt_ += [loss_recon_inwt.mean(dim=0).item()]
                mask_      += [loss_mask.mean(dim=0).item()]
                slam_      += [loss_slam.mean(dim=0).item()]
                total_     += [loss_recon.mean(dim=0).item()]

#             print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)
#             print_and_log_scalar(writer, 'valid/elbo', mn(elbo_), writes)
#             print_and_log_scalar(writer, 'valid/kl_cost_', mn(kl_cost_), writes)
#             print_and_log_scalar(writer, 'valid/kl_obj_', mn(kl_obj_), writes)
#             print_and_log_scalar(writer, 'valid/recon', mn(recon_), writes)
            print_and_log_scalar(writer, 'valid/recon_act_', mn(recon_act_), writes)
            print_and_log_scalar(writer, 'valid/recon_gwt_', mn(recon_gwt_), writes)
            print_and_log_scalar(writer, 'valid/recon_inwt_',mn(recon_inwt_), writes)
            print_and_log_scalar(writer, 'valid/mask_',      mn(mask_),      writes)
            print_and_log_scalar(writer, 'valid/slam_',      mn(slam_),      writes)
            print_and_log_scalar(writer, 'valid/total_',     mn(total_),     writes)

            #loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
            recon_act_, recon_gwt_, recon_inwt_, mask_, slam_, total_ = [[] for _ in range(6)]
            gc.collect()

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

    # assert False
    old_model_file =  os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch-1))
    new_model_file = os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch))
    torch.save(model.state_dict(), new_model_file)
    if os.path.exists(old_model_file):
        os.remove(old_model_file)


    old_optim_file = os.path.join(args.base_dir, 'optims/gen_{}.pth'.format(epoch-1))
    new_optim_file = os.path.join(args.base_dir, 'optims/gen_{}.pth'.format(epoch))
    optim_state = {'epoch':epoch, 'mask_factor': mask_factor, 'slam_factor':slam_factor, 'optimizer': optim.state_dict()}
    torch.save(optim_state, new_optim_file)
    if os.path.exists(old_optim_file):
        os.remove(old_optim_file)

    if mask_factor < 1:
        mask_factor = mask_factor + 0.5
    if slam_factor < 2000:
        slam_factor = slam_factor + 50
    gc.collect()
