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
from tqdm import tqdm

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

LIDAR_RANGE = 100



# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()

model = VAE(args).cuda()
#model = Unet(args).cuda()
MODEL_USED_DATA_PARALLEL = False
# model_file = '/home/sabyasachi/Projects/ati/ati_motors/adversarial_based/prashVAE/small_map_unet_more_layers_restarted/models/gen_300.pth'
model_file = 'second_attempt_triple_data_1024/models/gen_50.pth'
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
# assert False

# network=torch.load(model_file)
# model.load_state_dict(network)
# model.eval()

def getint(name):
    return int(name.split('.')[0])

# STATIC_TRAIN_FOLDER_PATH  = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/static_out_npy/'
# DYNAMIC_TRAIN_FOLDER_PATH = '/home/sabyasachi/Projects/ati/data/data/datasets/Carla/small_map/110k/pair/dynamic_out_npy/'
STATIC_PREPROCESS_DATA_PATH  =  "../training_data/small_map/dynamic_high_med/static_prepreprocess_dir"
DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map/dynamic_high_med/dynamic_prepreprocess_dir"

train_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[0]
val_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[-1]

print("Loading dynamic file")
dataset_val = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)
print("Loading static file")
static_dataset_val = np.load(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)

class Pairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset1,dataset2):
        super(Pairdata, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __len__(self):
        return self.dataset1.shape[0]

    def __getitem__(self, index):
        
        return index, self.dataset1[index],self.dataset2[index]

val_data = Pairdata(dataset_val, static_dataset_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=False, num_workers=20, drop_last=False)

process_input = from_polar if args.no_polar else lambda x : x
process_output = lambda x : x if args.no_polar else from_polar


for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
    dynamic_img = img_data[1].cuda()
    static_img = img_data[2].cuda()
    recon, kl_cost, hidden_z = model(process_input(dynamic_img))
    static_img = process_input(static_img)
    print((recon - static_img).abs().sum(-1).sum(-1).sum(-1))
    break
