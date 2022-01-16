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

LIDAR_RANGE = 100
min_a, max_a = -70, 70
min_b, max_b = -70, 70


# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()

#model = VAE_filtered(args, n_filters=64).cuda()
model = Unet_filtered(args, n_filters=64).cuda()
#model = Unet(args).cuda()
MODEL_USED_DATA_PARALLEL = False
# model_file = '/home/sabyasachi/Projects/ati/ati_motors/adversarial_based/prashVAE/small_map_unet_more_layers_restarted/models/gen_300.pth'
model_file = 'trying_new_unet_correctly_64f/models/gen_42.pth'
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
STATIC_PREPROCESS_DATA_PATH  =  "../training_data/small_map/SEM_8_24_48_cropped/static_prepreprocess_dir"
DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map/SEM_8_24_48_cropped/dynamic_prepreprocess_dir"
#STATIC_PREPROCESS_DATA_PATH  =  "../training_data/colon_neg_three_small_map/dynamic_high_med/static_prepreprocess_dir"
#DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/colon_neg_three_small_map/dynamic_high_med/dynamic_prepreprocess_dir"


train_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[0]
#val_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[-1]
val_file = train_file

print("Loading dynamic file")
dataset_val = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)
print("Loading static file")
static_dataset_val = np.load(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), allow_pickle=True)

#dataset_val = dataset_val[:10000]
#static_dataset_val = static_dataset_val[:10000]
gc.collect()

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

val_data = Pairdata(dataset_val, static_dataset_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=False, num_workers=8, drop_last=False)


#val_loader    = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
process_input = from_polar if args.no_polar else lambda x : x
process_output = lambda x : x if args.no_polar else from_polar
# print 


if not os.path.exists("samples"):
    os.makedirs("samples/reconstructed")
    os.makedirs("samples/dynamic")
    os.makedirs("samples/masked_dynamic")
    os.makedirs("samples/masked_reconstructed")
    os.makedirs("samples/static")
else:
    shutil.rmtree("samples")
    os.makedirs("samples/reconstructed")
    os.makedirs("samples/dynamic")
    os.makedirs("samples/static")
    os.makedirs("samples/masked_dynamic")
    os.makedirs("samples/masked_reconstructed")

recons=[]
original=[]
staticc=[]

def masked_dynamic_recon(dynamic, recon, mask):
    shape_tuple = (mask.shape[0], 1, mask.shape[2], mask.shape[3])
    bin_mask = mask[:,1].round().view(shape_tuple)
    bin_mask = torch.cat([bin_mask, bin_mask], axis=1)
    new_recon = (dynamic * (1-bin_mask)) + (bin_mask * recon)
    new_dynamic = dynamic * (1-bin_mask)
    return new_dynamic, new_recon

#def masked_dynamic_recon(dynamic, recon, mask):
    # bin_mask = (mask[:,0] - mask[:,1]).round().view((mask.shape[0], 1, mask.shape[2], mask.shape[3]))
#    bin_mask = mask[:,1].round().view((mask.shape[0], 1, mask.shape[2], mask.shape[3]))
#    masked_dynamic = (dynamic * (1-mask))
#    masked_recon = (dynamic * (1-mask)) + (mask * recon)
#    return masked_dynamic, masked_recon

for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
#     if i != len(val_loader) - 1:
#         continue
#     print(i)
    # if(i!=10):
    # 	continue

    # print(type(img))								
    # exit(1)
    dynamic_img = img_data[1].cuda()
    static_img = img_data[2].cuda()
#     print(type(dynamic_img))								#torch.Tensor
    # print(img.shape)       						#[64,2,40,256]

    # print(((process_input(img))).shape)  # 			[64,3,40,256] 

#     print(((process_input(dynamic_img))[0:1,:,:,:]).shape)  #[1,3,40,256] 
    #print(show_pc_lite((process_input(img))[0:1,:,:,:]))
    # exit(1)
    recon, mask = model(process_input(dynamic_img))
    masked_dynamic, masked_recon = masked_dynamic_recon(dynamic_img, recon, mask)
#     print(new_recon.shape)
#     print(type(recon))
    recons=recon
    md_recons=masked_dynamic
    mr_recons=masked_recon
    original=dynamic_img
    staticc=static_img
    # print(recon.detach().shape)  					[64,2,40,256]
    # print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
    # print((recon[0:1,:,:,:]).shape)				([1, 2, 40, 256]
    #print(show_pc_lite(from_polar((recon[0:1,:,:,:]).detach())))

#     break

    md_recons_temp=np.array(md_recons.detach().cpu())
    mr_recons_temp=np.array(mr_recons.detach().cpu())
    recons_temp=np.array(recons.detach().cpu())    			#(64, 2, 40, 256)
    original_temp=np.array(original.detach().cpu())    	    #(64, 2, 40, 256)
    staticc_temp=np.array(staticc.detach().cpu())
    # recons_temp=np.array(recons)    			#(64, 2, 40, 256)
    # original_temp=np.array(original)    			#(64, 2, 40, 256)

    # print((recons_temp).shape)
    # print((original_temp).shape)


    for frame_num in range(md_recons_temp.shape[0]):
        frame=from_polar(md_recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        # frame=process_output(recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([min_a,max_a])
        axes.set_ylim([min_b,max_b])
        plt.title("Masked Dynamic")
        plt.grid()
        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='c')
        plt.savefig('samples/masked_dynamic/'+str(i*args.batch_size+frame_num)+'.png') 
        # plt.show()
        plt.close()

    for frame_num in range(mr_recons_temp.shape[0]):
        frame=from_polar(mr_recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        # frame=process_output(recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([min_a,max_a])
        axes.set_ylim([min_b,max_b])
        plt.title("Masked Reconstructed")
        plt.grid()
        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='m')
        plt.savefig('samples/masked_reconstructed/'+str(i*args.batch_size+frame_num)+'.png') 
        # plt.show()
        plt.close()



    for frame_num in range(recons_temp.shape[0]):
        frame=from_polar(recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        # frame=process_output(recons[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([min_a,max_a])
        axes.set_ylim([min_b,max_b])
        plt.title("Reconstructed")
        plt.grid()
        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='g')
        plt.savefig('samples/reconstructed/'+str(i*args.batch_size+frame_num)+'.png') 
        # plt.show()
        plt.close()

    for frame_num in range(original_temp.shape[0]):
        frame=from_polar(original[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        # frame=process_input(original[frame_num:frame_num+1,:,:,:]).detach().cpu()  * LIDAR_RANGE
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([min_a,max_a])
        axes.set_ylim([min_b,max_b])
        plt.title("Dynamic")
        plt.grid()
        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='b')
        plt.savefig('samples/dynamic/'+str(i*args.batch_size+frame_num)+'.png') 
        # plt.show()
        plt.close()

    for frame_num in range(staticc_temp.shape[0]):
        frame=from_polar(staticc[frame_num:frame_num+1,:,:,:]).detach().cpu() * LIDAR_RANGE
        # frame=process_input(staticc[frame_num:frame_num+1,:,:,:]).detach().cpu()  * LIDAR_RANGE
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([min_a,max_a])
        axes.set_ylim([min_b,max_b])
        plt.title("Static")
        plt.grid()
        plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='r')
        plt.savefig('samples/static/'+str(i*args.batch_size+frame_num)+'.png') 
        # plt.show()
        plt.close()


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




