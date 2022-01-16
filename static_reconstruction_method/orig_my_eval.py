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





# Encoder must be trained with all types of frames,dynmaic, static all

args = parser.parse_args()
model = VAE(args).cuda()
model_file = '/home/sabyasachi/Projects/ati/ati_motors/adversarial_based/prashVAE/unet/models/gen_5.pth'
print("Loading model from {}".format(model_file))
network=torch.load(model_file)
model.load_state_dict(network)
model.eval()




# kitti='kitti_data/'
# data=np.load(kitti+'lidar_test.npz')
# # print(data.shape)  		#(832,60,512,4)


# x=onePC=data[0:1,:,:,:]
# x=onePC.reshape([60,512,4])

# dataset_val   = np.load('../lidar_generation/kitti_data/lidar_val.npz') 

def getint(name):
    return int(name.split('.')[0])

# dataset_val   = np.load('/home/prashant/P/carla-0.8.4/PythonClient/_out_npy/carlaData/0_2/val.npy') 
test_folder = "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/few_dynamic_runs/110k/pair/dynamic_out_npy"
test_file = os.path.join(test_folder, sorted(os.listdir(test_folder), key=getint)[-1])
dataset_val   = np.load(test_file)
   



# print(dataset_val.shape)    #(154,60,512,4)
# exit(1)
# print(dataset_val[0:1,:,:,:].shape)  #[1,60,512,4] 
dataset_val   = preprocess(dataset_val).astype('float32')
# print("Here")
# print(type(dataset_val))
# exit(1)
# dataset_val1   = preprocess(dataset_val[0:1,:,:,:]).astype('float32')
# print(dataset_val1.shape)      #(1,2,40,256)

# print(dataset_val.shape)          #(152,2,40,256)
# exit(1)

val_loader    = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)
process_input = from_polar if args.no_polar else lambda x : x
# print 





recons=[]
original=[]


for i, img in enumerate(val_loader):
	print(i)
	# if(i!=10):
	# 	continue

	# print(type(img))								
	# exit(1)
	img = img.cuda()
	print(type(img))								#torch.Tensor
	# print(img.shape)       						#[64,2,40,256]

	# print(((process_input(img))).shape)  # 			[64,3,40,256] 
	
	print(((process_input(img))[0:1,:,:,:]).shape)  #[1,3,40,256] 
	#print(show_pc_lite((process_input(img))[0:1,:,:,:]))
	# exit(1)
	recon, kl_cost,hidden_z = model(process_input(img))
	print(type(recon))
	recons=recon
	original=img
	# print(recon.detach().shape)  					[64,2,40,256]
	# print((from_polar(recon.detach())).shape)     [64, 3, 40, 256]      #this is is polar
	# print((recon[0:1,:,:,:]).shape)				([1, 2, 40, 256]
	#print(show_pc_lite(from_polar((recon[0:1,:,:,:]).detach())))
	break
	


recons_temp=np.array(recons.detach().cpu())    			#(64, 2, 40, 256)
original_temp=np.array(original.detach().cpu())    	    #(64, 2, 40, 256)
# recons_temp=np.array(recons)    			#(64, 2, 40, 256)
# original_temp=np.array(original)    			#(64, 2, 40, 256)

# print((recons_temp).shape)
# print((original_temp).shape)

if not os.path.exists("samples/reconstructed"):
    os.makedirs("samples/reconstructed")
else:
    shutil.rmtree("samples/reconstructed")
    os.makedirs("samples/reconstructed")
if not os.path.exists("samples/original"):
    os.makedirs("samples/original")
else:
    shutil.rmtree("samples/original")
    os.makedirs("samples/original")

for frame_num in range(recons_temp.shape[0]):
	frame=from_polar(recons[frame_num:frame_num+1,:,:,:]).detach().cpu()
	plt.figure()
	plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
	plt.savefig('samples/reconstructed/'+str(frame_num)+'.jpg') 
	# plt.show()
	plt.close()

for frame_num in range(original_temp.shape[0]):
	frame=from_polar(original[frame_num:frame_num+1,:,:,:]).detach().cpu()
	plt.figure()
	plt.scatter(frame[:, 0], frame[:, 1], s=0.7, color='k')
	plt.savefig('samples/original/'+str(frame_num)+'.jpg') 
	# plt.show()
	plt.close()


print('final exit from my_eval')

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




