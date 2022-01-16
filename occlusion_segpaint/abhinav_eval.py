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
from utils import * 
from underfit_models import * 
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='VAE training of LiDAR')
parser.add_argument('--batch_size',         type=int,   default=64,             help='size of minibatch used during training')
parser.add_argument('--use_selu',           type=int,   default=0,              help='replaces batch_norm + act with SELU')
parser.add_argument('--base_dir',           type=str,                           help='root of experiment directory')
parser.add_argument('--no_polar',           type=int,   default=0,              help='if True, the representation used is (X,Y,Z), instead of (D, Z), where D=sqrt(X^2+Y^2)')
parser.add_argument('--lr',                 type=float, default=1e-3,           help='learning rate value')
parser.add_argument('--z_dim',              type=int,   default=256,             help='size of the bottleneck dimension in the VAE, or the latent noise size in GAN')
parser.add_argument('--autoencoder',        type=int,   default=1,              help='if True, we do not enforce the KL regularization cost in the VAE')
parser.add_argument('--atlas_baseline',     type=int,   default=0,              help='If true, Atlas model used. Also determines the number of primitives used in the model')
parser.add_argument('--panos_baseline',     type=int,   default=0,              help='If True, Model by Panos Achlioptas used')
parser.add_argument('--kl_warmup_epochs',   type=int,   default=150,            help='number of epochs before fully enforcing the KL loss')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--preprocessed', type = int, default = 0)
parser.add_argument('--path', type = str)
parser.add_argument('--dataset', type = str, default = 'carla')
parser.add_argument('--model_path', type = str, default = '')
# ------------------------ARGS stuff----------------------------------------------
def getint(name):
    try:
        print(name.split('.'))
        return int(name.split('.')[0])
    except Exception as e:
        print("Not sorting".format(name))
    return name


class add_noise_partial():
    def __init__(self):
        self.noise = list(range(5,40,5))/100

    def __call__(self, sample):
        idx, img, mask = sample
        print("asdsa")
        print(idx, img.shape)
        h,w = mask.shape
        ratio = np.random.randint(5,30)/100.0
        noise = np.random.choice(self.noise)
        numdyn = int(ratio*h*w)
        numdyn = numdyn - numdyn%3
        indices = np.random.randint(0,h*w,size = numdyn)
        inda,indb,indc = np.split(indices,3)
        mask.view(-1)[indices] = 1
        means = img.reshape((2, -1)).mean(-1)
        stds  = img.reshape((2, -1)).std(-1)
        noise_tensora = torch.zeros((numdyn//3,1)).normal_(0, noise)
        noise_tensorb = torch.zeros((numdyn//3,1)).normal_(0, noise)
        noise_tensorc = torch.zeros((numdyn,2)).normal_(0, noise)
        means, stds = [x.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) for x in [means, stds]]
         # normalize data
        norm = (img - means) / (stds + 1e-9)
        # add the noise
        norm.reshape((2,-1))[0][inda] = norm.reshape((2,-1))[0][inda] + noise_tensora
        norm.reshape((2,-1))[1][indb] = norm.reshape((2,-1))[1][indb] + noise_tensorb
        norm.reshape((2,-1))[indc] = norm.reshape((2,-1))[indc] + noise_tensorc
        # unnormalize
        unnorm = norm * (stds + 1e-9) + means
        print(unnorm.shape)
        return idx, unnorm, mask

class Pairdata(torch.utils.data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dataset1):
        super(Pairdata, self).__init__()
        self.dataset1 = dataset1

    def __len__(self):
        return self.dataset1.shape[0]

    def __getitem__(self, index):
        img = self.dataset1[index]
        if img.shape == (2,64,512):
            return index, img[:,5:45,:]
        elif img.shape == (2, 40, 256):
            img = F.interpolate(img,(img.shape[0], img.shape[1], 512))
        elif img.shape == (2, 64, 1024):
            return index,img[:,5:45,::2] 
        return index, img          
    def shape(self):
        return self.dataset1[0].shape

args = parser.parse_args()
model = VAE_dynseg(args).cuda()
if args.model_path == '':
    if args.dataset == 'carla':
        #model_file = 'carla_models/gen_9.pth'
        model_file = 'models_carla_610/gen_56.pth'
else:
    model_file = args.model_path
print("Loading model from {}".format(model_file))

network=torch.load(model_file)
model.load_state_dict(network)
print(summary(model, input_size=(2, 40, 512)))
model.eval()
preprocessed_path = args.path + "/preprocessed"
if os.path.exists(preprocessed_path):
    print("Preprocessed existsts")
 
if args.preprocessed and not os.path.exists(preprocessed_path):
    LIDAR_RANGE = 120
    print(os.listdir(args.path))
    npyList = sorted(os.listdir(args.path), key=getint)
    print(npyList) 
    os.makedirs(preprocessed_path)
    print("Processing:")
    for file in tqdm(npyList):
        print(file)
        dynamic_dataset_train = np.load(os.path.join(args.path,file))
        dynamic_dataset_train = preprocess(dynamic_dataset_train, LIDAR_RANGE).astype('float32')
        gc.collect()
        np.save(os.path.join(preprocessed_path, file),dynamic_dataset_train[:,:2])
        del dynamic_dataset_train
        gc.collect()


if os.path.exists(preprocessed_path):
    print("Have already preprocessed datasets at {}".format(preprocessed_path))
else:
    print("No preprocessed datasets at {} ".format(preprocessed_path))
    print("Considering data as preprocessed")
    preprocessed_path = args.path

npyList = sorted(os.listdir(preprocessed_path), key=getint)
print(npyList)

print("Loading and creating training datalaoders !")
train_loader_list = []
for file in tqdm(npyList):
    dynamic_dataset_train = np.load(os.path.join(preprocessed_path, file))
    gc.collect()
    train_data = Pairdata(dynamic_dataset_train)
    #print(train_data.shape())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                shuffle=False, num_workers=4, drop_last=False)#, transforms = transforms.Compose([add_noise_partial()]))
    gc.collect()
    train_loader_list.append(train_loader)
    gc.collect()

def loss_fn(model_output,target):
    loss=nn.CrossEntropyLoss()
    loss_value=loss(model_output,target[:,0].long())
    return loss_value
    
# Output is always in polar, however input can be (X,Y,Z) or (D,Z)
process_input = from_polar if args.no_polar else lambda x : x

#runmselist = []
#runacclist = []
for idx, train_loader in enumerate(train_loader_list):
    #batchmselist = []
    #batchacclist = []
    recons_list = []
    print(npyList[idx])
    mask_name = npyList[idx].split('.')[0]+'_mask.npy'
    for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        dynamic_img = img_data[1].cuda()
        print(dynamic_img.shape)
        recon, xQ = model(process_input(dynamic_img[:,:2,:,:]))
        recon = np.array(recon.detach().cpu())
        #recon[recon[:,0,:,:]>=0.51] = 1
        #recon[recon[:,1,:,:]<0.51] = 0
        #recon[recon[:,0,:,:]>=0.51] = 1
        #recon[recon[:,1,:,:]<0.51] = 0
        recons_list.append(recon)
    recons_array=np.vstack(recons_list)
    np.save(os.path.join(args.path, mask_name),recons_array)


   #     mse_batch = np.sum(np.square(1 - recon[:,1,:,:]))/recon[:,1,:,:].size
   #     acc_batch = recon[:,1,:,:][recon[:,1,:,:]>0.51].size/recon[:,1,:,:].size
    #    batchmselist.append(mse_batch)
    #    batchacclist.append(acc_batch)
    #batchmsearr = np.array(batchmselist)
    #batchaccarr = np.array(batchacclist)
    #print("For file {}: MSE:{} Acc: {}".format(npyList[idx], np.sum(batchmsearr)/batchmsearr.size, np.sum(batchaccarr)/batchaccarr.size))
#    runmselist.append(np.sum(batchmsearr)/batchmsearr.size)
#    runacclist.append(np.sum(batchaccarr)/batchaccarr.size)
#print("************************")
#runmsearr = np.array(runmselist)
#runaccarr = np.array(runacclist)
#print("Average for all runs: MSE: {} Acc:{}".format(np.sum(runmsearr)/runmsearr.size, np.sum(runaccarr)/runaccarr.size))
