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

model = VAE(args, n_filters=64).cuda()
#model = Unet(args).cuda()
print(model)
print(summary(model, input_size=(2, 16, 1024)))
gc.collect()
# assert False

model.apply(weights_init)
optim = optim.Adam(model.parameters(), lr=args.lr)#, amsgrad=True) 

MODEL_PATH = 'second_attempt_filtered_32f_triple_data_restarted_again1_correctly_1024/models/gen_42.pth'
if(os.path.exists(MODEL_PATH)):
    network_state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(network_state_dict)
    print("Previous weights loaded from {}".format(MODEL_PATH))
else:
    print("Starting from scratch")
#assert False
gc.collect()

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

PREPROCESS_FRESHLY = True
# IMAGE_HEIGHT = -3

# STATIC_PREPROCESS_DATA_PATH  =  "../training_data/small_map/dynamic_high/static_prepreprocess_dir"
# DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map/dynamic_high/dynamic_prepreprocess_dir"
STATIC_PREPROCESS_DATA_PATH  =  "../training_data/small_map/64beam/static_prepreprocess_dir"
DYNAMIC_PREPROCESS_DATA_PATH = "../training_data/small_map/64beam/dynamic_prepreprocess_dir"


if PREPROCESS_FRESHLY:
    STATIC_TRAIN_FOLDER_PATH  = '/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pair_transform/static_out_npy/'
    DYNAMIC_TRAIN_FOLDER_PATH = '/home/saby/Projects/ati/data/data/datasets/Carla/64beam-Data/pair_transform/dynamic_out_npy/'
    STATIC_VALID_FOLDER_PATH  =  STATIC_TRAIN_FOLDER_PATH
    DYNAMIC_VALID_FOLDER_PATH = DYNAMIC_TRAIN_FOLDER_PATH
    LIDAR_RANGE = 100

    #train_file  = sorted(os.listdir( STATIC_TRAIN_FOLDER_PATH), key=getint)[0]
    val_file  = sorted(os.listdir( STATIC_TRAIN_FOLDER_PATH), key=getint)[-1]
    npyList = sorted(os.listdir(STATIC_TRAIN_FOLDER_PATH), key=getint)[:-1]

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

    assert False

else:

    train_file  = sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[0]
    val_file  =   sorted(os.listdir( STATIC_PREPROCESS_DATA_PATH), key=getint)[-1]
#     npyList = sorted(os.listdir(STATIC_TRAIN_FOLDER_PATH), key=getint)[:-1]
    if os.path.exists(STATIC_PREPROCESS_DATA_PATH) and os.path.exists(DYNAMIC_PREPROCESS_DATA_PATH):
        print("Have already preprocessed datasets at {} and {}".format(STATIC_PREPROCESS_DATA_PATH, DYNAMIC_PREPROCESS_DATA_PATH))
    else:
        print("No preprocessed datasets at {} and {}".format(STATIC_PREPROCESS_DATA_PATH, DYNAMIC_PREPROCESS_DATA_PATH))
        assert False
    print("Loading static validation dataset")
    with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, val_file), 'rb') as pkl_file:
        static_dataset_val = pickle.load(pkl_file)
    gc.collect()
    print("Loading dynamic validation dataset")
    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, val_file), 'rb') as pkl_file:
        dynamic_dataset_val = pickle.load(pkl_file)
    gc.collect()

    print("Loading static training dataset")
    with open(os.path.join(STATIC_PREPROCESS_DATA_PATH, train_file), 'rb') as pkl_file:
        static_dataset_train = pickle.load(pkl_file)
    gc.collect()
    print("Loading dynamic training dataset")
    with open(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, train_file), 'rb') as pkl_file:
        dynamic_dataset_train = pickle.load(pkl_file)
    gc.collect()

# assert False

# -------------------------------------------------------------------------------------------
#class to load own dataset

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

# static_val_loader = torch.utils.data.DataLoader(static_dataset_val, batch_size=args.batch_size,
#                     shuffle=False, num_workers=20, drop_last=False)
# dynamic_val_loader    = torch.utils.data.DataLoader(dynamic_dataset_val, batch_size=args.batch_size,
#                     shuffle=False, num_workers=20, drop_last=False)

train_data = Pairdata(dynamic_dataset_train, static_dataset_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                    shuffle=True, num_workers=8, drop_last=False)
gc.collect()

val_data = Pairdata(dynamic_dataset_val, static_dataset_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                    shuffle=True, num_workers=8, drop_last=False)
gc.collect()

    # VAE training
# ----------------------- AE TRAINING -------------------------------------------------------
rangee=150 if args.autoencoder else 300
# print("Begin training:")
for epoch in range(500):
    print('Epoch #%s' % epoch)
    model.train()
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]

    # Output is always in polar, however input can be (X,Y,Z) or (D,Z)
    process_input = from_polar if args.no_polar else lambda x : x

    # FOR EVERY SMALL FILE
    print("Training: ")
    # for file in tqdm(npyList):
        # Load corresponding dataset batch
        # static_dataset_train = np.load(os.path.join(STATIC_PREPROCESS_DATA_PATH, file), allow_pickle=True)
        # dynamic_dataset_train = np.load(os.path.join(DYNAMIC_PREPROCESS_DATA_PATH, file), allow_pickle=True)
        
        # static_train_loader  = torch.utils.data.DataLoader(staticic_dataset_train, batch_size=args.batch_size,
        #                     shuffle=True, num_workers=20, drop_last=True)
        # dynamic_train_loader  = torch.utils.data.DataLoader(dynamic_dataset_train, batch_size=args.batch_size,
        #                     shuffle=True, num_workers=20, drop_last=True)

        # train_data = Pairdata(dynamic_dataset_train, static_dataset_train)
        # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
        #                     shuffle=False, num_workers=20, drop_last=False)

        # build loss function
    if args.atlas_baseline or args.panos_baseline:
        loss_fn = get_chamfer_dist()
    else:
        loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1) 

    # TRAIN HERE
    for i, img_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        # if i == 100:
        #    break
        dynamic_img = img_data[1].cuda()
        static_img  = img_data[2].cuda()
        
        recon = model(process_input(dynamic_img))
        static_img = process_input(static_img)
        # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
        loss_recon = loss_fn(recon, static_img)


        if args.autoencoder:
            kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
        else:
            kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                            torch.clamp(kl_cost, min=5)

        loss = (kl_obj  + loss_recon).mean(dim=0)
        elbo = (kl_cost + loss_recon).mean(dim=0)

        loss_    += [loss.item()]
        elbo_    += [elbo.item()]
        kl_cost_ += [kl_cost.mean(dim=0).item()]
        kl_obj_  += [kl_obj.mean(dim=0).item()]
        recon_   += [loss_recon.mean(dim=0).item()]

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
    print_and_log_scalar(writer, 'train/loss', mn(loss_), writes)
    print_and_log_scalar(writer, 'train/elbo', mn(elbo_), writes)
    print_and_log_scalar(writer, 'train/kl_cost_', mn(kl_cost_), writes)
    print_and_log_scalar(writer, 'train/kl_obj_', mn(kl_obj_), writes)
    print_and_log_scalar(writer, 'train/recon', mn(recon_), writes)
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
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
    loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
    with torch.no_grad():
        model.eval()
        if epoch % 1 == 0:
            # print('test set evaluation')
            for i, img_data in tqdm(enumerate(val_loader), total=len(val_loader)):
                dynamic_img = img_data[1].cuda()
                static_img  = img_data[2].cuda()

                recon = model(process_input(dynamic_img))
                static_img = process_input(static_img)
           
                # loss_recon = loss_fn(recon[:IMAGE_HEIGHT], static_img[:IMAGE_HEIGHT])
                loss_recon = loss_fn(recon, static_img)

                if args.autoencoder:
                    kl_obj, kl_cost = [torch.zeros_like(loss_recon)] * 2
                else:
                    kl_obj  =  min(1, float(epoch+1) / args.kl_warmup_epochs) * \
                                    torch.clamp(kl_cost, min=5)
                
                loss = (kl_obj  + loss_recon).mean(dim=0)
                elbo = (kl_cost + loss_recon).mean(dim=0)

                loss_    += [loss.item()]
                elbo_    += [elbo.item()]
                kl_cost_ += [kl_cost.mean(dim=0).item()]
                kl_obj_  += [kl_obj.mean(dim=0).item()]
                recon_   += [loss_recon.mean(dim=0).item()]

            print_and_log_scalar(writer, 'valid/loss', mn(loss_), writes)
            print_and_log_scalar(writer, 'valid/elbo', mn(elbo_), writes)
            print_and_log_scalar(writer, 'valid/kl_cost_', mn(kl_cost_), writes)
            print_and_log_scalar(writer, 'valid/kl_obj_', mn(kl_obj_), writes)
            print_and_log_scalar(writer, 'valid/recon', mn(recon_), writes)
            loss_, elbo_, kl_cost_, kl_obj_, recon_ = [[] for _ in range(5)]
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
    torch.save(model.state_dict(), os.path.join(args.base_dir, 'models/gen_{}.pth'.format(epoch)))
    gc.collect()
