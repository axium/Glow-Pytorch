import torch 
from torchvision import datasets
import torchvision.transforms as transforms
from glow import Glow
import numpy as np
import skimage.io as sio
import matplotlib.pyplot as plt
import os
import json


# constants
K           = 16
L           = 3
coupling    = "affine"
last_zeros  = True
batchsize   = 256
size        = 16
lr          = 1e-4
n           = size*size
n_bits_x    = 8
epochs      = 1000
warmup_iter = 0
sample_freq = 50
save_freq   = 1000
device      = "cuda"
save_path   = "mnist/mnist_%dx%d/training/glowmodel.pt"%(size,size)

# saving configurations
config_path = "mnist/mnist_%dx%d/training/configs.json"%(size,size)
configs     = {"K":K,
               "L":L,
               "coupling":"affine",
               "last_zeros":True,
               "batchsize":batchsize,
               "size":size,
               "lr": lr,
               "n_bits_x":n_bits_x,
               "warmup_iter":warmup_iter}
with open(config_path, 'w') as f:
    json.dump(configs, f,sort_keys=True,indent=4, ensure_ascii=False)

# setting up dataloader
trans      = transforms.Compose([transforms.Resize((size,size)), 
                                 transforms.ToTensor()])
dataset    = datasets.MNIST(root="./mnist/data/", transform=trans)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                            drop_last=True, shuffle=True)
# loading GLOW model
glow = Glow((1,size,size),
            K=K,L=L,coupling=coupling,n_bits_x=n_bits_x,
            nn_init_last_zeros=last_zeros,
            device=device)

# loading pre-trained model
if os.path.exists(save_path):
    glow.load_state_dict(torch.load(save_path))
    glow.set_actnorm_init()
    print("Pre-Trained Model Loaded")
    print("Actnorm Initied")

# setting up optimizer and learning rate scheduler
opt          = torch.optim.Adam(glow.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",
                                                          factor=0.5,
                                                          patience=1000,
                                                          verbose=True,
                                                          min_lr=1e-8)
# training code 
global_step = 0
global_loss = []
warmup_completed = False
for i in range(epochs):
    Loss_epoch = []
    for j, data in enumerate(dataloader):
        opt.zero_grad()
        glow.zero_grad()
        
        # loading batch
        x = data[0].cuda()*255
        
        # pre-processing data
        x = glow.preprocess(x)
        n,c,h,w = x.size()
        nll,logdet,logpz,z_mu,z_std = glow.nll_loss(x)
        if global_step == 0:
            global_step += 1
            continue
        
        # backpropogating loss and gradient clipping
        nll.backward()
        torch.nn.utils.clip_grad_value_(glow.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(glow.parameters(), 100)
        
        # linearly increase learning rate till warmup_iter to lr
        if global_step <= warmup_iter:
            warmup_lr = lr / warmup_iter * global_step
            for params in opt.param_groups:
                params["lr"] = warmup_lr
        
        # taking optimizer step                        
        opt.step()
        
        # learning rate scheduling after warm up iterations
        if global_step > warmup_iter:
            lr_scheduler.step(nll)
            if not warmup_completed:
                print("\nWarm up Completed")
            warmup_completed = True
            
        # printing training metrics 
        print("\repoch=%0.2d..nll=%0.2f..logdet=%0.2f..logpz=%0.2f..mu=%0.2f..std=%0.2f"
              %(i,nll.item(),logdet,logpz,z_mu,z_std),end="\r")
        try:
            if j % sample_freq == 0:
                with torch.no_grad():
                    z_sample, z_sample_t = glow.generate_z(n=50,mu=0,std=0.7,to_torch=True)
                    x_gen = glow(z_sample_t, reverse=True)
                    x_gen = glow.postprocess(x_gen)
                    x_gen = x_gen.data.cpu().numpy()
                    x_gen = x_gen.transpose([0,2,3,1])
                    if x_gen.shape[-1] == 1:
                        x_gen = x_gen[...,0]
                    sio.imshow_collection(x_gen)
                    plt.savefig("./fig/%d.jpg"%global_step)
                    plt.close()
        except:
            print("\nFailed at Global Step = %d"%global_step)
        global_step = global_step + 1
        global_loss.append(nll.item())
        if global_step % save_freq == 0:
            torch.save(glow.state_dict(), save_path)
    
# model visualization 
temperature = [0.8]
for temp in temperature:
    with torch.no_grad():
        glow.eval()
        z_sample, z_sample_t = glow.generate_z(n=50,mu=0,std=0.7,to_torch=True)
        x_gen = glow(z_sample_t, reverse=True)
        x_gen = glow.postprocess(x_gen)
        x_gen = x_gen.data.cpu().numpy()
        x_gen = x_gen.transpose([0,2,3,1])
        if x_gen.shape[-1] == 1:
            x_gen = x_gen[...,0]
        sio.imshow_collection(x_gen)
        
# saving model weights
torch.save(glow.state_dict(), save_path)    

