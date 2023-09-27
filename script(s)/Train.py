# The following section MUST exist at the beginning of the script
# ==============================================================================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# ==============================================================================

import numpy
import torch

from torch.nn import Module, MSELoss
from torchvision.transforms import ToTensor
from traceback import print_exc
from time import time, localtime
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

from Network import LocalizerNet
from Dataset import HPatchesSquenceDataset

class KeypointLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = MSELoss()

    def forward(self, prediction, target):
        prediction = prediction.to(target.device)
        loss = self.loss_fn(prediction, target)
        return loss

class DescriptorLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = MSELoss()

    def forward(self, prediction, target):
        prediction = prediction.to(target.device)
        loss = self.loss_fn(prediction, target)
        return loss

class TrainLoss(Module):
    def __init__(self):
        super().__init__()
        self.KeypointLoss = KeypointLoss()
        self.DescriptorLoss = DescriptorLoss()
        
    def forward(self, prediction, target):
        prediction_keypoints, prediction_descriptors = prediction
        target_keypoints, target_descriptors = target, prediction_descriptors
        train_loss = self.KeypointLoss(prediction_keypoints, target_keypoints) \
                   + self.DescriptorLoss(prediction_descriptors, target_descriptors)
        return train_loss
    
if __name__ == "__main__":
    print(f"="*80)
    print(f"Self Localization Training Demo")
    print(f"-"*80)
    
    parser = ArgumentParser(
                prog="Self Localization Network Demo",
                description='Self Localization Network Description',
                epilog='Self Localization Network Footer',
                )
    args = parser.parse_args()
    args.device = None
    args.dataset = True
    args.input = None

    print(f"*"*80)
    print(f"Provided Args...")
    print(f"-"*80)
    print(f"\t{'device':20s}: {args.device}")
    print(f"\t{'dataset':20s}: {args.dataset}")
    print(f"\t{'input':20s}: {args.input}")
    print(f"*"*80)

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    logging_interval = 1 # in epochs
    checkpoint_interval = 1 # in epochs
    max_epochs = 100
    lrn_rate = 0.05
    split_size = 2000

    print(f"*"*80)
    print(f"Training Parameters...")
    print(f"-"*80)
    print(f"\t{'device':20s}: {device}")
    print(f"\t{'logging interval':20s}: {logging_interval}")
    print(f"\t{'checkpoint interval':20s}: {checkpoint_interval}")
    print(f"\t{'epochs':20s}: {max_epochs}")
    print(f"\t{'learning rate':20s}: {lrn_rate}")
    print(f"\t{'split size':20s}: {split_size}")
    print(f"*"*80)
    
    timestamp = lambda timestamp: f"{timestamp.tm_year:04d}-{timestamp.tm_mon:02d}-{timestamp.tm_mday:02d} {timestamp.tm_hour:02d}:{timestamp.tm_min:02d}:{timestamp.tm_sec:02d}"
    duration = lambda duration: f"{int(duration/(24*60*60)): 2d} Days {int(duration/(60*60)%24): 2d} Hour {int(duration/(60))%60: 2d} Min {duration-int(duration/60): 5.2f} Seconds"
    
    dataset = HPatchesSquenceDataset()
    
    ... # TODO Implement any training pre-requisites
    
    net = LocalizerNet(device)
    
    _weight_pattern = f"Network_", "*", ".pt"
    _weights_path = Path("".join(_weight_pattern))
    _weights_path = sorted(Path.cwd().glob(str(_weights_path)))
    _weights_path = _weights_path[-1] if len(_weights_path) > 0 else None
    if _weights_path and _weights_path.exists():
        net.load_state_dict(torch.load(_weights_path))
        print(f"Loaded `{_weights_path.stem}.{_weights_path.suffix}` checkpoint")
        
    loss_func = TrainLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lrn_rate)
    
    train_start = time()
    print(f"\nTraining Started @ {timestamp(localtime(train_start))}")
    try:
        with torch.enable_grad(): # enable_grad() or no_grad()
            net.train() # sets model network to TRAINING mode
            for epoch in tqdm(range(0, max_epochs)):
                epoch_loss = 0  # for one full epoch
                for batch in tqdm(dataset, leave=False):
                    image = ToTensor()(batch['image']).unsqueeze(0)                                                             # (1, C, H, W) with values normalized {0.0, 1.0}
                    target = torch.from_numpy(numpy.load(f"{batch['path']}.npy")).unsqueeze(0).unsqueeze(0).type(torch.float32) # (1, 1, H, W)
                    image_patches = net._preprocess(image)
                    target_patches = net._preprocess(torch.cat((target, target, target), dim = 1))
                    for patch, patch_target in zip(torch.split(image_patches, split_size), torch.split(target_patches, split_size)):
                        patch_predicted = net(patch)
                        loss_val = loss_func(patch_predicted, patch_target)
                        epoch_loss += loss_val.item()
                        loss_val.backward()                 # compute/accumulate gradients
                        optimizer.step()                    # update weights
                        optimizer.zero_grad()               # reset gradients to zeros
            
                if epoch % logging_interval == logging_interval-1:
                    print(f"epoch = {epoch:4d} loss = {epoch_loss:10.4f}")
                if epoch % checkpoint_interval == checkpoint_interval-1:
                    _weight_pattern = f"Network_", "*", ".pt"
                    prefix, timestamp, extension = _weight_pattern
                    timestamp = localtime(time())
                    timestamp = f"{timestamp.tm_year:04d}{timestamp.tm_mon:02d}{timestamp.tm_mday:02d}_{timestamp.tm_hour:02d}{timestamp.tm_min:02d}{timestamp.tm_sec:02d}"
                    _weights_path = Path("".join((prefix, timestamp, extension)))
                    torch.save(net.state_dict(), _weights_path)
                    print(f"checkpoint `{_weights_path.stem}{_weights_path.suffix}` saved")
    except:
        print_exc()
    train_end = time()
    print(f"\nTraining Finished @ {timestamp(localtime(train_end))}")
    print(f"Train Duration {duration(train_end - train_start)}")
    input(f"Press Enter To Terminate") # Halt to see GPU Utilization in case of failures
    