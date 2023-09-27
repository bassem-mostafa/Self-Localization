import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512" # linux export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

import torch
import numpy

from pathlib import Path
from Memory_Monitor import _memory_monitor
from nets.aliked import ALIKED
from tqdm import tqdm
from Dataset import HPatchesSquenceDataset

if __name__ == "__main__":
    print(f"Training Prepare Demo")
    
    train_ldr = HPatchesSquenceDataset()
    net = ALIKED(model_name="aliked-n16rot",
                         device="cpu",
                         top_k=10,
                         scores_th=0.2,
                         n_limit=200,
                         )
    # for index, batch in tqdm(enumerate(train_ldr)):
    for batch in tqdm(train_ldr):
        # print(f"\n\t[{index}] >> image: {batch['path']}, shape: {batch['image'].shape}")

        with torch.no_grad():
            if f"cuda" in net.device:
                torch.cuda.empty_cache()
                torch.cuda.synchronize(net.device)
            image = batch['image']
            image = torch.Tensor(image).to(net.device).unsqueeze(0)
            output = net(image)
            output = output["keypoints"][0]
            _, _, h, w = image.shape
            wh = torch.tensor([w - 1, h - 1],device=output.device)
            output = wh*(output+1)/2
            output = output.cpu().numpy()
            output = output.astype(int)
            
            # Mark the target pixels to be keypoints
            if not Path(f"{batch['path']}.npy").exists():
                target = numpy.zeros(image.shape[2:])
                # print(f"\nInitialized shape {target.shape}")
            else:
                target = numpy.load(f"{batch['path']}.npy")
                # print(f"\nLoaded shape {target.shape}")
                # os.remove(f"{batch['path']}.npy")
                # print(f"Deleted {batch['path']}.npy")
                # continues
            target[output[:, 1], output[:, 0]] = 1
            numpy.save(f"{batch['path']}", target)
            # print(f"\nSaved shape {target.shape}")