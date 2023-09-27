# The following section MUST exist at the beginning of the script
# ==============================================================================
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:512"
# ==============================================================================

import cv2
import torch

from torch.nn import Module, Unfold, Fold, AvgPool2d, Upsample, Sequential, Conv2d, BatchNorm2d, ReLU, Softmax2d
from torchvision.transforms import ToTensor
from traceback import print_exc
from time import time, localtime
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

class _preprocess(Module):
    """
    Divides a batch of images into batch of patches
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.unfold_param = dict(
                                kernel_size = (20, 20),         # kernel size (height, width), which represents the patch spatial (height, width)
                                dilation = (1, 1),              # always (1, 1), means no dilation for kernel cells
                                padding = (10, 10),             # floor(`kernel_size` / 2)
                                stride = (10, 10),              # floor(`kernel_size` / 2)
                                )
        self.unfold = Unfold(**self.unfold_param)
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, images_batch):
        images_batch = images_batch.to(self.device)                                     # Move `images_batch` to used device
        N, C, H, W = images_batch.shape                                                 # N, C, H, W
        N, C, H, W                                                                      # dummy line to remove editor unused warning
        kH, kW = self.unfold_param['kernel_size']                                       # kH, kW
        images_batch = images_batch.view(-1, 1, *images_batch.shape[-2:])               # NxC, 1, H, W
        patches = self.unfold(images_batch)                                             # NxC, 1xkHxkW, L
        patches = patches.permute(0, 2, 1)                                              # NxC, L, 1xkHxkW
        patches = patches.contiguous()                                                  # FIX contiguous memory
        patches = patches.view(-1, 1, kH, kW)                                           # NxCxL, 1, kH, kW
        return patches # NxCxL, 1, kH, kW

class _feature_encoder(Module):
    """
    Extracts multi scale features from patches
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.div_1 = Sequential(
                               Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                     in_channels = 1,
                                     out_channels = 4,
                                     kernel_size = (3, 3),
                                     stride = (1, 1),
                                     padding = (1, 1),
                                     dilation = (1, 1),
                                     groups = 1,
                                     bias = True,
                                     padding_mode = "zeros",
                                     ),
                               BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                          num_features = 4, # previous `out_channels`
                                          ),
                               ReLU( # number of parameters = 0
                                   inplace = True,
                                   ),
                               )
        self.div_2 = Sequential(
                               AvgPool2d( # number of parameters = 0
                                        kernel_size=(1, 2),
                                        ),
                               Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                     in_channels = 1,
                                     out_channels = 4,
                                     kernel_size = (3, 3),
                                     stride = (1, 1),
                                     padding = (1, 1),
                                     dilation = (1, 1),
                                     groups = 1,
                                     bias = True,
                                     padding_mode = "zeros",
                                     ),
                               BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                          num_features = 4, # previous `out_channels`
                                          ),
                               ReLU( # number of parameters = 0
                                   inplace = True,
                                   ),
                               )
        self.div_3 = Sequential(
                               AvgPool2d( # number of parameters = 0
                                        kernel_size=(2, 1),
                                        ),
                               Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                     in_channels = 1,
                                     out_channels = 4,
                                     kernel_size = (3, 3),
                                     stride = (1, 1),
                                     padding = (1, 1),
                                     dilation = (1, 1),
                                     groups = 1,
                                     bias = True,
                                     padding_mode = "zeros",
                                     ),
                               BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                          num_features = 4, # previous `out_channels`
                                          ),
                               ReLU( # number of parameters = 0
                                   inplace = True,
                                   ),
                               )
        self.div_4 = Sequential(
                               AvgPool2d( # number of parameters = 0
                                        kernel_size=(2, 2),
                                        ),
                               Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                     in_channels = 1,
                                     out_channels = 4,
                                     kernel_size = (3, 3),
                                     stride = (1, 1),
                                     padding = (1, 1),
                                     dilation = (1, 1),
                                     groups = 1,
                                     bias = True,
                                     padding_mode = "zeros",
                                     ),
                               BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                          num_features = 4, # previous `out_channels`
                                          ),
                               ReLU( # number of parameters = 0
                                   inplace = True,
                                   ),
                               )
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, patches):
        patches = patches.to(self.device)
        features = tuple(div_x(patches) for div_x in [self.div_1, self.div_2, self.div_3, self.div_4])
        return features # x1_feaures, x2_features, x3_features, ...

class _feature_aggregator(Module):
    """
    Aggregates multi scale features into combined features
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.div_1 = Sequential(
                               Upsample(
                                       size = None,                     #
                                       scale_factor = (1, 1),           #
                                       mode = "bilinear",               #
                                       align_corners = True,            #
                                       recompute_scale_factor = False,  #
                                       )
                               )
        self.div_2 = Sequential(
                               Upsample(
                                       size = None,                     #
                                       scale_factor = (1, 2),           #
                                       mode = "bilinear",               #
                                       align_corners = True,            #
                                       recompute_scale_factor = False,  #
                                       )
                               )
        self.div_3 = Sequential(
                               Upsample(
                                       size = None,                     #
                                       scale_factor = (2, 1),           #
                                       mode = "bilinear",               #
                                       align_corners = True,            #
                                       recompute_scale_factor = False,  #
                                       )
                               )
        self.div_4 = Sequential(
                               Upsample(
                                       size = None,                     #
                                       scale_factor = (2, 2),           #
                                       mode = "bilinear",               #
                                       align_corners = True,            #
                                       recompute_scale_factor = False,  #
                                       )
                               )
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, features):
        features = tuple(feature.to(self.device) for feature in features)
        features = tuple(div_x(feature) for feature, div_x in zip(features, [self.div_1, self.div_2, self.div_3, self.div_4]))
        features = torch.cat(features, dim = 1) # N, (C), H, W
        return features

class _keypoint_decoder(Module):
    """
    Detects patch keypoints
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.div_1 = Sequential(
                              # The following two convolutions applies the `depth-wise convolution` technique adapted from MobileNet
                              Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                    in_channels = 16,
                                    out_channels = 16,      # MUST be same as `in_channels` to achieve depth-wise convolution
                                    kernel_size = (3, 3),
                                    stride = (1, 1),
                                    padding = (1, 1),
                                    dilation = (1, 1),
                                    groups = 16,            # MUST be same as `in_channels` to achieve depth-wise convolution
                                    bias = True,
                                    padding_mode = "zeros",
                                    ),
                              Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                    in_channels = 16,
                                    out_channels = 1,
                                    kernel_size = (1, 1),   # MUST be `(1, 1)` to achieve point-wise convolution
                                    stride = (1, 1),
                                    padding = (0, 0),
                                    dilation = (1, 1),
                                    groups = 1,             # MUST be `1` to achieve point-wise convolution
                                    bias = True,
                                    padding_mode = "zeros",
                                    ),
                              BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                         num_features = 1, # previous `out_channels`
                                         ),
                              Softmax2d( # number of parameters = 0
                                       ),
                              )
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, features):
        features = features.to(self.device)
        keypointness_map = self.div_1(features)
        return keypointness_map

class _descriptor_decoder(Module):
    """
    Detects patch descriptors
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.div_1 = Sequential(
                              # AvgPool2d( # number of parameters = 0
                              #          kernel_size=(2, 2),
                              #          ),
                              Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                    in_channels = 16,
                                    out_channels = 16,      # MUST be same as `in_channels` to achieve depth-wise convolution
                                    kernel_size = (3, 3),
                                    stride = (1, 1),
                                    padding = (1, 1),
                                    dilation = (1, 1),
                                    groups = 16,            # MUST be same as `in_channels` to achieve depth-wise convolution
                                    bias = True,
                                    padding_mode = "zeros",
                                    ),
                              Conv2d( # number of parameters = `out_channels` * `kernel_size` * `in_channels`
                                    in_channels = 16,
                                    out_channels = 16,
                                    kernel_size = (1, 1),   # MUST be `(1, 1)` to achieve point-wise convolution
                                    stride = (1, 1),
                                    padding = (0, 0),
                                    dilation = (1, 1),
                                    groups = 1,             # MUST be `1` to achieve point-wise convolution
                                    bias = True,
                                    padding_mode = "zeros",
                                    ),
                              BatchNorm2d( # number of parameters = 2 * `num_features` where 2 stands for `mean` and `variance` per feature
                                         num_features = 16, # previous `out_channels`
                                         ),
                              ReLU( # number of parameters = 0
                                  inplace = True,
                                  ),
                              )
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, features):
        features = features.to(self.device)
        description_map = self.div_1(features)
        return description_map

class _postprocess(Module):
    """
    Combines patches of features into batch of features
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self.fold_param = dict(
                              output_size=None,               # Note: Initially `None` but will be overwritten in forward
                              # The following MUST be same as `_preprocess` unfold paramaters
                              kernel_size = (20, 20),         #
                              dilation = (1, 1),              #
                              padding = (10, 10),             #
                              stride = (10, 10),              #
                              )
        self.fold = Fold(**self.fold_param)
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, patches, images_batch):
        # patches = patches.to(self.device)
        # Note: Overwrite fold's `output_size` as it was NOT known on initialization
        self.fold.output_size = images_batch.shape[-2:]
        N, C, H, W = images_batch.shape                                 # N, C, H, W
        kH, kW = self.fold_param['kernel_size']                         # kH, kW
        _, pC, _, _ = patches.shape                                     # NxCxL, pC, kH, kW
        patches = patches.view(N*C, -1, pC*kH*kW)                       # NxC, L, pCxkHxkW
        patches = patches.permute(0, 2, 1)                              # NxC, pCxkHxkW, L
        batch = self.fold(patches)                                      # NxC, pC, H, W
        batch = batch.view(N, C, -1, H, W)                              # N, C, pC, H, W
        return batch

class LocalizerNet(Module):
    """
    Executes the whole network pipeline
    """
    def __init__(self, device = None):
        super().__init__()
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        ...
        self._split_size = 100
        self._preprocess = _preprocess(self.device)
        self._feature_encoder = _feature_encoder(self.device)
        self._feature_aggregator = _feature_aggregator(self.device)
        self._keypoint_decoder = _keypoint_decoder(self.device)
        self._descriptor_decoder = _descriptor_decoder(self.device)
        self._postprocess = _postprocess(self.device)
        ...
        self.to(self.device) # MUST BE LAST statement on `__init__(...)`

    def forward(self, images_batch):
        """
        Args:
            `images_batch`: Tensor of shape [N, C, H, W],
                            where `N`: number of images,
                                  `C`: image channels,
                                  `H`: image height,
                                  `W`: image width
        Returns:
            `keypoints`:
            `descriptors`:
        """
        if "cuda" in self.device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
        patches = self._preprocess(images_batch)     
        pN, _, pH, pW = patches.shape # NxCxL, 1, kH, kW
        block_start, block_end = iter(range(0, pN, self._split_size)), iter(range(self._split_size, pN + self._split_size, self._split_size))
        keypoints = torch.zeros(pN, 1, pH, pW)      # MUST follow `_keypoint_decoder` output shape (pN, (`1`), pH, pW)
        descriptors = torch.zeros(pN, 16, pH, pW)   # MUST follow `_descriptor_decoder` output shape (pN, (`16`), pH, pW)
        for batch in tqdm(torch.split(patches, self._split_size), leave = False):
            batch_features = self._feature_encoder(batch)
            batch_features = self._feature_aggregator(batch_features)
            batch_keypoints = self._keypoint_decoder(batch_features)
            batch_descriptors = self._descriptor_decoder(batch_features)
            start, end = next(block_start), next(block_end)
            keypoints[start:end] = batch_keypoints
            descriptors[start:end] = batch_descriptors
            
        keypoints = self._postprocess(keypoints, images_batch)          # N, C, 1, H, W
        descriptors = self._postprocess(descriptors, images_batch)      # N, C, 16, H, W
        keypoints = torch.max(keypoints, dim=1).values                  # N, 1, H, W
        descriptors = torch.mean(descriptors, dim=1)                    # N, 16, H, W
        return keypoints, descriptors

if __name__ == '__main__':
    print(f"="*80)
    print(f"Self Localization Network Demo")
    print(f"-"*80)
    
    parser = ArgumentParser(
                prog="Self Localization Network Demo",
                description='Self Localization Network Description',
                epilog='Self Localization Network Footer',
                )
    args = parser.parse_args()
    args.device = None
    args.input = "..\dataset(s)\hpatches\hpatches-sequences-release\i_dc"

    print(f"*"*80)
    print(f"Provided Args...")
    print(f"-"*80)
    print(f"\t{'device':20s}: {args.device}")
    print(f"\t{'input':20s}: {args.input}")
    print(f"*"*80)

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    net = LocalizerNet(device)
        
    _weight_pattern = f"Network_", "*", ".pt"
    _weights_path = Path("".join(_weight_pattern))
    _weights_path = sorted(Path.cwd().glob(str(_weights_path)))
    _weights_path = _weights_path[-1] if len(_weights_path) > 0 else None
    if _weights_path and _weights_path.exists():
        net.load_state_dict(torch.load(_weights_path))
    else:
        prefix, timestamp, extension = _weight_pattern
        timestamp = localtime(time())
        timestamp = f"{timestamp.tm_year:04d}{timestamp.tm_mon:02d}{timestamp.tm_mday:02d}_{timestamp.tm_hour:02d}{timestamp.tm_min:02d}{timestamp.tm_sec:02d}"
        _weights_path = Path("".join((prefix, timestamp, extension)))
        torch.save(net.state_dict(), _weights_path)

    if args.input:
        folder = Path(args.input)
        images = sorted(folder.glob("**/*.ppm"))
        assert len(images) > 0, f"Couldn't find images in `{args.input}`"
    else:
        raise NotImplementedError("Please provide source of images")

    print(f"*"*80)
    print(f"Evaluation Parameters...")
    print(f"-"*80)
    print(f"\t{'input':20s}: `{folder}`...")
    print(f"\t{'device':20s}: {device}")
    print(f"\t{'pre-trained':20s}: {_weights_path.stem + _weights_path.suffix}")
    print(f"\t{'# of Parameters':20s}: {sum(p.numel() for p in net.parameters())}")
    print(f"*"*80)
    
    try:
        with torch.no_grad(): # enable_grad() or no_grad()
            net.eval()  # sets model network to EVALUATION mode
            for image in tqdm(images):
                image_path = image
                print(f"*"*80)
                image = cv2.imread(str(image))
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Optional Conversion
                image = ToTensor()(image).unsqueeze(0)
                print(f"Image `{image_path}`")
                print(f"\tShape: {image.shape}")
                keypoints, descriptors = net(image)
                print(f"")
                print(f"Keypoints:")
                print(f"\t{keypoints.shape}")
                print(f"Descriptors:")
                print(f"\t{descriptors.shape}")
                print(f"*"*80)
    except:
        print_exc()
    input("Press Enter To Continue") # To Monitor GPU Utilization For That Batch
    print(f"Terminated")