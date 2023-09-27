# ------------------------------------------------------------------------------
#  Copyright (C) 2023 - Bassem Mostafa
# 
#  This software is distributed under the terms and conditions of the 'Apache-2.0'
#  license which can be found in the file 'LICENSE.txt' in this package distribution
#  or at 'http://www.apache.org/licenses/LICENSE-2.0'.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Self Localization Script
# --    Objective:
# --        Make use of surrounding objects/features for self-localization in GPS not available areas
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Reference(s)
# --    - 
# ------------------------------------------------------------------------------

try:

# ------------------------------------------------------------------------------
# -- Import Section
# ------------------------------------------------------------------------------

    import traceback
    import os
    import time
    import math
    import numpy as np
    import cv2
    import torch

# ------------------------------------------------------------------------------
# -- Configuration Section
# ------------------------------------------------------------------------------

    _VERSION = ['{:04}{:02}{:02}-{:02}{:02}'.format( *( _VERSION.tm_year, _VERSION.tm_mon, _VERSION.tm_mday, _VERSION.tm_hour, _VERSION.tm_min ) ) for _VERSION in [ time.localtime( os.path.getmtime( __file__ ) ) ] ][0]

    _ESC_KEY = 27
    _CAMERA_ID = 0

    # Q matrix - Camera parameters - Can also be found using stereoRectify
    Q = np.array(([1.0, 0.0, 0.0, -160.0],
                  [0.0, 1.0, 0.0, -120.0],
                  [0.0, 0.0, 0.0, 350.0],
                  [0.0, 0.0, 1.0/90.0, 0.0]),dtype=np.float32)


    # Load a MiDas model for depth estimation
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

# ------------------------------------------------------------------------------
# -- Wrapper Section
# ------------------------------------------------------------------------------

    # section wrappers
    def _print(*args):
        print(*args)
        pass

# ------------------------------------------------------------------------------
# -- Initialization Section
# ------------------------------------------------------------------------------

    _print( "{:<20} {}".format( "Self-Localization Script version", _VERSION ) )
    _print( "{:<20} {}".format( "OpenCV version", cv2.__version__ ) )

    # initialize random seed
    np.random.seed(42)
    
    # section camera initialization
    _Camera = cv2.VideoCapture(_CAMERA_ID)
    _print("Camera Ready: {}".format("OK" if _Camera.isOpened() else "No"))

# ------------------------------------------------------------------------------
# -- Methodology Section
# ------------------------------------------------------------------------------

    while _Camera.isOpened():

        # 1) Capture a frame
        _frame_available, _frame = _Camera.read()

        if not _frame_available:
            _print("\tFrame Not Captured !!")
            continue

        # 2) Convert frame content type into RGB
        _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)

        # 3) Save time snapshot for the start point
        _time_start = time.time()

        # Apply input transforms
        _depth_input = transform(_frame).to(device)

        # Prediction and resize to original resolution
        with torch.no_grad():
            _depth_prediction = midas(_depth_input)

            _depth_prediction = torch.nn.functional.interpolate(
                _depth_prediction.unsqueeze(1),
                size=_frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        _depth_map = _depth_prediction.cpu().numpy()

        _depth_map = cv2.normalize(_depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        #Reproject points into 3D
        # points_3D = cv2.reprojectImageTo3D(_depth_map, Q, handleMissingValues=False)

        #Get rid of points with value 0 (i.e no depth)
        # mask_map = _depth_map > 0.4

        #Mask colors and points. 
        # output_points = points_3D[mask_map]
        # output_colors = _frame[mask_map]
    
        _depth_map = (_depth_map*255).astype(np.uint8)
        _depth_map = cv2.applyColorMap(_depth_map , cv2.COLORMAP_MAGMA)

        # 7) Save time snapshot for the end point
        _time_end = time.time()

        # 8) Compute total time
        _time_total = _time_end - _time_start

# ------------------------------------------------------------------------------
# -- Display Section
# ------------------------------------------------------------------------------

        _output_fps = 1 / _time_total if _time_total > 0 else math.inf

        # section display output
        _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
        
        cv2.putText(_frame, f'FPS: {_output_fps}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Self Localization - Scene', _frame)
        cv2.imshow('Self Localization - Depth', _depth_map)

        # section user interaction
        if cv2.waitKey(1) & 0xFF == _ESC_KEY:
            break

# Note: Nothing Should Be Done After This Point
    _print( "\n{}".format( "Success" ) )

except Exception as e:
    _print( "\n{}".format( "Failed" ) )
    _print( "\n{}".format( e ) )
    traceback.print_exc()
    
finally:
    _print( "\n{}".format( "Cleaning Up..." ) )
    
    _Camera.release()
    cv2.destroyAllWindows()

    _print( "\n{}".format( "Done" ) )
    input("Press Any Key To Continue...")