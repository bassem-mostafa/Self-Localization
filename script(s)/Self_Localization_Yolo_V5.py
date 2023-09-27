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
# --    - https://github.com/ultralytics/yolov5#pretrained-checkpoints
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
    import re
    import functools

# ------------------------------------------------------------------------------
# -- Configuration Section
# ------------------------------------------------------------------------------

    _VERSION = ['{:04}{:02}{:02}-{:02}{:02}'.format( *( _VERSION.tm_year, _VERSION.tm_mon, _VERSION.tm_mday, _VERSION.tm_hour, _VERSION.tm_min ) ) for _VERSION in [ time.localtime( os.path.getmtime( __file__ ) ) ] ][0]

    _ESC_KEY = 27
    _CAMERA_ID = 0

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

    # load our YOLO object detector trained on COCO dataset (80 classes)
    _print("[INFO] loading YOLO from disk...")
    _YOLO_Net = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

    # use GPU settings
    _print(cv2.getBuildInformation())
    
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

        # 4)
        _YOLO_Output = _YOLO_Net([_frame])

        # 5) Save time snapshot for the end point
        _time_end = time.time()

        # 6) Compute total time
        _time_total = _time_end - _time_start

# ------------------------------------------------------------------------------
# -- Display Section
# ------------------------------------------------------------------------------

        _print("[INFO] YOLO took {:.6f} seconds".format(_time_total))

        _frame = np.asarray(_YOLO_Output.render())[0]

        _output_fps = 1 / _time_total if _time_total > 0 else math.inf

        # section display output
        _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
        
        cv2.putText(_frame, f'FPS: {_output_fps}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Self Localization', _frame)

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