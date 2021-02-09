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

    _YOLO_V3             = "C:/workspace/sourcetree/darknet/"
    _YOLO_CONFIDENCE     = 0.5
    _YOLO_THRESHOLD      = 0.3

    # derive the paths to the YOLO weights and model configuration
    _YOLO_LABEL_PATH     = os.path.sep.join([_YOLO_V3, "data/coco.names"])
    _YOLO_WEIGHT_PATH    = os.path.sep.join([_YOLO_V3, "yolov3.weights"])
    _YOLO_CONFIG_PATH    = os.path.sep.join([_YOLO_V3, "cfg/yolov3.cfg"])

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
    
    ## load the COCO class labels our YOLO model was trained on
    _YOLO_Labels = open(_YOLO_LABEL_PATH).read().strip().split("\n")
    
    ## initialize a list of colors to represent each possible class label
    _YOLO_Colors = np.random.randint(0, 255, size=(len(_YOLO_Labels), 3), dtype="uint8")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    _print("[INFO] loading YOLO from disk...")
    _YOLO_Net = cv2.dnn.readNetFromDarknet(_YOLO_CONFIG_PATH, _YOLO_WEIGHT_PATH)

    # use GPU settings
    _print(cv2.getBuildInformation())
    _YOLO_Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    _YOLO_Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    _YOLO_Output_Layer = _YOLO_Net.getLayerNames()
    _YOLO_Output_Layer = [_YOLO_Output_Layer[layer - 1] for layer in _YOLO_Net.getUnconnectedOutLayers()]
    
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
        # Construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        _YOLO_Blob = cv2.dnn.blobFromImage(_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # 5) Pass Input Reference Into YOLO Network
        _YOLO_Net.setInput(_YOLO_Blob)

        # 6) Evaluate YOLO Output
        _YOLO_Output = _YOLO_Net.forward(_YOLO_Output_Layer)

        # 7) Save time snapshot for the end point
        _time_end = time.time()

        # 8) Compute total time
        _time_total = _time_end - _time_start

# ------------------------------------------------------------------------------
# -- Display Section
# ------------------------------------------------------------------------------

        _print("[INFO] YOLO took {:.6f} seconds".format(_time_total))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        _YOLO_Boxes = []
        _YOLO_Confidences = []
        _YOLO_Class_IDs = []

        (_frame_Height, _frame_Width) = _frame.shape[:2]

        # loop over each of the layer outputs
        for output in _YOLO_Output:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > _YOLO_CONFIDENCE:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([_frame_Width, _frame_Height, _frame_Width, _frame_Height])
                    (centerX, centerY, width, height) = box.astype("int")
           
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
           
                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    _YOLO_Boxes.append([x, y, int(width), int(height)])
                    _YOLO_Confidences.append(float(confidence))
                    _YOLO_Class_IDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        _YOLO_Indeces = cv2.dnn.NMSBoxes(_YOLO_Boxes, _YOLO_Confidences, _YOLO_CONFIDENCE, _YOLO_THRESHOLD)

        # ensure at least one detection exists
        if len(_YOLO_Indeces) > 0:
            # loop over the indexes we are keeping
            for i in _YOLO_Indeces.flatten():
                # extract the bounding box coordinates
                (x, y) = (_YOLO_Boxes[i][0], _YOLO_Boxes[i][1])
                (w, h) = (_YOLO_Boxes[i][2], _YOLO_Boxes[i][3])
          
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in _YOLO_Colors[_YOLO_Class_IDs[i]]]
                cv2.rectangle(_frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(_YOLO_Labels[_YOLO_Class_IDs[i]], _YOLO_Confidences[i])
                cv2.putText(_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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