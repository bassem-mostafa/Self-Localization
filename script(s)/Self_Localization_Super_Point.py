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
# --    - https://github.com/magicleap/SuperGluePretrainedNetwork
# --    - https://paperswithcode.com/paper/superpoint-self-supervised-interest-point
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

    from pathlib import Path
    import matplotlib.cm as cm

    from SuperGluePretrainedNetwork.models.matching import Matching
    from SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer,
                              make_matching_plot_fast, frame2tensor)

# ------------------------------------------------------------------------------
# -- Configuration Section
# ------------------------------------------------------------------------------

    _VERSION = ['{:04}{:02}{:02}-{:02}{:02}'.format( *( _VERSION.tm_year, _VERSION.tm_mon, _VERSION.tm_mday, _VERSION.tm_hour, _VERSION.tm_min ) ) for _VERSION in [ time.localtime( os.path.getmtime( __file__ ) ) ] ][0]

    _ESC_KEY = 27
    _CAMERA_ID = 0

    _update_anchor_every_n_frames = 10              # update anchor every n frames

    class _Dummy: pass
    opt = _Dummy()

    opt.input = 0                                   # 'ID of a USB webcam, URL of an IP camera, '
                                                    # 'or path to an image directory or movie file'
    opt.output_dir = None                           # 'Directory where to write output frames (If None, no output)'
    opt.image_glob = ['*.png', '*.jpg', '*.jpeg']   # 'Glob if a directory of images is specified'
    opt.skip = 1                                    # 'Images to skip if input is a movie or directory'
    opt.max_length = 1000000                        # 'Maximum length if input is a movie or directory'
    opt.resize = [-1]                               # 'Resize the input image before running inference. If two numbers, '
                                                    # 'resize to the exact dimensions, if one number, resize the max '
                                                    # 'dimension, if -1, do not resize'
    opt.superglue = 'outdoor'                       # 'SuperGlue weights' {'indoor', 'outdoor'}
    opt.max_keypoints = -1                          # 'Maximum number of keypoints detected by Superpoint'
                                                    # ' (\'-1\' keeps all keypoints)'
    opt.keypoint_threshold = 0.005                  # 'SuperPoint keypoint detector confidence threshold'
    opt.nms_radius = 5                              # 'SuperPoint Non Maximum Suppression (NMS) radius'
                                                    # ' (Must be positive)'
    opt.sinkhorn_iterations = 20                    # 'Number of Sinkhorn iterations performed by SuperGlue'
    opt.match_threshold = 0.9                       # 'SuperGlue match threshold'
    opt.show_keypoints = False                      # 'Show the detected keypoints'
    opt.no_display = False                          # 'Do not display images to screen. Useful if running remotely'
    opt.force_cpu = False                           # 'Force pytorch to run in CPU mode.'

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

    torch.set_grad_enabled(False)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        _print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        _print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        _print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    _print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    if opt.output_dir is not None:
        _print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        _print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    _print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq/ESC: quit')

    timer = AverageTimer()

# ------------------------------------------------------------------------------
# -- Methodology Section
# ------------------------------------------------------------------------------

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            _print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        # update anchor frame every n frames
        if _update_anchor_every_n_frames > 0 and stem1 % _update_anchor_every_n_frames == 0:
            # update set the current frame as anchor
            last_data = {k+'0': pred[k+'1'] for k in keys}
            last_data['image0'] = frame_tensor
            last_frame = frame
            last_image_id = (vs.i - 1)

# ------------------------------------------------------------------------------
# -- Display Section
# ------------------------------------------------------------------------------

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q' or key == _ESC_KEY:
                vs.cleanup()
                _print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                _print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                _print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            _print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()

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