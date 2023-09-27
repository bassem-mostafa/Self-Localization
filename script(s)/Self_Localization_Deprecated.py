# ------------------------------------------------------------------------------
# BSD 3-Clause License
# 
# Copyright (C) 2023, Bassem Mostafa
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Self Localization Script
# --    Objective:
# --        Make use of surrounding objects/features for self-localization in GPS not available areas
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Reference(s)
# --    - https://arxiv.org/pdf/2304.06194.pdf
# --    - https://arxiv.org/pdf/2304.03608.pdf
# --    - https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w9/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.pdf
# --    - https://openaccess.thecvf.com/content_CVPR_2020/papers/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.pdf
# --
# --    - https://github.com/facebookresearch/silk
# --    - https://github.com/Shiaoming/ALIKED
# --    - https://github.com/magicleap/SuperPointPretrainedNetwork
# --    - https://github.com/magicleap/SuperGluePretrainedNetwork
# --
# --    - https://github.com/pjreddie/darknet
# --    - https://github.com/ultralytics/yolov5#pretrained-checkpoints
# --    - https://github.com/WongKinYiu/yolov7
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Note(s)
# --    - Localization
# --            Reference(s):
# --            - `https://www.youtube.com/playlist?list=PLAwxTw4SYaPkCSYXw6-a_aAoXVKLDwnHK`
# --    - Autonomous Navigation:
# --            Reference(s):
# --            - `https://www.youtube.com/playlist?list=PLn8PRpmsu08rLRGrnF-S6TyGrmcA2X7kg`
# --    - Sensor Fusion:
# --            Reference(s):
# --            - `https://www.youtube.com/playlist?list=PLn8PRpmsu08rneZErjW_NIBs0Rl_vcgSw`
# --    - State Estimation (Kalman Filter):
# --            Reference(s):
# --            - `https://www.youtube.com/playlist?list=PLn8PRpmsu08pzi6EMiYnR-076Mh-q3tWr`
# --    - Optical Flow
# --            Reference(s):
# --            - `https://medium.com/building-autonomous-flight-software/math-behind-optical-flow-1c38a25b1fe8`
# --    - Camera Calibration
# --            Reference(s):
# --                - `https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-for-camera-calibration-in-computer-vision/`
# --                - `https://medium.com/analytics-vidhya/camera-calibration-theory-and-implementation-b253dad449fb`
# --                - `https://towardsdatascience.com/camera-calibration-fda5beb373c3`
# --    - MiDaS model output is relative depth and cannot be mapped to absolute depth directly but it could be aligned to other models which provides absolute depth
# --            Reference(s):
# --                - `https://github.com/isl-org/MiDaS/issues/171#issue-1242702825`
# --                - `https://github.com/isl-org/MiDaS/issues/101#issuecomment-934494602`
# --    - MiDaS model output alignment operation
# --            Reference(s)
# --                - `https://github.com/S-B-Iqbal/ViT-for-Monocular-Depth-Estimation/blob/85bbc01ab50db06a9690d4a3669a01c4e7d1264f/src/utils.py#L46`
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# -- Legend
# --    - local variables, functions, classes ... etc is prefixed by underscore '_'
# ------------------------------------------------------------------------------

try:

# ------------------------------------------------------------------------------
# -- Extendable Object Section
# --    Provides a minimal interface that could be extended later compared to builtin object
# ------------------------------------------------------------------------------

    class _object: pass

# ------------------------------------------------------------------------------
# -- Logger Section
# --    Provides an interface for logging
# ------------------------------------------------------------------------------

    class _logger:
        _trace   = 0
        _debug   = 1
        _info    = 2
        _warning = 3
        _error   = 4
        _fatal   = 5

        def __init__(self):
            _opt = _object()
            _opt.level = _logger._trace

            self.level = _opt.level

        def trace(self, *args, **kwargs):
            if self.level > _logger._trace: return
            print(('\x1b[34m'+'[{:^9}] '+len(args)*'\x1b[34m{}\x1b[39m'+len(kwargs)*'\x1b[34m{}\x1b[39m').format('TRACE', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

        def debug(self, *args, **kwargs):
            if self.level > _logger._debug: return
            print(('\x1b[36m'+'[{:^9}] '+len(args)*'\x1b[36m{}\x1b[39m'+len(kwargs)*'\x1b[36m{}\x1b[39m').format('DEBUG', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

        def info(self, *args, **kwargs):
            if self.level > _logger._info: return
            print(('\x1b[32m'+'[{:^9}] '+len(args)*'\x1b[32m{}\x1b[39m'+len(kwargs)*'\x1b[32m{}\x1b[39m').format('INFO', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

        def warning(self, *args, **kwargs):
            if self.level > _logger._warning: return
            print(('\x1b[33m'+'[{:^9}] '+len(args)*'\x1b[33m{}\x1b[39m'+len(kwargs)*'\x1b[33m{}\x1b[39m').format('WARNING', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

        def error(self, *args, **kwargs):
            if self.level > _logger._error: return
            print(('\x1b[35m'+'[{:^9}] '+len(args)*'\x1b[35m{}\x1b[39m'+len(kwargs)*'\x1b[35m{}\x1b[39m').format('ERROR', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

        def fatal(self, *args, **kwargs):
            if self.level > _logger._fatal: return
            print(('\x1b[31m'+'[{:^9}] '+len(args)*'\x1b[31m{}\x1b[39m'+len(kwargs)*'\x1b[31m{}\x1b[39m').format('FATAL', *args, *['{}={}'.format(k,v) for k, v in kwargs.items()]))

# ------------------------------------------------------------------------------
# -- Performance Section
# --    Provides an interface for logging using builtin print
# ------------------------------------------------------------------------------

    class _performance:
        def __init__(self):
            _opt = _object()
            _opt.smoothing = 0.3                # weight of current value and (1 - weight) of stored value
            self.smoothing = _opt.smoothing
            self.milestones = OrderedDict()
            self.timestamp_last = time.time()

        def update(self, *args, **kwargs):
            _timestamp_current = time.time()
            _timestamp_delta = _timestamp_current - self.timestamp_last
            for milestone, _ in kwargs.items():
                if milestone in self.milestones:
                    _timestamp_delta = self.smoothing * _timestamp_delta + (1 - self.smoothing) * self.milestones[milestone]
                self.milestones[milestone] = _timestamp_delta
            self.timestamp_last = _timestamp_current

        def evaluate(self, *args, **kwargs):
            _time_total = 0.0
            for milestone, timestamp_delta in self.milestones.items():
                _logger.debug('{:<30} {:6.3f} seconds'.format(milestone, timestamp_delta))
                _time_total += timestamp_delta
            _logger.info('{:<30} {:6.3f} seconds {{{:6.1f} FPS}}'.format('Total', _time_total, 1.0/_time_total), '\x1b[{}A'.format(1+len(self.milestones)))

        def summary(self, *args, **kwargs):
            _time_total = 0.0
            for milestone, timestamp_delta in self.milestones.items():
                _logger.debug('{:<30} {:6.3f} seconds'.format(milestone, timestamp_delta))
                _time_total += timestamp_delta
            _logger.info('{:<30} {:6.3f} seconds {{{:6.1f} FPS}}'.format('Total', _time_total, 1.0/_time_total))

# ------------------------------------------------------------------------------
# -- Scene Section
# --    Provides an interface for scene operations
# ------------------------------------------------------------------------------

    class _scene:

        class _source:
            def __init__(self):
                pass
        
            def capture(self, *args, **kwargs):
                pass

        class _camera(_source):
            def __init__(self):
                _opt = _object()
                _opt.camera_id = 0
                self.camera = cv2.VideoCapture(_opt.camera_id)
                if self.camera is None:
                    raise Exception('Camera Initialize Failed')
                if not self.camera.isOpened():
                    self.camera = None
                    raise Exception('Camera Open Failed')
                self.capture()
        
            def capture(self, *args, **kwargs):
                if self.camera is None:
                    raise Exception('Camera NOT Initialized')
                _frame_available, _frame = self.camera.read()
                _frame = cv2.flip(_frame, 1) # mirror frame horizontly
                if not _frame_available:
                    raise Exception('Camera Capture Failed')
                return _frame

        class _kitti(_source):
            def __init__(self):
                _opt = _object()
                _opt.dataset = 'C:/Users/MLLD1740/OneDrive - orange.com/Workspace/smart-tag/KITTI_raw'
                self.dataset = _opt.dataset
                # filter(lambda directory : re.compile('(?!data_splits|.*\.md)').match(directory), sorted(os.listdir(path=self.dataset))) # Note: skip traversing all dataset folders
                self.dataset_root = '{}/{}'.format(self.dataset, '2011_09_26')
                self.dataset_folder_gen = ('{}/{}/image_02/data'.format(self.dataset_root, folder) for folder in filter(lambda directory : re.compile('.*?(?=_sync$)').match(directory), sorted(os.listdir(path=self.dataset_root))))
                self.dataset_folder = next(self.dataset_folder_gen)
                self.dataset_frame_gen = ('{}/{}'.format(self.dataset_folder, image) for image in filter(lambda file : re.compile('.*?(?=\.png$)').match(file), sorted(os.listdir(path=self.dataset_folder))))
                self.dataset_frame = next(self.dataset_frame_gen)

                cv2.imread(self.dataset_frame) # verify existance
        
            def capture(self, *args, **kwargs):
                try:
                    _frame = cv2.imread(self.dataset_frame)
                    self.dataset_frame = next(self.dataset_frame_gen)
                except:
                    try:
                        self.dataset_folder = next(self.dataset_folder_gen)
                        self.dataset_frame_gen = ('{}/{}'.format(self.dataset_folder, image) for image in filter(lambda file : re.compile('.*?(?=\.png$)').match(file), sorted(os.listdir(path=self.dataset_folder))))
                        self.dataset_frame = next(self.dataset_frame_gen)
                    except:
                        # send halt signal to indicate demo end
                        pass
                return _frame


        def __init__(self):
            _opt = _object()
            _opt.source = 'kitti' # 'camera' or 'kitti' or other

            self.source = _opt.source
            if self.source == 'camera':
                self.source = self._camera()
            elif self.source == 'kitti':
                self.source = self._kitti()
            else:
                raise Exception("Un-supported source")

        def capture(self, *args, **kwargs):
            _frame = self.source.capture()
            return _frame

        def render(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            _frame = kwargs['scene_frame']
            return _frame

# ------------------------------------------------------------------------------
# -- Scene Object Section
# --    Provides an interface for scene object operations
# ------------------------------------------------------------------------------

    class _scene_object:
        def __init__(self):
            _opt = _object()
            _opt.model_type = 'yolov5s' # yolov5s or yolov5m, yolov5l, yolov5x, custom
            _opt.conf = 0.65            # NMS confidence threshold, valid values are between 0.0 and 1.0'
            _opt.iou = 0.45             # NMS IoU threshold, valid values are between 0.0 and 1.0'
            _opt.agnostic = True        # NMS class-agnostic, Detect objects without classifying
            _opt.multi_label = True     # NMS multiple labels per box
            _opt.classes = None         # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
            _opt.max_det = 1000         # maximum number of detections per image
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model = torch.hub.load('ultralytics/yolov5', _opt.model_type)
            self.model.conf = _opt.conf
            self.model.iou = _opt.iou
            self.model.agnostic = _opt.agnostic
            self.model.multi_label = _opt.multi_label
            self.model.classes = _opt.classes
            self.model.max_det = _opt.max_det
            self.model.to(self.device)
            self.model.eval()

        def detect(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            _frame = kwargs['scene_frame'].copy()
            _detections = self.model([_frame])
            _detections_xyxyn = _detections.xyxyn[0].tolist()
            for i, detection in enumerate(_detections_xyxyn):
                detection = {k:v for k, v in zip(['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'], detection)}
                detection['class-name'] = self.model.names[detection['class']] if 'class' in detection.keys() else None
                _detections_xyxyn[i] = detection
            return _detections_xyxyn, _detections

        def render(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            if 'scene_objects' not in kwargs: raise Exception('A Keyword Argument `scene_objects` Is Required')
            _frame = kwargs['scene_frame']
            _detections_xyxyn, _detections = kwargs['scene_objects']
            # _detections_image = _scene_frame.copy()
            # for detection in _detections_xyxyn:
            #     detection.update({key:int(value*detection[key]) for key, value in zip(['xmin', 'ymin', 'xmax', 'ymax'], [_scene_frame.shape[1], _scene_frame.shape[0], _scene_frame.shape[1], _scene_frame.shape[0]])})
            #     cv2.rectangle(_detections_image, (detection['xmin'], detection['ymin']), (detection['xmax'], detection['ymax']), color=(0, 255, 0), thickness=2)
            #     cv2.putText(_detections_image, detection['class-name'], (detection['xmin'], detection['ymin']), 0, 1, (255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            _detections_image = numpy.asarray(_detections.render())[0]
            return _detections_image

# ------------------------------------------------------------------------------
# -- Scene Depth Section
# --    Provides an interface for scene depth operations
# ------------------------------------------------------------------------------

    class _scene_depth:
        def __init__(self):
            _opt = _object()
            _opt.model_type = 'MiDaS_small'     # MiDaS model for depth estimation, could be one of the following {'DPT_Large', 'DPT_Hybrid', 'MiDaS_small'}
                                                # 'DPT_Large'     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
                                                # 'DPT_Hybrid'    # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
                                                # 'MiDaS_small'   # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            self.model = torch.hub.load('intel-isl/MiDaS', _opt.model_type)
            self.model.to(self.device)
            self.model.eval()
            # Load transforms to resize and normalize the image
            _midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if _opt.model_type in ['DPT_Large', 'DPT_Hybrid']:
                self.transform = _midas_transforms.dpt_transform
            else:
                self.transform = _midas_transforms.small_transform
            self.depth_to_meters = None

        def compute(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            _frame = kwargs['scene_frame']
            # Apply input transforms
            _frame_transformed = self.transform(_frame).to(self.device)
            # Prediction and resize to original resolution
            with torch.no_grad():
                _inverse_depth = self.model(_frame_transformed)
                _inverse_depth = torch.nn.functional.interpolate(
                    _inverse_depth.unsqueeze(1),
                    size=_frame.shape[:2],
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
            _inverse_depth = _inverse_depth.cpu().numpy()
            _inverse_depth[_inverse_depth < 0] = 0 # filter out -ve values generated from interpolation
## Start of Alignment (Not Used for the moment)
            # x = _frame[:, :, 0].copy().flatten()
            # y = _inverse_depth.copy().flatten()
            # A = numpy.vstack([x, numpy.ones(len(x))]).T
            # s,t = numpy.linalg.lstsq(A,y, rcond=None)[0]
            # _aligned_depth = (_inverse_depth -t)/s
## End of Alignment
            # _depth_map = 1 / _inverse_depth
            _depth_map = _inverse_depth.max() - _inverse_depth
            return _depth_map

        def render(self, *args, **kwargs):
            if 'scene_depth' not in kwargs: raise Exception('A Keyword Argument `scene_depth` Is Required')
            _depth_map = kwargs['scene_depth']
            # _inverse_depth = 1 / _depth_map
            _inverse_depth = _depth_map.max() - _depth_map
            _inverse_depth = cv2.normalize(_inverse_depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            _inverse_depth = (_inverse_depth*255).astype(numpy.uint8)
            _inverse_depth = cv2.applyColorMap(_inverse_depth , cv2.COLORMAP_MAGMA)
            return _inverse_depth

# ------------------------------------------------------------------------------
# -- Scene Feature Section
# --    Provides an interface for scene feature operations
# ------------------------------------------------------------------------------

    class _scene_feature:
        def __init__(self):
            _opt = _object()
            _opt.superglue = 'outdoor'                       # 'SuperGlue weights' {'indoor', 'outdoor'}
            _opt.max_keypoints = -1                          # 'Maximum number of keypoints detected by Superpoint'
                                                            # ' (\'-1\' keeps all keypoints)'
            _opt.keypoint_threshold = 0.005                  # 'SuperPoint keypoint detector confidence threshold'
            _opt.nms_radius = 5                              # 'SuperPoint Non Maximum Suppression (NMS) radius'
                                                            # ' (Must be positive)'
            _opt.sinkhorn_iterations = 20                    # 'Number of Sinkhorn iterations performed by SuperGlue'
            _opt.match_threshold = 0.9                       # 'SuperGlue match threshold'
            _opt.force_cpu = False                           # 'Force pytorch to run in CPU mode.'
            torch.set_grad_enabled(False)
            self.device = 'cuda' if torch.cuda.is_available() and not _opt.force_cpu else 'cpu'
            _config = {
                'superpoint': {
                    'nms_radius': _opt.nms_radius,
                    'keypoint_threshold': _opt.keypoint_threshold,
                    'max_keypoints': _opt.max_keypoints
                },
                'superglue': {
                    'weights': _opt.superglue,
                    'sinkhorn_iterations': _opt.sinkhorn_iterations,
                    'match_threshold': _opt.match_threshold,
                }
            }
            self.model = Matching(_config).eval().to(self.device)

        def detect(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            _scene_frame = kwargs['scene_frame']
            _input = cv2.cvtColor(_scene_frame, cv2.COLOR_RGB2GRAY)
            _input_tensor = frame2tensor(_input, self.device)
            _output = self.model.superpoint({'image': _input_tensor})
            _output['frame'] = _input
            _output['image'] = _input_tensor
            return _output

        def match(self, *args, **kwargs):
            if 'scene_features' not in kwargs: raise Exception('A Keyword Argument `scene_features` Is Required')
            _scene_features = kwargs['scene_features']
            _input = {}
            _input.update({k+'1':v for k,v in _scene_features[0].items()})
            _input.update({k+'0':v for k,v in _scene_features[1].items()})
            _output = self.model(_input)
            _output['keypoints0'] = _input['keypoints0']
            _output['keypoints1'] = _input['keypoints1']
            _output['frame0'] = _input['frame0']
            _output['frame1'] = _input['frame1']
            return _output

        def render(self, *args, **kwargs):
            if 'scene_features_match' not in kwargs: raise Exception('A Keyword Argument `scene_features_match` Is Required')
            _scene_features_match = kwargs['scene_features_match']
            _input = _scene_features_match['frame0'], _scene_features_match['frame1']
            _kpts0 = _scene_features_match['keypoints0'][0].cpu().numpy()
            _kpts1 = _scene_features_match['keypoints1'][0].cpu().numpy()
            _matches = _scene_features_match['matches0'][0].cpu().numpy()
            _confidence = _scene_features_match['matching_scores0'][0].cpu().numpy()
            _valid = _matches > -1
            _mkpts0 = _kpts0[_valid]
            _mkpts1 = _kpts1[_matches[_valid]]
            _color = cm.jet(_confidence[_valid])
            _out = make_matching_plot_fast(*_input, _kpts0, _kpts1, _mkpts0, _mkpts1, _color, [], path=None, show_keypoints=True, margin=0)
            return _out

# ------------------------------------------------------------------------------
# -- Scene Fusion Section
# --    Provides an interface for scene fusion operations
# ------------------------------------------------------------------------------

    class _scene_fusion:
        def __init__(self):
            pass

        def compute(self, *args, **kwargs):
            if 'scene_frame' not in kwargs: raise Exception('A Keyword Argument `scene_frame` Is Required')
            if 'scene_objects' not in kwargs: raise Exception('A Keyword Argument `scene_objects` Is Required')
            if 'scene_depth' not in kwargs: raise Exception('A Keyword Argument `scene_depth` Is Required')
            if 'scene_features' not in kwargs: raise Exception('A Keyword Argument `scene_features` Is Required')
            if 'scene_features_match' not in kwargs: raise Exception('A Keyword Argument `scene_features_match` Is Required')
            _frame = kwargs['scene_frame']
            _objects = kwargs['scene_objects']
            _depth = kwargs['scene_depth']
            _features = kwargs['scene_features']
            _features_match = kwargs['scene_features_match']

            _fusion = _frame.copy()
            _detections_xyxyn, _ = _objects
            for detection in _detections_xyxyn:
                detection.update({key:int(value*detection[key]) for key, value in zip(['xmin', 'ymin', 'xmax', 'ymax'], [_fusion.shape[1], _fusion.shape[0], _fusion.shape[1], _fusion.shape[0]])})
                detection['depth'] = _depth[detection['ymin'] : detection['ymax'] + 1, detection['xmin'] : detection['xmax'] + 1].mean()
                detection['depth'] = str(detection['depth'] if not numpy.isinf(detection['depth']) else 'Undetermined')
                cv2.rectangle(
                    _fusion,
                    (detection['xmin'], detection['ymin']),
                    (detection['xmax'], detection['ymax']),
                    color=(0, 255, 0),
                    thickness=2)
                _text = '{} {} meter(s)'.format(detection['class-name'], detection['depth'])
                _text_font = 0
                _text_color = (255, 0, 0)
                _text_scale = 2.0/3.0
                _text_thickness = 1
                _text_width, _text_height = cv2.getTextSize(_text, _text_font, fontScale=_text_scale, thickness=_text_thickness)[0]
                _text_org = (detection['xmin'], detection['ymin'] + _text_height)
                cv2.putText(
                    _fusion,
                    _text,
                    _text_org,
                    _text_font,
                    _text_scale,
                    _text_color,
                    thickness=_text_thickness,
                    lineType=cv2.LINE_AA)

            _kpts0 = _features_match['keypoints0'][0].cpu().numpy()
            _kpts1 = _features_match['keypoints1'][0].cpu().numpy()
            _matches = _features_match['matches0'][0].cpu().numpy()
            _confidence = _features_match['matching_scores0'][0].cpu().numpy()
            _valid = _matches > -1
            _mkpts0 = _kpts0[_valid]
            _mkpts1 = _kpts1[_matches[_valid]]

            _mkpts0, _mkpts1 = numpy.round(_mkpts0).astype(int), numpy.round(_mkpts1).astype(int)

            for (x0, y0), (x1, y1) in zip(_mkpts0, _mkpts1):
                cv2.line(
                    _fusion,
                    (x0, y0),
                    (x1, y1),
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA)
                # display line end-points as circles
                cv2.circle(
                    _fusion,
                    (x0, y0),
                    2,
                    (255, 0, 0),
                    -1,
                    lineType=cv2.LINE_AA)
                cv2.circle(
                    _fusion,
                    (x1, y1),
                    2,
                    (0, 255, 0),
                    -1,
                    lineType=cv2.LINE_AA)

            _fusion = cv2.addWeighted(_fusion, 0.7, _scene_depth.render(scene_depth=_depth), 0.5, 0)
            return _fusion

        def render(self, *args, **kwargs):
            if 'scene_fusion' not in kwargs: raise Exception('A Keyword Argument `scene_fusion` Is Required')
            _fusion = kwargs['scene_fusion']
            return _fusion

# ------------------------------------------------------------------------------
# -- User Interface Section
# --    Provides an interface for user either display output or keyboard input
# ------------------------------------------------------------------------------

    class _user_interface:
        _is_enabled = True
        _KEY_HALT = [27, ord('q'), ord('Q')]

        def __init__(self):
            _opt = _object()
            _opt.width = 1360                           # user interface screen width
            _opt.height = 760                           # user interface screen height
            _opt.depth = 3                              # user interface screen layers (RGB = 3)
            _opt.background_color = (255, 255, 255)     # user interface screen initial background color (Blue, Red, Green)
            self.screen = numpy.random.randint(255, size=(_opt.height, _opt.width, _opt.depth), dtype=numpy.uint8)
            if 'background_color' in dir(_opt): self.screen[:, :] = _opt.background_color
            self.screen = cv2.resize(self.screen, (_opt.width, _opt.height), interpolation = cv2.INTER_LINEAR)
            self.screen_frame_counter = None
            self.show()

        def render(self, *args, **kwargs):
            if not _user_interface._is_enabled:
                return
            _screen_height, _screen_width, _screen_depth = self.screen.shape
            _cell_height, _cell_width, _cell_depth = int(_screen_height / 2), int(_screen_width / 3), int(_screen_depth) # split the screen into cells
            _scene_frame = None
            _scene_objects = None
            _depth = None
            _scene_features = None
            _scene_features_match = None
            _fusion = None
            if 'scene_frame' in kwargs:
                _scene_frame = kwargs['scene_frame']
                if _scene_frame is not None: _scene_frame = cv2.resize(_scene_frame, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            if 'scene_objects' in kwargs:
                _scene_objects = kwargs['scene_objects']
                _scene_objects = _scene_object.render(scene_frame=_scene_frame, scene_objects=_scene_objects)
                if _scene_objects is not None: _scene_objects = cv2.resize(_scene_objects, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            if 'scene_depth' in kwargs:
                _depth = kwargs['scene_depth']
                _depth = _scene_depth.render(scene_depth=_depth)
                if _depth is not None: _depth = cv2.resize(_depth, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            # if 'scene_features' in kwargs:
            #     _scene_features = kwargs['scene_features']
            #     if _scene_features is not None: _scene_features = cv2.resize(_scene_features, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            if 'scene_features_match' in kwargs:
                _scene_features_match = kwargs['scene_features_match']
                _scene_features_match = _scene_feature.render(scene_features_match=_scene_features_match)
                if _scene_features_match is not None: _scene_features_match = cv2.resize(_scene_features_match, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            if 'scene_fusion' in kwargs:
                _fusion = kwargs['scene_fusion']
                _fusion = _scene_fusion.render(scene_fusion=_fusion)
                if _fusion is not None: _fusion = cv2.resize(_fusion, (_cell_width, _cell_height), interpolation = cv2.INTER_LINEAR)
            _cell_row, _cell_column = 0, 0
            for _cell in [_scene_frame, _scene_objects, _depth, _scene_features, _scene_features_match, _fusion]:
                if _cell is None:
                    continue
                self.screen[(_cell_row)*_cell_height:(_cell_row + 1)*_cell_height, (_cell_column)*_cell_width:(_cell_column + 1)*_cell_width] = _cell
                _cell_column = _cell_column + 1
                if _cell_column*_cell_width >= _screen_width - _cell_width:
                    _cell_column = 0
                    _cell_row = _cell_row + 1
            if self.screen_frame_counter is None:
                self.screen_frame_counter = 1
            else:
                self.screen_frame_counter = self.screen_frame_counter + 1
            return self.screen

        def show(self, *args, **kwargs):
            if not _user_interface._is_enabled:
                return
            cv2.imshow('Self Localization - Screen', self.screen)

        def save(self, *args, **kwargs):
            if not _user_interface._is_enabled:
                return
            if self.screen_frame_counter is None:
                raise Exception('`render()` should be called before `save()`')
            cv2.imwrite('Self Localization - Screen {:04d}.jpg'.format(self.screen_frame_counter), self.screen)

        def halt(self, *args, **kwargs):
            if not _user_interface._is_enabled:
                return False
            _key = (cv2.waitKey(1) & 0xFF)
            if _key in _user_interface._KEY_HALT:
                return True
            return False

# ------------------------------------------------------------------------------
# -- Import Section
# ------------------------------------------------------------------------------

    import traceback
    import os
    import sys
    import time
    import math
    import numpy
    import cv2
    import torch
    import re
    import functools
    import readchar
    sys.path.append(os.path.join(os.getcwd(), '..', 'submodule(s)'))
    from collections import OrderedDict
    from pathlib import Path
    from matplotlib import cm
    from SuperGluePretrainedNetwork.models.matching import Matching
    from SuperGluePretrainedNetwork.models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor, process_resize)

# ------------------------------------------------------------------------------
# -- Configuration Section
# ------------------------------------------------------------------------------

    _VERSION = ['{:04}{:02}{:02}-{:02}{:02}'.format( *( _VERSION.tm_year, _VERSION.tm_mon, _VERSION.tm_mday, _VERSION.tm_hour, _VERSION.tm_min ) ) for _VERSION in [ time.localtime( os.path.getmtime( __file__ ) ) ] ][0]

# ------------------------------------------------------------------------------
# -- Initialization Section
# ------------------------------------------------------------------------------

    _logger = _logger()

    _logger.info( "{:<40} {}".format( "Self-Localization Script version", _VERSION ) )
    _logger.debug( "{:<40} {}".format( "OpenCV version", cv2.__version__ ) )

    _logger.trace(cv2.getBuildInformation())

    numpy.random.seed(42)

    _performance    = _performance()
    _scene          = _scene()
    _scene_object   = _scene_object()
    _scene_depth    = _scene_depth()
    _scene_feature  = _scene_feature()
    _scene_fusion   = _scene_fusion()
    _user_interface = _user_interface()

# ------------------------------------------------------------------------------
# -- Methodology Section
# ------------------------------------------------------------------------------

    _history = {}
    while not _user_interface.halt():
        _frame = _scene.capture()
        _performance.update(scene_capture=time.time())
        _objects = _scene_object.detect(scene_frame=_frame)
        _performance.update(scene_object_detect=time.time())
        # _objects_image = _scene_object.render(scene_objects=_objects)
        # _performance.update(scene_object_render=time.time())
        _depth = _scene_depth.compute(scene_frame=_frame)
        _performance.update(scene_depth_compute=time.time())
        _features = _scene_feature.detect(scene_frame=_frame)
        _performance.update(scene_feature_detect=time.time())
        _features_match = _scene_feature.match(scene_features=[_features, _history['_features'] if '_features' in _history.keys() else _features ])
        _performance.update(scene_feature_match=time.time())
        # _features_match_image = _scene_feature.render(scene_features_match=_features_match)
        # _performance.update(scene_feature_render=time.time())
        _fusion = _scene_fusion.compute(scene_frame=_frame, scene_objects=_objects, scene_depth=_depth, scene_features=_features, scene_features_match=_features_match)
        _performance.update(scene_fusion=time.time())
        _user_interface.render(scene_frame=_frame, scene_objects=_objects, scene_depth=_depth, scene_features=_features, scene_features_match=_features_match, scene_fusion=_fusion)
        _performance.update(user_interface_render=time.time())
        _user_interface.show()
        _performance.update(user_interface_show=time.time())
        # _user_interface.save()
        # _performance.update(user_interface_save=time.time())
        _performance.evaluate()
        _history = {
            '_frame': _frame,
            '_objects': _objects,
            '_depth': _depth,
            '_features': _features,
        }
    _performance.summary()

# ------------------------------------------------------------------------------
# -- Final Section
# --    Nothing Should Be Done After This Point
# ------------------------------------------------------------------------------

    _logger.info( "Success" )

except Exception as e:
    _logger.fatal( "Failed With Exception {}".format(e) )
    traceback.print_exc()

finally:
    _logger.debug( "Cleaning Up..." )