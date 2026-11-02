"""Lazy imports let ROI editing and geometry tests run without model downloads."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from vision import (Face, Person, affine_points, expanded_box,
                    nms_faces, rotate_expanded)


class PersonTracker:
    def __init__(self, cfg, fps, output_dir, device='auto'):
        from ultralytics import YOLO
        from ultralytics.utils import ROOT
        import torch

        self.cfg = cfg
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        # Start from the installed package's complete config, not a partial copy.
        with open(Path(ROOT) / 'cfg' / 'trackers' / 'botsort.yaml', encoding='utf-8') as f:
            tracker_cfg = yaml.safe_load(f)
        tracker_cfg.update(cfg['tracker'])
        tracker_cfg['tracker_type'] = 'botsort'
        self.lost_frames = max(1, math.ceil(fps * cfg['lost_seconds']))
        tracker_cfg['track_buffer'] = self.lost_frames
        if cfg['confidence'] > tracker_cfg['track_low_thresh']:
            raise ValueError('person.confidence must be <= track_low_thresh for low-confidence recovery.')
        if not (0 <= tracker_cfg['track_low_thresh'] < tracker_cfg['track_high_thresh']
                <= tracker_cfg['new_track_thresh'] <= 1):
            raise ValueError('Require 0 <= track_low_thresh < track_high_thresh <= new_track_thresh <= 1.')
        self.tracker_file = Path(output_dir) / 'botsort_runtime.yaml'
        self.tracker_file.write_text(yaml.safe_dump(tracker_cfg), encoding='utf-8')
        self.model = YOLO(cfg['model'])

    def update(self, frame, window):
        x1, y1, x2, y2 = map(int, window)
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        result = self.model.track(crop, persist=True, classes=[0],
                                  tracker=str(self.tracker_file), conf=self.cfg['confidence'],
                                  iou=self.cfg['iou'], imgsz=self.cfg['imgsz'], device=self.device,
                                  verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy() + np.array([x1, y1, x1, y1])
        return [Person(int(tid), box.astype(float), float(score)) for tid, box, score in
                zip(boxes.id.detach().cpu().numpy(), xyxy, boxes.conf.detach().cpu().numpy())]


class FaceDetector:
    def __init__(self, cfg, device='auto', base_dir='.'):
        # torch first also helps expose compatible CUDA libraries to ONNX Runtime.
        import torch  # noqa: F401
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        from insightface import model_zoo

        self.cfg = cfg
        available = ort.get_available_providers()
        cuda = device != 'cpu' and 'CUDAExecutionProvider' in available
        if device == 'cuda' and not cuda:
            raise RuntimeError('CUDAExecutionProvider unavailable. Install compatible onnxruntime-gpu or use --face-device cpu.')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        # CPU fallback is deliberate in auto mode, but never silent.
        model_path = cfg.get('onnx_path')
        if model_path:
            path = (Path(base_dir) / model_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.detector = model_zoo.get_model(str(path), providers=providers)
            if self.detector is None or self.detector.taskname != 'detection':
                raise ValueError('face.onnx_path must point to a compatible SCRFD detection model.')
            self.detector.prepare(ctx_id=0 if cuda else -1, input_size=tuple(cfg['input_size']),
                                  det_thresh=cfg['confidence'])
        else:
            app = FaceAnalysis(name=cfg['model_pack'], allowed_modules=['detection'], providers=providers)
            app.prepare(ctx_id=0 if cuda else -1, det_size=tuple(cfg['input_size']),
                        det_thresh=cfg['confidence'])
            self.detector = app.det_model
        active = self.detector.session.get_providers()
        print('Face detector ONNX providers:', active)
        if device == 'cuda' and 'CUDAExecutionProvider' not in active:
            raise RuntimeError('ONNX fell back to CPU; check CUDA/cuDNN installation or select CPU explicitly.')
        # Experimental miss-only recovery is opt-in. No default detection changes.
        # The normal SCRFD, association, ROI and quality gates remain authoritative.
        self._recovery = self._read_recovery_config(cfg)

    @staticmethod
    def _read_recovery_config(cfg):
        recovery = cfg.get('recovery', {})
        if not isinstance(recovery, dict):
            raise ValueError('face.recovery must be a mapping.')
        allowed = {'enabled', 'retry_upper_crop', 'upper_crop_height_fraction',
                   'retry_rotations_on_unscheduled_empty'}
        extras = set(recovery) - allowed
        if extras:
            raise ValueError(f'Unknown face.recovery settings: {sorted(extras)}')
        enabled = recovery.get('enabled', False)
        upper = recovery.get('retry_upper_crop', True)
        rotate = recovery.get('retry_rotations_on_unscheduled_empty', False)
        for name, val in [('enabled', enabled), ('retry_upper_crop', upper),
                          ('retry_rotations_on_unscheduled_empty', rotate)]:
            if type(val) is not bool:
                raise ValueError(f'face.recovery.{name} must be a boolean.')
        try:
            height_fraction = float(recovery.get('upper_crop_height_fraction', 0.65))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('face.recovery.upper_crop_height_fraction must be in (0,1).') from exc
        if not math.isfinite(height_fraction) or not 0 < height_fraction < 1:
            raise ValueError('face.recovery.upper_crop_height_fraction must be in (0,1).')
        # Upper crop must truly be shorter than the primary crop to avoid a
        # pointless duplicate SCRFD call (which could also increase NMS noise).
        if enabled and upper and height_fraction >= float(cfg['person_crop_height_fraction']):
            raise ValueError('Recovery upper crop must be shorter than the primary person crop.')
        return {'enabled': enabled, 'retry_upper_crop': upper,
                'upper_crop_height_fraction': height_fraction,
                'retry_rotations_on_unscheduled_empty': rotate}

    def _detect_crop(self, crop, offset, angle):
        if angle:
            view, inverse = rotate_expanded(crop, angle)
        else:
            view, inverse = crop, np.array([[1., 0., 0.], [0., 1., 0.]])
        boxes, landmarks = self.detector.detect(view, max_num=0)
        if len(boxes) and landmarks is None:
            raise RuntimeError('Use a detection model with FIVE landmarks (e.g. buffalo_l/det_10g.onnx).')
        found = []
        h, w = crop.shape[:2]
        for row, kps in zip(boxes, [] if landmarks is None else landmarks):
            if not np.isfinite(row).all() or not np.isfinite(kps).all():
                continue
            x1, y1, x2, y2 = row[:4]
            corners = affine_points([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], inverse)
            points = affine_points(kps, inverse)
            # Reject detections in rotation padding instead of clipping them into real faces.
            if ((points < [0, 0]) | (points >= [w, h])).any():
                continue
            points += offset
            box = np.r_[corners.min(0) + offset, corners.max(0) + offset]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            found.append(Face(box, points, float(row[4]), float(angle)))
        return found

    def detect(self, frame, people, frame_index, diagnostics=None):
        """Detect faces inside person crops, preserving the original gates.

        When ``diagnostics`` is a dict, populate JSON-safe per-track details.
        An optional, explicitly enabled recovery trial retries a shorter head
        crop only after the normal detector (and scheduled rotations) found no
        face. Extra detections STILL pass global NMS, association, ROI and quality.
        """
        all_faces = []
        h, w = frame.shape[:2]
        # Lazy fallback also supports test doubles built with __new__.
        recovery = getattr(self, '_recovery', None)
        if recovery is None:
            recovery = self._read_recovery_config(self.cfg)
        if diagnostics is not None:
            if not isinstance(diagnostics, dict):
                raise TypeError('diagnostics must be a dict or None.')
            diagnostics.clear()
            diagnostics['people'] = []
        for person in people:
            box = expanded_box(person.box, self.cfg['crop_padding'], w, h,
                               self.cfg['person_crop_height_fraction'])
            x1, y1, x2, y2 = map(int, box)
            audit = None
            if diagnostics is not None:
                audit = {'track_id': int(person.track_id),
                         'crop_box_xyxy': [x1, y1, x2, y2],
                         'crop_width_px': x2 - x1, 'crop_height_px': y2 - y1,
                         'normal_detections': 0, 'scheduled_rotation_attempts': 0,
                         'scheduled_rotation_detections': 0,
                         'recovery_upper_attempts': 0, 'recovery_upper_detections': 0,
                         'recovery_rotation_attempts': 0,
                         'recovery_rotation_detections': 0,
                         'faces_before_global_nms': 0}
                diagnostics['people'].append(audit)
            if x2 - x1 < 12 or y2 - y1 < 12:
                if audit is not None:
                    audit['reason'] = 'crop_too_small'
                continue
            crop = frame[y1:y2, x1:x2]
            offset = np.array([x1, y1])
            found = self._detect_crop(crop, offset, 0)
            if audit is not None:
                audit['normal_detections'] = len(found)
            # Original scheduled rotation policy (unchanged).
            scheduled = frame_index % self.cfg['rotation_every_n_frames'] == 0
            if scheduled and (not found or not self.cfg['rotate_only_if_empty']):
                for angle in self.cfg['rotation_angles']:
                    if angle:
                        rotated = self._detect_crop(crop, offset, float(angle))
                        found.extend(rotated)
                        if audit is not None:
                            audit['scheduled_rotation_attempts'] += 1
                            audit['scheduled_rotation_detections'] += len(rotated)
            # Miss-only recovery is an A/B option, OFF by default. No extra
            # detector calls when normal/scheduled detection already succeeded.
            if recovery['enabled'] and not found:
                if recovery['retry_upper_crop']:
                    upper_box = expanded_box(
                        person.box, self.cfg['crop_padding'], w, h,
                        recovery['upper_crop_height_fraction'])
                    ux1, uy1, ux2, uy2 = map(int, upper_box)
                    if (ux2 - ux1 >= 12 and uy2 - uy1 >= 12
                            and (ux1, uy1, ux2, uy2) != (x1, y1, x2, y2)):
                        upper_crop = frame[uy1:uy2, ux1:ux2]
                        recovered = self._detect_crop(
                            upper_crop, np.array([ux1, uy1]), 0)
                        found.extend(recovered)
                        if audit is not None:
                            audit['recovery_upper_attempts'] += 1
                            audit['recovery_upper_detections'] += len(recovered)
                if not found and not scheduled and recovery['retry_rotations_on_unscheduled_empty']:
                    for angle in self.cfg['rotation_angles']:
                        if angle:
                            recovered = self._detect_crop(crop, offset, float(angle))
                            found.extend(recovered)
                            if audit is not None:
                                audit['recovery_rotation_attempts'] += 1
                                audit['recovery_rotation_detections'] += len(recovered)
            if audit is not None:
                audit['faces_before_global_nms'] = len(found)
                audit['reason'] = ('face_found' if found else 'detector_no_face')
            all_faces.extend(found)
        final = nms_faces(all_faces, self.cfg['nms_iou'])
        if diagnostics is not None:
            diagnostics['summary'] = {
                'person_crops': len(people),
                'detections_before_global_nms': len(all_faces),
                'detections_after_global_nms': len(final),
                'recovery_enabled': bool(recovery['enabled']),
            }
        return final


class FaceRecognizer:
    """Extract L2-normalized ArcFace embeddings; identity matching belongs elsewhere.

    Input frames and aligned crops use OpenCV BGR uint8 format. Face.landmarks
    must contain five points in FULL processed-frame coordinates, as returned
    by FaceDetector. This class never decides or assigns a person identity.
    """

    def __init__(self, cfg, device='auto', base_dir='.'):
        # Keep inference imports lazy, as in FaceDetector.
        import torch  # noqa: F401
        import onnxruntime as ort
        from insightface import model_zoo
        from insightface.utils import ensure_available, face_align

        self.cfg = cfg
        available = ort.get_available_providers()
        cuda = device != 'cpu' and 'CUDAExecutionProvider' in available
        if device == 'cuda' and not cuda:
            raise RuntimeError('CUDAExecutionProvider unavailable. Install compatible onnxruntime-gpu or use CPU recognition.')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']

        model_path = cfg.get('onnx_path')
        if model_path:
            path = (Path(base_dir) / model_path).resolve()
        else:
            # Reuse the same downloaded model pack as SCRFD, but load ONLY ArcFace.
            # No second face detector is allocated on the Jetson.
            pack = cfg.get('model_pack', 'buffalo_l')
            model_dir = Path(ensure_available('models', pack)).expanduser()
            path = model_dir / 'w600k_r50.onnx'
        if not path.is_file():
            raise FileNotFoundError(f'ArcFace recognition ONNX model not found: {path}')

        self.model = model_zoo.get_model(str(path), providers=providers)
        if self.model is None or self.model.taskname != 'recognition':
            raise ValueError('recognition.onnx_path must point to an InsightFace-compatible recognition model.')
        self.model.prepare(ctx_id=0 if cuda else -1)
        active = self.model.session.get_providers()
        print('Face recognizer ONNX providers:', active)
        if device == 'cuda' and 'CUDAExecutionProvider' not in active:
            raise RuntimeError('Recognition ONNX fell back to CPU; check CUDA/cuDNN installation or select CPU explicitly.')

        input_size = tuple(self.model.input_size)
        if len(input_size) != 2 or input_size[0] != input_size[1] or (input_size[0] % 112 and input_size[0] % 128):
            raise ValueError('ArcFace alignment requires a square model input divisible by 112 or 128.')
        self.input_size = int(input_size[0])
        self.model_path = str(path)
        self._face_align = face_align

    def get_embedding(self, frame, face):
        """Return unit-L2 float32 embedding, or None for unusable landmarks.

        Use the original, unannotated BGR frame. The five landmarks are in
        full-frame coordinates (not relative to a person or face crop).
        """
        self._validate_image(frame)
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
            return None
        h, w = frame.shape[:2]
        if (landmarks < 0).any() or (landmarks[:, 0] >= w).any() or (landmarks[:, 1] >= h).any():
            return None
        if np.linalg.norm(landmarks[0] - landmarks[1]) < 1.0:
            return None
        aligned = self._face_align.norm_crop(frame, landmark=landmarks, image_size=self.input_size)
        return self.get_embedding_from_aligned(aligned)

    def get_embedding_from_aligned(self, aligned_bgr):
        """Embedding for an ALREADY aligned face (avoid aligning twice)."""
        self._validate_image(aligned_bgr)
        if aligned_bgr.shape[:2] != (self.input_size, self.input_size):
            raise ValueError(f'Expected aligned face of size {self.input_size}x{self.input_size}.')
        # InsightFace get_feat owns RGB channel swapping and ONNX normalization.
        vector = np.asarray(self.model.get_feat(np.ascontiguousarray(aligned_bgr)), dtype=np.float32).reshape(-1)
        if not vector.size or not np.isfinite(vector).all():
            return None
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            return None
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    @staticmethod
    def _validate_image(image):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.shape[0] or not image.shape[1]:
            raise ValueError('Expected non-empty OpenCV BGR uint8 image with shape (height, width, 3).')
