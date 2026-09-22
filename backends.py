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

    def detect(self, frame, people, frame_index):
        all_faces = []
        h, w = frame.shape[:2]
        for person in people:
            box = expanded_box(person.box, self.cfg['crop_padding'], w, h,
                               self.cfg['person_crop_height_fraction'])
            x1, y1, x2, y2 = box
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            crop = frame[y1:y2, x1:x2]
            offset = np.array([x1, y1])
            found = self._detect_crop(crop, offset, 0)
            # Budget expensive rotation search; ordinary detection still runs each face interval.
            if frame_index % self.cfg['rotation_every_n_frames'] == 0 and (
                    not found or not self.cfg['rotate_only_if_empty']):
                for angle in self.cfg['rotation_angles']:
                    if angle:
                        found.extend(self._detect_crop(crop, offset, float(angle)))
            all_faces.extend(found)
        return nms_faces(all_faces, self.cfg['nms_iou'])
