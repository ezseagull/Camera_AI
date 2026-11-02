"""Camera AI sample: ROI -> person tracking -> face detection -> top-K per track."""
from __future__ import annotations

import argparse
from copy import deepcopy
from collections import Counter, deque
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import yaml

from backends import FaceDetector, FaceRecognizer, PersonTracker
from gallery import Gallery, write_image, write_json
from vision import (Rectifier, associate_faces, box_iou, draw_regions, inside_polygon,
                    polygon_pixels, quality_metrics, recognition_quality_metrics, tracking_window,
                     face_geometry_diagnostics)


def read_config(path):
    with Path(path).open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if cfg['face']['every_n_frames'] < 1 or cfg['face']['rotation_every_n_frames'] < 1:
        raise ValueError('Frame intervals must be positive.')
    if cfg['gallery']['top_k'] < 1 or cfg['gallery']['min_sample_gap_s'] < 0:
        raise ValueError('top_k must be positive and sample gap nonnegative.')
    if cfg['person']['lost_seconds'] <= 0:
        raise ValueError('lost_seconds must be positive.')
    if not 0 < cfg['face']['person_crop_height_fraction'] <= 1:
        raise ValueError('person_crop_height_fraction must be in (0,1].')
    if not 0 <= cfg['roi']['context_margin_fraction'] <= 1:
        raise ValueError('context_margin_fraction must be in [0,1].')
    # Recognition is strictly opt-in; preserve legacy execution when disabled.
    recognition_cfg = cfg.get('recognition', {})
    if not isinstance(recognition_cfg, dict):
        raise ValueError('recognition must be a YAML mapping.')
    if type(recognition_cfg.get('enabled', False)) is not bool:
        raise ValueError('recognition.enabled must be true or false.')
    if recognition_cfg.get('enabled', False):
        if not recognition_cfg.get('database_path'):
            raise ValueError('recognition.database_path is required when recognition is enabled.')
        if not isinstance(cfg.get('identity'), dict):
            raise ValueError('identity must be a YAML mapping when recognition is enabled.')
    return cfg


def load_roi(cfg, base_dir, rectifier, size):
    path = (base_dir / cfg['roi']['file']).resolve()
    points = cfg['roi']['points']
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('rectification_signature') != rectifier.signature:
            raise ValueError('Lens settings changed. Re-draw ROI using --edit-roi.')
        old_size = data.get('frame_size')
        if old_size and abs((size[0] / size[1]) / (old_size[0] / old_size[1]) - 1) > 0.005:
            raise ValueError('Video aspect ratio changed. Re-draw ROI using --edit-roi.')
        points = data['points']
    return polygon_pixels(points, *size)


def load_tracking_window(cfg, base_dir, polygon, width, height):
    """Optional fixed cyan tracking window; green capture ROI stays independent.

    When roi.json has no tracking_window_pixels, preserve the original behavior.
    Fixed coordinates refer to roi.json frame_size and scale with the video size.
    """
    dynamic_window = tracking_window(
        polygon, width, height, cfg['roi']['context_margin_fraction'])
    roi_path = (Path(base_dir) / cfg['roi']['file']).resolve()
    if not roi_path.is_file():
        return dynamic_window
    roi_data = json.loads(roi_path.read_text(encoding='utf-8'))
    fixed = roi_data.get('tracking_window_pixels')
    if fixed is None:
        return dynamic_window
    reference_size = roi_data.get('frame_size')
    if (not isinstance(reference_size, list) or len(reference_size) != 2
            or not all(isinstance(v, (int, float)) and v > 0 for v in reference_size)):
        raise ValueError('A fixed tracking window requires a valid ROI frame_size.')
    coords = np.asarray(fixed, dtype=float)
    if coords.shape != (4,) or not np.isfinite(coords).all():
        raise ValueError('tracking_window_pixels must be four finite numbers [x1,y1,x2,y2].')
    scale = np.asarray([width / reference_size[0], height / reference_size[1]] * 2)
    window = np.rint(coords * scale).astype(int)
    x1, y1, x2, y2 = window
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError('Fixed tracking window is outside the processed video frame.')
    if (polygon[:, 0].min() < x1 or polygon[:, 0].max() >= x2
            or polygon[:, 1].min() < y1 or polygon[:, 1].max() >= y2):
        raise ValueError('Green face ROI extends outside fixed blue tracking window.')
    return window


def roi_exclusion_geometry(center, polygon):
    """Describe which polygon boundary rejected a face, without changing ROI.

    The nearest segment is valid for triangles, trapezoids and concave ROI.
    The side is the direction from the nearest boundary point to the face
    center, NOT a new inside/outside criterion.
    """
    center = np.asarray(center, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    best_distance = float('inf')
    best_point, best_edge = None, None
    for edge_index, start in enumerate(vertices):
        stop = vertices[(edge_index + 1) % len(vertices)]
        segment = stop - start
        length_sq = float(segment @ segment)
        ratio = np.clip(float((center - start) @ segment) / length_sq, 0., 1.) if length_sq else 0.
        nearest = start + ratio * segment
        distance = float(np.linalg.norm(center - nearest))
        if distance < best_distance:
            best_distance, best_point, best_edge = distance, nearest, edge_index
    offset = center - best_point
    if abs(offset[0]) >= abs(offset[1]):
        side = 'left' if offset[0] < 0 else 'right'
    else:
        side = 'top' if offset[1] < 0 else 'bottom'
    return {'side': side, 'distance_px': round(best_distance, 2),
            'nearest_edge_index': best_edge,
            'nearest_boundary_xy': [round(float(v), 1) for v in best_point]}


def face_size_bucket(face):
    """Native detected face size, not resized/aligned ArcFace input size."""
    size = float(np.min(np.asarray(face.box[2:]) - np.asarray(face.box[:2])))
    if size < 16:
        return 'lt16'
    if size < 20:
        return '16_19'
    if size < 24:
        return '20_23'
    if size < 28:
        return '24_27'
    if size < 32:
        return '28_31'
    return 'ge32'


def roi_save_payload(path, points, frame_size, rectification_signature):
    """Keep fixed cyan tracking bounds and existing metadata when editing green ROI."""
    path = Path(path)
    existing = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    existing.update({'points': points, 'frame_size': frame_size,
                     'rectification_signature': rectification_signature})
    return existing


def edit_roi(frame, cfg, base_dir, rectifier):
    h, w = frame.shape[:2]
    scale = min(1., 1280 / w, 780 / h)
    dw, dh = max(2, round(w * scale)), max(2, round(h * scale))
    display = cv2.resize(frame, (dw, dh))
    # Start from prior vertices only if the geometry still matches.
    try:
        previous = load_roi(cfg, base_dir, rectifier, (w, h)).astype(float) / [w - 1, h - 1]
    except ValueError:
        previous = np.asarray(cfg['roi']['points'], dtype=float)
    points = [list(p * [dw - 1, dh - 1]) for p in previous]
    state = {'drag': None}
    window_name = 'ROI editor'

    def mouse(event, x, y, flags, userdata):
        x, y = float(np.clip(x, 0, dw - 1)), float(np.clip(y, 0, dh - 1))
        near = min(range(len(points)), key=lambda i: np.linalg.norm(np.array(points[i]) - [x, y])) if points else None
        if event == cv2.EVENT_LBUTTONDOWN:
            if near is not None and np.linalg.norm(np.array(points[near]) - [x, y]) < 18:
                state['drag'] = near
            else:
                points.append([x, y])
                state['drag'] = len(points) - 1
        elif event == cv2.EVENT_MOUSEMOVE and state['drag'] is not None:
            points[state['drag']] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP:
            state['drag'] = None
        elif event == cv2.EVENT_RBUTTONDOWN and near is not None:
            points.pop(near)
            state['drag'] = None

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse)
    print('ROI: drag a vertex; left click adds; right click removes; R clears; Enter saves; Esc cancels.')
    try:
        while True:
            canvas = display.copy()
            if len(points) >= 2:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], len(points) >= 3, (0, 255, 0), 2)
            for i, p in enumerate(points):
                x, y = map(int, p)
                cv2.circle(canvas, (x, y), 7, (0, 255, 0), -1)
                cv2.putText(canvas, str(i + 1), (x + 8, y - 6), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            cv2.putText(canvas, 'Drag/Add: left | Remove: right | R: clear | Enter: save | Esc: cancel',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .53, (255, 255, 255), 2)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return
            if key in (ord('r'), ord('R')):
                points.clear()
                state['drag'] = None
            if key in (10, 13):
                normalized = np.asarray(points) / [dw - 1, dh - 1] if points else np.empty((0, 2))
                try:
                    polygon_pixels(normalized, w, h)
                except ValueError as exc:
                    print(exc)
                    continue
                path = (base_dir / cfg['roi']['file']).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, roi_save_payload(
                    path, normalized.tolist(), [w, h], rectifier.signature))
                print('Saved ROI:', path)
                return
    finally:
        cv2.destroyAllWindows()


def package_versions():
    data = {}
    for name in ('ultralytics', 'insightface', 'torch', 'onnxruntime', 'onnxruntime-gpu', 'numpy', 'opencv-python'):
        try:
            data[name] = version(name)
        except PackageNotFoundError:
            pass
    data['opencv_runtime'] = cv2.__version__
    return data



class AdaptiveFaceCapture:
    """Experimental H1 scheduler; tracking, face gates and identity are untouched.

    Baseline frames still scan everyone in ``boost_only`` mode. Extra detector
    passes on intermediate frames are capped and require a recently detected,
    sufficiently large face *inside the existing capture ROI*. The scheduler
    does not infer pose or person identity from bounding-box position.

    ``balanced`` is a separate, conservative throughput experiment: on baseline
    frames, downsample only tracks with three consecutive <16px observations
    outside ROI. Re-probe them periodically and immediately if their person
    bounding-box area grows; never permanently blacklist a track.
    """

    DEFAULTS = {
        'enabled': False,
        'mode': 'boost_only',
        'min_candidate_face_px': 24.0,
        'boost_hold_s': 1.0,
        'max_extra_people_per_frame': 2,
        'tiny_face_px': 16.0,
        'tiny_observations': 3,
        'far_probe_every_base_frames': 3,
        'reapproach_area_ratio': 1.3,
    }

    def __init__(self, face_cfg, fps, *, cli_enabled=False, cli_balanced=False):
        user = face_cfg.get('adaptive_capture', {})
        if not isinstance(user, dict):
            raise ValueError('face.adaptive_capture must be a mapping.')
        unknown = set(user) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f'Unknown face.adaptive_capture options: {sorted(unknown)}')
        opts = {**self.DEFAULTS, **user}
        if type(opts['enabled']) is not bool:
            raise ValueError('face.adaptive_capture.enabled must be boolean.')
        if cli_enabled or cli_balanced:
            opts['enabled'] = True
        if cli_balanced:
            opts['mode'] = 'balanced'
        if opts['mode'] not in ('boost_only', 'balanced'):
            raise ValueError('face.adaptive_capture.mode must be boost_only or balanced.')
        for key in ('max_extra_people_per_frame', 'tiny_observations',
                    'far_probe_every_base_frames'):
            if type(opts[key]) is not int or opts[key] < 1:
                raise ValueError(f'face.adaptive_capture.{key} must be a positive integer.')
        for key in ('min_candidate_face_px', 'boost_hold_s', 'tiny_face_px',
                    'reapproach_area_ratio'):
            try:
                opts[key] = float(opts[key])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Invalid face.adaptive_capture.{key}.') from exc
            if not math.isfinite(opts[key]) or opts[key] <= 0:
                raise ValueError(f'face.adaptive_capture.{key} must be positive and finite.')
        if opts['reapproach_area_ratio'] <= 1:
            raise ValueError('reapproach_area_ratio must exceed 1.')
        self.cfg = opts
        self.enabled = opts['enabled']
        self.interval = int(face_cfg['every_n_frames'])
        self.hold_frames = max(1, int(math.ceil(opts['boost_hold_s'] * fps)))
        self._track = {}
        self.counters = Counter()
        self.per_track = {}

    @staticmethod
    def _area(person):
        wh = np.maximum(np.asarray(person.box[2:]) - person.box[:2], 0.0)
        return float(wh[0] * wh[1])

    def _state(self, tid):
        if tid not in self._track:
            self._track[tid] = {'last_good_frame': None, 'last_good_size': 0.,
                                'last_observed_area': None,
                                'tiny_history': deque(maxlen=self.cfg['tiny_observations'])}
        return self._track[tid]

    def select(self, people, frame_index, identity_manager=None):
        """Return (people_to_detect, reason) without modifying tracker state.

        A fresh track is NEVER skipped until it has been observed to be tiny
        outside ROI repeatedly. The boost selection never suppresses a
        baseline face-detection opportunity in boost_only mode.
        """
        baseline = frame_index % self.interval == 0
        if not self.enabled:
            return (people if baseline else []), 'baseline' if baseline else 'idle'
        if baseline:
            selected = []
            for person in people:
                tid = int(person.track_id)
                state = self._state(tid)
                tiny = (len(state['tiny_history']) >= self.cfg['tiny_observations']
                        and all(state['tiny_history']))
                area = self._area(person)
                previously = state['last_observed_area']
                approaching = (previously is not None and
                               area >= self.cfg['reapproach_area_ratio'] * previously)
                scheduled_probe = ((frame_index // self.interval + tid) %
                                   self.cfg['far_probe_every_base_frames'] == 0)
                suppress = (self.cfg['mode'] == 'balanced' and tiny and
                            not approaching and not scheduled_probe)
                if suppress:
                    self.counters['far_person_crops_deferred'] += 1
                    self.per_track.setdefault(tid, Counter())['far_deferred'] += 1
                else:
                    selected.append(person)
                    if tiny and self.cfg['mode'] == 'balanced':
                        self.counters['far_probe_person_crops'] += 1
            self.counters['baseline_person_crops_selected'] += len(selected)
            return selected, 'baseline'

        eligible = []
        for person in people:
            tid = int(person.track_id)
            state = self._track.get(tid)
            if not state or state['last_good_frame'] is None:
                continue
            age = frame_index - state['last_good_frame']
            if not 0 < age <= self.hold_frames:
                continue
            # Extra frames are best used while evidence is still incomplete.
            # Confirmed tracks continue to get the *ordinary* baseline scans.
            evidence = 0
            if identity_manager is not None:
                result = identity_manager.get_result(tid)
                if result.get('identity_status') in ('confirmed', 'conflict'):
                    continue
                evidence = result.get('evidence_count', 0)
            eligible.append((int(evidence), -state['last_good_size'], age, tid, person))
        eligible.sort(key=lambda v: v[:4])
        selected = [item[-1] for item in eligible[:self.cfg['max_extra_people_per_frame']]]
        self.counters['boost_eligible_person_crops'] += len(eligible)
        self.counters['boost_person_crops_selected'] += len(selected)
        self.counters['boost_person_crops_budget_dropped'] += len(eligible) - len(selected)
        for person in selected:
            self.per_track.setdefault(int(person.track_id), Counter())['boost_frames'] += 1
        return selected, 'boost' if selected else 'idle'

    def record_face(self, person, face, inside_roi, frame_index):
        tid = int(person.track_id)
        state = self._state(tid)
        size = float(np.min(np.asarray(face.box[2:]) - np.asarray(face.box[:2])))
        size = max(0., size)
        area = self._area(person)
        state['last_observed_area'] = area
        state['tiny_history'].append(not inside_roi and size < self.cfg['tiny_face_px'])
        if inside_roi and size >= self.cfg['min_candidate_face_px']:
            state['last_good_frame'] = frame_index
            state['last_good_size'] = size
            self.counters['opportunity_face_observations'] += 1

    def forget(self, track_id):
        self._track.pop(int(track_id), None)

    def summary(self):
        return {'enabled': bool(self.enabled), 'config': dict(self.cfg),
                'counters': dict(self.counters),
                'tracks': [{'track_id': tid, **dict(counts)} for tid, counts
                           in sorted(self.per_track.items())]}


def apply_h1_detector_options(cfg, args):
    """Copy config and enable recovery only when explicitly requested.

    Runtime flags affect the persisted run config, never the on-disk YAML.
    """
    cfg = deepcopy(cfg)
    if getattr(args, 'h1_recovery', False):
        if not hasattr(FaceDetector, '_read_recovery_config'):
            raise RuntimeError('H1 Recovery requires the updated backends.py FaceDetector.')
        recovery = cfg['face'].setdefault('recovery', {})
        if not isinstance(recovery, dict):
            raise ValueError('face.recovery must be a mapping.')
        recovery['enabled'] = True
        recovery.setdefault('retry_upper_crop', True)
        recovery.setdefault('upper_crop_height_fraction', .65)
        recovery.setdefault('retry_rotations_on_unscheduled_empty', False)
    return cfg

def process_video(cap, first_frame, cfg, base_dir, args, rectifier, fps, output_dir,
                  tracker=None, face_detector=None, recognizer=None, identity_manager=None):
    """Backends may be injected for deterministic integration tests."""
    cfg = apply_h1_detector_options(cfg, args)
    h, w = first_frame.shape[:2]
    h1 = AdaptiveFaceCapture(cfg['face'], fps,
                             cli_enabled=getattr(args, 'h1_capture', False),
                             cli_balanced=getattr(args, 'h1_balanced', False))
    polygon = load_roi(cfg, base_dir, rectifier, (w, h))
    crop_window = load_tracking_window(cfg, base_dir, polygon, w, h)
    recognition_cfg = cfg.get('recognition', {})
    recognition_enabled = recognition_cfg.get('enabled', False)
    identity_records = []
    identity_evidence_records = []  # Final bounded selection audit per temporary track.
    if recognition_enabled:
        # Do not load recognition models or the enrollment database for ROI editing
        # or when running the original tracking/gallery-only pipeline.
        if identity_manager is None:
            from face_database import FaceDatabase
            from identity_manager import IdentityManager
            database_path = (Path(base_dir) / recognition_cfg['database_path']).resolve()
            if not database_path.is_file():
                raise FileNotFoundError(f'Enrollment database not found: {database_path}. Run enroll.py first.')
            database = FaceDatabase.load(database_path)
            identity_manager = IdentityManager(database, cfg['identity'])
        if recognizer is None:
            recognizer = FaceRecognizer(recognition_cfg, args.face_device, base_dir)
        # Fail closed: no silent identity matching with a different ArcFace model.
        identity_manager.verify_recognizer(recognizer)
    tracker = tracker or PersonTracker(cfg['person'], fps, output_dir, args.device)
    face_detector = face_detector or FaceDetector(cfg['face'], args.face_device, base_dir)
    gallery = Gallery(output_dir / 'faces', cfg['gallery'])
    writer = None
    if cfg['output']['save_video']:
        writer = cv2.VideoWriter(str(output_dir / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError('Cannot create annotated.mp4; install an OpenCV build with MP4 support.')
    roi_preview = draw_regions(rectifier.apply(first_frame).copy(), polygon, crop_window)
    write_image(output_dir / 'roi_preview.jpg', roi_preview)
    meta = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': args.source,
            'fps': fps, 'frame_size': [w, h], 'config': cfg, 'roi_pixels': polygon.tolist(),
            'tracking_window': crop_window.tolist(), 'rectification_signature': rectifier.signature,
            'versions': package_versions(), 'identity_scope': 'temporary_track_in_this_run',
            'coordinate_space': 'full_processed_frame', 'status': 'running'}
    if recognition_enabled:
        meta['identity_scope'] = 'enrolled_person_id_and_temporary_track'
        meta['recognition'] = {
            'enabled': True,
            'database_path': str((Path(base_dir) / recognition_cfg['database_path']).resolve()),
            'model_id': identity_manager.database.model_id,
            'enrolled_identities': len(identity_manager.database),
            'enrolled_templates': identity_manager.database.total_templates,
            'thresholds_configured': identity_manager.thresholds_configured,
        }
    meta['h1_capture'] = {'enabled': h1.enabled, 'mode': h1.cfg['mode'],
                          'recovery_enabled': bool(cfg['face'].get('recovery', {}).get('enabled', False))}
    write_json(output_dir / 'run.json', meta)
    counters, history, trails = Counter(), {}, {}
    # Diagnostic-only counters: no effect on ROI, matching, quality gates or identity decisions.
    track_capture = {}
    # Diagnostics are observation-only; no changes to ROI gates or IdentityManager.
    diagnostic_events = ((output_dir / 'face_diagnostic_events.jsonl').open('w', encoding='utf-8')
                         if recognition_enabled else None)
    last_identity_state = {}

    def log_event(event, **fields):
        if diagnostic_events is not None:
            diagnostic_events.write(json.dumps({'event': event, **fields},
                                               ensure_ascii=False, allow_nan=False) + '\n')

    def log_identity_transition(track_id, result, frame_no, time_s, source):
        key = (result.get('identity_status'), result.get('person_id'),
               result.get('candidate_person_id'), result.get('reason'))
        if last_identity_state.get(track_id) != key:
            last_identity_state[track_id] = key
            track_capture[track_id]['identity_state_changes'] += 1
            log_event('identity_state', track_id=track_id, frame_index=frame_no,
                      timestamp_s=round(time_s, 3), source=source,
                      identity_status=result.get('identity_status'),
                      person_id=result.get('person_id'),
                      candidate_person_id=result.get('candidate_person_id'),
                      reason=result.get('reason'),
                      match_score=result.get('match_score'),
                      second_match_score=result.get('second_match_score'),
                      score_margin=result.get('score_margin'),
                      vote_fraction=result.get('vote_fraction'),
                      evidence_count=result.get('evidence_count'),
                      high_count=result.get('high_count'),
                      medium_count=result.get('medium_count'))

    started = time.perf_counter()
    frame_index, frame = 0, first_frame
    finish_reason = 'end_of_video'
    try:
        while frame is not None:
            clean = rectifier.apply(frame)
            # Every video frame updates the tracker, including frames without detections.
            people = tracker.update(clean, crop_window)
            timestamp = frame_index / fps  # Constant-frame-rate recorded video.
            for person in people:
                gallery.observe(person.track_id, frame_index, timestamp)
                capture = track_capture.setdefault(person.track_id, Counter())
                capture['person_frames'] += 1
                trail = trails.setdefault(person.track_id, deque(maxlen=45))
                trail.append(tuple(np.rint((person.box[:2] + person.box[2:]) / 2).astype(int)))
            face_labels = []
            detect_people, capture_mode = h1.select(
                people, frame_index, identity_manager if recognition_enabled else None)
            baseline_due = frame_index % cfg['face']['every_n_frames'] == 0
            # If adaptive capture is disabled, this is identical to the old
            # baseline schedule. In boost mode, SCRFD runs on a capped subset.
            if (baseline_due and not h1.enabled) or (detect_people and (baseline_due or h1.enabled)):
                detect_ids = {int(person.track_id) for person in detect_people}
                counters['h1_person_crops_submitted'] += len(detect_people)
                if capture_mode == 'boost':
                    counters['h1_boost_frames'] += 1
                detect_started = time.perf_counter()
                # New SCRFD diagnostics are available only on the real FaceDetector;
                # injected fake backends keep the unchanged detect() contract.
                detector_audit = ({} if getattr(args, 'diagnose_association', False)
                                  and isinstance(face_detector, FaceDetector) else None)
                if detector_audit is None:
                    faces = face_detector.detect(clean, detect_people, frame_index)
                else:
                    faces = face_detector.detect(clean, detect_people, frame_index,
                                                 diagnostics=detector_audit)
                    for item in detector_audit['people']:
                        tid = item['track_id']
                        capture = track_capture[tid]
                        capture['detector_crop_attempts'] += 1
                        capture['detector_normal_faces'] += item['normal_detections']
                        capture['detector_rotation_faces'] += item['scheduled_rotation_detections']
                        capture['detector_recovery_upper_attempts'] += item['recovery_upper_attempts']
                        capture['detector_recovery_upper_faces'] += item['recovery_upper_detections']
                        capture['detector_recovery_rotation_attempts'] += item['recovery_rotation_attempts']
                        capture['detector_recovery_rotation_faces'] += item['recovery_rotation_detections']
                        if item['reason'] in ('crop_too_small', 'detector_no_face'):
                            capture[item['reason'] if item['reason'] == 'detector_no_face'
                                    else 'detector_crop_too_small'] += 1
                        if item['reason'] != 'face_found' or item['recovery_upper_detections'] or item['recovery_rotation_detections']:
                            log_event('face_detector', frame_index=frame_index,
                                      timestamp_s=round(timestamp, 3), **item)
                counters['h1_detector_elapsed_s'] += time.perf_counter() - detect_started
                counters['detected_faces'] += len(faces)
                if h1.enabled and capture_mode == 'boost':
                    counters['h1_boost_faces_detected'] += len(faces)
                if h1.enabled:
                    log_event('h1_capture_schedule', frame_index=frame_index,
                              timestamp_s=round(timestamp, 3), capture_mode=capture_mode,
                              selected_track_ids=sorted(detect_ids),
                              other_people_on_frame=len(people) - len(detect_people))
                # Optional, observation-only association audit; no changed thresholds
                # or person-face assignments. Off by default for baseline/FPS parity.
                association_audit = {} if getattr(args, 'diagnose_association', False) else None
                if association_audit is None:
                    pairs = associate_faces(people, faces, cfg['association'], history, frame_index,
                                            max_history_age=max(1, round(0.5 * fps)))
                else:
                    pairs = associate_faces(people, faces, cfg['association'], history, frame_index,
                                            max_history_age=max(1, round(0.5 * fps)),
                                            diagnostics=association_audit)
                counters['unassigned_faces'] += len(faces) - len(pairs)
                matched_person_indices = {pi for pi, _ in pairs}
                matched_face_indices = {fi for _, fi in pairs}
                for pi, person in enumerate(people):
                    capture = track_capture[person.track_id]
                    # In H1, a deferred crop is not a SCRFD miss. Still retain
                    # all people for conservative face-person association.
                    if (h1.enabled and person.track_id not in detect_ids
                            and pi not in matched_person_indices):
                        continue
                    capture['face_detection_opportunities'] += 1
                    if pi not in matched_person_indices:
                        capture['face_detection_frames_no_match'] += 1
                    if association_audit is not None:
                        reason = association_audit['people'][pi]['reason']
                        diagnostic_key = (reason if reason.startswith('association_')
                                          else 'association_' + reason)
                        capture[diagnostic_key] += 1
                # Unassigned faces cannot safely be attributed to any track.
                for fi, face in enumerate(faces):
                    if fi not in matched_face_indices:
                        association_details = (association_audit['faces'][fi]
                                               if association_audit is not None else {})
                        log_event('unassigned_face', frame_index=frame_index,
                                  timestamp_s=round(timestamp, 3),
                                  face_center_xy=[round(float(v), 1) for v in
                                                  (face.box[:2] + face.box[2:]) / 2],
                                  native_face_size_bucket=face_size_bucket(face),
                                  **({'association_reason': association_details['reason'],
                                      'geometric_candidates': association_details['geometric_candidates'],
                                      'accepted_candidates': association_details['accepted_candidates'],
                                      'geometric_gate_counts': association_details['gate_counts']}
                                     if association_audit is not None else {}))
                for pi, fi in pairs:
                    person, face = people[pi], faces[fi]
                    track_capture[person.track_id]['face_matches'] += 1
                    center = (face.box[:2] + face.box[2:]) / 2
                    face_in_roi = inside_polygon(center, polygon)
                    if h1.enabled:
                        h1.record_face(person, face, face_in_roi, frame_index)
                        if capture_mode == 'boost':
                            track_capture[person.track_id]['h1_boost_face_matches'] += 1
                            counters['h1_boost_face_matches'] += 1
                    relative = (center - person.box[:2]) / np.maximum(person.box[2:] - person.box[:2], 1)
                    history[person.track_id] = (frame_index, relative)
                    if not face_in_roi:
                        if h1.enabled and capture_mode == 'boost':
                            counters['h1_boost_faces_outside_roi'] += 1
                        counters['outside_roi'] += 1
                        capture = track_capture[person.track_id]
                        capture['face_outside_green_roi'] += 1
                        exit_info = roi_exclusion_geometry(center, polygon)
                        capture['outside_' + exit_info['side']] += 1
                        capture['outside_distance_sum_px'] += exit_info['distance_px']
                        capture['outside_distance_max_px'] = max(
                            capture['outside_distance_max_px'], exit_info['distance_px'])
                        log_event('outside_green_roi', frame_index=frame_index,
                                  timestamp_s=round(timestamp, 3), track_id=person.track_id,
                                  face_center_xy=[round(float(v), 1) for v in center],
                                  native_face_size_bucket=face_size_bucket(face), **exit_info)
                        continue
                    track_capture[person.track_id]['face_inside_green_roi'] += 1
                    if h1.enabled and capture_mode == 'boost':
                        counters['h1_boost_faces_inside_roi'] += 1
                    track_capture[person.track_id][
                        'native_face_size_' + face_size_bucket(face)] += 1
                    metrics, reason = quality_metrics(clean, face, cfg['quality'], rectifier.valid_mask)
                    counters['quality_' + reason] += 1
                    saved = False
                    if metrics is not None:
                        saved = gallery.offer(person, face, metrics, clean, frame_index, timestamp)
                        counters['gallery_updates'] += int(saved)
                    if recognition_enabled:
                        # Keep relaxed recognition evidence independent of the
                        # original (unchanged) strict top-K gallery selection.
                        rec_metrics, tier = recognition_quality_metrics(
                            clean, face, cfg['quality'], rectifier.valid_mask)
                        counters['recognition_quality_' + tier] += 1
                        track_capture[person.track_id]['quality_' + tier] += 1
                        if h1.enabled and capture_mode == 'boost':
                            counters['h1_boost_quality_' + tier] += 1
                        if association_audit is not None and rec_metrics is None:
                            geometry = face_geometry_diagnostics(face, clean.shape, cfg['quality'])
                            log_event('recognition_quality_rejected', frame_index=frame_index,
                                      timestamp_s=round(timestamp, 3), track_id=person.track_id,
                                      quality_reason=tier, **geometry)
                        if rec_metrics is not None:
                            if h1.enabled and capture_mode == 'boost':
                                counters['h1_boost_recognition_offers'] += 1
                                track_capture[person.track_id]['h1_boost_recognition_offers'] += 1
                            track_capture[person.track_id]['recognition_offers'] += 1
                            archived = gallery.offer_recognition(
                                person, face, rec_metrics, clean, frame_index, timestamp, tier=tier)
                            counters['recognition_gallery_updates'] += int(archived)
                            counters['recognition_attempts'] += 1
                            before = identity_manager.get_result(person.track_id)
                            after = identity_manager.observe_face(
                                person, face, rec_metrics, clean, frame_index, timestamp,
                                recognizer, tier=tier, association_ok=True)
                            # A zero bank-size delta can mean replacement OR rejection.
                            # Exact selection reasons require instrumenting identity_manager.py.
                            log_event('recognition_offer', frame_index=frame_index,
                                      timestamp_s=round(timestamp, 3), track_id=person.track_id,
                                      capture_mode=capture_mode,
                                      tier=tier, quality_score=float(rec_metrics['score']),
                                      archived_to_gallery=bool(archived),
                                      evidence_count_before=before.get('evidence_count', 0),
                                      evidence_count_after=after.get('evidence_count', 0),
                                      last_evidence_frame_before=before.get('last_frame_index'),
                                      last_evidence_frame_after=after.get('last_frame_index'),
                                      candidate_person_id=after.get('candidate_person_id'),
                                      match_score=after.get('match_score'),
                                      second_match_score=after.get('second_match_score'),
                                      score_margin=after.get('score_margin'),
                                      vote_fraction=after.get('vote_fraction'),
                                      strong_support=after.get('strong_support'),
                                      identity_status=after.get('identity_status'),
                                      reason=after.get('reason'),
                                      evidence_action=after.get('evidence_last_action'),
                                      evidence_action_reason=after.get('evidence_last_reason'))
                            track_capture[person.track_id]['evidence_net_changes'] += int(
                                after.get('evidence_count', 0) != before.get('evidence_count', 0))
                            if h1.enabled and capture_mode == 'boost' and (
                                    after.get('evidence_count', 0) > before.get('evidence_count', 0)):
                                counters['h1_boost_evidence_added'] += 1
                            log_identity_transition(person.track_id, after,
                                                    frame_index, timestamp, 'recognition_offer')
                    face_labels.append((face, person.track_id, reason, saved))
            if recognition_enabled:
                # Only flag simultaneous ID collisions for visibly distinct,
                # non-overlapping tracks. Occluded/overlapping detections alone
                # cannot prove that two tracks represent different people.
                for index, person in enumerate(people):
                    result = identity_manager.get_result(person.track_id)
                    if result['identity_status'] != 'confirmed':
                        continue
                    for other in people[:index]:
                        other_result = identity_manager.get_result(other.track_id)
                        if (other_result['identity_status'] == 'confirmed'
                                and other_result['person_id'] == result['person_id']
                                and box_iou(person.box, other.box) < 0.1):
                            first_conflict = identity_manager.mark_conflict(
                                person.track_id, 'same_identity_on_distinct_simultaneous_tracks')
                            second_conflict = identity_manager.mark_conflict(
                                other.track_id, 'same_identity_on_distinct_simultaneous_tracks')
                            log_identity_transition(person.track_id, first_conflict,
                                                    frame_index, timestamp, 'simultaneous_conflict')
                            log_identity_transition(other.track_id, second_conflict,
                                                    frame_index, timestamp, 'simultaneous_conflict')
                            counters['identity_simultaneous_conflicts'] += 1
                            break
                # Gallery.finalize()/expire() write the track metadata; attach a
                # current JSON-safe identity summary before that happens.
                for person in people:
                    gallery.tracks[person.track_id]['identity'] = identity_manager.get_result(person.track_id)
            # Annotate only after cropping/saving from the clean frame.
            canvas = draw_regions(clean.copy(), polygon, crop_window)
            for person in people:
                x1, y1, x2, y2 = person.box.astype(int)
                track_color = (245, 180, 50)  # Original color for unconfirmed tracks (BGR).
                n = len(gallery.tracks[person.track_id]['samples'])
                track_label = f'Track {person.track_id} | faces {n}'
                if recognition_enabled:
                    result = identity_manager.get_result(person.track_id)
                    if result['identity_status'] == 'confirmed':
                        track_color = (255, 0, 255)  # Magenta: confirmed identity (BGR).
                        track_label += f" | ID {result['person_id']}"
                    elif (getattr(args, 'show_candidates', False)
                          and result.get('candidate_person_id')):
                        # Diagnostic display only: tentative candidate is NOT a confirmed ID.
                        score = result.get('match_score')
                        score_text = f' {score:.3f}' if score is not None else ''
                        track_label += f" | ?{result['candidate_person_id']}{score_text} (PRED)"
                    else:
                        # A tentative candidate is never displayed as a verified name.
                        track_label += f" | {result['identity_status'].upper()}"
                cv2.rectangle(canvas, (x1, y1), (x2, y2), track_color, 2)
                cv2.putText(canvas, track_label, (max(0, x1), max(50, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, track_color, 2)
                if len(trails[person.track_id]) >= 2:
                    cv2.polylines(canvas, [np.asarray(trails[person.track_id], dtype=np.int32)], False, (255, 170, 30), 2)
            for face, tid, reason, saved in face_labels:
                x1, y1, x2, y2 = face.box.astype(int)
                color = (30, 255, 70) if saved else (0, 200, 255)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                for x, y in face.landmarks.astype(int):
                    cv2.circle(canvas, (x, y), 2, color, -1)
                label = 'SAVED' if saved else reason
                cv2.putText(canvas, f'{tid}: {label}', (max(0, x1), max(75, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
            for tid in gallery.expire(frame_index, tracker.lost_frames):
                h1.forget(tid)
                history.pop(tid, None)
                trails.pop(tid, None)
                if recognition_enabled:
                    if hasattr(identity_manager, 'get_evidence_diagnostics'):
                        identity_evidence_records.append({
                            'track_id': tid,
                            **identity_manager.get_evidence_diagnostics(tid)})
                    final_summary = identity_manager.finalize_track(tid)
                    log_identity_transition(tid, final_summary, frame_index, timestamp, 'expired')
                    identity_records.append(final_summary)
                last_identity_state.pop(tid, None)
            counters['frames'] += 1
            elapsed = time.perf_counter() - started
            cv2.putText(canvas, f'Frame {frame_index} | processing {counters["frames"] / max(elapsed, .001):.1f} fps',
                        (15, h - 16), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            if writer is not None:
                writer.write(canvas)
            if not args.headless:
                scale = min(1., 1280 / w, 780 / h)
                cv2.imshow('Camera AI - press Q to stop', cv2.resize(canvas, (round(w * scale), round(h * scale))))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    finish_reason = 'user_stop'
                    break
            if args.max_frames and counters['frames'] >= args.max_frames:
                finish_reason = 'max_frames'
                break
            if counters['frames'] % max(1, round(5 * fps)) == 0:
                print(f'{counters["frames"]} frames | {counters["gallery_updates"]} gallery updates | {counters["frames"] / max(elapsed, .001):.1f} fps')
            ok, frame = cap.read()
            frame = frame if ok else None
            frame_index += 1
    except KeyboardInterrupt:
        finish_reason = 'keyboard_interrupt'
    except Exception:
        finish_reason = 'error'
        raise
    finally:
        if recognition_enabled:
            # Final results are written even for early stops and interruptions.
            for tid in list(gallery.tracks):
                if hasattr(identity_manager, 'get_evidence_diagnostics'):
                    identity_evidence_records.append({
                        'track_id': tid,
                        **identity_manager.get_evidence_diagnostics(tid)})
                summary = identity_manager.finalize_track(tid)
                log_identity_transition(tid, summary, frame_index, frame_index / fps, 'finalize')
                gallery.tracks[tid]['identity'] = summary
                identity_records.append(summary)
            identity_manager.close()
        gallery.close(finish_reason)
        h1_report = h1.summary()
        h1_report['runtime_counters'] = {key: value for key, value in counters.items()
                                         if key.startswith('h1_')}
        write_json(output_dir / 'h1_capture_diagnostics.json', h1_report)
        if recognition_enabled:
            write_json(output_dir / 'identities.json', {
                'status': finish_reason,
                'model_id': meta['recognition']['model_id'],
                'thresholds_configured': meta['recognition']['thresholds_configured'],
                'tracks': identity_records,
            })
            write_json(output_dir / 'identity_evidence_diagnostics.json', {
                'schema_version': 1,
                'notes': ('Bounded action history and counters per temporary track. '
                          'No face images or feature vectors. Similarity scores are not probabilities.'),
                'tracks': identity_evidence_records,
            })
            # Each face observation is diagnostic only, not a unique person.
            identity_by_track = {x['track_id']: x for x in identity_records}
            write_json(output_dir / 'face_capture_diagnostics.json', {
                'schema_version': 2,
                'notes': ('Counts are observations, not unique people or independent embeddings. '
                          'The JSONL event trace contains no embeddings; a net-zero evidence count '
                          'cannot distinguish bank replacement from discarded offers.'),
                'event_trace_file': 'face_diagnostic_events.jsonl',
                'tracks': [
                    {
                        'track_id': tid,
                        **{key: counts[key] for key in (
                            'person_frames', 'face_detection_opportunities',
                            'face_detection_frames_no_match', 'face_matches',
                            'face_inside_green_roi', 'face_outside_green_roi',
                            'outside_left', 'outside_right', 'outside_top',
                            'outside_bottom', 'quality_high', 'quality_medium',
                            'quality_small', 'quality_eyes_small',
                            'quality_confidence', 'quality_blur',
                            'recognition_offers', 'evidence_net_changes',
                            'identity_state_changes',
                            'native_face_size_lt16', 'native_face_size_16_19',
                            'native_face_size_20_23', 'native_face_size_24_27',
                            'native_face_size_28_31', 'native_face_size_ge32')},
                        **dict(counts),
                        'outside_distance_mean_px': (
                            round(counts['outside_distance_sum_px'] /
                                  counts['face_outside_green_roi'], 2)
                            if counts['face_outside_green_roi'] else None),
                        'evidence_count': identity_by_track.get(tid, {}).get('evidence_count', 0),
                        'identity_status': identity_by_track.get(tid, {}).get('identity_status', 'pending'),
                        'identity_reason': identity_by_track.get(tid, {}).get('reason', 'not_finalized'),
                    }
                    for tid, counts in sorted(track_capture.items())
                ],
            })
            counters['identity_tracks_finalized'] = len(identity_records)
            counters['identity_tracks_confirmed'] = sum(
                item['identity_status'] == 'confirmed' for item in identity_records)
            counters['identity_tracks_conflict'] = sum(
                item['identity_status'] == 'conflict' for item in identity_records)
        if diagnostic_events is not None:
            diagnostic_events.close()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        meta.update(status=finish_reason, counters=dict(counters), elapsed_seconds=time.perf_counter() - started)
        write_json(output_dir / 'run.json', meta)
    print('Output:', output_dir)
    print(json.dumps(dict(counters), ensure_ascii=False))
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Recorded video file; paths are relative to the current directory.')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--device', default='auto', help='YOLO device: auto, cpu or CUDA index such as 0')
    parser.add_argument('--face-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--edit-roi', action='store_true')
    mode.add_argument('--preview-roi', action='store_true', help='Save roi_preview.jpg without loading models.')
    parser.add_argument('--at-second', type=float, default=0., help='Seek for ROI editing/preview only.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--show-candidates', action='store_true',
                        help='Show unverified candidate IDs in annotated video for diagnostic review only.')
    parser.add_argument('--diagnose-association', action='store_true',
                        help='Write detailed association gate and face-size diagnostics; matching is unchanged.')
    parser.add_argument('--h1-capture', action='store_true',
                        help='H1 boost-only: capped extra SCRFD scans after a usable face appears inside ROI.')
    parser.add_argument('--h1-balanced', action='store_true',
                        help='H1 balanced: boosts plus periodic downsampling of persistently tiny outside-ROI faces.')
    parser.add_argument('--h1-recovery', action='store_true',
                        help='Independently enable experimental miss-only upper-crop SCRFD recovery (new backends.py).')
    args = parser.parse_args()
    if args.max_frames < 0 or args.at_second < 0:
        parser.error('--max-frames and --at-second must be nonnegative')
    if args.at_second and not (args.edit_roi or args.preview_roi):
        parser.error('--at-second is for ROI editing/preview only')
    if args.headless and args.edit_roi:
        parser.error('--edit-roi requires a desktop display')
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    cfg = read_config(config_path)
    # Custom YOLO checkpoint paths are relative to config; official model names stay unchanged.
    if (base_dir / cfg['person']['model']).is_file():
        cfg['person']['model'] = str((base_dir / cfg['person']['model']).resolve())
    cap = cv2.VideoCapture(args.source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {args.source}')
        if args.at_second:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.at_second * 1000)
        ok, first = cap.read()
        if not ok:
            raise RuntimeError('Cannot decode the requested frame.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = float(cfg['input']['fallback_fps'])
            print(f'Video FPS metadata missing; using configured {fps} FPS.')
        rectifier = Rectifier(cfg['undistort'], first.shape[1::-1], base_dir)
        processed = rectifier.apply(first)
        if args.edit_roi:
            edit_roi(processed, cfg, base_dir, rectifier)
        elif args.preview_roi:
            polygon = load_roi(cfg, base_dir, rectifier, first.shape[1::-1])
            window = load_tracking_window(cfg, base_dir, polygon, first.shape[1], first.shape[0])
            path = base_dir / 'roi_preview.jpg'
            write_image(path, draw_regions(processed.copy(), polygon, window))
            print('ROI preview:', path)
        else:
            output_dir = (base_dir / cfg['output']['directory'] /
                          (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
            output_dir.mkdir(parents=True, exist_ok=False)
            process_video(cap, first, cfg, base_dir, args, rectifier, fps, output_dir)
    finally:
        cap.release()


if __name__ == '__main__':
    main()