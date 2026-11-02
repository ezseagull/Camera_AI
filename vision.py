"""Geometry, calibrated undistortion, conservative association and face quality.

All boxes/landmarks below use the full processed frame, before any annotation.
No face identity model, generative restoration, or guessed lens coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Person:
    track_id: int
    box: np.ndarray
    confidence: float


@dataclass
class Face:
    box: np.ndarray
    landmarks: np.ndarray
    confidence: float
    rotation: float = 0.0


def clip_box(box, width, height):
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    return np.array([np.clip(np.floor(x1), 0, width),
                     np.clip(np.floor(y1), 0, height),
                     np.clip(np.ceil(x2), 0, width),
                     np.clip(np.ceil(y2), 0, height)], dtype=int)


def expanded_box(box, fraction, width, height, height_fraction=1.0):
    x1, y1, x2, y2 = np.asarray(box, dtype=float)
    bw, bh = x2 - x1, y2 - y1
    return clip_box([x1 - fraction * bw, y1 - fraction * bh,
                     x2 + fraction * bw, y1 + height_fraction * bh + fraction * bh],
                    width, height)


def coverage(inner, outer):
    """Fraction of INNER box contained in OUTER, not IoU."""
    lo = np.maximum(inner[:2], outer[:2])
    hi = np.minimum(inner[2:], outer[2:])
    return float(np.prod(np.maximum(hi - lo, 0)) /
                 max(float(np.prod(np.maximum(inner[2:] - inner[:2], 0))), 1e-6))


def box_iou(a, b):
    inter = float(np.prod(np.maximum(np.minimum(a[2:], b[2:]) -
                                     np.maximum(a[:2], b[:2]), 0)))
    union = float(np.prod(np.maximum(a[2:] - a[:2], 0)) +
                  np.prod(np.maximum(b[2:] - b[:2], 0)) - inter)
    return inter / max(union, 1e-6)


def nms_faces(faces, threshold=0.4):
    """Deduplicate detections from overlapping person crops and rotations."""
    remaining = sorted(faces, key=lambda f: f.confidence, reverse=True)
    kept = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [f for f in remaining if box_iou(best.box, f.box) < threshold]
    return kept


def affine_points(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    return p @ matrix[:, :2].T + matrix[:, 2]


def rotate_expanded(image, angle):
    """Rotate without clipping; return inverse map to the original crop."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(w * c + h * s)), int(np.ceil(h * c + w * s))
    matrix[:, 2] += [(nw - w) / 2, (nh - h) / 2]
    rotated = cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR)
    return rotated, cv2.invertAffineTransform(matrix)


def polygon_pixels(points, width, height):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError('ROI must contain at least three finite [x,y] vertices.')
    if (p < 0).any() or (p > 1).any():
        raise ValueError('ROI coordinates must be normalized to [0,1].')
    # Check nonadjacent edge intersections; concave simple polygons are allowed.
    def cross(a, b, c):
        u, v = b - a, c - a
        return u[0] * v[1] - u[1] * v[0]
    def on_segment(a, b, c):
        return abs(cross(a, b, c)) < 1e-10 and np.all(c >= np.minimum(a, b) - 1e-10) and np.all(c <= np.maximum(a, b) + 1e-10)
    for i in range(len(p)):
        if np.linalg.norm(p[i] - p[(i + 1) % len(p)]) < 1e-6:
            raise ValueError('ROI contains duplicate neighboring vertices.')
        for j in range(i + 1, len(p)):
            if j == i + 1 or (i == 0 and j == len(p) - 1):
                continue
            a, b, c, d = p[i], p[(i + 1) % len(p)], p[j], p[(j + 1) % len(p)]
            if ((cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0)
                    or any([on_segment(a, b, c), on_segment(a, b, d), on_segment(c, d, a), on_segment(c, d, b)])):
                raise ValueError('ROI edges intersect. Draw the vertices in perimeter order.')
    poly = np.rint(p * [width - 1, height - 1]).astype(np.int32)
    if abs(cv2.contourArea(poly)) < 16:
        raise ValueError('ROI is too small or degenerate.')
    return poly


def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0


def roi_point_diagnostics(point, polygon):
    """Describe a point relative to the existing green ROI, WITHOUT changing it.

    Distances are in full processed-frame pixels. ``signed_distance_px`` is
    positive inside, negative outside and zero on the boundary. ``side`` is
    only a geometric direction from the nearest edge toward an outside point;
    it is not a semantic entrance/exit direction for a concave polygon.
    """
    p = np.asarray(point, dtype=np.float64)
    vertices = np.asarray(polygon, dtype=np.float64)
    if p.shape != (2,) or not np.isfinite(p).all():
        raise ValueError('ROI diagnostic point must be a finite [x, y] pair.')
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3 or not np.isfinite(vertices).all():
        raise ValueError('ROI diagnostic polygon must have at least three finite vertices.')
    contour = np.ascontiguousarray(vertices.astype(np.float32))
    signed = float(cv2.pointPolygonTest(contour, tuple(map(float, p)), True))
    nearest_xy, nearest_index = None, None
    nearest_distance_sq = float('inf')
    for edge_index, a in enumerate(vertices):
        b = vertices[(edge_index + 1) % len(vertices)]
        ab = b - a
        length_sq = float(ab @ ab)
        t = float(np.clip(((p - a) @ ab) / length_sq, 0., 1.)) if length_sq else 0.
        nearest = a + t * ab
        dist_sq = float(np.sum((p - nearest) ** 2))
        if dist_sq < nearest_distance_sq:
            nearest_xy, nearest_index, nearest_distance_sq = nearest, edge_index, dist_sq
    side = None
    if signed < 0:
        offset = p - nearest_xy
        if abs(offset[0]) >= abs(offset[1]):
            side = 'left' if offset[0] < 0 else 'right'
        else:
            side = 'top' if offset[1] < 0 else 'bottom'
    return {
        'inside': bool(signed >= 0),
        'signed_distance_px': round(signed, 3),
        'distance_to_border_px': round(float(np.sqrt(nearest_distance_sq)), 3),
        'side': side,
        'nearest_edge_index': int(nearest_index),
        'nearest_boundary_xy': [round(float(v), 3) for v in nearest_xy],
    }


def tracking_window(polygon, width, height, margin_fraction):
    """Fixed crop: its origin must never move between tracker updates."""
    pad = np.array([width, height]) * margin_fraction
    return clip_box(np.r_[polygon.min(0) - pad, polygon.max(0) + pad + 1], width, height)


class Rectifier:
    def __init__(self, cfg, size, base_dir):
        self.size = tuple(size)
        self.mode = cfg['mode']
        self.maps = None
        signature = {'mode': self.mode}
        w, h = self.size
        self.valid_mask = np.ones((h, w), dtype=np.uint8)
        if self.mode != 'none':
            if self.mode not in ('pinhole', 'fisheye'):
                raise ValueError('undistort.mode must be none, pinhole or fisheye')
            path = (Path(base_dir) / cfg['calibration_file']).resolve()
            with np.load(path, allow_pickle=False) as data:
                k = np.asarray(data['K'], dtype=np.float64).copy()
                d = np.asarray(data['D'], dtype=np.float64).copy()
                cw, ch = map(int, data['image_size'])
                model = str(data['model'].item())
            if model != self.mode:
                raise ValueError(f'Calibration model is {model}, but config requests {self.mode}.')
            if k.shape != (3, 3) or not np.isfinite(k).all() or not np.isfinite(d).all():
                raise ValueError('Invalid calibration matrices.')
            if cw <= 0 or ch <= 0 or abs((w / h) / (cw / ch) - 1) > 0.005:
                raise ValueError('Video aspect ratio differs from calibration; recalibrate this camera mode.')
            k[0, :] *= w / cw
            k[1, :] *= h / ch
            if self.mode == 'fisheye':
                if d.size != 4:
                    raise ValueError('Fisheye calibration requires four distortion coefficients.')
                balance = float(cfg['balance'])
                if not 0 <= balance <= 1:
                    raise ValueError('balance must be in [0,1].')
                new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    k, d, self.size, np.eye(3), balance=balance)
                self.maps = cv2.fisheye.initUndistortRectifyMap(
                    k, d, np.eye(3), new_k, self.size, cv2.CV_32FC1)
                signature['balance'] = balance
            else:
                alpha = float(cfg['alpha'])
                if not 0 <= alpha <= 1:
                    raise ValueError('alpha must be in [0,1].')
                new_k, _ = cv2.getOptimalNewCameraMatrix(k, d, self.size, alpha, self.size)
                self.maps = cv2.initUndistortRectifyMap(k, d, None, new_k, self.size, cv2.CV_32FC1)
                signature['alpha'] = alpha
            mx, my = self.maps
            self.valid_mask = ((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1)).astype(np.uint8)
            signature['calibration_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()

    def apply(self, frame):
        if frame.shape[1::-1] != self.size:
            raise ValueError('Frame resolution changed; restart the pipeline and ROI setup.')
        if self.maps is None:
            return frame
        return cv2.remap(frame, *self.maps, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)


def associate_faces(people, faces, cfg, history=None, frame_index=0, max_history_age=15,
                    *, diagnostics=None):
    """One-to-one, gated Hungarian matching. Ambiguous face columns are rejected.

    history maps a track ID to (frame_index, face center relative to person box).
    This is geometric association, not an identity guarantee.

    Optional ``diagnostics`` must be a dict. It is populated with JSON-safe
    face/person reasons, but can NEVER override matching or attach an unmatched
    face to a person. The default path preserves the original matching results.
    """
    if diagnostics is not None and not isinstance(diagnostics, dict):
        raise TypeError('diagnostics must be a dict or None.')
    if not people or not faces:
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update({
                'schema_version': 1,
                'notes': 'Observations only. Unmatched detections are NOT identities.',
                'faces': [
                    {'face_index': int(j), 'person_index': None, 'track_id': None,
                     'reason': 'no_people_available', 'geometric_candidates': 0,
                     'accepted_candidates': 0, 'gate_counts': {}}
                    for j in range(len(faces))
                ],
                'people': [
                    {'person_index': int(i), 'track_id': int(person.track_id),
                     'face_index': None, 'reason': 'no_faces_detected',
                     'geometric_candidates': 0, 'accepted_candidates': 0,
                     'gate_counts': {}}
                    for i, person in enumerate(people)
                ],
                'summary': {'matched': 0, 'unmatched_faces': len(faces),
                            'unmatched_people': len(people), 'ambiguous_faces': 0},
            })
        return []
    history = history or {}
    invalid = 1e6
    cost = np.full((len(people), len(faces)), invalid, dtype=float)
    # Capture gate counts ONLY when diagnostics are explicitly requested.
    from collections import Counter
    face_gates = [Counter() for _ in faces] if diagnostics is not None else None
    person_gates = [Counter() for _ in people] if diagnostics is not None else None
    for i, person in enumerate(people):
        wh = np.maximum(person.box[2:] - person.box[:2], 1)
        pad = wh * cfg['person_padding']
        outer = np.r_[person.box[:2] - pad, person.box[2:] + pad]
        for j, face in enumerate(faces):
            center = (face.box[:2] + face.box[2:]) / 2
            if not np.all(center >= outer[:2]) or not np.all(center <= outer[2:]):
                if face_gates is not None:
                    face_gates[j]['center_outside_padded_person_box'] += 1
                    person_gates[i]['center_outside_padded_person_box'] += 1
                continue
            cov = coverage(face.box, outer)
            if cov < cfg['min_face_coverage']:
                if face_gates is not None:
                    face_gates[j]['insufficient_face_coverage'] += 1
                    person_gates[i]['insufficient_face_coverage'] += 1
                continue
            if np.max((face.box[2:] - face.box[:2]) / wh) > cfg['max_face_person_ratio']:
                if face_gates is not None:
                    face_gates[j]['face_person_ratio_exceeded'] += 1
                    person_gates[i]['face_person_ratio_exceeded'] += 1
                continue
            relative = (center - person.box[:2]) / wh
            # Soft prior only; no assumption that the head must be in the top third.
            distance = min(np.linalg.norm(relative - cfg['expected_head_xy']), 1.5) / 1.5
            value = 0.55 * (1 - cov) + 0.35 * distance + 0.10 * (1 - face.confidence)
            previous = history.get(person.track_id)
            if previous is not None and frame_index - previous[0] <= max_history_age:
                value += 0.20 * min(float(np.linalg.norm(relative - previous[1])), 1)
            cost[i, j] = value
    if diagnostics is not None:
        geometric = cost.copy()  # Preserve pre-ambiguity costs for explanation only.
    ambiguous = set()
    for j in range(len(faces)):
        choices = np.sort(cost[:, j][cost[:, j] < invalid])
        if len(choices) > 1 and choices[1] - choices[0] < cfg['ambiguity_margin']:
            cost[:, j] = invalid
            if diagnostics is not None:
                ambiguous.add(j)
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < cfg['max_cost']]
    if diagnostics is None:
        return pairs

    matched_by_face = {j: i for i, j in pairs}
    matched_by_person = {i: j for i, j in pairs}
    face_info = []
    person_info = []
    for j in range(len(faces)):
        geometric_count = int(np.count_nonzero(geometric[:, j] < invalid))
        accepted_count = int(np.count_nonzero(cost[:, j] < cfg['max_cost']))
        if j in matched_by_face:
            reason = 'matched'
        elif j in ambiguous:
            reason = 'ambiguous_competing_tracks'
        elif geometric_count == 0:
            reason = 'no_geometric_candidate'
        elif accepted_count == 0:
            reason = 'association_cost_over_threshold'
        else:
            reason = 'lost_one_to_one_assignment'
        i = matched_by_face.get(j)
        face_info.append({
            'face_index': int(j),
            'person_index': int(i) if i is not None else None,
            'track_id': int(people[i].track_id) if i is not None else None,
            'reason': reason,
            'geometric_candidates': geometric_count,
            'accepted_candidates': accepted_count,
            'gate_counts': dict(face_gates[j]),
        })
    for i, person in enumerate(people):
        geometric_count = int(np.count_nonzero(geometric[i, :] < invalid))
        accepted_count = int(np.count_nonzero(cost[i, :] < cfg['max_cost']))
        if i in matched_by_person:
            reason = 'matched'
        elif geometric_count == 0:
            reason = 'no_geometric_candidate'
        elif accepted_count == 0 and any(geometric[i, j] < invalid for j in ambiguous):
            reason = 'ambiguous_competing_tracks'
        elif accepted_count == 0:
            reason = 'association_cost_over_threshold'
        else:
            reason = 'lost_one_to_one_assignment'
        j = matched_by_person.get(i)
        person_info.append({
            'person_index': int(i), 'track_id': int(person.track_id),
            'face_index': int(j) if j is not None else None,
            'reason': reason,
            'geometric_candidates': geometric_count,
            'accepted_candidates': accepted_count,
            'gate_counts': dict(person_gates[i]),
        })
    diagnostics.clear()
    diagnostics.update({
        'schema_version': 1,
        'notes': 'Gate counts are per face/person PAIR; no inferred identity for an unmatched face.',
        'faces': face_info,
        'people': person_info,
        'summary': {'matched': len(pairs),
                    'unmatched_faces': len(faces) - len(pairs),
                    'unmatched_people': len(people) - len(pairs),
                    'ambiguous_faces': len(ambiguous)},
    })
    return pairs


def align_face(frame, landmarks, size=112):
    """Least-squares 2D similarity alignment; does not synthesize a frontal face."""
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=float) * (size / 112)
    source = np.asarray(landmarks, dtype=float)
    if source.shape != (5, 2) or not np.isfinite(source).all():
        return None
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    variance = float((x * x).sum() / 5)
    if variance < 1e-6:
        return None
    u, s, vt = np.linalg.svd(y.T @ x / 5)
    signs = np.array([1., np.sign(np.linalg.det(u @ vt))])
    rotation = u @ np.diag(signs) @ vt
    scale = float((s * signs).sum() / variance)
    matrix = np.column_stack((scale * rotation, target_mean - scale * rotation @ source_mean))
    error = float(np.sqrt(np.mean(np.sum((affine_points(source, matrix) - target) ** 2, axis=1))))
    if scale <= 0 or error > 0.15 * size:
        return None
    return cv2.warpAffine(frame, matrix, (size, size), flags=cv2.INTER_LINEAR)


def quality_metrics(frame, face, cfg, valid_mask=None):
    """Heuristic selection score. Thresholds need tuning on this camera."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_box(face.box, w, h)
    fw, fh = x2 - x1, y2 - y1
    if fw < cfg['min_face_px'] or fh < cfg['min_face_px']:
        return None, 'small'
    if face.confidence < cfg['min_det_score']:
        return None, 'confidence'
    if valid_mask is not None and valid_mask[y1:y2, x1:x2].mean() < 0.97:
        return None, 'dewarp_border'
    kps = face.landmarks
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        return None, 'landmarks'
    wh = np.array([fw, fh])
    if ((kps < [x1, y1] - wh * 0.15) | (kps > [x2, y2] + wh * 0.15)).any():
        return None, 'landmarks'
    eye_distance = float(np.linalg.norm(kps[0] - kps[1]))
    if eye_distance < cfg['min_eye_distance_px']:
        return None, 'eyes_small'
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (112, 112), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    brightness = float(gray.mean())
    if sharpness < cfg['min_sharpness']:
        return None, 'blur'
    if not cfg['min_brightness'] <= brightness <= cfg['max_brightness']:
        return None, 'exposure'
    a, b = np.linalg.norm(kps[2] - kps[0]), np.linalg.norm(kps[2] - kps[1])
    symmetry = float(min(a, b) / max(a, b, 1e-6))  # Soft cue, NOT a yaw/pitch estimator.
    score = (0.30 * face.confidence + 0.25 * min(np.sqrt(fw * fh) / 140., 1.)
             + 0.25 * min(np.log1p(sharpness) / np.log1p(500.), 1.)
             + 0.10 * max(0., 1. - abs(brightness - 128.) / 128.) + 0.10 * symmetry)
    return {'score': float(score), 'det_score': float(face.confidence),
            'width_px': int(fw), 'height_px': int(fh), 'eye_distance_px': eye_distance,
            'sharpness_112': sharpness, 'brightness': brightness, 'symmetry_hint': symmetry}, 'ok'


def face_geometry_diagnostics(face, frame_shape, quality_cfg):
    """Non-decision geometry for investigating rejected face evidence.

    Reports gaps to configured native-pixel face/eye thresholds, not ArcFace's
    112x112 aligned-image size. Negative deficits are clamped to zero. Does NOT
    imply that a face passing these dimensions meets blur/landmark requirements.
    """
    height, width = int(frame_shape[0]), int(frame_shape[1])
    x1, y1, x2, y2 = clip_box(face.box, width, height)
    fw, fh = int(x2 - x1), int(y2 - y1)
    native_size = min(fw, fh)
    high_size = float(quality_cfg['min_face_px'])
    high_eye = float(quality_cfg['min_eye_distance_px'])
    medium_cfg = quality_cfg.get('medium', {})
    medium_size = min(high_size, float(medium_cfg.get('min_face_px', 28)))
    medium_eye = min(high_eye, float(medium_cfg.get('min_eye_distance_px', 6)))
    landmarks = np.asarray(face.landmarks)
    eye_distance = None
    if landmarks.shape == (5, 2) and np.isfinite(landmarks).all():
        eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    return {
        'width_px': fw, 'height_px': fh, 'native_short_side_px': native_size,
        'det_score': float(face.confidence),
        'eye_distance_px': eye_distance,
        'high_size_shortfall_px': round(max(0., high_size - native_size), 3),
        'medium_size_shortfall_px': round(max(0., medium_size - native_size), 3),
        'high_eye_shortfall_px': (round(max(0., high_eye - eye_distance), 3)
                                  if eye_distance is not None else None),
        'medium_eye_shortfall_px': (round(max(0., medium_eye - eye_distance), 3)
                                    if eye_distance is not None else None),
        'within_high_size_gate': bool(fw >= high_size and fh >= high_size),
        'within_medium_size_gate': bool(fw >= medium_size and fh >= medium_size),
    }


def draw_regions(frame, polygon, window):
    layer = frame.copy()
    cv2.fillPoly(layer, [polygon], (0, 145, 35))
    frame = cv2.addWeighted(layer, 0.16, frame, 0.84, 0)
    cv2.polylines(frame, [polygon], True, (40, 255, 80), 2)
    x1, y1, x2, y2 = map(int, window)
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 190, 0), 2)
    cv2.putText(frame, 'GREEN: capture ROI | CYAN: tracking context', (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return frame


def recognition_quality_metrics(frame, face, cfg, valid_mask=None):
    """Classify face evidence as high, medium, or rejected without changing legacy gates.

    Returns (metrics, tier) for accepted faces, with tier 'high' or 'medium'.
    Rejected faces return (None, reason), preserving quality_metrics' reason strings.
    'High' means passing the existing heuristic, NOT verified identity quality.

    Optional cfg['medium'] overrides the initial relaxed thresholds. The
    geometrical landmark check and undistortion-border check are NEVER relaxed.
    Neither tier grants permission to confirm a person's identity by itself.
    """
    metrics, strict_reason = quality_metrics(frame, face, cfg, valid_mask)
    if metrics is not None:
        return {**metrics, 'quality_tier': 'high'}, 'high'

    # Never allow a weaker quality threshold to bypass an unsafe crop/landmark.
    if strict_reason in ('landmarks', 'dewarp_border'):
        return None, strict_reason

    # Tentative starting points for small overhead-camera faces, NOT calibrated
    # face-recognition thresholds. Tune with held-out, identity-labeled footage.
    medium_defaults = {
        'min_det_score': 0.45,
        'min_face_px': 28,
        'min_eye_distance_px': 6,
        'min_sharpness': 7.0,
        'min_brightness': 20.0,
        'max_brightness': 235.0,
    }
    user_medium = cfg.get('medium', {})
    if not isinstance(user_medium, dict):
        raise TypeError('quality.medium must be a mapping of threshold overrides.')
    unexpected = set(user_medium) - set(medium_defaults)
    if unexpected:
        raise ValueError(f'Unknown quality.medium settings: {sorted(unexpected)}')

    relaxed_cfg = dict(cfg)
    for name, default in medium_defaults.items():
        requested = float(user_medium.get(name, default))
        if not np.isfinite(requested):
            raise ValueError(f'quality.medium.{name} must be finite.')
        if name == 'max_brightness':
            relaxed_cfg[name] = max(float(cfg[name]), requested)
        else:
            relaxed_cfg[name] = min(float(cfg[name]), requested)

    # Do not use invalid/degenerate parameters to admit arbitrarily bad faces.
    if not (0 <= relaxed_cfg['min_det_score'] <= 1
            and relaxed_cfg['min_face_px'] >= 1
            and relaxed_cfg['min_eye_distance_px'] > 0
            and relaxed_cfg['min_sharpness'] >= 0
            and 0 <= relaxed_cfg['min_brightness'] <= relaxed_cfg['max_brightness'] <= 255):
        raise ValueError('Invalid medium-quality thresholds.')

    medium_metrics, medium_reason = quality_metrics(frame, face, relaxed_cfg, valid_mask)
    if medium_metrics is None:
        return None, medium_reason
    return {**medium_metrics, 'quality_tier': 'medium',
            'strict_rejection_reason': strict_reason}, 'medium'
