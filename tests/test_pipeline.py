from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from backends import FaceDetector
from gallery import Gallery
from run import load_roi, process_video
from vision import (Face, Person, Rectifier, affine_points, align_face, associate_faces,
                    inside_polygon, nms_faces, polygon_pixels, quality_metrics,
                    rotate_expanded, tracking_window)


@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text())


def make_face(box=(40, 30, 90, 90), confidence=.95):
    b = np.array(box, dtype=float)
    wh = b[2:] - b[:2]
    kps = np.array([[.30, .35], [.70, .35], [.50, .55], [.35, .78], [.65, .78]]) * wh + b[:2]
    return Face(b, kps, confidence)


def test_roi_scale_and_tracking_context():
    poly = polygon_pixels([[.25, .25], [.75, .25], [.75, .75], [.25, .75]], 401, 201)
    assert inside_polygon((100, 50), poly)  # boundary counts as inside
    assert not inside_polygon((10, 10), poly)
    assert np.array_equal(poly[0], [100, 50])
    window = tracking_window(poly, 401, 201, .1)
    assert window[0] < 100 and window[2] > 300


@pytest.mark.parametrize('points', [
    [[0, 0], [1, 1], [0, 1], [1, 0]],
    [[0, 0], [1, 0]], [[0, 0], [1.1, 0], [0, 1]],
    [[0, 0], [0, 0], [1, 1]],
])
def test_invalid_roi_rejected(points):
    with pytest.raises(ValueError):
        polygon_pixels(points, 400, 300)


def test_face_association_distinct_and_ambiguous(cfg):
    a = Person(1, np.array([10, 10, 110, 220.]), .9)
    b = Person(2, np.array([130, 10, 230, 220.]), .9)
    faces = [make_face(), make_face((160, 30, 210, 90))]
    assert associate_faces([a, b], faces, cfg['association']) == [(0, 0), (1, 1)]
    duplicate = Person(3, a.box.copy(), .9)
    assert associate_faces([a, duplicate], faces[:1], cfg['association']) == []
    assert associate_faces([a], faces[1:], cfg['association']) == []


def test_global_nms_prevents_duplicate_face_assignment():
    face = make_face()
    duplicate = make_face((41, 31, 91, 91), .8)
    assert nms_faces([duplicate, face]) == [face]


@pytest.mark.parametrize('angle', [-90, -30, 30, 90])
def test_rotated_detection_is_mapped_back_to_full_frame(angle):
    crop = np.zeros((150, 130, 3), np.uint8)
    face = make_face()
    rotated, inverse = rotate_expanded(crop, angle)
    forward = cv2.invertAffineTransform(inverse)
    kps = affine_points(face.landmarks, forward)
    # Model returns an axis-aligned bounding box in the rotated view.
    original = np.array([[40, 30], [90, 30], [90, 90], [40, 90.]])
    mapped = affine_points(original, forward)
    row = np.r_[mapped.min(0), mapped.max(0), .95][None]
    detector = FaceDetector.__new__(FaceDetector)
    detector.detector = SimpleNamespace(detect=lambda image, max_num=0: (row, kps[None]))
    results = detector._detect_crop(crop, np.array([200, 100]), angle)
    assert len(results) == 1
    np.testing.assert_allclose(results[0].landmarks, face.landmarks + [200, 100], atol=1e-8)


def test_alignment_similarity_and_degenerate_landmarks():
    template = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]])
    frame = np.zeros((250, 250, 3), np.uint8)
    source = template * 1.5 + [10, 5]
    for x, y in source:
        cv2.circle(frame, (round(x), round(y)), 4, (255, 255, 255), -1)
    aligned = align_face(frame, source)
    assert aligned.shape == (112, 112, 3)
    for x, y in template:
        assert aligned[round(y), round(x)].mean() > 200
    assert align_face(frame, np.ones((5, 2))) is None


def test_real_calibration_map_and_roi_change_guard(tmp_path, cfg):
    path = tmp_path / 'camera_calibration.npz'
    np.savez(path, K=np.array([[300., 0, 160], [0, 300, 120], [0, 0, 1]]),
             D=np.zeros(5), image_size=np.array([320, 240]), model=np.array('pinhole'))
    c = {**cfg['undistort'], 'mode': 'pinhole'}
    rectifier = Rectifier(c, (640, 480), tmp_path)
    assert rectifier.apply(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)
    assert rectifier.valid_mask.mean() > .95
    with pytest.raises(ValueError, match='aspect ratio'):
        Rectifier(c, (640, 360), tmp_path)
    with pytest.raises(ValueError, match='Calibration model'):
        Rectifier({**c, 'mode': 'fisheye'}, (320, 240), tmp_path)
    (tmp_path / 'roi.json').write_text(json.dumps({'points': cfg['roi']['points'], 'rectification_signature': 'stale'}))
    with pytest.raises(ValueError, match='Lens settings'):
        load_roi(cfg, tmp_path, rectifier, (640, 480))


def test_fisheye_map(tmp_path, cfg):
    np.savez(tmp_path / 'camera_calibration.npz', K=np.array([[200., 0, 160], [0, 200, 120], [0, 0, 1]]),
             D=np.array([-.02, .001, 0, 0]), image_size=np.array([320, 240]), model=np.array('fisheye'))
    rectifier = Rectifier({**cfg['undistort'], 'mode': 'fisheye'}, (320, 240), tmp_path)
    assert np.isfinite(rectifier.maps[0]).all()
    assert rectifier.apply(np.zeros((240, 320, 3), np.uint8)).shape == (240, 320, 3)


def test_quality_rejects_native_small_blank_and_bad_border(cfg):
    image = np.random.default_rng(42).integers(30, 220, (240, 320, 3), dtype=np.uint8)
    metrics, reason = quality_metrics(image, make_face(), cfg['quality'])
    assert reason == 'ok' and metrics['score'] > 0
    assert quality_metrics(image, make_face((10, 10, 30, 30)), cfg['quality'])[1] == 'small'
    assert quality_metrics(np.zeros_like(image), make_face(), cfg['quality'])[1] == 'blur'
    assert quality_metrics(image, make_face(), cfg['quality'], np.zeros((240, 320), np.uint8))[1] == 'dewarp_border'


def test_gallery_temporal_diversity_top_k_and_empty_track(tmp_path, cfg):
    c = {**cfg['gallery'], 'top_k': 2, 'min_track_observations': 1}
    gallery = Gallery(tmp_path / 'faces', c)
    frame = np.full((240, 320, 3), 128, np.uint8)
    p, face = Person(7, np.array([10, 10, 110, 220.]), .9), make_face()
    for index, timestamp, score in [(0, 0., .4), (1, .1, .5), (10, 1., .6), (20, 2., .7)]:
        gallery.observe(7, index, timestamp)
        assert gallery.offer(p, face, {'score': score}, frame, index, timestamp)
    assert [s['quality']['score'] for s in gallery.tracks[7]['samples']] == [.7, .6]
    assert len(list((tmp_path / 'faces' / 'track_000007').glob('*_crop.jpg'))) == 2
    gallery.observe(8, 20, 2.)
    assert set(gallery.expire(100, 10)) == {7, 8}
    summaries = [json.loads(line) for line in (tmp_path / 'tracks.jsonl').read_text().splitlines()]
    assert len(summaries) == 2 and summaries[1]['samples'] == []


def test_pipeline_outputs_clean_faces_updates_every_frame_and_respects_roi(tmp_path, cfg):
    cfg = deepcopy(cfg)
    cfg['roi']['points'] = [[.05, .01], [.4, .01], [.4, .95], [.05, .95]]
    cfg['gallery']['min_track_observations'] = 1
    # Textured synthetic input; detectors below are explicitly stubs, not accuracy tests.
    source = np.random.default_rng(8).integers(30, 220, (240, 320, 3), dtype=np.uint8)
    frames = [source.copy() for _ in range(20)]
    class Capture:
        def __init__(self):
            self.index = 1
        def read(self):
            if self.index >= len(frames):
                return False, None
            result = frames[self.index]
            self.index += 1
            return True, result
    class Tracker:
        lost_frames = 60
        def __init__(self):
            self.calls = 0
        def update(self, frame, window):
            self.calls += 1
            if 4 <= self.calls <= 6:
                return []
            return [Person(1, np.array([10, 10, 110, 220.]), .9),
                    Person(2, np.array([140, 10, 240, 220.]), .9)]
    class Detector:
        def detect(self, frame, people, index):
            return [make_face(), make_face((170, 30, 220, 90))] if people else []
    args = SimpleNamespace(source='synthetic', device='cpu', face_device='cpu', headless=True, max_frames=0)
    rectifier = Rectifier(cfg['undistort'], (320, 240), tmp_path)
    tracker = Tracker()
    meta = process_video(Capture(), frames[0], cfg, tmp_path, args, rectifier, 10, tmp_path,
                         tracker=tracker, face_detector=Detector())
    assert tracker.calls == 20 and meta['counters']['frames'] == 20
    assert meta['counters']['outside_roi'] > 0
    assert not (tmp_path / 'faces' / 'track_000002').exists()
    track_dir = tmp_path / 'faces' / 'track_000001'
    info = json.loads((track_dir / 'meta.json').read_text())
    assert 1 <= len(info['samples']) <= cfg['gallery']['top_k']
    sample = info['samples'][0]
    # The JPEG must be exactly encoded from the clean source, not annotated pixels.
    from vision import expanded_box
    x1, y1, x2, y2 = expanded_box(make_face().box, cfg['gallery']['crop_padding'], 320, 240)
    encoded = cv2.imencode('.jpg', source[y1:y2, x1:x2])[1]
    assert (track_dir / sample['crop_file']).read_bytes() == encoded.tobytes()
    video = cv2.VideoCapture(str(tmp_path / 'annotated.mp4'))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 20
    ok, image = video.read()
    video.release()
    assert ok and image.shape == source.shape
