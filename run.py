"""Camera AI sample: ROI -> person tracking -> face detection -> top-K per track."""
from __future__ import annotations

import argparse
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

from backends import FaceDetector, PersonTracker
from gallery import Gallery, write_image, write_json
from vision import (Rectifier, associate_faces, draw_regions, inside_polygon,
                    polygon_pixels, quality_metrics, tracking_window)


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
                write_json(path, {'points': normalized.tolist(), 'frame_size': [w, h],
                                  'rectification_signature': rectifier.signature})
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


def process_video(cap, first_frame, cfg, base_dir, args, rectifier, fps, output_dir,
                  tracker=None, face_detector=None):
    """Backends may be injected for deterministic integration tests."""
    h, w = first_frame.shape[:2]
    polygon = load_roi(cfg, base_dir, rectifier, (w, h))
    crop_window = tracking_window(polygon, w, h, cfg['roi']['context_margin_fraction'])
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
    write_json(output_dir / 'run.json', meta)
    counters, history, trails = Counter(), {}, {}
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
                trail = trails.setdefault(person.track_id, deque(maxlen=45))
                trail.append(tuple(np.rint((person.box[:2] + person.box[2:]) / 2).astype(int)))
            face_labels = []
            if frame_index % cfg['face']['every_n_frames'] == 0:
                faces = face_detector.detect(clean, people, frame_index)
                counters['detected_faces'] += len(faces)
                pairs = associate_faces(people, faces, cfg['association'], history, frame_index,
                                        max_history_age=max(1, round(0.5 * fps)))
                counters['unassigned_faces'] += len(faces) - len(pairs)
                for pi, fi in pairs:
                    person, face = people[pi], faces[fi]
                    center = (face.box[:2] + face.box[2:]) / 2
                    relative = (center - person.box[:2]) / np.maximum(person.box[2:] - person.box[:2], 1)
                    history[person.track_id] = (frame_index, relative)
                    if not inside_polygon(center, polygon):
                        counters['outside_roi'] += 1
                        continue
                    metrics, reason = quality_metrics(clean, face, cfg['quality'], rectifier.valid_mask)
                    counters['quality_' + reason] += 1
                    saved = False
                    if metrics is not None:
                        saved = gallery.offer(person, face, metrics, clean, frame_index, timestamp)
                        counters['gallery_updates'] += int(saved)
                    face_labels.append((face, person.track_id, reason, saved))
            # Annotate only after cropping/saving from the clean frame.
            canvas = draw_regions(clean.copy(), polygon, crop_window)
            for person in people:
                x1, y1, x2, y2 = person.box.astype(int)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (245, 180, 50), 2)
                n = len(gallery.tracks[person.track_id]['samples'])
                cv2.putText(canvas, f'Track {person.track_id} | faces {n}', (max(0, x1), max(50, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (245, 180, 50), 2)
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
                history.pop(tid, None)
                trails.pop(tid, None)
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
        gallery.close(finish_reason)
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
            window = tracking_window(polygon, first.shape[1], first.shape[0], cfg['roi']['context_margin_fraction'])
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
