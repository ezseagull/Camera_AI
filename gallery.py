"""Bounded top-K face storage; each run and tracker ID has a separate directory."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from vision import align_face, expanded_box


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_image(path, image):
    # Supports Unicode paths on Windows too.
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise IOError(f'Could not encode image: {path}')
    encoded.tofile(str(path))


class Gallery:
    def __init__(self, root, cfg):
        self.root, self.cfg = Path(root), cfg
        self.root.mkdir(parents=True, exist_ok=True)
        self.tracks = {}

    def observe(self, track_id, frame_index, timestamp):
        if track_id not in self.tracks:
            self.tracks[track_id] = {
                'track_id': track_id, 'first_frame': frame_index, 'last_frame': frame_index,
                'first_time_s': timestamp, 'last_time_s': timestamp,
                'observations': 0, 'samples': [], 'accepted_updates': 0,
                'coordinate_space': 'full_processed_frame',
            }
        track = self.tracks[track_id]
        track.update(last_frame=frame_index, last_time_s=timestamp)
        track['observations'] += 1
        return track

    def offer(self, person, face, metrics, frame, frame_index, timestamp):
        track = self.tracks[person.track_id]
        if track['observations'] < self.cfg['min_track_observations']:
            return False
        samples = track['samples']
        near = [s for s in samples if abs(s['timestamp_s'] - timestamp) < self.cfg['min_sample_gap_s']]
        # Within a time neighborhood retain only the better sample.
        if near and max(s['quality']['score'] for s in near) >= metrics['score']:
            return False
        evict = list(near)
        rest = [s for s in samples if s not in evict]
        if len(rest) >= self.cfg['top_k']:
            worst = min(rest, key=lambda s: s['quality']['score'])
            if worst['quality']['score'] >= metrics['score']:
                return False
            evict.append(worst)
        directory = self.root / f'track_{person.track_id:06d}'
        directory.mkdir(exist_ok=True)
        prefix = f'frame_{frame_index:09d}'
        crop_name = prefix + '_crop.jpg'
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = expanded_box(face.box, self.cfg['crop_padding'], w, h)
        write_image(directory / crop_name, frame[y1:y2, x1:x2])
        aligned_name = None
        if self.cfg['save_aligned']:
            aligned = align_face(frame, face.landmarks, self.cfg['aligned_size'])
            if aligned is not None:
                aligned_name = prefix + '_aligned.jpg'
                write_image(directory / aligned_name, aligned)
        sample = {
            'frame_index': frame_index, 'timestamp_s': timestamp,
            'person_box': person.box.tolist(), 'face_box': face.box.tolist(),
            'landmarks_5': face.landmarks.tolist(), 'quality': metrics,
            'detection_rotation_deg': face.rotation,
            'crop_file': crop_name, 'aligned_file': aligned_name,
        }
        track['samples'] = sorted([s for s in samples if s not in evict] + [sample],
                                  key=lambda s: s['quality']['score'], reverse=True)
        track['accepted_updates'] += 1
        # Write metadata before deleting old files, so an interruption loses no referenced image.
        write_json(directory / 'meta.json', track)
        for old in evict:
            for key in ('crop_file', 'aligned_file'):
                if old.get(key):
                    (directory / old[key]).unlink(missing_ok=True)
        return True

    def finalize(self, track_id, reason):
        track = self.tracks.pop(track_id)
        track['finalized_reason'] = reason
        if track['samples'] or any(track.get('recognition_samples', {}).values()):
            write_json(self.root / f'track_{track_id:06d}' / 'meta.json', track)
        with (self.root.parent / 'tracks.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(track, ensure_ascii=False, allow_nan=False) + '\n')

    def expire(self, frame_index, lost_frames):
        expired = [tid for tid, t in self.tracks.items() if frame_index - t['last_frame'] > lost_frames + 2]
        for tid in expired:
            self.finalize(tid, 'lost_timeout')
        return expired

    def close(self, reason='end_of_video'):
        for tid in list(self.tracks):
            self.finalize(tid, reason)

    def offer_recognition(self, person, face, metrics, frame, frame_index, timestamp, tier=None):
        """Keep additional high/medium face evidence without changing legacy top-K.

        Call after observe() and after recognition_quality_metrics() returned
        'high' or 'medium'. This is IMAGE selection, not identity confirmation.
        Samples are stored separately from offer() so legacy eviction cannot
        remove recognition evidence. Embeddings belong in IdentityManager.

        Optional gallery config keys (defaults): recognition_high_k=5,
        recognition_medium_k=10, recognition_min_sample_gap_s=min_sample_gap_s.
        A high-quality sample takes precedence over nearby medium samples.
        """
        tier = tier or metrics.get('quality_tier')
        if tier not in ('high', 'medium'):
            raise ValueError("Recognition tier must be 'high' or 'medium'.")
        if person.track_id not in self.tracks:
            raise KeyError('Call observe() before offer_recognition().')
        track = self.tracks[person.track_id]
        # Image archiving is independent of the IdentityManager evidence bank.
        # Preserve the old gate by default; optionally archive early high/medium
        # faces for reviewing identity decisions made in the first track frames.
        min_observations = self.cfg.get(
            'recognition_min_track_observations', self.cfg['min_track_observations'])
        if type(min_observations) is not int or min_observations < 1:
            raise ValueError('recognition_min_track_observations must be a positive integer.')
        if track['observations'] < min_observations:
            return False

        import math
        score = float(metrics['score'])
        timestamp = float(timestamp)
        if not (math.isfinite(score) and 0 <= score <= 1 and math.isfinite(timestamp)):
            raise ValueError('Recognition quality score and timestamp must be finite.')
        high_k = int(self.cfg.get('recognition_high_k', 5))
        medium_k = int(self.cfg.get('recognition_medium_k', 10))
        gap = float(self.cfg.get('recognition_min_sample_gap_s', self.cfg['min_sample_gap_s']))
        if high_k < 0 or medium_k < 0 or not math.isfinite(gap) or gap < 0:
            raise ValueError('Recognition quotas/gap must be nonnegative and finite.')
        limit = high_k if tier == 'high' else medium_k
        if limit == 0:
            return False

        bank = track.get('recognition_samples', {'high': [], 'medium': []})
        samples = bank[tier]
        if any(s['frame_index'] == frame_index for s in samples):
            return False  # Never overwrite an image already referenced by metadata.
        if tier == 'medium' and any(
                abs(s['timestamp_s'] - timestamp) < gap for s in bank['high']):
            return False  # Prefer a higher-quality observation at this moment.

        near = [s for s in samples if abs(s['timestamp_s'] - timestamp) < gap]
        if near and max(s['quality']['score'] for s in near) >= score:
            return False
        evict = list(near)
        rest = [s for s in samples if s not in evict]
        if len(rest) >= limit:
            worst = min(rest, key=lambda s: s['quality']['score'])
            if worst['quality']['score'] >= score:
                return False
            evict.append(worst)
            rest = [s for s in rest if s is not worst]
        medium_evict = ([s for s in bank['medium']
                         if abs(s['timestamp_s'] - timestamp) < gap]
                        if tier == 'high' else [])

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = expanded_box(face.box, self.cfg['crop_padding'], w, h)
        if x2 <= x1 or y2 <= y1:
            return False
        # Validate alignment before creating files. Never generate/restore faces.
        aligned = (align_face(frame, face.landmarks, self.cfg['aligned_size'])
                   if self.cfg['save_aligned'] else None)
        if self.cfg['save_aligned'] and aligned is None:
            return False

        track_dir = self.root / f'track_{person.track_id:06d}'
        directory = track_dir / 'recognition' / tier
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f'frame_{frame_index:09d}'
        crop_name = f'recognition/{tier}/{prefix}_crop.jpg'
        aligned_name = f'recognition/{tier}/{prefix}_aligned.jpg' if aligned is not None else None
        # The crop is taken from the unannotated processed frame, just as in offer().
        write_image(track_dir / crop_name, frame[y1:y2, x1:x2])
        if aligned_name:
            write_image(track_dir / aligned_name, aligned)
        sample = {
            'frame_index': frame_index, 'timestamp_s': timestamp,
            'person_box': person.box.tolist(), 'face_box': face.box.tolist(),
            'landmarks_5': face.landmarks.tolist(), 'quality': metrics,
            'quality_tier': tier, 'detection_rotation_deg': face.rotation,
            'crop_file': crop_name, 'aligned_file': aligned_name,
        }
        if 'recognition_samples' not in track:
            track['recognition_samples'] = bank
            track['recognition_accepted_updates'] = {'high': 0, 'medium': 0}
        bank[tier] = sorted(rest + [sample],
                            key=lambda s: s['quality']['score'], reverse=True)
        if medium_evict:
            bank['medium'] = [s for s in bank['medium'] if s not in medium_evict]
        track['recognition_accepted_updates'][tier] += 1
        # Commit metadata BEFORE deleting evicted images; keep reference integrity.
        write_json(track_dir / 'meta.json', track)
        for old in evict + medium_evict:
            for key in ('crop_file', 'aligned_file'):
                if old.get(key):
                    (track_dir / old[key]).unlink(missing_ok=True)
        return True
