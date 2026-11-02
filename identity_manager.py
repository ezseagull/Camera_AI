"""Conservative, bounded multi-frame identity decisions for Camera AI.

Compatibility (no changes to legacy files required):
- FaceDatabase.search(embedding, top_k=..., score_mode=..., template_top_k=...)
- FaceRecognizer.get_embedding(frame, face) on original, unannotated BGR frame
- recognition_quality_metrics(...): (metrics, 'high'|'medium')
- Gallery.offer_recognition(...) remains an independent IMAGE archive.

This module stores ephemeral embeddings in RAM; results contain no embeddings.
Track IDs are temporary and MUST NOT be treated as proof of identity. Explicit,
held-out-camera-calibrated ``min_match_score`` and ``min_margin`` are required
before returning a confirmed Person ID. Similarities are NOT probabilities.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

import numpy as np

from face_database import FaceDatabase


@dataclass
class _Evidence:
    frame_index: int
    timestamp_s: float
    tier: str
    quality_score: float
    embedding: np.ndarray
    scores: dict[str, float]
    top_person_id: str | None
    top_score: float | None
    top_margin: float | None


@dataclass
class _TrackState:
    track_id: int
    high: list[_Evidence] = field(default_factory=list)
    medium: list[_Evidence] = field(default_factory=list)
    last_frame_index: int | None = None
    last_timestamp_s: float | None = None
    status: str = 'pending'
    reason: str = 'collecting_evidence'
    person_id: str | None = None
    candidate_person_id: str | None = None
    match_score: float | None = None
    second_match_score: float | None = None
    score_margin: float | None = None
    vote_fraction: float | None = None
    strong_support: int = 0
    conflict_reason: str | None = None
    ever_confirmed_id: str | None = None


class IdentityManager:
    """Accumulate face evidence by *temporary* tracker ID, and reject uncertainty.

    ``cfg`` is the identity decision subsection of config.yaml (added later),
    NOT the complete recognition detector configuration. Example keys:

      min_match_score: null         # MUST be calibrated, cosine in [-1,1]
      min_margin: null              # MUST be calibrated, top-1 minus top-2
      min_evidence: 3               # relatively separated observations
      min_high_evidence: 1
      min_strong_support: 2         # high-scoring, separated frame votes
      min_vote_fraction: 0.75       # preliminary consistency rule, tunable
      min_span_s: 0.8
      min_sample_gap_s: 0.4
      high_k: 5
      medium_k: 10
      medium_weight: 0.5
      score_mode: centroid          # FaceDatabase.search mode
      template_top_k: 3
      allow_single_identity: false  # no top-2 margin possible with one ID

    NOTE: None defaults deliberately DISABLE confirmation, not matching.
    Thresholds must be selected on held-out video with known and unknown
    people and evaluated at the actual database size. No numerical threshold
    in this module guarantees a particular identification error rate.

    ``update_embedding`` allows direct, already-verified embeddings.
    Prefer ``observe_face`` in the video pipeline: it verifies that the
    ArcFace ONNX model matches the enrollment database before inference.
    """

    def __init__(self, database: FaceDatabase, cfg=None):
        if not isinstance(database, FaceDatabase):
            raise TypeError('database must be a FaceDatabase instance.')
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise TypeError('IdentityManager cfg must be a mapping.')
        self.database = database
        self.cfg = dict(cfg)
        self.min_match_score = self._optional_float('min_match_score', -1.0, 1.0)
        self.min_margin = self._optional_float('min_margin', 0.0, 2.0)
        if (self.min_match_score is None) != (self.min_margin is None):
            raise ValueError('Set BOTH min_match_score and min_margin using held-out calibration, or neither.')
        self.thresholds_configured = self.min_match_score is not None
        self.min_evidence = self._positive_int('min_evidence', 3)
        self.min_high_evidence = self._nonnegative_int('min_high_evidence', 1)
        self.min_strong_support = self._positive_int('min_strong_support', 2)
        self.high_k = self._nonnegative_int('high_k', 5)
        self.medium_k = self._nonnegative_int('medium_k', 10)
        if self.high_k + self.medium_k == 0:
            raise ValueError('At least one evidence bank must be enabled.')
        if self.min_evidence > self.high_k + self.medium_k:
            raise ValueError('min_evidence exceeds total evidence capacity.')
        if self.min_high_evidence > self.high_k:
            raise ValueError('min_high_evidence exceeds high_k.')
        if self.min_strong_support > self.high_k + self.medium_k:
            raise ValueError('min_strong_support exceeds total evidence capacity.')
        self.min_span_s = self._float('min_span_s', 0.8, 0.0, math.inf)
        self.min_sample_gap_s = self._float('min_sample_gap_s', 0.4, 0.0, math.inf)
        self.medium_weight = self._float('medium_weight', 0.5, 0.0, 1.0)
        if self.medium_weight == 0:
            raise ValueError('medium_weight must be positive; disable medium with medium_k=0 instead.')
        self.min_vote_fraction = self._float('min_vote_fraction', 0.75, 0.0, 1.0)
        self.allow_single_identity = self.cfg.get('allow_single_identity', False)
        if type(self.allow_single_identity) is not bool:
            raise ValueError('allow_single_identity must be a boolean.')
        self.score_mode = self.cfg.get('score_mode', 'centroid')
        if self.score_mode not in ('centroid', 'template_topk', 'max_template'):
            raise ValueError('Unknown FaceDatabase score_mode.')
        self.template_top_k = self._positive_int('template_top_k', 3)
        # Experimental bank selection. 'quality' is EXACTLY the legacy default;
        # temporal_diverse must be enabled explicitly for a controlled A/B test.
        self.evidence_selection = self.cfg.get('evidence_selection', 'quality')
        if self.evidence_selection not in ('quality', 'temporal_diverse'):
            raise ValueError('evidence_selection must be quality or temporal_diverse.')
        self.diversity_min_quality_ratio = self._float(
            'diversity_min_quality_ratio', 0.8, 0.0, 1.0)
        self.diversity_min_utility_gain = self._float(
            'diversity_min_utility_gain', 0.01, 0.0, 1.0)
        self.diagnostic_max_events = self._positive_int('diagnostic_max_events', 64)
        self._tracks: dict[int, _TrackState] = {}
        # Counters and bounded JSON-safe event histories; NEVER store raw frames
        # or extra embeddings here. Released when a temporary track finalizes.
        self._evidence_logs: dict[int, dict] = {}
        self._verified_recognizer_key = None

    def _positive_int(self, key, default):
        value = self.cfg.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer.')
        return value

    def _nonnegative_int(self, key, default):
        value = self.cfg.get(key, default)
        if type(value) is not int or value < 0:
            raise ValueError(f'{key} must be a nonnegative integer.')
        return value

    def _float(self, key, default, low, high):
        try:
            value = float(self.cfg.get(key, default))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{key} must be finite and in [{low}, {high}].') from exc
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be finite and in [{low}, {high}].')
        return value

    def _optional_float(self, key, low, high):
        if self.cfg.get(key) is None:
            return None
        return self._float(key, None, low, high)

    @staticmethod
    def _track_id(track_id):
        if isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer)) or track_id < 0:
            raise ValueError('track_id must be a nonnegative integer.')
        return int(track_id)

    @staticmethod
    def _position(frame_index, timestamp):
        if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)) or frame_index < 0:
            raise ValueError('frame_index must be a nonnegative integer.')
        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('timestamp must be a nonnegative finite number.') from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('timestamp must be a nonnegative finite number.')
        return int(frame_index), timestamp

    def _record_evidence_action(self, track_id, frame_index, timestamp, tier,
                                action, reason, quality=None, *,
                                removed_frames=(), top_person_id=None,
                                top_score=None, top_margin=None):
        """Bounded audit record; no raw image, crop, embedding or probability."""
        log = self._evidence_logs.get(track_id)
        if log is None:
            log = {'counts': Counter(), 'events': deque(maxlen=self.diagnostic_max_events),
                   'last_action': None, 'last_reason': None}
            self._evidence_logs[track_id] = log
        log['counts'][reason] += 1
        log['counts']['action_' + action] += 1
        state = self._tracks.get(track_id)
        event = {'frame_index': int(frame_index), 'timestamp_s': float(timestamp),
                 'tier': tier, 'action': action, 'reason': reason,
                 'quality_score': float(quality) if quality is not None else None,
                 'removed_frames': [int(x) for x in removed_frames],
                 'offered_top_person_id': top_person_id,
                 'offered_top_score': top_score, 'offered_top_margin': top_margin,
                 'identity_status': state.status if state is not None else 'pending',
                 'identity_reason': state.reason if state is not None else 'no_face_evidence',
                 'candidate_person_id': state.candidate_person_id if state is not None else None,
                 'fused_match_score': state.match_score if state is not None else None,
                 'fused_second_match_score': state.second_match_score if state is not None else None,
                 'fused_margin': state.score_margin if state is not None else None,
                 'vote_fraction': state.vote_fraction if state is not None else None,
                 'strong_support': state.strong_support if state is not None else 0,
                 'high_count': len(state.high) if state is not None else 0,
                 'medium_count': len(state.medium) if state is not None else 0}
        log['events'].append(event)
        log['last_action'], log['last_reason'] = action, reason

    @staticmethod
    def _bank_utility(samples):
        """A/B-only heuristic balancing quality, time spread and embedding novelty.

        This is NOT a recognition-confidence metric and is never used with
        evidence_selection='quality'. Embeddings stay in temporary RAM banks.
        """
        if not samples:
            return 0.0
        quality = float(np.mean([s.quality_score for s in samples]))
        if len(samples) == 1:
            return 0.70 * quality
        temporal, visual = [], []
        for i, sample in enumerate(samples):
            others = [item for j, item in enumerate(samples) if i != j]
            nearest_dt = min(abs(sample.timestamp_s - other.timestamp_s) for other in others)
            temporal.append(min(nearest_dt / 3.0, 1.0))
            nearest_distance = min(max(0.0, 1.0 - float(np.dot(sample.embedding, other.embedding)))
                                   for other in others)
            visual.append(min(nearest_distance / 0.3, 1.0))
        return 0.70 * quality + 0.15 * float(np.mean(temporal)) + 0.15 * float(np.mean(visual))

    def get_evidence_diagnostics(self, track_id):
        """JSON-safe bounded audit for one track; includes only retained sample metadata."""
        tid = self._track_id(track_id)
        log = self._evidence_logs.get(tid)
        state = self._tracks.get(tid)
        samples = [] if state is None else state.high + state.medium
        return {'selection_mode': self.evidence_selection,
                'counts': dict(log['counts']) if log else {},
                'last_action': log['last_action'] if log else None,
                'last_reason': log['last_reason'] if log else None,
                'recent_events': list(log['events']) if log else [],
                'events_truncated': bool(log and
                                         sum(v for k, v in log['counts'].items()
                                             if k.startswith('action_')) > len(log['events'])),
                'retained_samples': [
                    {'frame_index': e.frame_index, 'timestamp_s': e.timestamp_s,
                     'tier': e.tier, 'quality_score': e.quality_score,
                     'top_person_id': e.top_person_id,
                     'top_score': e.top_score, 'top_margin': e.top_margin,
                     'strong_vote': self._strong_vote(e)}
                    for e in sorted(samples, key=lambda item: item.timestamp_s)]}

    def verify_recognizer(self, recognizer):
        """Reject model/preprocessing mismatches with enrolled templates.

        Enrollment records the ONNX file SHA256. Hash it only once per model
        path per IdentityManager, not once per video frame. Without a stored
        digest, require the same ``model_id`` naming convention as enroll.py.
        """
        model_path = Path(recognizer.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f'Recognition model is missing: {model_path}')
        stat = model_path.stat()
        key = (str(model_path), stat.st_size, stat.st_mtime_ns)
        if key == self._verified_recognizer_key:
            return True
        cfg = getattr(recognizer, 'cfg', {})
        if not isinstance(cfg, dict):
            raise ValueError('Recognizer config must be a mapping.')
        model_id = cfg.get('model_id') or f"insightface/{cfg.get('model_pack', 'buffalo_l')}/{model_path.stem}"
        if model_id != self.database.model_id:
            raise ValueError('Recognizer model_id differs from FaceDatabase enrollment model_id.')
        if self.database.model_sha256:
            sha = hashlib.sha256()
            with model_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            if sha.hexdigest() != self.database.model_sha256:
                raise ValueError('Recognizer ONNX SHA256 differs from FaceDatabase enrollment SHA256.')
        self._verified_recognizer_key = key
        return True

    def observe_face(self, person, face, metrics, frame, frame_index, timestamp,
                     recognizer, *, tier=None, association_ok=True):
        """Extract an ArcFace embedding and incorporate a validated face.

        Call after associate_faces() selected an unambiguous person-face pair
        AND recognition_quality_metrics() accepted high/medium quality.
        Never use a Gallery JPEG for input if the original BGR frame exists.
        For suspected ID switches call mark_conflict() or reset_track(); do NOT
        pass a possibly reassigned face to the existing track without review.
        """
        track_id = self._track_id(person.track_id)
        if not association_ok or metrics is None:
            reason = 'association_rejected' if not association_ok else 'quality_unavailable'
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', reason)
            return self.get_result(track_id)
        self.verify_recognizer(recognizer)
        embedding = recognizer.get_embedding(frame, face)
        if embedding is None:
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'embedding_unavailable',
                                         quality=metrics.get('score'))
            return self.get_result(track_id)
        return self.update_embedding(track_id, embedding, metrics, frame_index,
                                     timestamp, tier=tier, association_ok=True)

    def update_embedding(self, track_id, embedding, metrics, frame_index, timestamp,
                         *, tier=None, association_ok=True):
        """Offer one embedding from a geometrically associated face.

        The face was previously verified to belong to the correct track;
        association_ok=False discards the sample. Does not save raw faces.
        DO NOT pass AdaFace vectors to an ArcFace-enrolled database.
        """
        track_id = self._track_id(track_id)
        frame_index, timestamp = self._position(frame_index, timestamp)
        if not association_ok or metrics is None:
            reason = 'association_rejected' if not association_ok else 'quality_unavailable'
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', reason)
            return self.get_result(track_id)
        if not isinstance(metrics, dict):
            raise TypeError('metrics must be the dict from recognition_quality_metrics().')
        tier = tier or metrics.get('quality_tier')
        if tier not in ('high', 'medium'):
            raise ValueError("Quality tier must be 'high' or 'medium'.")
        try:
            quality = float(metrics['score'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('metrics must contain a finite quality score in [0,1].') from exc
        if not math.isfinite(quality) or not 0 < quality <= 1:
            raise ValueError('Quality score must be finite and in (0,1].')
        if ((tier == 'high' and self.high_k == 0)
                or (tier == 'medium' and self.medium_k == 0)):
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'tier_bank_disabled', quality)
            return self.get_result(track_id)
        try:
            embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError('Embedding must be a numerical vector.') from exc
        if not embedding.size or not np.isfinite(embedding).all():
            raise ValueError('Embedding must be finite and nonempty.')
        if self.database.embedding_dim is not None and embedding.size != self.database.embedding_dim:
            raise ValueError(f'Embedding dimension mismatch: expected {self.database.embedding_dim}.')
        norm = float(np.linalg.norm(embedding.astype(np.float64)))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError('Embedding must have a nonzero finite norm.')
        embedding = np.ascontiguousarray(embedding / norm, dtype=np.float32)

        state = self._tracks.setdefault(track_id, _TrackState(track_id=track_id))
        if state.conflict_reason is not None:
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'conflict_latched', quality)
            return self.get_result(track_id)  # Conflict is latched until explicit reset.
        if state.last_frame_index is not None:
            if frame_index < state.last_frame_index or timestamp < state.last_timestamp_s:
                raise ValueError('Track evidence must arrive in chronological order.')
            if frame_index == state.last_frame_index:
                self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                             'rejected', 'duplicate_frame', quality)
                return self.get_result(track_id)  # Prevent repeat use of the same video frame.
        state.last_frame_index = frame_index
        state.last_timestamp_s = timestamp
        if len(self.database) == 0:
            state.status, state.reason = 'unknown', 'empty_database'
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'empty_database', quality)
            return self.get_result(track_id)

        bank = state.high if tier == 'high' else state.medium
        other_bank = state.medium if tier == 'high' else state.high
        if tier == 'medium' and any(abs(s.timestamp_s - timestamp) < self.min_sample_gap_s
                                    for s in state.high):
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'medium_shadowed_by_high', quality)
            return self.get_result(track_id)  # High observation takes priority.
        near = [s for s in bank if abs(s.timestamp_s - timestamp) < self.min_sample_gap_s]
        if near and max(s.quality_score for s in near) >= quality:
            self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                         'rejected', 'near_higher_quality', quality)
            return self.get_result(track_id)
        near_ids = {id(s) for s in near}
        keep = [s for s in bank if id(s) not in near_ids]
        removed = list(near)
        limit = self.high_k if tier == 'high' else self.medium_k
        selection_reason = 'replaced_near' if near else 'accepted_new'
        if len(keep) >= limit:
            # Default path is deliberately unchanged: quality-only evicts the
            # lowest quality sample when and only when the offer is better.
            if self.evidence_selection == 'quality' or limit == 1:
                worst = min(keep, key=lambda s: s.quality_score)
                if worst.quality_score >= quality:
                    self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                                 'rejected', 'bank_full_lower_quality', quality)
                    return self.get_result(track_id)
                keep = [s for s in keep if s is not worst]
                removed.append(worst)
                selection_reason = 'replaced_worst_quality'
            else:
                # Experimental: prospective sample has no identity scores yet.
                # Only accepted samples are searched against the database.
                probe = _Evidence(frame_index, timestamp, tier, quality, embedding,
                                  {}, None, None, None)
                worst_quality = min(s.quality_score for s in keep)
                if quality < worst_quality * self.diversity_min_quality_ratio:
                    self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                                 'rejected', 'diversity_quality_floor', quality)
                    return self.get_result(track_id)
                baseline = self._bank_utility(keep)
                best_gain, evict_index = self.diversity_min_utility_gain, None
                for index in range(len(keep)):
                    trial = keep[:index] + keep[index + 1:] + [probe]
                    gain = self._bank_utility(trial) - baseline
                    if gain > best_gain:
                        best_gain, evict_index = gain, index
                if evict_index is None:
                    self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                                 'rejected', 'diversity_no_utility_gain', quality)
                    return self.get_result(track_id)
                removed.append(keep[evict_index])
                keep = [s for i, s in enumerate(keep) if i != evict_index]
                selection_reason = 'replaced_temporal_diverse'

        candidates = self.database.search(embedding, top_k=len(self.database),
                                          score_mode=self.score_mode,
                                          template_top_k=self.template_top_k)
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        evidence = _Evidence(frame_index, timestamp, tier, quality, embedding,
                             {c['person_id']: float(c['score']) for c in candidates},
                             top['person_id'] if top else None,
                             float(top['score']) if top else None,
                             float(top['score'] - second['score']) if second else None)
        bank[:] = keep + [evidence]
        if tier == 'high':
            evicted_medium = [s for s in other_bank
                              if abs(s.timestamp_s - timestamp) < self.min_sample_gap_s]
            removed.extend(evicted_medium)
            other_bank[:] = [s for s in other_bank
                             if abs(s.timestamp_s - timestamp) >= self.min_sample_gap_s]
        self._recompute(state)
        self._record_evidence_action(track_id, frame_index, timestamp, tier,
                                     'replaced' if removed else 'accepted', selection_reason,
                                     quality, removed_frames=[s.frame_index for s in removed],
                                     top_person_id=evidence.top_person_id,
                                     top_score=evidence.top_score,
                                     top_margin=evidence.top_margin)
        return self.get_result(track_id)

    def _weight(self, evidence):
        return evidence.quality_score * (1.0 if evidence.tier == 'high' else self.medium_weight)

    def _strong_vote(self, evidence):
        if not self.thresholds_configured or evidence.top_person_id is None:
            return False
        if evidence.top_score < self.min_match_score:
            return False
        return (evidence.top_margin is not None and evidence.top_margin >= self.min_margin
                or evidence.top_margin is None and self.allow_single_identity
                and len(self.database) == 1)

    def _recompute(self, state):
        samples = state.high + state.medium
        state.person_id = None
        state.candidate_person_id = None
        state.match_score = None
        state.second_match_score = None
        state.score_margin = None
        state.vote_fraction = None
        state.strong_support = 0
        if not samples:
            state.status, state.reason = 'pending', 'collecting_evidence'
            return
        if state.conflict_reason is not None:
            state.status, state.reason = 'conflict', state.conflict_reason
            return
        # A strong contradiction across distinct times/frames is NOT resolved
        # by majority vote. Reject the track until human review or track reset.
        strong = [s for s in samples if self._strong_vote(s)]
        strong_ids = {s.top_person_id for s in strong}
        if len(strong_ids) > 1:
            state.conflict_reason = 'contradictory_strong_face_matches'
            state.status, state.reason = 'conflict', state.conflict_reason
            return

        weights = np.array([self._weight(s) for s in samples], dtype=np.float64)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            state.status, state.reason = 'pending', 'zero_evidence_weight'
            return
        # Every accepted sample is scored against the same enrolled identities;
        # do not average embeddings from potentially different people.
        person_ids = sorted({pid for s in samples for pid in s.scores})
        ranked = []
        for pid in person_ids:
            if any(pid not in s.scores for s in samples):
                continue
            fused = sum(self._weight(s) * s.scores[pid] for s in samples) / total_weight
            ranked.append((pid, float(fused)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if not ranked:
            state.status, state.reason = 'unknown', 'no_valid_candidate'
            return
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else None
        margin = float(best_score - second_score) if second_score is not None else None
        votes = float(sum(self._weight(s) for s in samples
                          if s.top_person_id == best_id)) / total_weight
        strong_support = sum(1 for s in strong if s.top_person_id == best_id)
        state.candidate_person_id = best_id
        state.match_score = best_score
        state.second_match_score = second_score
        state.score_margin = margin
        state.vote_fraction = votes
        state.strong_support = strong_support

        if not self.thresholds_configured:
            state.status, state.reason = 'pending', 'thresholds_not_configured'
            return

        ordered_times = [s.timestamp_s for s in samples]
        if (len(samples) < self.min_evidence or len(state.high) < self.min_high_evidence
                or max(ordered_times) - min(ordered_times) < self.min_span_s):
            state.status, state.reason = 'pending', 'insufficient_independent_evidence'
            return
        if best_score < self.min_match_score:
            state.status, state.reason = 'unknown', 'below_calibrated_match_threshold'
            return
        if margin is None and not (len(self.database) == 1 and self.allow_single_identity):
            state.status, state.reason = 'pending', 'cannot_validate_top2_margin'
            return
        if margin is not None and margin < self.min_margin:
            state.status, state.reason = 'pending', 'insufficient_top2_margin'
            return
        if votes < self.min_vote_fraction or strong_support < self.min_strong_support:
            state.status, state.reason = 'pending', 'insufficient_consensus'
            return
        if state.ever_confirmed_id is not None and state.ever_confirmed_id != best_id:
            state.conflict_reason = 'previously_confirmed_identity_changed'
            state.status, state.reason = 'conflict', state.conflict_reason
            return
        state.ever_confirmed_id = best_id
        state.person_id = best_id
        state.status, state.reason = 'confirmed', 'consensus_passed'

    def get_result(self, track_id):
        """JSON-safe identity summary; person_id is non-null ONLY if confirmed."""
        track_id = self._track_id(track_id)
        state = self._tracks.get(track_id)
        if state is None:
            return {'track_id': track_id, 'person_id': None, 'identity_status': 'pending',
                    'reason': 'no_face_evidence', 'candidate_person_id': None,
                    'match_score': None, 'second_match_score': None, 'score_margin': None,
                    'vote_fraction': None, 'strong_support': 0, 'evidence_count': 0,
                    'high_count': 0, 'medium_count': 0, 'last_frame_index': None,
                    'last_timestamp_s': None, 'thresholds_configured': self.thresholds_configured,
                    'score_mode': self.score_mode,
                    'evidence_last_action': self._evidence_logs.get(track_id, {}).get('last_action'),
                    'evidence_last_reason': self._evidence_logs.get(track_id, {}).get('last_reason'),
                    'evidence_action_counts': dict(self._evidence_logs.get(track_id, {}).get('counts', {}))}
        return {'track_id': state.track_id, 'person_id': state.person_id if state.status == 'confirmed' else None,
                'identity_status': state.status, 'reason': state.reason,
                'candidate_person_id': state.candidate_person_id,
                'match_score': state.match_score, 'second_match_score': state.second_match_score,
                'score_margin': state.score_margin, 'vote_fraction': state.vote_fraction,
                'strong_support': state.strong_support,
                'evidence_count': len(state.high) + len(state.medium),
                'high_count': len(state.high), 'medium_count': len(state.medium),
                'last_frame_index': state.last_frame_index,
                'last_timestamp_s': state.last_timestamp_s,
                'thresholds_configured': self.thresholds_configured,
                'score_mode': self.score_mode,
                'evidence_last_action': self._evidence_logs.get(track_id, {}).get('last_action'),
                'evidence_last_reason': self._evidence_logs.get(track_id, {}).get('last_reason'),
                'evidence_action_counts': dict(self._evidence_logs.get(track_id, {}).get('counts', {}))}

    def mark_conflict(self, track_id, reason='suspected_tracker_id_switch'):
        """Immediately revoke a track identity when evidence association is unsafe.

        A conflict is LATCHED: it cannot be cleared by the next good face.
        Use reset_track() only after the caller has established a *new* clean
        tracking segment; this module never guesses when that is safe.
        """
        track_id = self._track_id(track_id)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Conflict reason must be a nonempty string.')
        state = self._tracks.setdefault(track_id, _TrackState(track_id=track_id))
        state.conflict_reason = reason
        state.status, state.person_id, state.reason = 'conflict', None, reason
        return self.get_result(track_id)

    def check_active_conflicts(self, active_track_ids):
        """Revoke duplicates confirmed on DISTINCT concurrently active tracks.

        Call only if active_track_ids refer to DISTINCT visible people in the
        same video frame. A tracker can occasionally duplicate one person;
        this module intentionally does NOT perform spatial verification.
        Returns JSON-safe results of the tracks that were revoked.
        """
        by_person: dict[str, list[int]] = {}
        for raw_tid in set(active_track_ids):
            tid = self._track_id(raw_tid)
            state = self._tracks.get(tid)
            if state is not None and state.status == 'confirmed':
                by_person.setdefault(state.person_id, []).append(tid)
        changed = []
        for group in by_person.values():
            if len(group) > 1:
                for tid in sorted(group):
                    changed.append(self.mark_conflict(tid, 'same_identity_on_simultaneous_tracks'))
        return changed

    def reset_track(self, track_id):
        """Discard old evidence when a tracker ID is explicitly reused/reset."""
        tid = self._track_id(track_id)
        self._tracks.pop(tid, None)
        self._evidence_logs.pop(tid, None)

    def finalize_track(self, track_id):
        """Return last JSON-safe summary and release all in-memory embeddings."""
        track_id = self._track_id(track_id)
        summary = self.get_result(track_id)
        # Keep output schema consistent with Gallery's most recent track
        # summary. Detailed bounded events are available through
        # get_evidence_diagnostics() before finalize_track() if the caller
        # explicitly exports them (never copied into every video frame).
        self.reset_track(track_id)
        return summary

    def close(self):
        """Discard ALL temporary tracking evidence at end of one video run."""
        self._tracks.clear()
        self._evidence_logs.clear()
