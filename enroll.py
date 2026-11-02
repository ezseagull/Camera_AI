"""Read-only face enrollment for Camera AI (one folder = one known person).

Example::

    python enroll.py --dataset faces_data --config config.yaml \
        --output data/face_database.npz

The dataset MUST be extracted first. Images are NEVER modified: this script
stores only ArcFace embeddings and a JSON audit report. It supports different
numbers of photographs per person and filename extensions such as .JPG.

Only trustworthy single-face images are enrolled. High-quality images are the
default; --include-medium opts in to less reliable enrollment images. All
rejection reasons and pairwise intra-identity similarities are reported.
Enrollment does not determine recognition thresholds or CONFIRM identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import yaml

from face_database import FaceDatabase, discover_identity_images
from vision import nms_faces, recognition_quality_metrics


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_image(path):
    """Load BGR without mangling Unicode filenames on Windows."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
    except (OSError, ValueError, cv2.error) as exc:
        raise ValueError(f'Cannot decode image: {type(exc).__name__}: {exc}') from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Cannot decode image as a 3-channel BGR photograph.')
    return image


def _write_json_atomic(path, data):
    """Create an owner-only report on POSIX; no arbitrary image writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(data, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def detect_enrollment_faces(detector, image, *, rotation_angles=(-30, 30)):
    """Reuse FaceDetector's SCRFD detection/rotation geometry on the full photo.

    This intentionally calls the existing FaceDetector._detect_crop helper: it
    produces Face objects with full-image landmarks and avoids reimplementing
    detection, affine coordinate transforms and rotation-padding checks.
    """
    offset = np.array([0., 0.], dtype=np.float64)
    faces = detector._detect_crop(image, offset, 0)
    # Rotation is only a fallback for failed detection, NOT pitch correction.
    if not faces:
        for angle in rotation_angles:
            if float(angle):
                faces.extend(detector._detect_crop(image, offset, float(angle)))
    return nms_faces(faces, 0.4)


def _intra_identity_stats(embeddings):
    """Diagnostic only; low similarity is not necessarily a wrong person."""
    if len(embeddings) < 2:
        return {'min_pairwise_cosine': None, 'mean_pairwise_cosine': None}
    rows = np.asarray(embeddings, dtype=np.float32)
    similarities = rows @ rows.T
    values = similarities[np.triu_indices(len(rows), k=1)]
    return {'min_pairwise_cosine': float(values.min()),
            'mean_pairwise_cosine': float(values.mean())}


def _cross_identity_diagnostics(database, limit=10):
    """Most similar templates from DIFFERENT folders, for human review.

    These are not automatic duplicate/identity decisions: lookalikes exist and
    score cutoffs need calibration on the actual enrollment population.
    """
    people = list(database.counts())
    pairs = []
    for i, left in enumerate(people):
        lv = database.get_templates(left)
        for right in people[i + 1:]:
            rv = database.get_templates(right)
            similarity = lv @ rv.T
            a, b = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
            pairs.append({'person_a': left, 'person_b': right,
                          'max_template_cosine': float(similarity[a, b]),
                          'source_a': database.get_metadata(left)['sources'][int(a)],
                          'source_b': database.get_metadata(right)['sources'][int(b)]})
    pairs.sort(key=lambda x: -x['max_template_cosine'])
    return pairs[:limit]


def _duplicate_hashes_between_people(groups):
    """Find byte-identical photos labeled as two *different* people.

    Such contradictory labels must not be inserted automatically. A failed
    hash/read is handled later by the image decoder/report.
    """
    identities_by_hash = {}
    hashes_by_path = {}
    for person_id, images in groups.items():
        for path in images:
            try:
                digest = _sha256_file(path)
            except OSError:
                continue
            hashes_by_path[path] = digest
            identities_by_hash.setdefault(digest, set()).add(person_id)
    conflicting = {h for h, owners in identities_by_hash.items() if len(owners) > 1}
    return hashes_by_path, conflicting


def enroll_dataset(dataset_root, database_path, *, detector, recognizer, quality_cfg,
                   face_cfg=None, report_path=None, include_medium=False,
                   overwrite=False, review_cosine=None, require_all_identities=False,
                   model_id=None, rotation_angles=None,
                   enrollment_min_face_px=64, enrollment_min_eye_distance_px=12):
    """Enroll real photos using already initialized detector and recognizer.

    Dependency injection makes this usable with real SCRFD/ArcFace or with
    lightweight unit-test stubs. Returns a JSON-serializable audit report.

    Neither file count nor pose is inferred from the filename. Persons with
    zero accepted embeddings are NEVER registered. An optional review cosine
    threshold can hold an ENTIRE inconsistent folder out for manual review;
    no default threshold is guessed from unrelated datasets.
    """
    root = Path(dataset_root).expanduser().resolve()
    output = Path(database_path).expanduser().resolve()
    if output.suffix.lower() != '.npz':
        raise ValueError('database_path must end with .npz.')
    report_file = (Path(report_path).expanduser().resolve() if report_path is not None
                   else output.with_suffix('.report.json'))
    if report_file == output:
        raise ValueError('Report and database paths must differ.')
    for path in (output, report_file):
        if path.exists() and not overwrite:
            raise FileExistsError(f'Output exists; use --overwrite explicitly: {path}')
    if review_cosine is not None and not -1.0 <= float(review_cosine) <= 1.0:
        raise ValueError('review_cosine must be within [-1, 1].')
    if not isinstance(quality_cfg, dict):
        raise TypeError('quality_cfg must be a mapping.')
    if enrollment_min_face_px < 1 or enrollment_min_eye_distance_px <= 0:
        raise ValueError('Enrollment size and eye-distance minima must be positive.')
    # Registration photos should be better than the minimum surveillance frame.
    # These are provisional quality gates, NOT recognition similarity thresholds.
    enroll_quality_cfg = dict(quality_cfg)
    enroll_quality_cfg['min_face_px'] = max(float(quality_cfg['min_face_px']),
                                            float(enrollment_min_face_px))
    enroll_quality_cfg['min_eye_distance_px'] = max(float(quality_cfg['min_eye_distance_px']),
                                                    float(enrollment_min_eye_distance_px))
    groups = discover_identity_images(root)
    face_cfg = {} if face_cfg is None else face_cfg
    angles = (face_cfg.get('rotation_angles', (-30, 30)) if rotation_angles is None
              else rotation_angles)
    if angles is None:
        angles = ()
    if any(not np.isfinite(float(a)) for a in angles):
        raise ValueError('rotation angles must be finite.')

    model_path = Path(recognizer.model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(f'Recognition model file is missing: {model_path}')
    model_digest = _sha256_file(model_path)
    pack = getattr(recognizer, 'cfg', {}).get('model_pack', 'buffalo_l')
    identity_model = model_id or f'insightface/{pack}/{model_path.stem}'
    db = FaceDatabase(model_id=identity_model, model_sha256=model_digest)

    image_hashes, cross_identity_hashes = _duplicate_hashes_between_people(groups)
    report = {
        'schema_version': 1,
        'dataset_root': str(root),
        'database_path': str(output),
        'model_id': identity_model,
        'model_sha256': model_digest,
        'include_medium': bool(include_medium),
        'review_cosine': review_cosine,
        'enrollment_quality_min_face_px': enroll_quality_cfg['min_face_px'],
        'enrollment_quality_min_eye_distance_px': enroll_quality_cfg['min_eye_distance_px'],
        'identities': {},
        'images': [],
    }
    seen_within_identity = set()
    image_totals = {'high': 0, 'medium': 0, 'rejected': 0}
    for person_id, paths in groups.items():
        vectors, sources, qualities = [], [], []
        identity_images = []
        for path in paths:
            source = path.relative_to(root).as_posix()
            item = {'person_id': person_id, 'source': source,
                    'status': 'rejected', 'reason': None, 'tier': None}
            image_hash = image_hashes.get(path)
            if image_hash in cross_identity_hashes:
                item['reason'] = 'same_photo_in_different_identity_folders'
            elif image_hash is not None and (person_id, image_hash) in seen_within_identity:
                item['reason'] = 'duplicate_photo_within_identity'
            else:
                if image_hash is not None:
                    seen_within_identity.add((person_id, image_hash))
                try:
                    image = _read_image(path)
                except ValueError:
                    item['reason'] = 'invalid_or_unreadable_image'
                else:
                    # Inference/runtime errors are intentionally not suppressed.
                    faces = detect_enrollment_faces(detector, image, rotation_angles=angles)
                    if len(faces) == 0:
                        item['reason'] = 'no_face'
                    elif len(faces) != 1:
                        item['reason'] = 'multiple_faces_requires_review'
                    else:
                        metrics, tier = recognition_quality_metrics(image, faces[0], enroll_quality_cfg)
                        if metrics is None:
                            item['reason'] = f'quality_{tier}'
                        elif tier == 'medium' and not include_medium:
                            item['reason'] = 'medium_requires_opt_in'
                            item['tier'] = tier
                            item['quality_score'] = float(metrics['score'])
                        else:
                            try:
                                vector = recognizer.get_embedding(image, faces[0])
                            except (ValueError, cv2.error):
                                vector = None
                            if vector is None:
                                item['reason'] = 'invalid_embedding_or_alignment'
                            else:
                                arr = np.asarray(vector, dtype=np.float32)
                                if (arr.ndim != 1 or not arr.size or not np.isfinite(arr).all()
                                        or float(np.linalg.norm(arr)) <= 1e-12
                                        or (vectors and len(arr) != len(vectors[0]))
                                        or (db.embedding_dim is not None and len(arr) != db.embedding_dim)):
                                    item['reason'] = 'invalid_embedding_or_alignment'
                                else:
                                    vectors.append(arr)
                                    sources.append(source)
                                    qualities.append(float(metrics['score']))
                                    item.update(status='accepted', reason=None, tier=tier,
                                                quality_score=float(metrics['score']))
            identity_images.append(item)
            report['images'].append(item)
        stats = _intra_identity_stats(vectors)
        held_for_review = (review_cosine is not None and
                           stats['min_pairwise_cosine'] is not None and
                           stats['min_pairwise_cosine'] < review_cosine)
        if held_for_review:
            # No vector is saved from a folder with conflicting evidence.
            for item in identity_images:
                if item['status'] == 'accepted':
                    item.update(status='held_for_review', reason='intra_identity_similarity_review')
        elif vectors:
            db.add_identity(person_id, np.stack(vectors), sources=sources, qualities=qualities)
        report['identities'][person_id] = {
            'total_images': len(paths),
            'accepted_templates': len(vectors) if not held_for_review else 0,
            'candidate_templates': len(vectors),
            'status': ('held_for_review' if held_for_review else
                       'enrolled' if vectors else 'no_usable_faces'),
            **stats,
        }
    for item in report['images']:
        if item['status'] == 'accepted':
            image_totals[item['tier']] += 1
        else:
            image_totals['rejected'] += 1
    report['summary'] = {
        'person_folders': len(groups),
        'persons_enrolled': len(db),
        'persons_not_enrolled': len(groups) - len(db),
        'photos_seen': len(report['images']),
        'templates_saved': db.total_templates,
        'high_photos_accepted': image_totals['high'],
        'medium_photos_accepted': image_totals['medium'],
        'photos_rejected_or_held': image_totals['rejected'],
    }
    report['cross_identity_review_candidates'] = _cross_identity_diagnostics(db)
    if not len(db) or (require_all_identities and len(db) != len(groups)):
        report['database_published'] = False
        _write_json_atomic(report_file, report)
        if not len(db):
            raise RuntimeError(f'No person has usable images. Database was not published; see report: {report_file}')
        missing = [pid for pid, value in report['identities'].items()
                   if value['status'] != 'enrolled']
        raise RuntimeError(f'Unenrolled identities {missing}; database not published; see report: {report_file}')
    # Persist the model-specific template DB and its explicit audit trail.
    report['database_published'] = True
    db.save(output)
    _write_json_atomic(report_file, report)
    return report


def _load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not isinstance(cfg.get('face'), dict) or not isinstance(cfg.get('quality'), dict):
        raise ValueError('Configuration must contain face and quality mappings.')
    recognition = cfg.get('recognition') or {}
    if not isinstance(recognition, dict):
        raise ValueError('recognition configuration must be a mapping.')
    return path, cfg, recognition


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build a multi-template face database from identity folders.')
    parser.add_argument('--dataset', required=True, help='Extracted dataset root: one subfolder per person.')
    parser.add_argument('--config', default='config.yaml', help='Existing Camera AI config.yaml.')
    parser.add_argument('--output', default='face_database.npz', help='Where to save the embedding database.')
    parser.add_argument('--report', help='Optional path for detailed enrollment JSON report.')
    parser.add_argument('--face-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--recognition-device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--include-medium', action='store_true',
                        help='Explicitly allow medium-quality images for ENROLLMENT (higher risk).')
    parser.add_argument('--min-enrollment-face-px', type=float, default=64,
                        help='Tentative minimum face side length for HIGH enrollment, default 64 px.')
    parser.add_argument('--min-enrollment-eye-px', type=float, default=12,
                        help='Tentative eye separation for HIGH enrollment, default 12 px.')
    parser.add_argument('--review-cosine', type=float, default=None,
                        help='Optional uncalibrated within-person cosine REVIEW gate; no default assumed.')
    parser.add_argument('--require-all-identities', action='store_true',
                        help='Fail without writing database if any folder has no usable templates.')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly replace existing output files.')
    parser.add_argument('--scan-only', action='store_true',
                        help='Count folders/photos without downloading or loading any models.')
    args = parser.parse_args(argv)

    if args.scan_only:
        groups = discover_identity_images(args.dataset)
        for pid, images in groups.items():
            print(f'{pid}: {len(images)} images')
        print(f'Found {len(groups)} folders, {sum(len(v) for v in groups.values())} candidate photos.')
        return 0

    config_path, cfg, rec_cfg = _load_config(args.config)
    # Same model pack as the detection pipeline; recognition can override only
    # its own ONNX path/model pack in config (added later in config.yaml).
    rec_cfg = dict(rec_cfg)
    rec_cfg.setdefault('model_pack', cfg['face'].get('model_pack', 'buffalo_l'))
    from backends import FaceDetector, FaceRecognizer  # lazy model imports

    detector = FaceDetector(cfg['face'], device=args.face_device,
                            base_dir=config_path.parent)
    recognizer = FaceRecognizer(rec_cfg, device=args.recognition_device,
                                base_dir=config_path.parent)
    report = enroll_dataset(
        args.dataset, args.output, detector=detector, recognizer=recognizer,
        quality_cfg=cfg['quality'], face_cfg=cfg['face'], report_path=args.report,
        include_medium=args.include_medium, overwrite=args.overwrite,
        review_cosine=args.review_cosine,
        require_all_identities=args.require_all_identities,
        model_id=rec_cfg.get('model_id'),
        enrollment_min_face_px=args.min_enrollment_face_px,
        enrollment_min_eye_distance_px=args.min_enrollment_eye_px)
    print(json.dumps(report['summary'], indent=2, ensure_ascii=False))
    print('Database:', Path(args.output).resolve())
    print('Report:', Path(args.report).resolve() if args.report else Path(args.output).resolve().with_suffix('.report.json'))
    if report['summary']['persons_not_enrolled']:
        print('WARNING: Some people were NOT enrolled. See the JSON report before deployment.')
    print('The embeddings are enrollment templates, not calibrated identification decisions.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
