"""Identity template database for the Camera AI face-recognition pipeline.

One directory in an extracted enrollment dataset represents one known person;
each directory may contain ANY POSITIVE number of photos. Image discovery is
read-only; image decoding, face detection, alignment and embedding extraction
will be handled by enroll.py. No recognition model is loaded by this module.

FaceDatabase stores multiple unit-L2 embeddings per person, but ranks PEOPLE
rather than ranking every photo as a separate identity. Search returns
candidates/similarities only, never a confirmed identity or a probability.
Unknown/confirmed/conflict decisions belong to identity_manager.py after
threshold calibration on held-out, identity-labeled video.

Biometric embeddings are sensitive data. Restrict access to database files,
keep enrollment images under access control, and apply a retention policy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np


IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'})
DATABASE_VERSION = 1


def discover_identity_images(dataset_root, *, recursive=False, exclude_folders=None):
    """Return {folder_name: [image_paths]} for a directory of identity folders.

    The number of images is NOT assumed to be equal across people. Empty
    identity folders remain in the result so that enroll.py can REPORT them
    rather than silently treating their identity as enrolled. Unsupported
    extensions and symlinks are ignored; no file contents are modified.

    For a ZIP dataset, extract it first, then pass the directory containing
    the person folders. ``recursive=True`` scans subfolders *within* each
    person directory, without treating those as additional identities.
    """
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f'Enrollment directory not found: {root}')
    excluded = {'__MACOSX'}
    if exclude_folders is not None:
        excluded.update(str(name) for name in exclude_folders)
    persons = [p for p in root.iterdir()
               if p.is_dir() and not p.is_symlink()
               and not p.name.startswith('.') and p.name not in excluded]
    persons.sort(key=lambda p: (p.name.casefold(), p.name))
    if not persons:
        raise ValueError('No person subfolders found. Expected dataset_root/person_id/image.jpg.')

    # Detect case/Unicode collisions before creating an ambiguous identity DB.
    seen = {}
    results = {}
    for folder in persons:
        person_id = _validate_person_id(folder.name)
        key = person_id.casefold()
        if key in seen:
            raise ValueError(f'Conflicting identity folder names: {seen[key]!r}, {person_id!r}')
        seen[key] = person_id
        candidates = folder.rglob('*') if recursive else folder.iterdir()
        images = [p for p in candidates
                  if p.is_file() and not p.is_symlink() and not p.name.startswith('.')
                  and p.suffix.lower() in IMAGE_EXTENSIONS]
        results[person_id] = sorted(images, key=lambda p: str(p.relative_to(folder)).casefold())
    return results


def _validate_person_id(person_id):
    if not isinstance(person_id, str) or not person_id or person_id != person_id.strip():
        raise ValueError('person_id must be a nonempty string without surrounding spaces.')
    if person_id in ('.', '..') or any(ch in person_id for ch in ('/', '\\')):
        raise ValueError('person_id must not contain path separators or be a dot directory.')
    if any(ord(ch) < 32 for ch in person_id):
        raise ValueError('person_id must not contain control characters.')
    return person_id


def _unit_rows(vectors, dim=None):
    try:
        values = np.asarray(vectors, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Embeddings must be a numeric array.') from exc
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError('Expected a non-empty embedding matrix with shape (N, D).')
    if dim is not None and values.shape[1] != dim:
        raise ValueError(f'Embedding dimension mismatch: expected {dim}, got {values.shape[1]}.')
    if not np.isfinite(values).all():
        raise ValueError('Embeddings must contain only finite values.')
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError('Embeddings must have nonzero finite norms.')
    return np.ascontiguousarray(values / norms[:, None], dtype=np.float32)


def _metadata_list(values, count, name):
    if values is None:
        return [None] * count
    if isinstance(values, (str, bytes)):
        raise TypeError(f'{name} must be a list with one entry per embedding.')
    result = list(values)
    if len(result) != count:
        raise ValueError(f'{name} must have {count} entries, got {len(result)}.')
    if name == 'sources':
        if any(item is not None and not isinstance(item, str) for item in result):
            raise TypeError('sources may contain only strings or None.')
    else:
        for item in result:
            if item is not None and (not isinstance(item, (int, float, np.number))
                                     or not np.isfinite(item) or not 0 <= item <= 1):
                raise ValueError('qualities must be finite values in [0, 1] or None.')
        result = [None if item is None else float(item) for item in result]
    return result


class FaceDatabase:
    """In-memory database; persist it as one atomic, non-pickle .npz file.

    ``model_id`` identifies the embedding model AND its preprocessing. A
    database created by ArcFace R50 must not be queried using AdaFace or a
    differently preprocessed model. For stronger protection, set model_sha256
    to the ONNX file's SHA256 and verify it when loading.

    A person may have 1, 3, 5, 20, ... embeddings. No padding, duplication,
    minimum fixed image count or automatic sample truncation is performed.
    """

    def __init__(self, model_id='insightface/buffalo_l/w600k_r50',
                 embedding_dim=None, model_sha256=None):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError('model_id must be a nonempty string.')
        if embedding_dim is not None and (type(embedding_dim) is not int or embedding_dim <= 0):
            raise ValueError('embedding_dim must be a positive integer or None.')
        if model_sha256 is not None and (not isinstance(model_sha256, str)
                                         or len(model_sha256) != 64
                                         or any(c not in '0123456789abcdef' for c in model_sha256.lower())):
            raise ValueError('model_sha256 must be a 64-character hexadecimal SHA256 digest.')
        self.model_id = model_id
        self.embedding_dim = embedding_dim
        self.model_sha256 = model_sha256.lower() if model_sha256 else None
        self._vectors = {}       # person_id -> (number_of_photos, embedding_dim)
        self._sources = {}       # one relative filename / label per photo, optional
        self._qualities = {}     # optional image quality metadata; NOT probabilities

    def __len__(self):
        return len(self._vectors)

    @property
    def total_templates(self):
        return sum(len(vectors) for vectors in self._vectors.values())

    def counts(self):
        """Counts may differ across identities; each person is indexed once."""
        return {person_id: len(self._vectors[person_id])
                for person_id in sorted(self._vectors)}

    def add_identity(self, person_id, embeddings, *, sources=None, qualities=None,
                     replace=False):
        """Store N>=1 embedding templates for one known person.

        Reject duplicate person IDs unless replace=True is explicit. A source
        entry should be a filename relative to the enrollment dataset root.
        """
        person_id = _validate_person_id(person_id)
        if any(existing.casefold() == person_id.casefold() and existing != person_id
               for existing in self._vectors):
            raise ValueError(f'Identity differs only by letter case: {person_id!r}.')
        if person_id in self._vectors and not replace:
            raise ValueError(f'Identity already exists: {person_id!r}; use add_template or replace=True.')
        vectors = _unit_rows(embeddings, self.embedding_dim)
        image_sources = _metadata_list(sources, len(vectors), 'sources')
        image_qualities = _metadata_list(qualities, len(vectors), 'qualities')
        if self.embedding_dim is None:
            self.embedding_dim = vectors.shape[1]
        self._vectors[person_id] = vectors.copy()
        self._sources[person_id] = image_sources
        self._qualities[person_id] = image_qualities

    def add_template(self, person_id, embedding, *, source=None, quality=None):
        """Append one new angle/photo without replacing the existing photos."""
        person_id = _validate_person_id(person_id)
        vector = _unit_rows(embedding, self.embedding_dim)
        if len(vector) != 1:
            raise ValueError('add_template accepts exactly one embedding.')
        if person_id not in self._vectors:
            return self.add_identity(person_id, vector, sources=[source], qualities=[quality])
        image_source = _metadata_list([source], 1, 'sources')[0]
        image_quality = _metadata_list([quality], 1, 'qualities')[0]
        self._vectors[person_id] = np.concatenate((self._vectors[person_id], vector))
        self._sources[person_id].append(image_source)
        self._qualities[person_id].append(image_quality)

    def get_templates(self, person_id):
        """Return a COPY; outside changes cannot corrupt the database."""
        return self._vectors[person_id].copy()

    def get_metadata(self, person_id):
        return {'person_id': person_id,
                'sources': list(self._sources[person_id]),
                'qualities': list(self._qualities[person_id]),
                'template_count': len(self._vectors[person_id])}

    def search(self, embedding, *, top_k=5, score_mode='centroid', template_top_k=3):
        """Return person candidates ranked by cosine similarity; NEVER confirm.

        Default ``centroid``: one L2-normalized mean embedding per person, so
        the database does not simply take the maximum of a larger photo set.
        For high pose variation, ``template_topk`` averages the top matching
        template scores per identity; ``max_template`` is diagnostic only.
        Any mode still requires held-out, gallery-size-aware calibration for
        open-set recognition. Scores are NOT calibrated probabilities.
        """
        if type(top_k) is not int or top_k <= 0:
            raise ValueError('top_k must be a positive integer.')
        if type(template_top_k) is not int or template_top_k <= 0:
            raise ValueError('template_top_k must be a positive integer.')
        if score_mode not in ('centroid', 'template_topk', 'max_template'):
            raise ValueError('score_mode must be centroid, template_topk or max_template.')
        query = _unit_rows(embedding, self.embedding_dim)
        if len(query) != 1:
            raise ValueError('search expects one embedding, not multiple frames.')
        if not self._vectors:
            return []
        query = query[0]
        results = []
        for person_id, vectors in self._vectors.items():
            scores = vectors @ query
            best_index = int(np.argmax(scores))
            best_score = float(scores[best_index])
            mean_vector = vectors.astype(np.float64).mean(axis=0)
            mean_norm = float(np.linalg.norm(mean_vector))
            # A near-zero mean indicates mutually inconsistent templates.
            # Do not choose an identity based on an arbitrary centroid.
            centroid_score = (float(mean_vector @ query / mean_norm)
                              if mean_norm > 1e-8 else None)
            strongest = np.sort(scores)[-min(template_top_k, len(scores)):]
            topk_score = float(np.mean(strongest))
            if score_mode == 'centroid':
                if centroid_score is None:
                    continue  # Enrollment needs review; no reliable centroid.
                score = centroid_score
            elif score_mode == 'template_topk':
                score = topk_score
            else:
                score = best_score
            results.append({
                'person_id': person_id,
                'score': float(score),
                'centroid_score': centroid_score,
                'topk_template_score': topk_score,
                'best_template_score': best_score,
                'best_template_source': self._sources[person_id][best_index],
                'template_count': len(vectors),
                'score_mode': score_mode,
            })
        results.sort(key=lambda result: (-result['score'], result['person_id']))
        return results[:top_k]

    def save(self, path):
        """Atomically save normalized vectors and JSON metadata as .npz.

        Never uses pickle; the saved file contains embeddings, not raw images.
        Its file permissions are owner-only on POSIX where supported.
        """
        path = Path(path).expanduser()
        if path.suffix.lower() != '.npz':
            raise ValueError('Face database filename must end with .npz.')
        path.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        blocks = []
        offset = 0
        for person_id in sorted(self._vectors):
            matrix = self._vectors[person_id]
            count = len(matrix)
            entries.append({'person_id': person_id, 'offset': offset, 'count': count,
                            'sources': self._sources[person_id],
                            'qualities': self._qualities[person_id]})
            blocks.append(matrix)
            offset += count
        vectors = (np.concatenate(blocks, axis=0) if blocks
                   else np.empty((0, self.embedding_dim or 0), dtype=np.float32))
        manifest = {'version': DATABASE_VERSION, 'model_id': self.model_id,
                    'model_sha256': self.model_sha256,
                    'embedding_dim': self.embedding_dim, 'entries': entries}
        manifest_text = json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.npz', dir=path.parent)
        try:
            with os.fdopen(descriptor, 'wb') as file:
                np.savez_compressed(file, embeddings=vectors,
                                    manifest=np.array(manifest_text))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path

    @classmethod
    def load(cls, path, *, expected_model_id=None, expected_model_sha256=None):
        """Load the database, rejecting mismatched model versions or bad data."""
        path = Path(path).expanduser()
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {'embeddings', 'manifest'}:
                raise ValueError('Invalid face database arrays.')
            raw_manifest = archive['manifest']
            if raw_manifest.ndim != 0 or raw_manifest.dtype.kind != 'U':
                raise ValueError('Invalid face database manifest.')
            manifest = json.loads(str(raw_manifest.item()))
            vectors = archive['embeddings']
        if not isinstance(manifest, dict) or manifest.get('version') != DATABASE_VERSION:
            raise ValueError('Unsupported face database version.')
        if expected_model_id is not None and manifest.get('model_id') != expected_model_id:
            raise ValueError('Recognition model differs from the enrollment model.')
        if expected_model_sha256 is not None and manifest.get('model_sha256') != expected_model_sha256:
            raise ValueError('Recognition model file SHA256 differs from the enrollment model.')
        db = cls(model_id=manifest.get('model_id'),
                 embedding_dim=manifest.get('embedding_dim'),
                 model_sha256=manifest.get('model_sha256'))
        if vectors.ndim != 2 or vectors.dtype.kind != 'f' or not np.isfinite(vectors).all():
            raise ValueError('Invalid embedding matrix in database.')
        if vectors.shape[1] != (db.embedding_dim or 0):
            raise ValueError('Embedding dimensions do not match the manifest.')
        entries = manifest.get('entries')
        if not isinstance(entries, list):
            raise ValueError('Invalid database entries.')
        offset = 0
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError('Invalid person entry.')
            count = entry.get('count')
            if (type(count) is not int or count <= 0
                    or type(entry.get('offset')) is not int or entry['offset'] != offset
                    or offset + count > len(vectors)):
                raise ValueError('Invalid or overlapping template offsets.')
            person_id = entry.get('person_id')
            db.add_identity(person_id, vectors[offset:offset + count],
                            sources=entry.get('sources'),
                            qualities=entry.get('qualities'))
            offset += count
        if offset != len(vectors):
            raise ValueError('Unclaimed embedding rows in database.')
        return db
