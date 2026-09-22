"""Calibrate the SAME camera/lens/resolution mode using checkerboard images."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, help='Quoted glob, e.g. "calib/*.jpg"')
    parser.add_argument('--model', choices=['pinhole', 'fisheye'], required=True)
    parser.add_argument('--cols', type=int, default=9, help='Checkerboard INNER corners across')
    parser.add_argument('--rows', type=int, default=6, help='Checkerboard INNER corners down')
    parser.add_argument('--square-size', type=float, default=25., help='Square side; any consistent physical unit')
    parser.add_argument('--out', default='camera_calibration.npz')
    args = parser.parse_args()
    if args.cols < 3 or args.rows < 3 or args.square_size <= 0:
        parser.error('Use at least 3x3 inner corners and a positive square size.')
    if Path(args.out).suffix != '.npz':
        parser.error('--out must end in .npz')
    files = sorted(glob.glob(args.images))
    if not files:
        raise FileNotFoundError(f'No images matched {args.images}')
    board = (args.cols, args.rows)
    object_template = np.zeros((args.rows * args.cols, 3), dtype=np.float64)
    object_template[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_size
    objects, corners_list, accepted = [], [], []
    image_size = None
    for file in files:
        image = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print('Cannot decode, skipped:', file)
            continue
        size = image.shape[::-1]
        if image_size is not None and size != image_size:
            raise ValueError('All calibration images must have the same resolution and camera mode.')
        image_size = size
        found, corners = cv2.findChessboardCornersSB(image, board,
                                                     flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
        if found:
            objects.append(object_template.copy())
            corners_list.append(corners.astype(np.float64))
            accepted.append(file)
        else:
            print('No complete checkerboard, skipped:', file)
    if len(accepted) < 12:
        raise RuntimeError(f'Only {len(accepted)} usable images. Capture at least 12 (preferably 20-30) varied views.')
    if args.model == 'fisheye':
        op = [p.reshape(1, -1, 3) for p in objects]
        ip = [p.reshape(1, -1, 2) for p in corners_list]
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            op, ip, image_size, np.eye(3), np.zeros((4, 1)), flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-7))
        def project(obj, rv, tv):
            return cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), rv, tv, k, d)[0].reshape(-1, 2)
    else:
        rms, k, d, rvecs, tvecs = cv2.calibrateCamera(
            [p.astype(np.float32) for p in objects],
            [p.astype(np.float32) for p in corners_list], image_size, None, None)
        def project(obj, rv, tv):
            return cv2.projectPoints(obj, rv, tv, k, d)[0].reshape(-1, 2)
    errors = [float(np.sqrt(np.mean(np.sum((project(obj, rv, tv) - pts.reshape(-1, 2)) ** 2, axis=1))))
              for obj, pts, rv, tv in zip(objects, corners_list, rvecs, tvecs)]
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise RuntimeError('Calibration did not converge to finite parameters.')
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, K=k, D=d, image_size=np.array(image_size), model=np.array(args.model),
             rms_px=np.array(rms))
    report = {'model': args.model, 'image_size': list(image_size), 'rms_px': float(rms),
              'views': [{'file': f, 'rms_px': e} for f, e in zip(accepted, errors)]}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved {output}; {len(accepted)} views; reprojection RMS = {rms:.3f} pixels.')
    print('Inspect rectified straight lines and edge faces; low RMS alone does not guarantee a good calibration.')


if __name__ == '__main__':
    main()
