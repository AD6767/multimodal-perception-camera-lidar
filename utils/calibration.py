import numpy as np

def load_calibration(calib_path):
    """Parse KITTI calibration file"""
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if ':' in line:
                key, value = line.split(':', 1)
            else:
                parts = line.split()
                key, value = parts[0], ' '.join(parts[1:])
            calib[key] = np.array([float(x) for x in value.split()])
    return calib

def get_calib_matrices(calib):
    """Extract matrices"""
    P2 = calib['P2'].reshape(3, 4)  # Camera Projection Matrix (intrinsics + rectified frame) (Projects 3D points (in rectified camera frame) to 2D image plane)
    R0 = calib['R_rect'].reshape(3, 3) # Rectification Matrix (Aligns the camera to a rectified coordinate frame)
    Tr = calib['Tr_velo_cam'].reshape(3, 4)  # LiDAR to Camera Extrinsic (Converts points from LiDAR frame -> camera frame)
    return P2, R0, Tr
