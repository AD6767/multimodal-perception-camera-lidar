from dataclasses import dataclass
from typing import List


@dataclass
class KittiObject:
    frame: int
    cls : str
    x: float
    y: float
    z: float
    w: float
    l: float
    h: float
    yaw: float

def parse_kitti_label_file(label_file_path: str) -> List[KittiObject]:
    """
    frame  track_id  type
    truncated  occluded  alpha
    bbox_left  bbox_top  bbox_right  bbox_bottom
    h  w  l
    x  y  z
    rotation_y
    """
    objects = []
    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            fields = line.strip().split()
            frame = int(fields[0])
            cls = fields[2]
            if (cls == 'DontCare'): # skip since annotation is unreliable.
                continue

            x, y, z = map(float, fields[13:16])
            h, w, l = map(float, fields[10:13])
            yaw = float(fields[16])
            objects.append(KittiObject(frame, cls, x, y, z, w, l, h, yaw))
    return objects
