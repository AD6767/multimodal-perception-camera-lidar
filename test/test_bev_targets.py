from dataset.kitti_labels import parse_kitti_label_file
from bev.bev_utils import kitti_object_to_bev_box
from bev.bev_targets import create_bev_targets
from bev.bev_config import BEVConfig
from visualization.bev_viz import visualize_heatmap

LABEL_PATH = "data/KITTI/tracking/training/label_02/0000.txt"

objects = parse_kitti_label_file(LABEL_PATH)
# FRAME_ID = 126
# print("Total objects in file:", len(objects))
# objects = [o for o in objects if o.frame == FRAME_ID]
# print(f"Objects in frame {FRAME_ID}:", len(objects))

bev_boxes = []
for obj in objects:
    if obj.cls == "Car":
        bev_box = kitti_object_to_bev_box(obj)
        bev_boxes.append(bev_box)

config = BEVConfig()
targets = create_bev_targets(bev_boxes, config)

visualize_heatmap(targets["heatmap"], title="Car Center Heatmap")
