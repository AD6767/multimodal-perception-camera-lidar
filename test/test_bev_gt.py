from dataset.kitti_labels import parse_kitti_label_file
from bev.bev_utils import kitti_object_to_bev_box
from visualization.bev_viz import visualize_bev


LABEL_PATH = 'data/KITTI/tracking/training/label_02/0000.txt'

objects = parse_kitti_label_file(LABEL_PATH)

bev_boxes = []
for obj in objects:
    if obj.cls == "Car":
        # print("Processing object:", obj)
        bev_box = kitti_object_to_bev_box(obj)
        bev_boxes.append(bev_box)

visualize_bev(bev_boxes, title='KITTI GT Cars (BEV)')
