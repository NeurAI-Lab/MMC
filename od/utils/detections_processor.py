import json
import numpy as np
import os
from vizer.draw import draw_boxes
from xml.dom.minidom import parseString
import xml.etree.cElementTree as ET


def convert_to_format(fmt, filename, image, boxes, labels, scores, class_names):
    if fmt == 'img':
        return draw_boxes(image, boxes, labels, scores, class_names).astype(np.uint8)

    detection_list = create_detection_list(boxes, labels, scores, class_names)
    if fmt == 'json':
        result = convert_to_json(detection_list)
    elif fmt == 'json_nie':
        result = convert_to_json_nie(detection_list)
    elif fmt == 'txt':
        result = "\n".join(['%s' % ','.join(map(str, detection)) for detection in detection_list])
    elif fmt == 'xml':
        result = convert_to_xml(filename, image, detection_list)
    else:
        raise ValueError('{} unsupported format.')

    return result


def create_detection_list(boxes, labels, scores, class_names):
    """Convert detections in a consolidated detection list."""
    detection_list = []
    for i in range(len(boxes)):
        detection_per_box = []
        cat_code = labels[i]
        detection_per_box.append(cat_code)
        detection_per_box.append(class_names[cat_code])
        box = [int(coord) for coord in list(boxes[i])]
        detection_per_box.extend(box)
        detection_per_box.append(scores[i])
        detection_list.append(detection_per_box)

    return detection_list


def convert_to_json(detection_list):
    """Convert to NavInfo JSON standard format."""
    result_dict = dict()
    result_dict["cv_task"] = 1
    result_dict["obj_num"] = len(detection_list)
    objects = list()

    # Get the unique category IDs available in the detection list.
    unique_cat_id = sorted(set(detection[0] for detection in detection_list))

    for cat_id in unique_cat_id:
        # Find a better algorithm as the detection_list is sorted already.
        per_cat_list = [detection for detection in detection_list if detection[0] == cat_id]
        sub_cat_dict = dict()
        sub_cat_dict["f_name"] = str(per_cat_list[0][1])
        sub_cat_dict["f_code"] = int(per_cat_list[0][0])
        obj_points = list()
        for detection in per_cat_list:
            per_detection_dict = dict()
            per_detection_dict["x"] = int(detection[2])
            per_detection_dict["y"] = int(detection[3])
            per_detection_dict["w"] = int(detection[4] - detection[2])
            per_detection_dict["h"] = int(detection[5] - detection[3])
            per_detection_dict["f_conf"] = float(detection[6])
            obj_points.append(per_detection_dict)
        sub_cat_dict["obj_points"] = obj_points
        objects.append(sub_cat_dict)
    result_dict["objects"] = objects

    return json.dumps(result_dict, indent=4)


def convert_to_json_nie(detection_list):
    """Convert to NavInfo Europe JSON format."""
    result_dict = dict()
    result_dict["cv_task"] = 1
    result_dict["obj_num"] = len(detection_list)
    objects = list()
    for detection in detection_list:
        object = dict()
        object["f_name"] = detection[1]
        object["f_code"] = int(detection[0])
        obj_points = dict()
        obj_points["x"] = int(detection[2])
        obj_points["y"] = int(detection[3])
        obj_points["w"] = int(detection[4] - detection[2])
        obj_points["h"] = int(detection[5] - detection[3])
        object["obj_points"] = obj_points
        object["f_conf"] = float(detection[6])
        objects.append(object)
    result_dict["objects"] = objects

    return json.dumps(result_dict, indent=4)


def convert_to_xml(filename, image, detection_list):
    node_root = ET.Element('annotation')
    node_folder = ET.SubElement(node_root, 'folder')
    node_folder.text = os.path.dirname(os.path.abspath(filename))
    node_filename = ET.SubElement(node_root, 'filename')
    node_filename.text = os.path.basename(filename)
    node_size = ET.SubElement(node_root, 'size')
    img_width = ET.SubElement(node_size, 'width')
    img_width.text = str(image.shape[1])
    img_height = ET.SubElement(node_size, 'height')
    img_height.text = str(image.shape[0])
    img_depth = ET.SubElement(node_size, 'depth')
    img_depth.text = str(image.shape[2])

    for detection in detection_list:
        node_object = ET.SubElement(node_root, 'object')
        object_name = ET.SubElement(node_object, 'name')
        object_name.text = str(detection[1])
        object_difficulty = ET.SubElement(node_object, 'difficult')
        object_difficulty.text = '0'
        object_bndbox = ET.SubElement(node_object, 'bndbox')
        xmin = ET.SubElement(object_bndbox, 'xmin')
        xmin.text = str(detection[2])
        ymin = ET.SubElement(object_bndbox, 'ymin')
        ymin.text = str(detection[3])
        xmax = ET.SubElement(object_bndbox, 'xmax')
        xmax.text = str(detection[4])
        ymax = ET.SubElement(object_bndbox, 'ymax')
        ymax.text = str(detection[5])
        object_confidence = ET.SubElement(node_object, 'confidence')
        object_confidence.text = str(detection[6])

    detection_tree = ET.tostring(node_root)
    dom = parseString(detection_tree)
    xml_out = dom.toprettyxml()

    return xml_out
