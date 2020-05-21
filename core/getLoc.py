import os
import sys

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt'

NUM_CLASSES = 1
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categorys = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categorys)


def getPageLoc(path):
	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			for i in range(1,2):
				image = cv.imread(path.format(i))
				image_np_expanded = np.expand_dims(image, axis=0)
				image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
				boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
				scores = detection_graph.get_tensor_by_name('detection_scores:0')
				classes = detection_graph.get_tensor_by_name('detection_classes:0')
				num_detections = detection_graph.get_tensor_by_name('num_detections:0')

				(boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],
																	feed_dict={image_tensor: image_np_expanded})
																	
				if len(boxes[scores>0.9]) == 0:
					continue
				posX = int(boxes[scores>0.9][0][1] * 320)
				posY = int(boxes[scores>0.9][0][0] * 160)
				posXmax = int(boxes[scores>0.9][0][3] * 320)
				posYmax = int(boxes[scores>0.9][0][2] * 160)
				
				cv.rectangle(
					image, 
					(posX,posY), #左上角
					(posXmax,posYmax), #右下角
					(0,255,0),
					2
				)
				print("左上角：",posX,posY)
				print("右下角:",posXmax,posYmax)
				cv.putText(
					image, 		#图片
					str(posX), 	#添加的文字
					(posX,posY - 5), #左上角坐标
					cv.FONT_HERSHEY_SIMPLEX, #字体
					1, #字体大小
					(0,0,255), #颜色
					2 #字体粗细
				)


				cv.imshow("SSD - drag Detector Demo{}".format(i), image)
			cv.waitKey(0)
			cv.destroyAllWindows()
	return (posX,posY,posXmax,posYmax)
