# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb
python infer_detections_using_tfrecord.py \
    --input_tfrecord_paths=pet_val.record \
    --output_tfrecord_path=detections.tfrecord \
    --inference_graph=frozen_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import cv2
# This is needed since the notebook is stored in the object_detection folder.
TF_API="/home/zhangbin/eclipse-workspace-python/TF_models/src/models/research/object_detection"
sys.path.append(os.path.split(TF_API)[0])
sys.path.append(TF_API)
# from object_detection.inference import detection_inference
import detection_inference

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')
tf.flags.DEFINE_boolean('show_image_on_run', True,
                        'run with plt.show()')

FLAGS = tf.flags.FLAGS

def bboxes_select(classes, scores, bboxes, threshold=0.1):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    mask = scores > threshold
    classes = classes[mask]
    scores = scores[mask]
    bboxes = bboxes[mask]
    return classes, scores, bboxes

def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2):
    shape = img.shape
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        # Draw bounding box...
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        cv2.rectangle(img, p1[::-1], p2[::-1], (247, 182, 210), thickness)
        # Draw text...
        s = '%s/%.3f' % (classes[i], scores[i])
        p1 = (p1[0]-5, p1[1])
        cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 0, 0), 1)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  with tf.Session() as sess:
    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
    serialized_example_tensor, image_tensor = detection_inference.build_input(
        input_tfrecord_paths)
    tf.logging.info('Reading graph and building model...')
    (detected_boxes_tensor, detected_scores_tensor,
     detected_labels_tensor) = detection_inference.build_inference_graph(
         image_tensor, FLAGS.inference_graph)

    tf.logging.info('Running inference and writing output to {}'.format(
        FLAGS.output_tfrecord_path))
    sess.run(tf.local_variables_initializer())
    tf.train.start_queue_runners()
    with tf.python_io.TFRecordWriter(
        FLAGS.output_tfrecord_path) as tf_record_writer:
      try:
        for counter in itertools.count():
          tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                 counter)
          tf_example = detection_inference.infer_detections_and_add_to_example(
              serialized_example_tensor, detected_boxes_tensor,
              detected_scores_tensor, detected_labels_tensor,
              FLAGS.discard_image_pixels)
          if(FLAGS.show_image_on_run):
              png_string = tf_example.features.feature['image/encoded'].bytes_list.value[0]
              decoded_img = tf.image.decode_jpeg(png_string, channels=3)
              img_data_jpg = tf.expand_dims(tf.image.convert_image_dtype(decoded_img, dtype=tf.float32), 0)
              
              label = tf_example.features.feature['image/detection/label'].int64_list.value[:]
              
              xmin = tf_example.features.feature['image/detection/bbox/xmin'].float_list.value[:]
              xmax = tf_example.features.feature['image/detection/bbox/xmax'].float_list.value[:]
              ymin = tf_example.features.feature['image/detection/bbox/ymin'].float_list.value[:]
              ymax = tf_example.features.feature['image/detection/bbox/ymax'].float_list.value[:]
    
              scores = tf_example.features.feature['image/detection/score'].float_list.value[:]
              boxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], 0))

              classes, scores, boxes = bboxes_select( np.asarray(label),  np.asarray(scores),  np.asarray(boxes.eval()), threshold=0.5)
              print(boxes)
              print(classes)
              print(scores)    
              
              #method 1,bboxes_draw_on_img
#               decoded_img =sess.run(decoded_img)
#               bboxes_draw_on_img(decoded_img, classes, scores, boxes, thickness=1)
#               plt.imshow(decoded_img)
              
              #method 2,tf.image.draw_bounding_boxes
              boxes = tf.expand_dims(boxes, 0)
              result = tf.image.draw_bounding_boxes(img_data_jpg, boxes)  
              plt.imshow(result[0].eval())

              plt.show()
          
          tf_record_writer.write(tf_example.SerializeToString())
      except tf.errors.OutOfRangeError:
        tf.logging.info('Finished processing records')


if __name__ == '__main__':
  tf.app.run()
