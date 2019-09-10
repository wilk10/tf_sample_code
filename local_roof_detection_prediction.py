import os
import cv2
import json
import base64
import pathlib
import subprocess
import numpy as np
import tensorflow as tf
from pipeline.inria.image_preparation import ImagePreparation
from pipeline.inria.ground_truths import InriaGroundTruths


class RoofPrediction:
    BASE_DIM = 300
    DETECTION_THRESHOLD = 0

    def __init__(self):
        self.sg_dir = pathlib.Path.home() / "Documents/sg"
        self.images_dir = self.sg_dir / "images/test_images"
        self.image_names = os.listdir(str(self.images_dir))[1:]
        self.predict_instance_filename = "inputs.json"
        self.predictions_dir = self.sg_dir / "sgrepo/cv_research/object_detection/results/predictions"
        self.predict_instance_filepath = self.predictions_dir / self.predict_instance_filename
        self.saved_model_dir = self.predictions_dir.parent / "output/saved_model"
        self.frozen_graph_filepath = self.saved_model_dir.parent / "frozen_inference_graph.pb"
        self.tf_research_dir = self.sg_dir / "sgrepo/cv_research/models/research"

    def save_images_to_json(self):
        with open(str(self.predict_instance_filepath), "w") as fp:
            for image_name in self.image_names:
                image_path = self.images_dir / image_name
                loaded_image = cv2.imread(str(image_path))
                resized_image = ImagePreparation.resize_image(loaded_image, self.BASE_DIM)
                _, buffer = cv2.imencode('.jpg', resized_image)
                image_64_encode = base64.b64encode(buffer)
                image_to_save = image_64_encode.decode("utf-8")
                fp.write(json.dumps({"b64": image_to_save}) + "\n")

    def use_gcloud_subprocess(self):
        prediction_files = os.listdir(str(self.predictions_dir))
        if self.predict_instance_filename not in prediction_files:
            self.save_images_to_json()
        command = ["gcloud", "ml-engine", "local", "predict", "--model-dir", str(self.saved_model_dir)]
        json_instance = "--json-instances={}".format(self.predict_instance_filepath)
        command.append(json_instance)
        process = subprocess.Popen(command, cwd=str(self.tf_research_dir), stdout=subprocess.PIPE)
        output = process.stdout.read().strip()
        print(output)

    def load_frozen_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(str(self.frozen_graph_filepath), 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
        return od_graph_def

    def run_inference_for_single_image(self, image):
        od_graph_def = self.load_frozen_model()
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.import_graph_def(od_graph_def, name='')
            ops = graph.get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_boxes', 'detection_scores']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict

    def use_frozen_model(self):
        for image_name in self.image_names:
            image_path = self.images_dir / image_name
            loaded_image = cv2.imread(str(image_path))
            resized_image = ImagePreparation.resize_image(loaded_image, self.BASE_DIM)
            image_array = np.expand_dims(resized_image, axis=0)
            output_dict = self.run_inference_for_single_image(image_array)
            accepted_detections = [
                score for score in output_dict["detection_scores"] if score > self.DETECTION_THRESHOLD]
            accepted_boxes = output_dict["detection_boxes"][:len(accepted_detections)]
            boxes_to_scale = accepted_boxes * self.BASE_DIM
            flat_boxes = InriaGroundTruths.make_xy_in_flat_box_to_tuples(boxes_to_scale, invert_xy=True)
            InriaGroundTruths.show_images(image1=resized_image, boxes=flat_boxes)


if __name__ == "__main__":
    # RoofPrediction().use_gcloud_subprocess()
    RoofPrediction().use_frozen_model()
