import os
import cv2
import json
import boto3
import base64
import subprocess
import numpy as np
import tensorflow as tf
import googleapiclient.discovery


class TestPredictions:
    BASE_DIM = 256
    PROJECT = "solargenio"
    MODEL = "roof_segmentation"
    VERSION = "mvp_full_v2"
    TIMESTAMP = "2019_06_03_h11m36"
    NUM = "1559562065"

    def __init__(self):
        self.bucket_name = "solargenio-live-app"
        self.aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
        self.aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
        self.image_key = "images/input_roof_segmentation/3a_schützallee_berlin_germany.png"
        self.saved_model_dir = f"gs://solargenio/custom_keras/models/model_{self.TIMESTAMP}/export/exporter/{self.NUM}/"
        self.instances_filepath = f"/Users/anselmosampietro/Downloads/instances_{self.TIMESTAMP}.json"
        sg_dir = "/Users/anselmosampietro/Documents/sg/"
        self.local_image_path = sg_dir + self.image_key

    def get_s3_object(self):
        session = boto3.session.Session(
            self.aws_access_key_id,
            self.aws_secret_access_key)
        s3 = session.resource('s3', use_ssl=False)
        return s3.Object(self.bucket_name, self.image_key)

    def read_s3_response(self):
        s3_object = self.get_s3_object()
        response = s3_object.get()
        return response['Body'].read()

    def load_image_from_s3(self):
        buffer = self.read_s3_response()
        image_array = np.frombuffer(buffer, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def resize_image(original_image, base_dim):
        old_size = original_image.shape[:2]
        ratio = float(base_dim) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        resized_image = cv2.resize(original_image, (new_size[1], new_size[0]))
        delta_w = base_dim - new_size[1]
        delta_h = base_dim - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        color = [0, 0, 0]
        return cv2.copyMakeBorder(
            resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    def send_prediction(self, instances):
        service = googleapiclient.discovery.build('ml', 'v1')
        name = 'projects/{}/models/{}/versions/{}'.format(self.PROJECT, self.MODEL, self.VERSION)
        response = service.projects().predict(
            name=name,
            body={'instances': instances}
        ).execute()
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response['predictions']

    @staticmethod
    def show_image(original, predictions):
        predictions_array = np.array(predictions[0]['conv2d_23'])
        prediction_mask = (predictions_array > 0.5).astype(np.uint8)
        prediction_mask *= 255
        _, contours, _ = cv2.findContours(prediction_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        output_image = original.copy()
        if contours:
            for contour in contours:
                cv2.drawContours(output_image, [contour], 0, (0, 0, 255), 1)
        window_name = 'Output'
        cv2.namedWindow(window_name)
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, np.hstack([output_image]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run_online(self):
        image = self.load_image_from_s3()
        resized_image = self.resize_image(image, self.BASE_DIM)
        buffer = cv2.imencode('.png', resized_image)[1].tostring()
        bytes_image = base64.b64encode(buffer).decode('ascii')
        instances = [{"image_bytes": {"b64": bytes_image}}]
        predictions = self.send_prediction(instances)
        self.show_image(resized_image, predictions)

    @staticmethod
    def signature_def_to_tensors(graph, signature_def):
        return {k: graph.get_tensor_by_name(v.name) for k, v in signature_def.inputs.items()}, \
               {k: graph.get_tensor_by_name(v.name) for k, v in signature_def.outputs.items()}

    def run_locally(self):
        image = self.load_image_from_s3()
        resized_image = self.resize_image(image, self.BASE_DIM)
        buffer = cv2.imencode('.png', resized_image)[1].tostring()
        bytes_image = base64.b64encode(buffer).decode()
        with tf.Session(graph=tf.Graph()) as sess:
            model = tf.saved_model.loader.load(sess, ["serve"], str(self.saved_model_dir))
            graph = tf.get_default_graph()
            input_dict, output_dict = self.signature_def_to_tensors(graph, model.signature_def['serving_default'])
            predictions = sess.run(output_dict, feed_dict={input_dict['image_bytes']: [bytes_image]})
        print(predictions)

    # based on: https://github.com/mhwilder/tf-keras-gcloud-deployment/blob/master/image_to_json.py
    def run_from_cli(self):
        with open(self.local_image_path, 'rb') as f:
            bytes_image = base64.b64encode(f.read()).decode('ascii')
        instances = {"image_bytes": {"b64": bytes_image}}
        with open(str(self.instances_filepath), 'w+') as outfile:
            json.dump(instances, outfile)
        base_command = ["gcloud", "ai-platform", "local", "predict"]
        flags = [
            "--model-dir={}".format(str(self.saved_model_dir)),
            "--json-instances={}".format(str(self.instances_filepath)),
            "--verbosity",
            "debug"]
        command = base_command + flags
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = process.stdout.read().strip()
        print(output)


if __name__ == "__main__":
    TestPredictions().run_online()
    #TestPredictions().run_locally()
    #TestPredictions().run_from_cli()
