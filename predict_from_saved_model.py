import os
import cv2
import pathlib
import argparse
import numpy as np
import tensorflow as tf
from pipeline.inria.image_preparation import ImagePreparation


class PredictCustomKerasModel:
    BASE_DIM = 256

    def __init__(self, timestamp, num):
        self.timestamp = timestamp
        self.num = num
        self.saved_model_dir = f"gs://solargenio/custom_keras/models/model_{self.timestamp}/export/exporter/{self.num}/"
        self.prediction_images_dir = self.get_prediction_images_dir()
        self.images_to_predict = self.get_images_to_predict()

    @staticmethod
    def get_prediction_images_dir():
        cwd = pathlib.Path.cwd()
        return cwd.parent / "images/test_images_zoom20/test"

    def get_images_to_predict(self):
        prediction_images = os.listdir(self.prediction_images_dir)
        prediction_images = [image for image in prediction_images if "png" in image]
        images = []
        for image_name in prediction_images:
            image_path = self.prediction_images_dir / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            resized_image = ImagePreparation.resize_image(image, self.BASE_DIM)
            float32_image = resized_image.astype('float32', casting='same_kind')
            final_image = float32_image / 255.0
            images.append(final_image)
        return np.array(images)

    @staticmethod
    def show_contour_predictions(predict_images, predictions):
        zipped_images = zip(predict_images, predictions)
        for original, prediction in zipped_images:
            masked_prediction = (prediction > 0.5).astype(np.uint8)
            _, contours, hierarchies = cv2.findContours(masked_prediction, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            output_image = (original * 255).astype(np.uint8)
            if contours:
                for contour, hierarchy in zip(contours, hierarchies[0]):
                    if hierarchy[-1] == -1:
                        cv2.drawContours(output_image, [contour], 0, (0, 0, 255), 1)
            color_prediction = cv2.cvtColor(masked_prediction, cv2.COLOR_GRAY2BGR)
            window_name = 'Output'
            cv2.namedWindow(window_name)
            cv2.moveWindow(window_name, 0, 0)
            cv2.imshow(window_name, np.hstack([output_image, color_prediction]))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def predict(self):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.saved_model_dir)
            graph = tf.get_default_graph()
            input_tensor = graph.get_tensor_by_name("input_image:0")
            model = graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
            predictions = sess.run(model, {input_tensor: self.images_to_predict})
        self.show_contour_predictions(self.images_to_predict, predictions)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("timestamp", type=str)
    parser.add_argument("num", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    PredictCustomKerasModel(**vars(arguments)).predict()
