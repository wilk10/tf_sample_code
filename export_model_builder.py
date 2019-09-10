import datetime
import tensorflow as tf
from custom_keras_model.trainer.model import Model


class ModelBuilder:
    TIMESTAMP = "2019_03_04_h11m28"
    NUM = "1551701418"
    BASE_DIM = 256
    COLOR_CHANNELS = 3
    BASE_FILTERS = 4

    def __init__(self):
        self.new_timestamp = datetime.datetime.now().strftime("%Y_%m_%d_h%Hm%M")
        self.saved_model_dir = f"gs://solargenio/custom_keras/models/model_{self.TIMESTAMP}/export/exporter/{self.NUM}/"
        self.target_model_dir = "gs://solargenio/custom_keras/models/mvp_model/14/"
        self.new_checkpoint_dir = f"gs://solargenio/custom_keras/models/model_{self.new_timestamp}/"
        self.input_shape = [self.BASE_DIM, self.BASE_DIM, self.COLOR_CHANNELS]

    def preprocess_image(self, image_buffer):
        image = tf.image.decode_png(image_buffer, channels=self.COLOR_CHANNELS)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.BASE_DIM, self.BASE_DIM], align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.cast(image, tf.float32)
        return image

    def serving_input_receiver_fn(self):
        input_ph = tf.placeholder(dtype=tf.string, shape=[None])
        images_tensor = tf.map_fn(self.preprocess_image, input_ph, back_prop=False, dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver({'input_1': images_tensor}, {'image_bytes': input_ph})

    def run_saved_model_builder(self):
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.saved_model_dir)
            graph = tf.get_default_graph()
            input_tensor = graph.get_tensor_by_name("image_bytes:0")
            output_tensor = graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
            tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
            tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tensor)
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'inputs': tensor_info_input},
                    outputs={'outputs': tensor_info_output},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
            builder = tf.saved_model.builder.SavedModelBuilder(self.target_model_dir)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={'serving_default': prediction_signature, })
            builder.save()

    def export(self):
        estimator = Model.keras_estimator(
            model_dir=self.new_checkpoint_dir,
            input_shape=self.input_shape,
            base_filters=self.BASE_FILTERS)
        estimator.export_savedmodel(self.target_model_dir, self.serving_input_receiver_fn)


if __name__ == "__main__":
    model_builder = ModelBuilder()
    #model_builder.export()
    model_builder.run_saved_model_builder()
