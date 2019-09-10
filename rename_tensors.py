import tensorflow as tf


TIMESTAMP = "2019_03_04_h11m28"
NUM = "1551701418"
SAVED_MODEL_DIR = f"gs://solargenio/custom_keras/models/model_{TIMESTAMP}/export/exporter/{NUM}/"
TARGET_MODEL_DIR = "gs://solargenio/custom_keras/models/mvp_model/saved_model_v8/"


def preprocess_image(image_buffer):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def rename_tensors():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], SAVED_MODEL_DIR)
        graph = tf.get_default_graph()
        #input_tensor = graph.get_tensor_by_name("input_image:0")
        output_tensor = graph.get_tensor_by_name("conv2d_23/Sigmoid:0")
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {
            'image/encoded': tf.FixedLenFeature(
                shape=[], dtype=tf.string),
        }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        jpegs = tf_example['image/encoded']
        images = tf.map_fn(preprocess_image, jpegs, dtype=tf.float32)
        input_tensor = tf.identity(images, name='input_image')
        tensor_info_input = tf.saved_model.utils.build_tensor_info(input_tensor)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(output_tensor)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'image_bytes': tensor_info_input},
                outputs={'output_bytes': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        builder = tf.saved_model.builder.SavedModelBuilder(TARGET_MODEL_DIR)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_images': prediction_signature, })
        builder.save()


if __name__ == "__main__":
    rename_tensors()
