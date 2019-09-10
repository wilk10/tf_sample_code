import argparse
import tensorflow as tf
from .model import Model
tf.logging.set_verbosity(tf.logging.INFO)


class Task:
    BASE_DIM = 256
    COLOR_CHANNELS = 3
    GT_CHANNELS = 1

    def __init__(self, model_dir, share_dataset, n_epochs, base_filters, batch_size):
        self.model_dir = model_dir
        self.share_dataset = share_dataset
        self.share_dataset_str = str(int(self.share_dataset * 100))
        self.n_epochs = n_epochs
        self.base_filters = base_filters
        self.batch_size = batch_size
        self.data_filepath_template = "gs://solargenio/custom_keras/data/{}_{}_percent.tfrecord"
        self.train_file = self.data_filepath_template.format("train", self.share_dataset_str)
        self.test_file = self.data_filepath_template.format("test", self.share_dataset_str)
        self.n_training_items = self.get_n_items(self.train_file)
        self.n_test_items = self.get_n_items(self.test_file)
        self.n_training_steps = self.n_training_items // self.batch_size * self.n_epochs
        self.n_test_steps = self.n_test_items // self.batch_size
        self.features = {
            "filename": tf.FixedLenFeature([], tf.string),
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "gt_depth": tf.FixedLenFeature([], tf.int64),
            "roof_depth": tf.FixedLenFeature([], tf.int64),
            "roof_image_raw": tf.FixedLenFeature([], tf.string),
            "gt_image_raw": tf.FixedLenFeature([], tf.string)}

    def get_n_items(self, tfrecord_filepath):
        n_items = 0
        for _ in tf.python_io.tf_record_iterator(str(tfrecord_filepath)):
            n_items += 1
        return n_items

    def process_image_from_tf_example(self, image_str_tensor, n_channels=3):
        image = tf.image.decode_image(image_str_tensor)
        image.set_shape([self.BASE_DIM, self.BASE_DIM, n_channels])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def read_and_decode(self, serialized):
        parsed_example = tf.parse_single_example(serialized=serialized, features=self.features)
        roof_image = self.process_image_from_tf_example(parsed_example["roof_image_raw"], self.COLOR_CHANNELS)
        gt_image = self.process_image_from_tf_example(parsed_example["gt_image_raw"], self.GT_CHANNELS)
        return roof_image, gt_image

    def input_fn(self, filepath, mode):
        dataset = tf.data.TFRecordDataset(filepath)
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(self.n_training_items, self.n_epochs))
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(self.read_and_decode, self.batch_size, num_parallel_batches=8))
            dataset = dataset.prefetch(2)
        if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
            dataset = dataset.apply(
                tf.contrib.data.map_and_batch(self.read_and_decode, self.batch_size, num_parallel_batches=8))
        iterator = dataset.make_one_shot_iterator()
        roof_images, gt_images = iterator.get_next()
        return roof_images, gt_images

    def process_image_from_buffer(self, image_buffer):
        image = tf.image.decode_png(image_buffer, channels=self.COLOR_CHANNELS)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [self.BASE_DIM, self.BASE_DIM], align_corners=False)
        image = tf.squeeze(image, [0])
        image = tf.cast(image, tf.float32)
        return image

    def serving_input_receiver_fn(self):
        input_ph = tf.placeholder(dtype=tf.string, shape=[None])
        images_tensor = tf.map_fn(self.process_image_from_buffer, input_ph, back_prop=False, dtype=tf.float32)
        return tf.estimator.export.ServingInputReceiver({'input_1': images_tensor}, {'image_bytes': input_ph})

    def train_and_evaluate(self):
        input_shape = [self.BASE_DIM, self.BASE_DIM, self.COLOR_CHANNELS]
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: self.input_fn(self.train_file, mode=tf.estimator.ModeKeys.TRAIN),
            max_steps=self.n_training_steps)
        exporter = tf.estimator.LatestExporter("exporter", self.serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: self.input_fn(self.test_file, mode=tf.estimator.ModeKeys.EVAL),
            steps=self.n_test_steps,
            exporters=[exporter],
            start_delay_secs=10,
            throttle_secs=10)
        estimator = Model.keras_estimator(
            model_dir=self.model_dir,
            input_shape=input_shape,
            base_filters=self.base_filters)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--share_dataset", type=float)
    parser.add_argument("--n_epochs", type=int, nargs='?', default=1)
    parser.add_argument("--base_filters", type=int, nargs='?', default=4)
    parser.add_argument("--batch_size", type=int, nargs='?', default=4)
    parser.add_argument("--job-dir", type=str, nargs='?', default="")
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    delattr(arguments, 'job_dir')
    tf.logging.set_verbosity('INFO')
    task = Task(**vars(arguments))
    task.train_and_evaluate()
