import os
import cv2
import pandas
import pathlib
import numpy as np
import tensorflow as tf
from .inria.image_preparation import ImagePreparation
from .inria.ground_truths import InriaGroundTruths


class InriaLoader:
    TRAIN_DIR = "/Users/anselmosampietro/AerialImageDataset/train"
    EXTENSION = "tfrecord"

    def __init__(self, research_type, data_dir, width_height_dir, image_ids, test_size, share_total_dataset):
        self.research_type = research_type
        assert research_type in ["object_detection", "instance_segmentation", "custom_keras"]
        self.data_dir = data_dir
        self.width_height_dir = width_height_dir
        self.image_ids = image_ids
        self.test_size = test_size
        self.share_total_dataset = share_total_dataset
        self.share_dataset_str = str(round(self.share_total_dataset * 100))
        self.train_dir = pathlib.Path(self.TRAIN_DIR)
        self.images_tif_dir = self.train_dir / "images"
        self.ground_truth_dir = self.train_dir / "gt"
        filenames = os.listdir(self.images_tif_dir)
        self.sampled_filenames = [filenames[i] for i in self.image_ids]

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    @staticmethod
    def remove_nans_from_features_dict_if_no_box(df, features_dict):
        assert len(features_dict["label"]) >= 1
        if np.isnan(features_dict["label"][0]):
            last_columns = df.columns.values[-6:]
            for column in last_columns:
                features_dict[column] = []
        else:
            features_dict["label"] = [int(label) for label in features_dict["label"]]
        return features_dict

    def make_example_for_pretrained_model(self, image_bytes, df, gt_images_bytes=None):
        features_dict = df.to_dict(orient="list")
        self.remove_nans_from_features_dict_if_no_box(df, features_dict)
        labels = [label.encode('utf8') for label in features_dict["class"]]
        filename = features_dict["filename"][0]
        image_format = filename.split(".")[-1]
        example_dict = {
            "image/width": self.int64_feature(features_dict["width"][0]),
            "image/height": self.int64_feature(features_dict["height"][0]),
            "image/filename": self.bytes_feature(filename.encode('utf8')),
            'image/source_id': self.bytes_feature(filename.encode('utf8')),
            "image/encoded": self.bytes_feature(image_bytes),
            'image/format': self.bytes_feature(image_format.encode('utf8')),
            "image/object/bbox/xmin": self.float_list_feature(features_dict["xmin"]),
            "image/object/bbox/ymin": self.float_list_feature(features_dict["ymin"]),
            "image/object/bbox/xmax": self.float_list_feature(features_dict["xmax"]),
            "image/object/bbox/ymax": self.float_list_feature(features_dict["ymax"]),
            "image/object/class/label": self.int64_list_feature(features_dict["label"]),
            "image/object/class/text": self.bytes_list_feature(labels)}
        if gt_images_bytes is not None:
            example_dict['image/object/mask'] = (self.bytes_list_feature(gt_images_bytes))
        return tf.train.Example(features=tf.train.Features(feature=example_dict))

    def make_example_for_custom_keras(self, filename, gt_image_shape, roof_image_bytes, gt_image_bytes):
        height, width, channel = gt_image_shape
        assert channel == 1
        roof_image_channels = channel * 3
        example_dict = {
            "filename": self.bytes_feature(filename.encode('utf8')),
            "height": self.int64_feature(height),
            "width": self.int64_feature(width),
            "gt_depth": self.int64_feature(channel),
            "roof_depth": self.int64_feature(roof_image_channels),
            "roof_image_raw": self.bytes_feature(roof_image_bytes),
            "gt_image_raw": self.bytes_feature(gt_image_bytes)}
        return tf.train.Example(features=tf.train.Features(feature=example_dict))

    def split_image_make_examples(self, ground_truth, n_splits):
        df = ground_truth.get_df_object_detection()
        base_dim = int(self.width_height_dir.name.split("x")[0])
        image_colour = ground_truth.load_full_image("colour")
        split_images = ImagePreparation.split(base_dim, image_colour)
        sampled_split_ids = np.random.choice(len(split_images), n_splits, replace=False)
        examples = []
        for split_id in sampled_split_ids:
            split_image = split_images[split_id]
            split_image_df = df.loc[df['split_id'] == split_id]
            image_path = ground_truth.get_split_or_single_image_filepath(split_id, "split")
            if not image_path.exists():
                cv2.imwrite(str(image_path), split_image)
            split_image_bytes = tf.gfile.GFile(str(image_path), 'rb').read()
            example = self.make_example_for_pretrained_model(split_image_bytes, split_image_df)
            examples.append(example)
        return examples

    @staticmethod
    def check_gt(gt_image, df, ground_truth_object):
        box = ground_truth_object.convert_from_df_row_to_coordinate_box(df.iloc[0, :])
        (x1, y1), (x2, y2) = box
        new_image = gt_image.copy()
        cv2.rectangle(new_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), -1)
        return new_image.sum() == 0

    def get_single_roofs_make_examples(self, ground_truth):
        single_roof_images = ground_truth.match_isolated_colour_and_gt_roofs(self.research_type)
        examples = []
        all_single_roofs_df = pandas.DataFrame(columns=ground_truth.COLUMNS)
        for single_roof_image_dict in single_roof_images:
            gt_image_path = single_roof_image_dict["gt_image_path"]
            gt_image = cv2.imread(str(gt_image_path), cv2.IMREAD_COLOR)
            colour_image_path = single_roof_image_dict["colour_image_path"]
            colour_image_bytes = tf.gfile.GFile(str(colour_image_path), 'rb').read()
            if self.research_type == "instance_segmentation":
                single_roof_df = ground_truth.get_df_with_image_data(gt_image)
                assert len(single_roof_df) == 1
                all_single_roofs_df = all_single_roofs_df.append(single_roof_df)
                cropped_gt_image_path = single_roof_image_dict["cropped_gt_image_path"]
                no_white_outside_of_box = self.check_gt(gt_image, single_roof_df, ground_truth)
                assert no_white_outside_of_box
                cropped_gt_image_bytes = tf.gfile.GFile(str(cropped_gt_image_path), 'rb').read()
                example = self.make_example_for_pretrained_model(
                    colour_image_bytes, single_roof_df, [cropped_gt_image_bytes])
            else:
                reshaped_gt_image = np.expand_dims(cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY), axis=3)
                gt_image_shape = reshaped_gt_image.shape
                gt_image_bytes = tf.gfile.GFile(str(gt_image_path), 'rb').read()
                example = self.make_example_for_custom_keras(
                    ground_truth.png_filename, gt_image_shape, colour_image_bytes, gt_image_bytes)
            examples.append(example)
        return examples

    def prepare_tfrecord_files(self, n_splits, batch_size):
        n_test_ids = int(len(self.image_ids) * self.test_size)
        test_filenames = np.random.choice(self.sampled_filenames, n_test_ids, replace=False)
        train_filenames = [
            filename for filename in self.sampled_filenames
            if filename not in test_filenames]
        filenames_by_dataset = {"train": train_filenames, "test": test_filenames}
        for dataset, filenames in filenames_by_dataset.items():
            tfrecord_filename = f"{dataset}_{self.share_dataset_str}_percent.{self.EXTENSION}"
            tfrecord_filepath = self.data_dir / tfrecord_filename
            writer = tf.python_io.TFRecordWriter(str(tfrecord_filepath))
            dataset_examples = []
            for i, tif_filename in enumerate(filenames):
                ground_truth = InriaGroundTruths(
                    width_height_dir=self.width_height_dir,
                    tif_filename=tif_filename)
                if self.research_type == "object_detection":
                    full_image_examples = self.split_image_make_examples(ground_truth, n_splits)
                else:
                    full_image_examples = self.get_single_roofs_make_examples(ground_truth)
                dataset_examples.extend(full_image_examples)
                print(f"{dataset}: {i + 1}/{len(filenames)}: {tif_filename} loaded")
            extra_examples = len(dataset_examples) % batch_size
            max_dataset_examples = len(dataset_examples) - extra_examples
            assert max_dataset_examples % batch_size == 0
            for example in dataset_examples[:max_dataset_examples]:
                writer.write(example.SerializeToString())
            writer.close()
            print(f"{dataset}: {max_dataset_examples} examples. {max_dataset_examples // batch_size} steps per epoch. "
                  f"{max_dataset_examples % batch_size} leftout examples")
