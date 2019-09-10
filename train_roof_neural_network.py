import pathlib
import argparse
import datetime
import subprocess
import numpy as np
import tensorflow as tf
from pipeline.inria_loader import InriaLoader


class RoofNeuralNetwork:
    TRAIN_DIR = "/Users/anselmosampietro/AerialImageDataset/train"
    TOTAL_N_BIG_IMAGES = 180
    DIM_BIG_IMAGES = 5000
    TEST_SIZE = 0.2
    BASE_DIM_BY_RESEARCH_TYPE = {"object_detection": 300, "instance_segmentation": 256, "custom_keras": 256}
    BATCH_SIZE_BY_RESEARCH_TYPE = {"object_detection": 24, "instance_segmentation": 1, "custom_keras": 32}

    def __init__(self, research_type: str, location: str, share_total_dataset: float, n_epochs: int, base_filters: int):
        self.research_type = research_type
        assert research_type in ["object_detection", "instance_segmentation", "custom_keras"]
        self.location = location
        assert location in ["local", "cloud"], "enter 'local' or 'cloud'"
        self.share_total_dataset = share_total_dataset
        self.share_dataset_str = str(round(self.share_total_dataset * 100))
        self.tfrecord_filenames = self.get_tfrecord_filenames()
        self.n_epochs = n_epochs
        self.base_filters = base_filters
        assert self.base_filters in [4, 8, 16, 32, 64]
        self.base_dim = self.BASE_DIM_BY_RESEARCH_TYPE[self.research_type]
        self.batch_size = self.BATCH_SIZE_BY_RESEARCH_TYPE[self.research_type]
        self.possible_splits_per_image = self.calculate_possible_splits_per_image()
        self.initial_n_splits = self.calculate_initial_n_splits()
        self.n_big_images = self.calculate_n_big_images()
        self.n_splits_per_image = self.initial_n_splits // self.n_big_images
        if self.research_type == "object_detection":
            assert self.n_splits_per_image <= self.possible_splits_per_image
        self.image_ids = np.random.choice(
            self.TOTAL_N_BIG_IMAGES, self.n_big_images, replace=False)
        self.train_dir = pathlib.Path(self.TRAIN_DIR)
        self.cwd = self.get_and_check_cwd(self.train_dir)
        self.cv_research_dir = self.cwd / "cv_research"
        self.tf_research_dir = self.cv_research_dir / "models/research"
        self.solargenio_research_dir = self.cv_research_dir / self.research_type
        self.data_dir = self.solargenio_research_dir / "data"
        self.models_dir = self.solargenio_research_dir / "models"
        self.timestamp = datetime.datetime.now().strftime("%Y_%m_%d_h%Hm%M")
        self.model_dir_name = "model_{}".format(self.timestamp)
        self.job_name = "{}_{}".format(self.research_type, self.timestamp)
        self.pipeline_config_filename = "{}_pipeline.config".format(self.location)
        self.pipeline_config_filepath = self.data_dir / self.pipeline_config_filename
        self.gcloud_data_dir_url = self.get_url_from_name_parts([self.research_type, "data"])
        self.gcloud_config_local_filepath = self.models_dir / "gcloud_config.yml"
        self.gcloud_model_dir_url = self.get_url_from_name_parts([self.research_type, "models", self.model_dir_name])

    def get_tfrecord_filenames(self):
        tfrecord_filenames = []
        tfrecord_filename_template = f"{self.share_dataset_str}_percent.tfrecord"
        for dataset_type in ["train", "test"]:
            tfrecord_filenames.append(f"{dataset_type}_{tfrecord_filename_template}")
        return tfrecord_filenames

    def calculate_possible_splits_per_image(self):
        n_splits_per_dimension = self.DIM_BIG_IMAGES // self.base_dim
        return n_splits_per_dimension**2

    def calculate_initial_n_splits(self):
        total_splits_full_dataset = self.TOTAL_N_BIG_IMAGES * self.possible_splits_per_image
        return int(total_splits_full_dataset * self.share_total_dataset)

    def calculate_n_big_images(self):
        if self.research_type == "object_detection":
            approx_n_big_images = np.sqrt(self.initial_n_splits)
        else:
            approx_n_big_images = self.TOTAL_N_BIG_IMAGES * self.share_total_dataset
        base = round(1 / self.TEST_SIZE)
        rounded_n_big_images = int(base * round(float(approx_n_big_images) / base))
        return min(self.TOTAL_N_BIG_IMAGES, max(rounded_n_big_images, base))

    def calculate_n_training_items(self):
        if self.research_type == "object_detection":
            n_test_images = int(self.n_big_images * self.TEST_SIZE)
            n_training_images = len(self.image_ids) - n_test_images
            n_training_items = n_training_images * self.n_splits_per_image
        else:
            n_training_items = 0
            assert "train" in self.tfrecord_filenames[0]
            tfrecord_filepath = self.data_dir / self.tfrecord_filenames[0]
            for _ in tf.python_io.tf_record_iterator(str(tfrecord_filepath)):
                n_training_items += 1
        return n_training_items

    @staticmethod
    def get_and_check_cwd(train_dir):
        cwd = pathlib.Path.cwd()
        cwd_from_train_dir = train_dir.parent.parent / "Documents/sg/sgrepo"
        message = "please run this script from {}".format(cwd_from_train_dir)
        assert cwd == cwd_from_train_dir, message
        return cwd

    def get_width_height_dir(self):
        csvs_dir = self.train_dir / "csvs"
        width_height_dir_name = "{}x{}".format(self.base_dim, self.base_dim)
        width_height_dir = csvs_dir / width_height_dir_name
        if not width_height_dir.exists():
            width_height_dir.mkdir()
        return width_height_dir

    def print_training_sizes(self, n_training_items, n_steps):
        if self.research_type == "object_detection":
            print(f"\n{len(self.image_ids)} total large images, each with {self.n_splits_per_image} splits")
            print(f"{n_training_items} split images in the training dataset")
            print(f"{self.n_epochs} epochs. training for {n_steps} steps\n")
        else:
            print(f"\n{len(self.image_ids)} total large images, {self.n_epochs} epochs")
            print(f"{n_training_items} training items, training for {n_steps} steps\n")

    def print_sizes_get_n_steps(self):
        n_training_items = self.calculate_n_training_items()
        n_steps = n_training_items // self.batch_size * self.n_epochs
        self.print_training_sizes(n_training_items, n_steps)
        return n_steps

    def create_train_dir(self):
        model_dir = self.models_dir / self.model_dir_name
        if not model_dir.exists():
            model_dir.mkdir()
        return model_dir

    def append_flags(self, command, value_by_flag, ):
        target_flags = ["runtime-version", "packages", "package-path", "module-name", "region", "config", ""]
        for flag, value in value_by_flag.items():
            if self.location == "cloud" and flag in target_flags:
                command.append("--{}".format(flag))
                if value is not None:
                    command.append(str(value))
            else:
                command.append("--{}={}".format(flag, str(value)))
        return command

    @staticmethod
    def run_command_print_output(command, cwd=None):
        if cwd is not None:
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE)
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE)
        output = process.stdout.read().strip()
        print(output)

    def train_locally(self, n_steps):
        model_dir = self.create_train_dir()
        value_by_flag = {
            "model_dir": model_dir,
            "pipeline_config_path": self.pipeline_config_filepath,
            "num_train_steps": n_steps}
        script = self.tf_research_dir / "object_detection/model_main.py"
        command = ["python", str(script)]
        command = self.append_flags(command, value_by_flag)
        command.append("--logtostderr")
        self.run_command_print_output(command)

    @staticmethod
    def get_url_from_name_parts(name_parts):
        directory = "/".join(name_parts)
        return "gs://solargenio/{}".format(directory)

    def copy_local_files_to_bucket(self):
        for filename in self.tfrecord_filenames:
            local_filepath = self.data_dir / filename
            print(f"copying {str(local_filepath)} to {str(self.gcloud_data_dir_url)}")
            copy_data_command = [
                "gsutil", "-o", "GSUtil:parallel_composite_upload_threshold=50M",
                "cp", str(local_filepath), str(self.gcloud_data_dir_url)]
            self.run_command_print_output(copy_data_command)

    def train_pretrained_model_in_the_cloud(self, n_steps):
        gcloud_pipeline_config_url = self.gcloud_data_dir_url + "/" + self.pipeline_config_filename
        value_by_flag = {
            "runtime-version": 1.9,
            "job-dir": self.gcloud_model_dir_url,
            "packages": "dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz",
            "module-name": "object_detection.model_main",
            "region": "europe-west1",
            "config": self.gcloud_config_local_filepath,
            "": None,
            "model_dir": self.gcloud_model_dir_url,
            "pipeline_config_path": gcloud_pipeline_config_url,
            "num_train_steps": n_steps}
        train_command = ["gcloud", "ml-engine", "jobs", "submit", "training", self.job_name]
        train_command = self.append_flags(train_command, value_by_flag)
        self.run_command_print_output(train_command, cwd=str(self.tf_research_dir))

    def train_custom_model_in_the_cloud(self):
        custom_keras_model_dir = self.cwd / "custom_keras_model"
        value_by_flag = {
            "runtime-version": 1.9,
            "job-dir": self.gcloud_model_dir_url,
            "module-name": "trainer.task",
            "package-path": "trainer",
            "region": "europe-west1",
            "config": self.gcloud_config_local_filepath,
            "": None,
            "model_dir": self.gcloud_model_dir_url,
            "share_dataset": self.share_total_dataset,
            "n_epochs": self.n_epochs,
            "base_filters": self.base_filters,
            "batch_size": self.batch_size}
        train_command = ["gcloud", "ml-engine", "jobs", "submit", "training", self.job_name]
        train_command = self.append_flags(train_command, value_by_flag)
        self.run_command_print_output(train_command, cwd=str(custom_keras_model_dir))

    def train(self):
        print("\n ----- TRAINING {} NEURAL NETWORK ----- \n".format(self.research_type.upper()))
        width_height_dir = self.get_width_height_dir()
        inria_loader = InriaLoader(
            self.research_type, self.data_dir, width_height_dir,
            self.image_ids, self.TEST_SIZE, self.share_total_dataset)
        new_tfrecord_files = input("Do you want to create new training and test tfrecord files? (Y/n) ")
        if new_tfrecord_files == "Y":
            inria_loader.prepare_tfrecord_files(self.n_splits_per_image, self.batch_size)
        n_steps = self.print_sizes_get_n_steps()
        if self.location == "local":
            self.train_locally(n_steps)
        else:
            if new_tfrecord_files == "Y":
                self.copy_local_files_to_bucket()
            if self.research_type in ["object_detection", "instance_segmentation"]:
                self.train_pretrained_model_in_the_cloud(n_steps)
            else:
                self.train_custom_model_in_the_cloud()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("research_type", type=str)
    parser.add_argument("location", type=str, nargs='?', default="local")
    parser.add_argument("share_total_dataset", nargs='?', type=float, default=0.01)
    parser.add_argument("n_epochs", type=int, nargs='?', default=1)
    parser.add_argument("base_filters", type=int, nargs='?', default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    roof_neural_network = RoofNeuralNetwork(**vars(arguments))
    roof_neural_network.train()
