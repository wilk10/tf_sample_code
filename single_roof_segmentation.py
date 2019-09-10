import os
import cv2
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from train_roof_neural_network import RoofNeuralNetwork
from pipeline.inria.ground_truths import InriaGroundTruths
from pipeline.inria.image_preparation import ImagePreparation
from pipeline.keras_models.image_segmentation import FullUnet


class SingleRoofSegmentation:
    TRAIN_DIR = "/Users/anselmosampietro/AerialImageDataset/train"
    TOTAL_N_BIG_IMAGES = 180
    DIM_BIG_IMAGES = 5000
    BASE_DIM = 256
    TEST_SIZE = 0.2
    BATCH_SIZE = 4
    MODEL = FullUnet

    def __init__(self, location, share_total_dataset, n_epochs):
        self.location = location  # will be used later for cloud training
        assert location in ["local", "cloud"], "enter 'local' or 'cloud'"
        self.share_total_dataset = share_total_dataset
        self.n_epochs = n_epochs
        self.n_big_images = self.calculate_n_big_images()
        self.image_ids = np.random.choice(
            self.TOTAL_N_BIG_IMAGES, self.n_big_images, replace=False)
        self.train_dir = pathlib.Path(self.TRAIN_DIR)
        self.images_tif_dir = self.train_dir / "images"
        self.ground_truth_dir = self.train_dir / "gt"
        filenames = os.listdir(self.images_tif_dir)
        self.sampled_filenames = [filenames[i] for i in self.image_ids]
        self.width_height_dir = RoofNeuralNetwork.get_width_height_dir(self.train_dir, self.BASE_DIM)
        self.predict_images_dir = pathlib.Path.home() / "Documents/sg/images/test_images_zoom20/test"

    def calculate_n_big_images(self):
        approx_n_big_images = self.TOTAL_N_BIG_IMAGES * self.share_total_dataset
        base = round(1 / self.TEST_SIZE)
        rounded_n_big_images = int(base * round(float(approx_n_big_images) / base))
        return min(self.TOTAL_N_BIG_IMAGES, max(rounded_n_big_images, base))

    @staticmethod
    def get_single_roof_images(ground_truth, base_dim):
        colour_and_gt_images = {"colour": [], "gt": []}
        single_roof_images = ground_truth.match_isolated_colour_and_gt_roofs("custom_keras")
        for single_roof_image_dict in single_roof_images:
            gt_image_path = single_roof_image_dict["gt_image_path"]
            gt_image = cv2.imread(str(gt_image_path), cv2.IMREAD_COLOR)
            resized_gt_image = ImagePreparation.resize_image(gt_image, base_dim)
            reshaped_gt_image = np.expand_dims(cv2.cvtColor(resized_gt_image, cv2.COLOR_BGR2GRAY), axis=3)
            colour_and_gt_images["gt"].append(reshaped_gt_image)
            colour_image_path = single_roof_image_dict["colour_image_path"]
            colour_image = cv2.imread(str(colour_image_path), cv2.IMREAD_COLOR)
            resized_colour_image = ImagePreparation.resize_image(colour_image, base_dim)
            colour_and_gt_images["colour"].append(resized_colour_image)
        return colour_and_gt_images

    def prepare_image_datasets(self):
        n_test_ids = int(len(self.image_ids) * self.TEST_SIZE)
        test_filenames = np.random.choice(self.sampled_filenames, n_test_ids, replace=False)
        train_filenames = [
            filename for filename in self.sampled_filenames
            if filename not in test_filenames]
        filenames_by_dataset = {"train": train_filenames, "test": test_filenames}
        images_by_dataset_by_type = {"train": {"colour": [], "gt": []}, "test": {"colour": [], "gt": []}}
        final_images_by_dataset_by_type = {"train": {"colour": [], "gt": []}, "test": {"colour": [], "gt": []}}
        for dataset, filenames in filenames_by_dataset.items():
            for i, tif_filename in enumerate(filenames):
                ground_truth = InriaGroundTruths(
                    width_height_dir=self.width_height_dir,
                    tif_filename=tif_filename)
                images_by_type = self.get_single_roof_images(ground_truth, self.BASE_DIM)
                for image_type, images in images_by_type.items():
                    images_by_dataset_by_type[dataset][image_type].extend(images)
                print(f"{dataset} {i + 1}/{len(filenames)}. {tif_filename}: {len(images_by_type['colour'])} roofs")
            n_dataset_images = len(images_by_dataset_by_type[dataset]["colour"])
            extra_images = n_dataset_images % self.BATCH_SIZE
            max_n_images = n_dataset_images - extra_images
            assert max_n_images % self.BATCH_SIZE == 0
            for image_type in ["colour", "gt"]:
                final_images_by_dataset_by_type[dataset][image_type] = \
                    images_by_dataset_by_type[dataset][image_type][:max_n_images]
            print(f"{dataset} dataset: {max_n_images} images in {max_n_images // self.BATCH_SIZE} batches")
        return final_images_by_dataset_by_type

    def prepare_predict_images(self):
        image_names = os.listdir(str(self.predict_images_dir))
        images = []
        for image_name in image_names:
            image_path = self.predict_images_dir / image_name
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            resized_image = ImagePreparation.resize_image(image, self.BASE_DIM)
            images.append(resized_image)
        return np.array(images)

    @staticmethod
    def show_predictions(predict_images, predictions):
        thresholds = [0.7, 0.6, 0.5, 0.4, 0.3]
        zipped_images = zip(predict_images, predictions)
        for original, prediction in zipped_images:
            masked_predictions = []
            for threshold in thresholds:
                masked_prediction = (prediction > threshold).astype(np.uint8)
                masked_predictions.append(masked_prediction)
            fig, axarr = plt.subplots(1, len(thresholds)+1)
            axarr[0].imshow(original)
            for i, masked_prediction in enumerate(masked_predictions):
                axarr[i+1].set_title(thresholds[i])
                axarr[i+1].imshow(masked_prediction.squeeze())
            plt.show()

    def run(self):
        predict_images = self.prepare_predict_images()
        image_arrays_by_dataset = self.prepare_image_datasets()
        train_images = np.array(image_arrays_by_dataset["train"]["colour"])
        test_images = np.array(image_arrays_by_dataset["test"]["colour"])
        train_gt = np.array(image_arrays_by_dataset["train"]["gt"])
        test_gt = np.array(image_arrays_by_dataset["test"]["gt"])
        input_shape = train_images.shape[1:]
        model = self.MODEL.declare(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(train_images, train_gt, batch_size=self.BATCH_SIZE, epochs=self.n_epochs)
        test_loss, test_acc = model.evaluate(test_images, test_gt)
        print('Test accuracy:', test_acc)
        predictions = model.predict(predict_images)
        self.show_predictions(predict_images, predictions)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str, nargs='?', default="local")
    parser.add_argument("share_total_dataset", nargs='?', type=float, default=0.01)
    parser.add_argument("n_epochs", type=int, nargs='?', default=1)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    single_roof_segmentation = SingleRoofSegmentation(**vars(arguments))
    single_roof_segmentation.run()
