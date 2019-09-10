import os
import cv2
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pipeline.inria.image_preparation import ImagePreparation
from pipeline.keras_models.image_segmentation import SimplifiedUnet


class CustomRoofSegmentation:
    TRAIN_DIR = "/Users/anselmosampietro/AerialImageDataset/train"
    BASE_DIM = 256
    N_IMAGES = 1
    IMAGE_IDS = np.random.randint(0, 180, N_IMAGES)
    MODEL = SimplifiedUnet
    EPOCHS = 1
    BATCH_SIZE = 16

    def __init__(self):
        self.train_dir_path = pathlib.Path(self.TRAIN_DIR)
        self.images_dir = self.train_dir_path / "images"
        self.ground_truth_dir = self.train_dir_path / "gt"
        self.predict_images_dir = pathlib.Path.home() / "Documents/sg/images/test_images"
        self.train_images = self.load_images_as_nparray(self.images_dir, "train")
        self.ground_truth_images = self.load_images_as_nparray(self.ground_truth_dir, "ground_truth")
        self.predict_images = self.load_images_as_nparray(self.predict_images_dir, "predict_images")

    def prepare_predict_image(self, image):
        for axis in [0, 1]:
            message = "the image size is smaller than {}, feed a larger image".format(self.BASE_DIM)
            assert image.shape[axis] >= self.BASE_DIM, message
            n_pixels_to_crop = image.shape[axis] - self.BASE_DIM
            id = int(n_pixels_to_crop / 2)
            image = image.take(indices=range(id, image.shape[axis]-id), axis=axis)
        assert image.shape[0] == image.shape[1]
        if self.MODEL.__name__ == "PVclassifier":
            expanded_image = np.expand_dims(image, axis=0)
            enhanced_image = self.MODEL.add_relative_luminescences(expanded_image)
            image = np.squeeze(enhanced_image, axis=0)
        return image

    def load_images_as_nparray(self, directory, flag):
        all_images = []
        counter = 0
        filenames = os.listdir(directory)
        if flag in ["train", "ground_truth"]:
            target_filenames = [filenames[i] for i in self.IMAGE_IDS]
        else:
            assert flag == "predict_images"
            target_filenames = [filename for filename in filenames if filename[0] != "."]
        for filename in target_filenames:
            filepath = directory / filename
            image = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
            if flag in ["train", "ground_truth"]:
                ready_images = ImagePreparation.split(self.BASE_DIM, image)
                if flag == "ground_truth":
                    ready_images = np.expand_dims(ready_images, axis=3)
                elif self.MODEL.__name__ == "PVclassifier":
                    ready_images = self.MODEL.add_relative_luminescences(ready_images)
            else:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                ready_images = self.prepare_predict_image(bgr_image)
            regolarized_images = ready_images / 255.0
            counter += 1
            print("{}: loaded {}/{} images".format(flag, counter, len(target_filenames)))
            if flag in ["train", "ground_truth"]:
                all_images.extend(regolarized_images)
            else:
                all_images.append(regolarized_images)
        return np.array(all_images)

    def check_images(self):
        ids = np.random.randint(0, len(self.train_images), 2)
        fig, axarr = plt.subplots(2, 2)
        axarr[0, 0].imshow(self.train_images[ids[0]])
        axarr[0, 1].imshow(self.ground_truth_images[ids[0]].squeeze())
        axarr[1, 0].imshow(self.train_images[ids[1]])
        axarr[1, 1].imshow(self.ground_truth_images[ids[1]].squeeze())
        plt.show()

    def show_predictions(self, predictions):
        thresholds = [0.5, 0.4, 0.3]
        zipped_images = zip(self.predict_images, predictions)
        for original, prediction in zipped_images:
            masked_predictions = []
            for threshold in thresholds:
                masked_prediction = (prediction > threshold).astype(np.uint8)
                masked_predictions.append(masked_prediction)
            fig, axarr = plt.subplots(1, 4)
            if self.MODEL.__name__ == "PVclassifier":
                original = original[:, :, 0:3]
            axarr[0].imshow(original)
            for i, masked_prediction in enumerate(masked_predictions):
                axarr[i+1].imshow(masked_prediction.squeeze())
            plt.show()

    def run(self):
        #self.check_images()
        train_images, test_images, train_gt, test_gt = train_test_split(
            self.train_images, self.ground_truth_images, test_size=0.2, random_state=42)
        print(train_images.shape)
        print(test_images.shape)
        print(train_gt.shape)
        print(test_gt.shape)
        print(self.predict_images.shape)
        input_shape = train_images.shape[1:]
        model = self.MODEL.declare(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        model.fit(train_images, train_gt, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)
        test_loss, test_acc = model.evaluate(test_images, test_gt)
        print('Test accuracy:', test_acc)
        predictions = model.predict(self.predict_images)
        self.show_predictions(predictions)


if __name__ == "__main__":
    CustomRoofSegmentation().run()
