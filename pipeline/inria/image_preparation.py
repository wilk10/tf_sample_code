import cv2
import numpy as np


class ImagePreparation:

    @staticmethod
    def split(dim, image):
        assert image.shape[0] == image.shape[1]
        n_pixels_to_crop = image.shape[0] % dim
        id = int(n_pixels_to_crop / 2)
        cropped_image = image.take(indices=range(id, image.shape[0] - id), axis=0)
        cropped_image = cropped_image.take(indices=range(id, image.shape[1] - id), axis=1)
        assert cropped_image.shape[0] == cropped_image.shape[1]
        n_divisions = int(cropped_image.shape[0] / dim)
        split_images = []
        for i in range(n_divisions):
            for j in range(n_divisions):
                split_image = cropped_image.take(indices=range(i * dim, (i + 1) * dim), axis=0)
                split_image = split_image.take(indices=range(j * dim, (j + 1) * dim), axis=1)
                split_images.append(split_image)
        return np.array(split_images)

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

    @staticmethod
    def crop_on_box(image, box, margin=0):
        (x1, y1), (x2, y2) = box
        height, width, _ = image.shape
        new_x1 = max(0, x1 - margin)
        new_x2 = min(x2 + margin + 1, width)
        new_y1 = max(0, y1 - margin)
        new_y2 = min(y2 + margin + 1, height)
        roof_image = image.take(indices=range(new_y1, new_y2), axis=0)
        return roof_image.take(indices=range(new_x1, new_x2), axis=1)
