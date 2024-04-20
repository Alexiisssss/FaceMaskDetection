import cv2
import numpy as np
import os

class DataLoader:
    def __init__(self, with_mask_dir, without_mask_dir):
        self.with_mask_dir = with_mask_dir
        self.without_mask_dir = without_mask_dir

    def load_images(self):
        with_mask_images = self.load_images_from_folder(self.with_mask_dir)
        without_mask_images = self.load_images_from_folder(self.without_mask_dir)
        images = np.array(with_mask_images + without_mask_images)
        labels = np.array([1] * len(with_mask_images) + [0] * len(without_mask_images))
        return images, labels

    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (100, 100))
                images.append(img)
        return images