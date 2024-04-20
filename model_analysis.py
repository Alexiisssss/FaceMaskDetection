
import cv2
import numpy as np

class ModelAnalyzer:
    @staticmethod
    def visualize_errors(model, data_loader):
        images, labels = data_loader.load_images()
        predictions = model.model.predict(images)

        # Находим индексы изображений, где произошла ошибка
        error_indices = np.where((predictions > 0.5) != labels)[0]

        # Выводим изображения, где модель ошиблась
        for index in error_indices:
            image = images[index]
            label = "Маска" if predictions[index] > 0.5 else "Без маски"
            cv2.imshow('Ошибка: ' + label, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def analyze_predictions(model, data_loader):
        images, labels = data_loader.load_images()
        predictions = model.model.predict(images)

        for i in range(len(images)):
            image = images[i]
            label = "Маска" if predictions[i] > 0.5 else "Без маски"
            probability = predictions[i]
            print(f"Изображение {i + 1}: {label}, Вероятность: {probability}")

    @staticmethod
    def explore_errors(model, data_loader):
        images, labels = data_loader.load_images()
        predictions = model.model.predict(images)

        for i in range(len(images)):
            image = images[i]
            label = "Маска" if predictions[i] > 0.5 else "Без маски"
            true_label = "Маска" if labels[i] == 1 else "Без маски"
            if label != true_label:
                cv2.imshow('Ошибка: Модель считает ' + label + ', но на самом деле ' + true_label, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
