import cv2
import numpy as np
import os

class Camera:
    def __init__(self, model):
        self.capture = cv2.VideoCapture(0)  # Используем камеру по умолчанию (0 - это индекс камеры)
        self.model = model.model  # Получаем модель из объекта класса Model

    def detect_mask(self):
        # Загрузка предварительно обученного каскадного классификатора для обнаружения лиц
        face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')

        while True:
            ret, frame = self.capture.read()  # Захватываем кадр с камеры

            # Преобразуем кадр в черно-белый для обнаружения лиц
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Обнаружение лиц на кадре
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Обработка каждого обнаруженного лица
            for (x, y, w, h) in faces:
                # Обрезаем область лица из кадра
                face_img = frame[y:y+h, x:x+w]

                # Изменяем размер изображения лица на размер, который принимает модель
                face_img = cv2.resize(face_img, (100, 100))

                # Преобразуем изображение лица в формат, подходящий для модели (добавляем измерение канала)
                face_img = np.expand_dims(face_img, axis=0)

                # Нормализация значений пикселей
                face_img = face_img.astype('float32') / 255.0

                # Классификация лица на наличие маски с использованием модели
                prediction = self.model.predict(face_img)

                # Отображение результатов классификации на кадре
                label = "Маска" if prediction > 0.5 else "Без маски"
                color = (0, 255, 0) if label == "Маска" else (0, 0, 255)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Выводим промежуточные результаты
                print(f"Координаты лица: ({x}, {y}), Размер: {w}x{h}, Класс: {label}, Вероятность: {prediction}")

            # Показываем кадр в окне
            cv2.imshow('Frame', frame)

            # Ожидаем нажатия клавиши 'q' для выхода из цикла
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем захват видеопотока и закрываем окна OpenCV
        self.capture.release()
        cv2.destroyAllWindows()
