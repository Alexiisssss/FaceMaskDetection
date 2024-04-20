#Face Mask Detection

Этот проект представляет собой систему обнаружения наличия медицинских масок на лицах людей с использованием компьютерного зрения и глубокого обучения.


#Описание

Данная работа разработана в качестве учебного проекта для обнаружения, надета ли у человека медицинская маска. Для более серьезный условий, в частности где требуется соблюдение мер безопасности, таких как общественные места, медицинские учреждения и т.д. и там где система может быть интегрирована в видеокамеры или существующие системы безопасности для автоматического мониторинга и контроля потрбуется более глубокое обучение модели.


#Функции

Обнаружение лиц на изображениях с использованием каскадов Хаара.
Классификация лиц на наличие маски с помощью глубокой нейронной сети.
Визуализация результатов обнаружения масок на лицах.
Тренировка модели на собственном наборе данных для улучшения качества обнаружения.


#Установка

1.Клонируйте репозиторий с помощью команды:

git clone https://github.com/Alexiisssss/FaceMaskDetection.git

2.Установите необходимые зависимости:

-opencv-python==4.5.3
-numpy==1.21.2
-tensorflow==2.6.0
-scikit-learn==0.24.2


#Использование

1.Запустите приложение, чтобы обучить модель:

python main.py train

2.Запустите приложение для тестирования модели:

python main.py test

3.Запустите камеру для обнаружения масок в реальном времени:

python main.py camera


#Вклад

Проект открыт для вклада и улучшения. Не стесняйтесь отправлять pull request с предложенными изменениями.
