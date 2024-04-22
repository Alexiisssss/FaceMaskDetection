from model_analysis import ModelAnalyzer
from src.data_loader import DataLoader
from src.model import Model
from src.app import App
from src.camera import Camera

def main():
    data_loader = DataLoader(with_mask_dir='dataset/with_mask', without_mask_dir='dataset/without_mask')
    model = Model()
    camera = Camera(model=model)
    app = App(data_loader=data_loader, model=model, camera=camera)

    history = app.train_model()
    print("Обучение модели завершено.")

    test_loss, test_accuracy = app.test_model()
    print("Потери на тестовом наборе данных:", test_loss)
    print("Точность на тестовом наборе данных:", test_accuracy)

    #ModelAnalyzer.visualize_errors(model, data_loader)
    #ModelAnalyzer.analyze_predictions(model, data_loader)
    #ModelAnalyzer.explore_errors(model, data_loader)

    # Теперь включим камеру и проведем анализ в реальном времени
    mask_detected = camera.detect_mask()
    if mask_detected:
        print("Маска обнаружена!")
    else:
        print("Маска не обнаружена.")

if __name__ == "__main__":
    main()
