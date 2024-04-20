from sklearn.model_selection import train_test_split

class App:
    def __init__(self, data_loader, model, camera):
        self.data_loader = data_loader
        self.model = model
        self.camera = camera

    def train_model(self):
        X_train, y_train = self.data_loader.load_images()
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        history = self.model.train(X_train, y_train, validation_data=(X_test, y_test))
        return history

    def test_model(self):
        X_test, y_test = self.data_loader.load_images()
        return self.model.evaluate(X_test, y_test)