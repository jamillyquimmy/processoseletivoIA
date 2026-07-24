import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation"
    )

    inputs = keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Iniciando treinamento do modelo...")
    model.fit(
        x_train, y_train,
        epochs=25,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
l
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nAcurácia no conjunto de teste: {test_acc * 100:.2f}%")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "model.h5")
    model.save(save_path)
    print(f"Modelo salvo com sucesso em: {save_path}")

if __name__ == "__main__":
    main()