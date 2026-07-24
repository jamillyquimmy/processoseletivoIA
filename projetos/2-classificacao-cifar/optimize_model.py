import os
import tensorflow as tf

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Modelo otimizado salvo em: {tflite_path}")

if __name__ == "__main__":
    main()