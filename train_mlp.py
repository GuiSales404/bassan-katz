import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np

def writeNNet(weights, biases, inputMins, inputMaxes, means, ranges, fileName):
    '''
    Write network data to the .nnet file format.

    Args:
        weights (list): Weight matrices in the network order 
        biases (list): Bias vectors in the network order
        inputMins (list): Minimum values for each input
        inputMaxes (list): Maximum values for each input
        means (list): Mean values for each input and a mean value for all outputs. Used to normalize inputs/outputs
        ranges (list): Range values for each input and a range value for all outputs. Used to normalize inputs/outputs
        fileName (str): File where the network will be written
    '''
    try:
        # Validate dimensions of weights and biases
        assert len(weights) == len(biases), "Number of weight matrices and bias vectors must match."
        
        # Open the file we wish to write
        with open(fileName, 'w') as f2:
            f2.write("// Neural Network File Format\n")

            numLayers = len(weights)
            inputSize = weights[0].shape[1]
            outputSize = len(biases[-1])
            maxLayerSize = max(inputSize, max(len(b) for b in biases))

            # Write network architecture info
            f2.write(f"{numLayers},{inputSize},{outputSize},{maxLayerSize},\n")
            f2.write(f"{inputSize}," + ",".join(str(len(b)) for b in biases) + ",\n")
            f2.write("0,\n")  # Unused flag

            # Write normalization information
            f2.write(",".join(map(str, inputMins)) + ",\n")
            f2.write(",".join(map(str, inputMaxes)) + ",\n")
            f2.write(",".join(map(str, means)) + ",\n")
            f2.write(",".join(map(str, ranges)) + ",\n")

            # Write weights and biases
            for w, b in zip(weights, biases):
                for i in range(w.shape[0]):
                    f2.write(",".join(f"{w[i, j]:.5e}" for j in range(w.shape[1])) + ",\n")
                for i in range(len(b)):
                    f2.write(f"{b[i]:.5e},\n")

    except Exception as e:
        print(f"Error writing NNet file: {e}")
        raise
# 1. Treina o modelo
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(30, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128)

# 2. Extrai pesos e bias
params = model.get_weights()
weights = params[0::2]
biases = params[1::2]
weights = [w.T for w in weights]

# 3. Normalização (identity)
input_size = 784
input_mins = [0.0] * input_size
input_maxes = [1.0] * input_size
means = [0.0] * input_size + [0.0]  # +1 para output
ranges = [1.0] * input_size + [1.0]

# 4. Escreve .nnet
writeNNet(weights, biases, input_mins, input_maxes, means, ranges, "mlp_mnist.nnet")

# 5. Salva o modelo Keras
model.save("mlp_mnist.h5")
print("Modelo treinado e salvo como 'mlp_mnist.nnet' e 'mlp_mnist.h5'.")