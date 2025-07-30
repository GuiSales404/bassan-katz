import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import numpy as np
import tensorflow as tf

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


# 1. Carregamento dos dados
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. Construção do modelo MLP
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(32, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
    Dense(10, activation='linear')  # importante: saída linear para Marabou
])

# 3. Compilação
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# 4. Treinamento com EarlyStopping
model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=2
)

# 5. Avaliação final
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")


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
writeNNet(weights, biases, input_mins, input_maxes, means, ranges, "mlp_mnist_robust.nnet")

# 6. Salvar modelo Keras
model.save("mlp_mnist_robust.h5")
