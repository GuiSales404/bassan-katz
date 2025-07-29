from maraboupy import Marabou
from tensorflow.keras.datasets import mnist
import numpy as np

# Configurações
idx = 0  # índice da imagem
free_pixel = 150  # pixel que ficará livre (0 a 783)

# Carrega rede e imagem
net = Marabou.read_nnet("mlp_mnist.nnet")
(_, _), (x_test, y_test) = mnist.load_data()
image = x_test[idx].astype(np.float32) / 255.0
label = y_test[idx]
flat_image = image.flatten()

# Define variáveis
input_vars = net.inputVars[0].flatten()
output_vars = net.outputVars[0].flatten()

# 1. Fixar todos os pixels exceto o pixel livre
for i in range(784):
    if i == free_pixel:
        net.setLowerBound(input_vars[i], 0.0)
        net.setUpperBound(input_vars[i], 1.0)
    else:
        net.setLowerBound(input_vars[i], flat_image[i])
        net.setUpperBound(input_vars[i], flat_image[i])

# 2. Impor restrição: output[label] ≥ output[i ≠ label]
for i in range(10):
    if i != label:
        net.addInequality([output_vars[i], output_vars[label]], [1, -1], 0)

# 3. Resolver
stats, vals, _ = net.solve()
print(f"Resultado com pixel {free_pixel} livre:", "SATISFÁVEL ✅" if stats=='sat' else "INSATISFÁVEL ❌")
