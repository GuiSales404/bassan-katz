from maraboupy import Marabou
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm

# === Configurações ===
idx = 0               # índice da imagem
delta = 0.05          # faixa de variação do pixel (reduzida)
epsilon = 0.01        # margem de separação nos logits
model_path = "mlp_mnist.nnet"

# === Dados ===
(_, _), (x_test, y_test) = mnist.load_data()
image = x_test[idx].astype(np.float32) / 255.0
label = int(y_test[idx])
flat_image = image.flatten()

# === Teste de singleton mais restrito ===
def is_contrastive_singleton(free_pixel):
    v = flat_image[free_pixel]
    lower = max(0.0, v - delta)
    upper = min(1.0, v + delta)
    if lower == upper:  # Não há espaço para perturbar
        return False

    for target in range(10):
        if target == label:
            continue

        net = Marabou.read_nnet(model_path)
        input_vars = net.inputVars[0].flatten()
        output_vars = net.outputVars[0].flatten()

        for i in range(784):
            if i == free_pixel:
                net.setLowerBound(input_vars[i], lower)
                net.setUpperBound(input_vars[i], upper)
            else:
                net.setLowerBound(input_vars[i], flat_image[i])
                net.setUpperBound(input_vars[i], flat_image[i])

        net.addInequality([output_vars[target], output_vars[label]], [1, -1], -epsilon)
        status, _, _ = net.solve()
        if status == 'sat':
            return True
    return False

# === Executar verificação ===
print(f"\n🔍 Verificando contrastive singletons com δ = {delta}...")
contrastive_singletons = []

for f in tqdm(range(784)):
    if is_contrastive_singleton(f):
        contrastive_singletons.append(f)

print(f"\n📌 Total de contrastive singletons encontrados: {len(contrastive_singletons)}")
print(contrastive_singletons)

# === Visualização ===
singleton_map = np.zeros(784, dtype=int)
singleton_map[contrastive_singletons] = 1
singleton_map = singleton_map.reshape((28, 28))

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(image, cmap='gray', alpha=0.7)
ax.imshow(singleton_map, cmap=cm.Reds, alpha=0.5)
ax.set_title(f"📌 Contrastive Singletons (δ = {delta})")
ax.axis('off')
plt.show()
