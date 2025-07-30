import numpy as np
from maraboupy import Marabou
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from itertools import combinations
import pulp
import os
import sys
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import matplotlib.pyplot as plt
import pandas as pd

# === Configurações Globais ===
MODEL_PATH = "mlp_mnist.nnet"
IMG_SIZE = 28
EPSILON = 0.01
DELTA_VALUES = [0.0005, 0.005, 0.05]
BLOCK_SIZES = [1, 4, 7, 14]
IMAGE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Exemplo de índices de imagens
MAX_WORKERS = 8
RESULTS_DIR = "exp_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# === Funções Utilitárias ===
def get_blocks(image, block_size):
    blocks = []
    indices = []
    for i in range(0, IMG_SIZE, block_size):
        for j in range(0, IMG_SIZE, block_size):
            block = []
            block_idx = []
            for bi in range(block_size):
                for bj in range(block_size):
                    if i + bi < IMG_SIZE and j + bj < IMG_SIZE:
                        block.append(image[i + bi, j + bj])
                        block_idx.append((i + bi) * IMG_SIZE + (j + bj))
            blocks.append(block)
            indices.append(block_idx)
    return blocks, indices

def run_marabou(image_flat, label, epsilon=EPSILON, delta=0.005):
    for target in range(10):
        if target == label:
            continue
        with suppress_stdout_stderr():
            net = Marabou.read_nnet(MODEL_PATH)
            input_vars = net.inputVars[0].flatten()
            output_vars = net.outputVars[0].flatten()
            for i in range(len(input_vars)):
                net.setLowerBound(input_vars[i], image_flat[i])
                net.setUpperBound(input_vars[i], image_flat[i])
            net.addInequality([output_vars[target], output_vars[label]], [1, -1], -epsilon)
            status, _, _ = net.solve(verbose=False)
            if status == 'sat':
                return {'status': 'sat'}
    return {'status': 'unsat'}

def save_mask_visualization(image, explanation_indices, path):
    mask = np.zeros_like(image)
    for idx in explanation_indices:
        i, j = divmod(idx, IMG_SIZE)
        mask[i, j] = 1
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.imshow(mask, cmap='jet', alpha=0.4)
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def save_binary_mask(image, explanation_indices, path):
    mask = np.zeros((IMG_SIZE, IMG_SIZE))
    for idx in explanation_indices:
        i, j = divmod(idx, IMG_SIZE)
        mask[i, j] = 1
    np.save(path, mask)

def save_csv_row(csv_path, image_idx, label, ub, lb, approx, image_path, block_size, delta):
    row = f"{image_idx},{label},{ub},{lb},{approx:.2f},{image_path},{block_size},{delta}\n"
    header = "image_idx,label,UB,LB,approximation,image_path,block_size,delta\n"
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(header)
    with open(csv_path, 'a') as f:
        f.write(row)

def minimum_vertex_cover_milp(pairs):
    vertices = sorted(set([v for edge in pairs for v in edge]))
    x = pulp.LpVariable.dicts("x", vertices, 0, 1, pulp.LpBinary)
    prob = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)
    prob += pulp.lpSum([x[v] for v in vertices])
    for u, v in pairs:
        prob += x[u] + x[v] >= 1, f"cover_{u}_{v}"
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    vertex_cover = {v for v in vertices if pulp.value(x[v]) >= 0.99}
    return vertex_cover


# === Função central de explicação ===
def run_upper_bound(image_flat, label, epsilon, delta, blocks, block_indices, max_workers=8):
    ub = len(blocks)
    free_blocks = []
    ub_lock = threading.Lock()

    def test_block(idx):
        altered = image_flat.copy()
        for pixel in block_indices[idx]:
            altered[pixel] = np.clip(altered[pixel] + np.random.uniform(-delta, delta), 0.0, 1.0)
        result = run_marabou(altered, label, epsilon, delta)
        if result['status'] == 'unsat':
            with ub_lock:
                free_blocks.append(idx)
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_block, i) for i in range(len(blocks))]
        for future in as_completed(futures):
            ub -= future.result()
    return ub, free_blocks

def run_lower_bound(image_flat, label, epsilon, delta, blocks, block_indices, max_workers=8):
    lb = 0
    singletons = []
    pairs = []
    lb_lock = threading.Lock()

    def test_singleton(idx):
        altered = image_flat.copy()
        for pixel in block_indices[idx]:
            altered[pixel] = np.clip(altered[pixel] + np.random.uniform(-delta, delta), 0.0, 1.0)
        result = run_marabou(altered, label, epsilon, delta)
        if result['status'] == 'sat':
            with lb_lock:
                singletons.append(idx)
            return 1
        return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(test_singleton, i) for i in range(len(blocks))]
        for future in as_completed(futures):
            lb += future.result()

    # Pairs
    candidates = [i for i in range(len(blocks)) if i not in singletons]
    all_pairs = list(combinations(candidates, 2))
    for u, v in all_pairs:
        altered = image_flat.copy()
        for pixel in block_indices[u] + block_indices[v]:
            altered[pixel] = np.clip(altered[pixel] + np.random.uniform(-delta, delta), 0.0, 1.0)
        result = run_marabou(altered, label, epsilon, delta)
        if result['status'] == 'sat':
            pairs.append((u, v))

    lb += len(minimum_vertex_cover_milp(pairs))
    return lb

def run_explanation(image, label, epsilon, delta, block_size, mask_path):
    image_flat = image.flatten()
    blocks, block_indices = get_blocks(image, block_size)

    ub, free_blocks = run_upper_bound(image_flat, label, epsilon, delta, blocks, block_indices, MAX_WORKERS)
    lb = run_lower_bound(image_flat, label, epsilon, delta, blocks, block_indices, MAX_WORKERS)

    # Monta explicação com blocos não livres
    explanation_indices = []
    for i in range(len(blocks)):
        if i not in free_blocks:
            explanation_indices.extend(block_indices[i])

    save_mask_visualization(image, explanation_indices, mask_path.replace('.npy', '.png'))
    save_binary_mask(image, explanation_indices, mask_path)

    return explanation_indices, ub, lb


def run_grid_search():
    (_, _), (x_test, y_test) = mnist.load_data()
    csv_path = os.path.join(RESULTS_DIR, "grid_results.csv")
    for idx in IMAGE_INDICES:
        image = x_test[idx] / 255.0
        label = int(y_test[idx])
        for delta in DELTA_VALUES:
            for block_size in BLOCK_SIZES:
                print(f"Processando img {idx}, block {block_size}, delta {delta}...")
                mask_path = os.path.join(RESULTS_DIR, f"mask_img{idx}_block{block_size}_delta{delta}.npy")
                explanation_indices, ub, lb = run_explanation(image, label, EPSILON, delta, block_size, mask_path)
                approx = ub / lb if lb > 0 else 0
                save_csv_row(csv_path, idx, label, ub, lb, approx, mask_path, block_size, delta)
                print(f"Finalizado: UB={ub}, LB={lb}, Approx={approx:.2f}")
                print(f"Explicação: {explanation_indices}")

if __name__ == '__main__':
    run_grid_search()
