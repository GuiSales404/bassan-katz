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

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = fnull
            sys.stderr = fnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# === Configurações ===
MODEL_PATH = "mlp_mnist_robust.nnet"
IMG_SIZE = 28
EPSILON = 0.01
DELTA = 0.005
MAX_WORKERS = 8

# === 1. Carregar imagem e label ===
(_, _), (x_test, y_test) = mnist.load_data()
image = x_test[0] / 255.0
true_label = int(y_test[0])

def is_classification_different_by_freedom(image_flat, label, free_index, epsilon=EPSILON, delta=DELTA):
    v = image_flat[free_index]
    lower = max(0.0, v - delta)
    upper = min(1.0, v + delta)
    if lower == upper:
        return False

    for target in range(10):
        if target == label:
            continue
        with suppress_stdout_stderr():
            net = Marabou.read_nnet(MODEL_PATH)
            input_vars = net.inputVars[0].flatten()
            output_vars = net.outputVars[0].flatten()
            for i in range(len(input_vars)):
                if i == free_index:
                    net.setLowerBound(input_vars[i], lower)
                    net.setUpperBound(input_vars[i], upper)
                else:
                    net.setLowerBound(input_vars[i], image_flat[i])
                    net.setUpperBound(input_vars[i], image_flat[i])
            net.addInequality([output_vars[target], output_vars[label]], [1, -1], -epsilon)
            status, _, _ = net.solve(verbose=False)
            if status == 'sat':
                return True
    return False

def run_marabou(image_flat, label, epsilon=EPSILON, delta=DELTA):
    return {'status': 'sat' if any(
        is_classification_different_by_freedom(image_flat, label, i, epsilon, delta)
        for i in range(len(image_flat))
    ) else 'unsat'}

def run_explanation(image, label): 
    sense_image = image.copy()
    image = image.flatten()

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

    def sort_by_sensibility(model_path='mlp_mnist_robust.h5', image=sense_image, label=label):
        model = tf.keras.models.load_model(model_path)
        with tf.GradientTape() as tape:
            input_tensor = tf.convert_to_tensor(image[None, ...])
            tape.watch(input_tensor)
            prediction = model(input_tensor)
            logit = prediction[0, label]
        grads = tape.gradient(logit, input_tensor).numpy()[0]
        flat_grads = np.abs(grads).flatten()
        return np.argsort(flat_grads)[::-1] 

    ub = len(image)
    free = []
    ub_lock = threading.Lock()

    def upper_bound(sorted_features_idx, image=image, label=label): 
        nonlocal ub, free
        def test_pixel(f):
            if not is_classification_different_by_freedom(image, label, f, EPSILON, DELTA):
                with ub_lock:
                    free.append(f)
                    ub_dec = 1
                    return ub_dec
            return 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(test_pixel, f) for f in sorted_features_idx]
            for future in as_completed(futures):
                ub -= future.result()
        return ub

    lb = 0
    lb_lock = threading.Lock()

    def lower_bound(image=image, label=label):
        nonlocal lb
        singletons = []
        pairs = []

        F = image.copy()

        def test_singleton(f_idx):
            altered_F = F.copy()
            altered_F[f_idx] = np.clip(altered_F[f_idx] + np.random.uniform(-DELTA, DELTA), 0.0, 1.0)
            result = run_marabou(altered_F, label, EPSILON, DELTA)
            if result['status'] == 'sat':
                with lb_lock:
                    singletons.append(f_idx)
                    lb_inc = 1
                    return lb_inc
            return 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(test_singleton, i) for i in range(len(F))]
            for future in as_completed(futures):
                lb += future.result()

        candidates = [i for i in np.arange(len(F)) if i not in singletons]
        all_pairs = list(combinations(candidates, 2))
        for pair in all_pairs:
            altered_F = F.copy()
            altered_F[pair[0]] = np.clip(altered_F[pair[0]] + np.random.uniform(-DELTA, DELTA), 0.0, 1.0)
            altered_F[pair[1]] = np.clip(altered_F[pair[1]] + np.random.uniform(-DELTA, DELTA), 0.0, 1.0)
            result_cp = run_marabou(altered_F, label, EPSILON, DELTA)
            if result_cp['status'] == 'sat':
                pairs.append(pair)

        mvc = minimum_vertex_cover_milp(pairs)
        lb += len(mvc)
        return lb

    sort_f = sort_by_sensibility()
    upb = upper_bound(sorted_features_idx=sort_f)
    lob = lower_bound()
    return [image[i] for i in np.arange(len(image)) if i not in free], upb, lob

# Executar
with suppress_stdout_stderr():
    exp, u, l = run_explanation(image, true_label)

print('Explanation len:', len(exp))
print('Explanation:', exp)
print(f'UB = {u} | LB = {l} | APPROXIMATION = {u/l if l>0 else 0:.2f}')
