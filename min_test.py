import numpy as np
from maraboupy import Marabou
from tensorflow.keras.datasets import mnist
import tensorflow as tf
from itertools import combinations
import pulp
import os
import sys
import contextlib

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
MODEL_PATH = "mlp_mnist.nnet"
IMG_SIZE = 28

# === 1. Carregar imagem e label ===
(_, _), (x_test, y_test) = mnist.load_data()
image = x_test[0] / 255.0  # normalizado
true_label = int(y_test[0])

def run_marabou(image_flat, label, epsilon=0.01):
    net = Marabou.read_nnet(MODEL_PATH)
    input_vars = net.inputVars[0].flatten()
    output_vars = net.outputVars[0].flatten()

    # Fixar entrada
    for i in range(len(input_vars)):
        net.setLowerBound(input_vars[i], image_flat[i])
        net.setUpperBound(input_vars[i], image_flat[i])

    # Impor que a saída esperada (label) tenha valor maior que todas as outras
    # for i in range(len(output_vars)):
    #     if i != label:
    #         net.addInequality(
    #             [output_vars[label], output_vars[i]],  # f_label - f_i >= ε
    #             [1, -1],
    #             -epsilon  # f_label - f_i ≥ ε  =>  f_label - f_i + (-ε) ≥ 0
    #         )
    
    # 2. Impor restrição: output[label] ≥ output[i ≠ label]
    for i in range(10):
        if i != label:
            net.addInequality([output_vars[i], output_vars[label]], [1, -1], 0)

    # Resolver
    status, vals, _ = net.solve(verbose=False)

    return {
        "status": status,
        "vals": vals,
    }



def run_explanation(image, label): 
    sense_image = image.copy()
    image = image.flatten()
    
    def minimum_vertex_cover_milp(pairs):
        # 1. Obter todos os vértices únicos
        vertices = sorted(set([v for edge in pairs for v in edge]))
        # 2. Variáveis binárias: x[i] = 1 se o vértice i está na cobertura
        x = pulp.LpVariable.dicts("x", vertices, 0, 1, pulp.LpBinary)
        # 3. Problema de minimização
        prob = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)
        # 4. Função objetivo: minimizar o número de vértices na cobertura
        prob += pulp.lpSum([x[v] for v in vertices])
        # 5. Restrições: para cada aresta (u, v), pelo menos um dos vértices deve estar no conjunto
        for u, v in pairs:
            prob += x[u] + x[v] >= 1, f"cover_{u}_{v}"
        # 6. Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        # 7. Extrair solução
        vertex_cover = {v for v in vertices if pulp.value(x[v]) >= 0.99}
        return vertex_cover

    def sort_by_sensibility(model='mlp_mnist.h5', image=image, label=label):
        with tf.GradientTape() as tape:
            input_tensor = tf.convert_to_tensor(image[None, ...])
            tape.watch(input_tensor)
            prediction = model(input_tensor)
            logit = prediction[0, label]
        grads = tape.gradient(logit, input_tensor).numpy()[0]
        flat_grads = np.abs(grads).flatten()
        return np.argsort(flat_grads)[::-1] 

    ub, lb = len(image), 0
    free = []

    def upper_bound(sorted_features_idx, image=image, label=label-1 if label>0 else label+1): 
        nonlocal ub
        nonlocal free
        F = sorted_features_idx.copy()
        for f in F:
            candidate_explanation = image.copy()
            if len(free) > 0:
                for freed_pixels in free:
                    candidate_explanation[freed_pixels] = np.random.uniform(0.0, 1.0)
            candidate_explanation[f] = np.random.uniform(0.0, 1.0)
            result = run_marabou(candidate_explanation, label)
            if result['status'] == 'unsat':
                free.append(f)
                ub -= 1
                print('=-'*50)
                print(free)
                exit()
        return ub
                
    def lower_bound(image=image, label=label-1 if label>0 else label+1):
        nonlocal lb
        singletons = []
        pairs = []
        
        # Contrastive Singletons
        F = image.copy()
        for f_idx in range(len(F)):
            altered_F = F.copy()
            altered_F[f_idx] = np.random.uniform(0.0, 1.0)
            result_cs = run_marabou(altered_F, label)
            if result_cs['status'] == 'sat':
                singletons.append(f_idx)
                lb += 1
        
        # All Contrastive Pairs
        candidates = [i for i in np.arange(len(F)) if i not in singletons]
        all_pairs = list(combinations(candidates, 2))
        for pair in all_pairs:
            altered_F = F.copy()
            altered_F[pair[0]] = np.random.uniform(0.0, 0.1)
            altered_F[pair[1]] = np.random.uniform(0.0, 0.1)
            result_cp = run_marabou(altered_F, label)
            if result_cp['status'] == 'sat':
                pairs.append(pair)
                
        mvc = minimum_vertex_cover_milp(pairs)
        lb += len(mvc)
        return lb
    
    sort_f = sort_by_sensibility()
    upb = upper_bound(sorted_features_idx=sort_f)
    lob = lower_bound()
    return [image[i] for i in np.arange(len(image)) if i not in free], upb, lob

# exp, u, l = run_explanation(image, true_label)
# print('Explanation len:', len(exp))
# print('Explanation:', exp)
# print(f'UB = {u} | LB = {l} | CORRECTNESS = {u/l if l>0 else 0}')

with suppress_stdout_stderr():
    res = run_marabou(image.flatten(), true_label, 0)
    
print(res['status'])