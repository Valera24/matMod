import random
import numpy as np

# Параметры системы
n = 3  # размерность системы
N = 1000  # длина цепи Маркова
m = 10000  # количество реализаций цепи

# Матрица A и вектор f для варианта 13
A = np.array([[0.75, 0.15, 0.1],
              [-0.15, 0.9, 0.35],
              [0.25, -0.05, 0.5]], dtype=float)

f = np.array([0.5, 2.5, 4], dtype=float)

# Вектор h (единичный вектор для оценки x_i, здесь для x0)
h = np.array([1.0, 0.0, 0.0], dtype=float)

# Начальное распределение цепи Маркова (равномерное)
pi = np.array([1/3, 1/3, 1/3], dtype=float)

# Матрица переходных вероятностей (равномерная)
P = np.array([[1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3],
              [1/3, 1/3, 1/3]], dtype=float)

# Инициализация генератора случайных чисел
random.seed()

# Моделирование m цепей Маркова длины N
ksi = np.zeros(m, dtype=float)

for j in range(m):
    # Начальное состояние цепи
    alpha = random.random()
    if alpha < pi[0]:
        i_prev = 0
    elif alpha < pi[0] + pi[1]:
        i_prev = 1
    else:
        i_prev = 2

    # Вычисление начального веса
    if pi[i_prev] > 0:
        Q = h[i_prev] / pi[i_prev]
    else:
        Q = 0.0

    # Накопитель для оценки
    ksi_j = Q * f[i_prev]

    # Проход по цепи Маркова
    for step in range(1, N + 1):
        # Следующее состояние
        alpha = random.random()
        if alpha < P[i_prev, 0]:
            i_curr = 0
        elif alpha < P[i_prev, 0] + P[i_prev, 1]:
            i_curr = 1
        else:
            i_curr = 2

        # Обновление веса
        if P[i_prev, i_curr] > 0:
            Q *= A[i_prev, i_curr] / P[i_prev, i_curr]
        else:
            Q = 0.0

        # Добавление вклада текущего состояния
        ksi_j += Q * f[i_curr]

        # Обновление предыдущего состояния
        i_prev = i_curr

    ksi[j] = ksi_j

# Оценка решения (для x0)
x0_estimate = np.mean(ksi)
print(f"Оценка для x0: {x0_estimate}")

# Для x1 и x2 нужно повторить с h = [0,1,0] и [0,0,1]
def estimate_component(h_vector):
    ksi = np.zeros(m, dtype=float)
    for j in range(m):
        alpha = random.random()
        if alpha < pi[0]:
            i_prev = 0
        elif alpha < pi[0] + pi[1]:
            i_prev = 1
        else:
            i_prev = 2

        if pi[i_prev] > 0:
            Q = h_vector[i_prev] / pi[i_prev]
        else:
            Q = 0.0

        ksi_j = Q * f[i_prev]

        for step in range(1, N + 1):
            alpha = random.random()
            if alpha < P[i_prev, 0]:
                i_curr = 0
            elif alpha < P[i_prev, 0] + P[i_prev, 1]:
                i_curr = 1
            else:
                i_curr = 2

            if P[i_prev, i_curr] > 0:
                Q *= A[i_prev, i_curr] / P[i_prev, i_curr]
            else:
                Q = 0.0

            ksi_j += Q * f[i_curr]
            i_prev = i_curr

        ksi[j] = ksi_j
    return np.mean(ksi)

# Оценки для x1 и x2
x1_estimate = estimate_component(np.array([0.0, 1.0, 0.0]))
x2_estimate = estimate_component(np.array([0.0, 0.0, 1.0]))

print(f"Оценка для x1: {x1_estimate}")
print(f"Оценка для x2: {x2_estimate}")

# Сравнение с точным решением через numpy
exact_solution = np.linalg.solve(A, f)
print(f"Точное решение (numpy): {exact_solution}")
print(f"Ошибки: |x0 - {exact_solution[0]}| = {abs(x0_estimate - exact_solution[0])}")
print(f"        |x1 - {exact_solution[1]}| = {abs(x1_estimate - exact_solution[1])}")
print(f"        |x2 - {exact_solution[2]}| = {abs(x2_estimate - exact_solution[2])}")