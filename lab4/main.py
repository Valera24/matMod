import random
import math
import matplotlib.pyplot as plt
from scipy import integrate # Только для получения эталонного точного значения

# ==========================================
# 1. ОПРЕДЕЛЕНИЕ ФУНКЦИЙ (ВАРИАНТ 13)
# ==========================================

def func_1(x):
    """Подынтегральная функция для I1: ln(x) * sin(x)"""
    return math.log(x) * math.sin(x)

def func_2(x, y):
    """Подынтегральная функция для I2: (x^3 + 3xy) / e^(-y)"""
    return (x**3 + 3*x*y) * math.exp(y)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (МАТЕМАТИКА ИЗ МЕТОДИЧКИ)
# ==========================================

def calculate_sample_variance(values, mean):
    """
    Вычисление несмещенной выборочной дисперсии S^2.
    Аналог функции var() из C++ примера на стр. 56.
    Formula: sum((xi - mean)^2) / (n - 1)
    """
    n = len(values)
    if n < 2: return 0.0
    
    sum_sq_diff = sum((x - mean)**2 for x in values)
    return sum_sq_diff / (n - 1)

def calc_prob_error(variance, n):
    """
    Вычисление вероятной ошибки (стр. 53, 57).
    tau = 0.6745 * sqrt(D / n)
    """
    if n == 0 or variance < 0: return 0.0
    return 0.6745 * math.sqrt(variance / n)

# ==========================================
# 3. РЕАЛИЗАЦИЯ МЕТОДА МОНТЕ-КАРЛО
# ==========================================

def solve_integral_1(n_iter):
    """
    Интеграл 1: [88, 99]. Длина интервала L = 11.
    Случайная величина Ksi = L * f(x).
    """
    a, b = 88, 99
    length = b - a
    
    # Массив для хранения реализаций СВ (как массив A в примере)
    samples = []
    
    for _ in range(n_iter):
        # Генерируем x равномерно на [a, b]
        x = a + length * random.random()
        # Вычисляем значение СВ
        val = length * func_1(x)
        samples.append(val)
    
    # Оценка интеграла (среднее)
    integral_val = sum(samples) / n_iter
    
    # Оценка дисперсии
    variance = calculate_sample_variance(samples, integral_val)
    
    # Вероятная ошибка
    error_prob = calc_prob_error(variance, n_iter)
    
    return integral_val, error_prob

def solve_integral_2(n_iter):
    """
    Интеграл 2: Область |x| + |y| < 1.
    Вписываем в квадрат [-1, 1]x[-1, 1]. Площадь S = 4.
    Случайная величина Ksi = S * f(x,y), если попали, иначе 0.
    """
    # Границы квадрата
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    area_box = (x_max - x_min) * (y_max - y_min) # = 4
    
    samples = []
    
    for _ in range(n_iter):
        x = x_min + (x_max - x_min) * random.random()
        y = y_min + (y_max - y_min) * random.random()
        
        # Проверка попадания в область D
        if abs(x) + abs(y) < 1:
            val = area_box * func_2(x, y)
        else:
            val = 0.0
            
        samples.append(val)
        
    integral_val = sum(samples) / n_iter
    variance = calculate_sample_variance(samples, integral_val)
    error_prob = calc_prob_error(variance, n_iter)
    
    return integral_val, error_prob

# ==========================================
# 4. ПОЛУЧЕНИЕ ТОЧНЫХ ЗНАЧЕНИЙ (ДЛЯ СРАВНЕНИЯ)
# ==========================================
def get_exact_values():
    # I1
    exact_1, _ = integrate.quad(func_1, 88, 99)
    # I2 (должен быть 0)
    def region_y_low(x):
        return -(1 - abs(x))
    
    def region_y_high(x):
        return (1 - abs(x))
        
    exact_2, _ = integrate.dblquad(lambda y, x: func_2(x,y), -1, 1, region_y_low, region_y_high)
    return exact_1, exact_2

# ==========================================
# 5. ОСНОВНОЙ ЦИКЛ И ВЫВОД
# ==========================================

if __name__ == "__main__":
    print("=== ЛАБОРАТОРНАЯ РАБОТА №4 (Вариант 13) ===\n")
    
    exact_1, exact_2 = get_exact_values()
    print(f"Точное значение I1: {exact_1:.6f}")
    print(f"Точное значение I2: {exact_2:.6f}")
    print("-" * 75)
    print(f"{'N':<8} | {'I1 (MC)':<10} | {'Prob.Err1':<10} | {'Real Err1':<10} | {'I2 (MC)':<10} | {'Prob.Err2':<10}")
    print("-" * 75)
    
    n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    
    # Списки для графиков
    prob_errors_1 = []
    prob_errors_2 = []
    
    for n in n_values:
        # Вычисления
        val_1, p_err_1 = solve_integral_1(n)
        val_2, p_err_2 = solve_integral_2(n)
        
        # Реальная ошибка (разница с точным) - для справки
        real_err_1 = abs(val_1 - exact_1)
        
        # Сохраняем для графиков
        prob_errors_1.append(p_err_1)
        prob_errors_2.append(p_err_2)
        
        print(f"{n:<8} | {val_1:<10.5f} | {p_err_1:<10.5f} | {real_err_1:<10.5f} | {val_2:<10.5f} | {p_err_2:<10.5f}")

    print("\nПояснение:")
    print("Prob.Err (Вероятная ошибка) рассчитана по формуле из методички: 0.6745 * sqrt(D/n).")
    print("Она показывает теоретический разброс метода Монте-Карло.")

    # Построение графиков
    plt.figure(figsize=(12, 5))
    
    # График для I1
    plt.subplot(1, 2, 1)
    plt.plot(n_values, prob_errors_1, 'o-', label='Probable Error I1')
    plt.title('Зависимость точности от N (Интеграл 1)')
    plt.xlabel('Количество итераций N')
    plt.ylabel('Вероятная ошибка (0.6745 * sigma / sqrt(n))')
    plt.grid(True)
    plt.legend()
    
    # График для I2
    plt.subplot(1, 2, 2)
    plt.plot(n_values, prob_errors_2, 's-', color='orange', label='Probable Error I2')
    plt.title('Зависимость точности от N (Интеграл 2)')
    plt.xlabel('Количество итераций N')
    plt.ylabel('Вероятная ошибка')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()