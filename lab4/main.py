import random
import math
import matplotlib.pyplot as plt
from scipy import integrate

# ==========================================
# 1. ОПРЕДЕЛЕНИЕ ФУНКЦИЙ
# ==========================================

def func_1(x):
    """Подынтегральная функция для I1: ln(x) * sin(x)"""
    return math.log(x) * math.sin(x)

def func_2(x, y):
    """Подынтегральная функция для I2: (x^3 + 3xy) / e^(-y)"""
    # e^(-y) в знаменателе это e^y в числителе
    return (x**3 + 3*x*y) * math.exp(y)

# ==========================================
# 2. МЕТОД МОНТЕ-КАРЛО (ЧИСТЫЙ PYTHON)
# ==========================================

def monte_carlo_integral_1(func, a, b, n_iter):
    """
    Вычисление определенного интеграла методом среднего.
    """
    sum_val = 0.0
    for _ in range(n_iter):
        # Генерируем случайное число от a до b
        x = a + (b - a) * random.random()
        sum_val += func(x)
    
    # Формула: (b - a) * среднее_значение
    length = b - a
    return length * (sum_val / n_iter)

def monte_carlo_integral_2(func, n_iter):
    """
    Вычисление двойного интеграла по области |x| + |y| < 1.
    Ограничивающий прямоугольник (Box): x in [-1, 1], y in [-1, 1].
    Площадь Box = 2 * 2 = 4.
    """
    # Границы описывающего квадрата
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    area_box = (x_max - x_min) * (y_max - y_min)
    
    sum_val = 0.0
    for _ in range(n_iter):
        # Генерируем точку в квадрате
        x = x_min + (x_max - x_min) * random.random()
        y = y_min + (y_max - y_min) * random.random()
        
        # Проверяем условие области |x| + |y| < 1
        if abs(x) + abs(y) < 1:
            sum_val += func(x, y)
        else:
            # Если не попали в область, значение функции считаем 0
            sum_val += 0.0
            
    return area_box * (sum_val / n_iter)

# ==========================================
# 3. ТОЧНЫЕ РЕШЕНИЯ (SCIPY)
# ==========================================

def get_exact_values():
    # Точное значение I1
    exact_1, _ = integrate.quad(lambda x: math.log(x) * math.sin(x), 88, 99)
    
    # Точное значение I2
    # Область |x| + |y| < 1 => -1 < x < 1, и для фиксированного x: -(1-|x|) < y < (1-|x|)
    # Или проще: так как функция нечетная по x на симметричной области, интеграл = 0.
    # Но честно посчитаем через dblquad для проверки
    def region_y_low(x):
        return -(1 - abs(x))
    
    def region_y_high(x):
        return (1 - abs(x))
        
    exact_2, _ = integrate.dblquad(
        lambda y, x: (x**3 + 3*x*y) * math.exp(y), 
        -1, 1, 
        region_y_low, 
        region_y_high
    )
    
    return exact_1, exact_2

# ==========================================
# 4. ОСНОВНАЯ ЧАСТЬ И ВИЗУАЛИЗАЦИЯ
# ==========================================

if __name__ == "__main__":
    print("=== ЛАБОРАТОРНАЯ РАБОТА №4 (Вариант 13) ===\n")
    
    # 1. Получаем "точные" значения
    exact_1, exact_2 = get_exact_values()
    print(f"Точное значение I1: {exact_1:.6f}")
    print(f"Точное значение I2: {exact_2:.6f} (ожидался 0 из-за симметрии)")
    print("-" * 50)
    
    # 2. Исследуем зависимость точности от числа итераций N
    n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
    errors_1 = []
    errors_2 = []
    
    print(f"{'N':<10} | {'I1 (MC)':<12} | {'Err I1':<10} | {'I2 (MC)':<12} | {'Err I2':<10}")
    print("-" * 65)
    
    for n in n_values:
        # Считаем I1
        res_1 = monte_carlo_integral_1(func_1, 88, 99, n)
        err_1 = abs(res_1 - exact_1)
        errors_1.append(err_1)
        
        # Считаем I2
        res_2 = monte_carlo_integral_2(func_2, n)
        err_2 = abs(res_2 - exact_2)
        errors_2.append(err_2)
        
        print(f"{n:<10} | {res_1:<12.5f} | {err_1:<10.5f} | {res_2:<12.5f} | {err_2:<10.5f}")

    # 3. Построение графиков
    plt.figure(figsize=(14, 6))
    
    # График для I1
    plt.subplot(1, 2, 1)
    plt.plot(n_values, errors_1, 'o-', label='Error I1', color='blue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Погрешность вычисления I1 от N')
    plt.xlabel('Количество итераций (N)')
    plt.ylabel('Абсолютная ошибка')
    plt.grid(True)
    plt.legend()
    
    # График для I2
    plt.subplot(1, 2, 2)
    plt.plot(n_values, errors_2, 's-', label='Error I2', color='red')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.title('Погрешность вычисления I2 от N')
    plt.xlabel('Количество итераций (N)')
    plt.ylabel('Абсолютная ошибка')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nВывод:")
    print("С ростом N ошибка уменьшается. Для I2 результат колеблется около 0,")
    print("что подтверждает теорию (интеграл нечетной функции по симметричной области).")