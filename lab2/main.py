import math
import random

# ==========================================
# 1. КОНСТАНТЫ ВАРИАНТА 13
# ==========================================
N = 1000          # Объем выборки
ALPHA = 0.05      # Уровень значимости

# Биномиальное распределение Bi(m, p)
BIN_M = 4
BIN_P = 0.3

# Отрицательное биномиальное Bi(r, p)
# (В методичке обозначено как m, p, но стандартно r - число успехов)
NB_R = 5
NB_P = 0.4

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (БЕЗ БИБЛИОТЕК)
# ==========================================

def factorial(n):
    """Факториал n!"""
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res

def nCr(n, k):
    """Число сочетаний из n по k"""
    if k < 0 or k > n: return 0
    return factorial(n) // (factorial(k) * factorial(n - k))

def get_normal_quantile(p):
    """
    Аппроксимация обратной функции нормального распределения.
    Нужна для получения критических точек Хи-квадрат без таблиц.
    """
    if p < 0.5: return -get_normal_quantile(1 - p)
    # Алгоритм Beasley-Springer-Moro
    a = [2.5066282, -18.6150006, 41.3911977, -25.4410605]
    b = [-8.4735109, 23.0833674, -21.0622410, 3.1308291]
    c = [-2.7871893, -2.2979648, 4.8501413, 2.3212129]
    d = [3.5438892, 1.6370678]
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        return y * (((a[3]*r + a[2])*r + a[1])*r + a[0]) / ((((b[3]*r + b[2])*r + b[1])*r + b[0])*r + 1.0)
    else:
        r = p
        if y > 0: r = 1 - p
        r = math.sqrt(-math.log(r))
        return (((c[3]*r + c[2])*r + c[1])*r + c[0]) / ((d[1]*r + d[0])*r + 1.0)

def get_chi2_critical(df, alpha):
    """
    Критическое значение Хи-квадрат (аппроксимация Уилсона-Гильферти).
    chi2 ~ df * (1 - 2/(9df) + z_p * sqrt(2/(9df)))^3
    """
    if df < 1: return 0.0
    z = get_normal_quantile(1 - alpha)
    return df * ((1 - 2/(9*df) + z * math.sqrt(2/(9*df)))**3)

# ==========================================
# 3. ФУНКЦИИ ВЕРОЯТНОСТИ (ТЕОРИЯ)
# ==========================================

def binom_pmf(k, params):
    """P(X=k) для Биномиального: C_m^k * p^k * (1-p)^(m-k)"""
    m, p = params['m'], params['p']
    if k < 0 or k > m: return 0.0
    return nCr(m, k) * (p**k) * ((1 - p)**(m - k))

def nbinom_pmf(k, params):
    """P(X=k) для Отриц. Биномиального: C_(k+r-1)^k * p^r * (1-p)^k"""
    r, p = params['r'], params['p']
    if k < 0: return 0.0
    # k - число неудач до r успехов
    return nCr(k + r - 1, k) * (p**r) * ((1 - p)**k)

# ==========================================
# 4. ГЕНЕРАТОРЫ (АЛГОРИТМЫ ИЗ МЕТОДИЧКИ)
# ==========================================

def generate_binomial(n_samples, params):
    """Алгоритм: сумма m испытаний Бернулли"""
    m, p = params['m'], params['p']
    sequence = []
    for _ in range(n_samples):
        successes = 0
        for _ in range(m):
            if random.random() < p:
                successes += 1
        sequence.append(successes)
    return sequence

def generate_nbinom(n_samples, params):
    """Алгоритм: счетчик неудач до r успехов"""
    r, p = params['r'], params['p']
    sequence = []
    for _ in range(n_samples):
        successes = 0
        failures = 0
        while successes < r:
            if random.random() < p:
                successes += 1
            else:
                failures += 1
        sequence.append(failures)
    return sequence

# ==========================================
# 5. КЛАСС СТАТИСТИЧЕСКОЙ ПРОВЕРКИ
# ==========================================

class PearsonTester:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def run_test(self, sample, pmf_func, params, silent=False):
        """
        Выполняет критерий Пирсона.
        Возвращает True, если гипотеза принята, False, если отвергнута.
        """
        n = len(sample)
        
        # 1. Считаем наблюдаемые частоты (Observed)
        counts = {}
        for x in sample:
            counts[x] = counts.get(x, 0) + 1
        
        max_val = max(sample)
        # Определяем теоретический предел цикла проверки
        if 'm' in params: # Биномиальное ограничено m
            limit = params['m'] + 1
        else: # Отрицательное биномиальное бесконечно, берем с запасом
            limit = max_val + 20

        # 2. Группировка интервалов (E >= 5)
        obs_groups = []
        exp_groups = []
        
        curr_obs = 0
        curr_exp = 0
        cum_prob = 0.0 # Сумма теоретических вероятностей
        
        for k in range(limit):
            # Наблюдаемое
            o_k = counts.get(k, 0)
            # Ожидаемое
            prob_k = pmf_func(k, params)
            e_k = n * prob_k
            
            curr_obs += o_k
            curr_exp += e_k
            cum_prob += prob_k
            
            # Условие закрытия группы:
            # Если ожидаемое >= 5 ИЛИ это последний возможный бин (для биномиального)
            is_last_fixed = ('m' in params) and (k == params['m'])
            
            if curr_exp >= 5 or is_last_fixed:
                # Если это последний бин, и там все равно мало, сливаем с предыдущим
                if is_last_fixed and curr_exp < 5 and len(exp_groups) > 0:
                    exp_groups[-1] += curr_exp
                    obs_groups[-1] += curr_obs
                else:
                    obs_groups.append(curr_obs)
                    exp_groups.append(curr_exp)
                curr_obs = 0
                curr_exp = 0

        # Для бесконечных распределений: хвост вероятности
        if 'm' not in params and cum_prob < 0.99999:
            tail_prob = 1.0 - cum_prob
            tail_exp = n * tail_prob
            # Наблюдений в хвосте 0 (так как мы ушли за max(sample))
            # Прибавляем хвост к последней группе
            if len(exp_groups) > 0:
                exp_groups[-1] += tail_exp
            else:
                exp_groups.append(tail_exp)
                obs_groups.append(0)

        # 3. Расчет статистики X^2
        chi2_stat = 0.0
        for o, e in zip(obs_groups, exp_groups):
            if e > 0:
                chi2_stat += ((o - e) ** 2) / e
                
        # 4. Критическое значение
        k_groups = len(obs_groups)
        df = k_groups - 1 # Число степеней свободы
        crit_val = get_chi2_critical(df, self.alpha)
        
        accepted = chi2_stat < crit_val
        
        if not silent:
            print(f"  Групп: {k_groups}, Степеней свободы: {df}")
            print(f"  Хи-квадрат стат: {chi2_stat:.4f}")
            print(f"  Критическое val: {crit_val:.4f}")
            print(f"  РЕЗУЛЬТАТ: {'ГИПОТЕЗА ПРИНЯТА' if accepted else 'ГИПОТЕЗА ОТВЕРГНУТА'}")
            
        return accepted

# ==========================================
# 6. ФУНКЦИЯ ПРОВЕРКИ ОШИБКИ 1 РОДА
# ==========================================

def check_type1_error(dist_name, gen_func, pmf_func, params, iterations=100):
    """
    Генерирует выборку много раз и считает процент отвергнутых гипотез.
    Должен стремиться к ALPHA (0.05).
    """
    print(f"\n>>> Проверка ошибки I рода для: {dist_name}")
    print(f"    Количество экспериментов: {iterations}")
    
    tester = PearsonTester(ALPHA)
    rejected_count = 0
    
    for i in range(iterations):
        # 1. Генерируем новую выборку
        sample = gen_func(N, params)
        # 2. Проверяем её (в тихом режиме)
        is_accepted = tester.run_test(sample, pmf_func, params, silent=True)
        
        if not is_accepted:
            rejected_count += 1
            
    error_rate = rejected_count / iterations
    print(f"    Отвергнуто гипотез: {rejected_count} из {iterations}")
    print(f"    Эмпирическая ошибка I рода: {error_rate:.3f}")
    print(f"    Теоретическая ошибка (alpha): {ALPHA}")
    
    if abs(error_rate - ALPHA) < 0.06: # Допуск для 100 экспериментов
        print("    ВЫВОД: Уровень значимости подтвержден.")
    else:
        print("    ВЫВОД: Отклонение велико (возможно, малая выборка для статистики).")

# ==========================================
# 7. ОСНОВНАЯ ПРОГРАММА
# ==========================================

if __name__ == "__main__":
    tester = PearsonTester(ALPHA)
    
    # --- БИНОМИАЛЬНОЕ ---
    print("="*60)
    print(f"1. БИНОМИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ (m={BIN_M}, p={BIN_P})")
    print("="*60)
    params_bin = {'m': BIN_M, 'p': BIN_P}
    
    # 1. Генерация и базовая проверка
    seq_bin = generate_binomial(N, params_bin)
    
    mean_est = sum(seq_bin)/N
    mean_theo = BIN_M * BIN_P
    var_est = sum([(x - mean_est)**2 for x in seq_bin])/(N-1)
    var_theo = BIN_M * BIN_P * (1 - BIN_P)
    
    print(f"Мат. ожидание: Оценка={mean_est:.4f}, Теор={mean_theo:.4f}")
    print(f"Дисперсия:     Оценка={var_est:.4f}, Теор={var_theo:.4f}")
    
    tester.run_test(seq_bin, binom_pmf, params_bin)
    
    # 2. Проверка ошибки I рода
    check_type1_error("Биномиальное", generate_binomial, binom_pmf, params_bin)


    # --- ОТРИЦАТЕЛЬНОЕ БИНОМИАЛЬНОЕ ---
    print("\n" + "="*60)
    print(f"2. ОТРИЦ. БИНОМИАЛЬНОЕ РАСПРЕДЕЛЕНИЕ (r={NB_R}, p={NB_P})")
    print("="*60)
    params_nb = {'r': NB_R, 'p': NB_P}
    
    # 1. Генерация и базовая проверка
    seq_nb = generate_nbinom(N, params_nb)
    
    mean_est = sum(seq_nb)/N
    mean_theo = NB_R * (1 - NB_P) / NB_P
    var_est = sum([(x - mean_est)**2 for x in seq_nb])/(N-1)
    var_theo = NB_R * (1 - NB_P) / (NB_P**2)
    
    print(f"Мат. ожидание: Оценка={mean_est:.4f}, Теор={mean_theo:.4f}")
    print(f"Дисперсия:     Оценка={var_est:.4f}, Теор={var_theo:.4f}")
    
    tester.run_test(seq_nb, nbinom_pmf, params_nb)
    
    # 2. Проверка ошибки I рода
    check_type1_error("Отриц. Биномиальное", generate_nbinom, nbinom_pmf, params_nb)

   