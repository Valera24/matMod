import math
import random

# ==========================================
# 1. КОНСТАНТЫ И ПАРАМЕТРЫ (ВАРИАНТ 13)
# ==========================================
N = 1000
ALPHA = 0.05

# 1) Нормальное N(m, s^2)
NORM_M = 2
NORM_S2 = 16
NORM_SIGMA = math.sqrt(NORM_S2)

# 2) Логнормальное LN(m, s^2) - параметры вложенного нормального
LOGN_M = -1
LOGN_S2 = 4
LOGN_SIGMA = math.sqrt(LOGN_S2)

# 3) Лапласа L(a)
LAPL_A = 1.5

# ==========================================
# 2. МАТЕМАТИЧЕСКОЕ ЯДРО (БЕЗ SCIPY)
# ==========================================

def erf_approx(x):
    """
    Аппроксимация функции ошибок erf(x).
    Нужна для вычисления CDF нормального распределения.
    Максимальная ошибка: 1.5e-7
    """
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    sign = 1
    if x < 0:
        sign = -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

    return sign * y

def normal_cdf(x, m, sigma):
    """Функция распределения F(x) для N(m, sigma)"""
    return 0.5 * (1 + erf_approx((x - m) / (sigma * math.sqrt(2))))

def lognormal_cdf(x, m, sigma):
    """Функция распределения F(x) для LN(m, sigma)"""
    if x <= 0: return 0.0
    return normal_cdf(math.log(x), m, sigma)

def laplace_cdf(x, a):
    """
    Функция распределения Лапласа L(a).
    F(x) = 0.5 * exp(a*x) при x < 0
    F(x) = 1 - 0.5 * exp(-a*x) при x >= 0
    (Центрировано в 0 по условию, параметр a - rate)
    """
    if x < 0:
        return 0.5 * math.exp(a * x)
    else:
        return 1.0 - 0.5 * math.exp(-a * x)

def get_normal_quantile(p):
    """Probit-функция для критических значений"""
    if p < 0.5: return -get_normal_quantile(1 - p)
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
    """Критическое значение Хи-квадрат"""
    if df < 1: return 0.0
    z = get_normal_quantile(1 - alpha)
    return df * ((1 - 2/(9*df) + z * math.sqrt(2/(9*df)))**3)

# ==========================================
# 3. ГЕНЕРАТОРЫ СЛУЧАЙНЫХ ВЕЛИЧИН
# ==========================================

def generate_normal(n, m, sigma):
    """Преобразование Бокса-Мюллера"""
    data = []
    for _ in range(n // 2 + 1): # Генерируем парами
        u1 = random.random()
        u2 = random.random()
        # Избегаем логарифма от 0
        if u1 == 0: u1 = 1e-10
        
        z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
        
        data.append(z0 * sigma + m)
        data.append(z1 * sigma + m)
    
    return data[:n]

def generate_lognormal(n, m, sigma):
    """Через нормальное: X = exp(Y), где Y ~ N(m, s^2)"""
    norm_data = generate_normal(n, m, sigma)
    return [math.exp(x) for x in norm_data]

def generate_laplace(n, a):
    """Метод обратной функции для Лапласа"""
    data = []
    for _ in range(n):
        u = random.random()
        # Обратная функция F^-1(u)
        if u < 0.5:
            val = (1/a) * math.log(2*u)
        else:
            val = -(1/a) * math.log(2*(1-u))
        data.append(val)
    return data

# ==========================================
# 4. СТАТИСТИЧЕСКИЕ КРИТЕРИИ
# ==========================================

class ContinuousTester:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def check_moments(self, sample, name, theo_m, theo_v):
        n = len(sample)
        mean_est = sum(sample) / n
        var_est = sum([(x - mean_est)**2 for x in sample]) / (n - 1)
        
        print(f"\n--- {name} ---")
        print(f"M[x]: Теор = {theo_m:.4f} | Оценка = {mean_est:.4f}")
        print(f"D[x]: Теор = {theo_v:.4f} | Оценка = {var_est:.4f}")

    def kolmogorov_test(self, sample, cdf_func, params):
        """
        Критерий Колмогорова для непрерывных СВ.
        D_n = max | F_n(x) - F(x) |
        """
        n = len(sample)
        sorted_sample = sorted(sample)
        d_max = 0.0
        
        for i in range(n):
            # F_n(x) делает скачок от i/n до (i+1)/n в точке x_i
            x = sorted_sample[i]
            
            f_teor = cdf_func(x, **params)
            
            # Отклонение слева (перед скачком) и справа (после скачка)
            d1 = abs(i/n - f_teor)
            d2 = abs((i+1)/n - f_teor)
            
            d_max = max(d_max, d1, d2)
            
        stat = math.sqrt(n) * d_max
        # Критическое значение для alpha=0.05
        crit = 1.36
        
        accepted = stat < crit
        return accepted, stat, crit

    def pearson_test(self, sample, cdf_func, params, k_bins=20):
        """
        Критерий Хи-квадрат для непрерывных СВ.
        """
        n = len(sample)
        min_val = min(sample)
        max_val = max(sample)
        
        # Формируем интервалы равной ширины
        step = (max_val - min_val) / k_bins
        boundaries = [min_val + i*step for i in range(k_bins + 1)]
        boundaries[-1] = float('inf') # Последний до бесконечности
        boundaries[0] = float('-inf') # Первый от минус бесконечности
        
        observed = [0] * k_bins
        expected = []
        
        # Подсчет наблюдаемых (O_i)
        for x in sample:
            for i in range(k_bins):
                # Проверка попадания в (boundaries[i], boundaries[i+1]]
                if boundaries[i] < x <= boundaries[i+1]:
                    observed[i] += 1
                    break
        
        # Подсчет теоретических вероятностей (E_i)
        # P(a < X < b) = F(b) - F(a)
        for i in range(k_bins):
            p_i = cdf_func(boundaries[i+1], **params) - cdf_func(boundaries[i], **params)
            expected.append(n * p_i)
            
        # Группировка интервалов (E >= 5)
        obs_g, exp_g = [], []
        curr_o, curr_e = 0, 0
        
        for o, e in zip(observed, expected):
            curr_o += o
            curr_e += e
            if curr_e >= 5:
                obs_g.append(curr_o)
                exp_g.append(curr_e)
                curr_o, curr_e = 0, 0
                
        # Хвост
        if curr_e > 0:
            if len(exp_g) > 0:
                exp_g[-1] += curr_e
                obs_g[-1] += curr_o
            else:
                exp_g.append(curr_e)
                obs_g.append(curr_o)
                
        # Статистика
        chi2 = sum([(o - e)**2 / e for o, e in zip(obs_g, exp_g)])
        df = len(obs_g) - 1
        crit = get_chi2_critical(df, self.alpha)
        
        accepted = chi2 < crit
        return accepted, chi2, crit

# ==========================================
# 5. ПРОВЕРКА ОШИБКИ I РОДА
# ==========================================

def check_type1_error_continuous(name, gen_func, cdf_func, params, iterations=100):
    print(f"\n>>> Ошибка I рода для: {name}")
    tester = ContinuousTester(ALPHA)
    
    rej_kolm = 0
    rej_pear = 0
    
    for _ in range(iterations):
        sample = gen_func(N, **params)
        
        # Колмогоров
        acc_k, _, _ = tester.kolmogorov_test(sample, cdf_func, params)
        if not acc_k: rej_kolm += 1
        
        # Пирсон
        acc_p, _, _ = tester.pearson_test(sample, cdf_func, params)
        if not acc_p: rej_pear += 1
        
    print(f"    Колмогоров: отвергнуто {rej_kolm}/{iterations} (Err rate: {rej_kolm/iterations:.2f})")
    print(f"    Пирсон:     отвергнуто {rej_pear}/{iterations} (Err rate: {rej_pear/iterations:.2f})")
    print(f"    Целевая alpha: {ALPHA}")

# ==========================================
# 6. ОСНОВНАЯ ПРОГРАММА
# ==========================================

if __name__ == "__main__":
    tester = ContinuousTester(ALPHA)
    
    print(f"=== ЛАБОРАТОРНАЯ РАБОТА №3 (Вариант 13, N={N}) ===\n")
    
    # --- 1. НОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ---
    params_norm = {'m': NORM_M, 'sigma': NORM_SIGMA}
    seq_norm = generate_normal(N, **params_norm)
    
    tester.check_moments(seq_norm, "Нормальное N(2, 16)", NORM_M, NORM_S2)
    
    acc_k, stat_k, crit_k = tester.kolmogorov_test(seq_norm, normal_cdf, params_norm)
    print(f"[Kolmogorov] Stat={stat_k:.4f}, Crit={crit_k:.4f} -> {'OK' if acc_k else 'FAIL'}")
    
    acc_p, stat_p, crit_p = tester.pearson_test(seq_norm, normal_cdf, params_norm)
    print(f"[Pearson]    Stat={stat_p:.4f}, Crit={crit_p:.4f} -> {'OK' if acc_p else 'FAIL'}")

    # --- 2. ЛОГНОРМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ ---
    params_logn = {'m': LOGN_M, 'sigma': LOGN_SIGMA}
    seq_logn = generate_lognormal(N, **params_logn)
    
    # Теоретические моменты:
    # E = exp(m + s^2/2)
    # D = (exp(s^2) - 1) * exp(2m + s^2)
    theo_m_logn = math.exp(LOGN_M + LOGN_S2/2)
    theo_v_logn = (math.exp(LOGN_S2) - 1) * math.exp(2*LOGN_M + LOGN_S2)
    
    tester.check_moments(seq_logn, "Логнормальное LN(-1, 4)", theo_m_logn, theo_v_logn)
    
    acc_k, stat_k, crit_k = tester.kolmogorov_test(seq_logn, lognormal_cdf, params_logn)
    print(f"[Kolmogorov] Stat={stat_k:.4f}, Crit={crit_k:.4f} -> {'OK' if acc_k else 'FAIL'}")
    
    acc_p, stat_p, crit_p = tester.pearson_test(seq_logn, lognormal_cdf, params_logn)
    print(f"[Pearson]    Stat={stat_p:.4f}, Crit={crit_p:.4f} -> {'OK' if acc_p else 'FAIL'}")

    # --- 3. РАСПРЕДЕЛЕНИЕ ЛАПЛАСА ---
    params_lapl = {'a': LAPL_A}
    seq_lapl = generate_laplace(N, **params_lapl)
    
    # Теоретические моменты: E=0, D=2/a^2
    theo_m_lapl = 0
    theo_v_lapl = 2 / (LAPL_A**2)
    
    tester.check_moments(seq_lapl, f"Лапласа L({LAPL_A})", theo_m_lapl, theo_v_lapl)
    
    acc_k, stat_k, crit_k = tester.kolmogorov_test(seq_lapl, laplace_cdf, params_lapl)
    print(f"[Kolmogorov] Stat={stat_k:.4f}, Crit={crit_k:.4f} -> {'OK' if acc_k else 'FAIL'}")
    
    acc_p, stat_p, crit_p = tester.pearson_test(seq_lapl, laplace_cdf, params_lapl)
    print(f"[Pearson]    Stat={stat_p:.4f}, Crit={crit_p:.4f} -> {'OK' if acc_p else 'FAIL'}")

    # --- 4. ПРОВЕРКА ОШИБКИ I РОДА ---
    check_type1_error_continuous("Нормальное", generate_normal, normal_cdf, params_norm)
    check_type1_error_continuous("Логнормальное", generate_lognormal, lognormal_cdf, params_logn)
    check_type1_error_continuous("Лапласа", generate_laplace, laplace_cdf, params_lapl)
