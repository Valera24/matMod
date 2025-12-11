import math

# ==========================================
# 1. ПАРАМЕТРЫ ВАРИАНТА 13
# ==========================================
VAR_M = 2**31
VAR_A0 = 65643
VAR_BETA = 65643
VAR_K = 256        # Объем таблицы для метода Макларена-Марсальи
N = 1000           # Объем выборки
ALPHA = 0.05       # Уровень значимости (epsilon)

# ==========================================
# 2. ГЕНЕРАТОРЫ (ДАТЧИКИ)
# ==========================================

class CongruentialGenerator:
    """
    Мультипликативный конгруэнтный метод (МКМ).
    Формула (3) из теории: alpha = (beta * alpha) mod M
    """
    def __init__(self, a0, beta, M):
        self.current = a0
        self.beta = beta
        self.M = M

    def next_val(self):
        self.current = (self.beta * self.current) % self.M
        return self.current / self.M # Нормировка в [0, 1)

    def generate_sequence(self, n):
        return [self.next_val() for _ in range(n)]

class MacLarenMarsagliaGenerator:
    """
    Метод Макларена-Марсальи.
    Комбинирует два датчика для устранения зависимостей.
    """
    def __init__(self, gen_vals, gen_index, K):
        self.gen_vals = gen_vals   # Датчик значений
        self.gen_index = gen_index # Датчик индексов
        self.K = K
        # Инициализация таблицы V (заполняем первыми числами)
        self.V = [self.gen_vals.next_val() for _ in range(K)]

    def next_val(self):
        # 1. Генерируем случайный индекс вторым датчиком
        y = self.gen_index.next_val()
        # 2. Вычисляем номер ячейки s = [y * K]
        s = int(y * self.K)
        # 3. Берем значение из таблицы
        result = self.V[s]
        # 4. Обновляем таблицу новым значением от первого датчика
        self.V[s] = self.gen_vals.next_val()
        return result

    def generate_sequence(self, n):
        return [self.next_val() for _ in range(n)]

# ==========================================
# 3. МАТЕМАТИЧЕСКИЕ УТИЛИТЫ (Вместо scipy)
# ==========================================

def get_normal_quantile(p):
    """
    Аппроксимация квантиля нормального распределения (алгоритм Beasley-Springer-Moro).
    Нужна для вычисления критических значений Хи-квадрат.
    """
    if p < 0.5:
        return -get_normal_quantile(1 - p)
    
    # Коэффициенты аппроксимации
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [-2.78718931138, -2.29796479134, 4.85014127135, 2.32121285912]
    d = [3.54388924762, 1.63706781897]

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
    Вычисление критического значения Хи-квадрат (Delta)
    с использованием аппроксимации Уилсона-Гильферти.
    Delta = df * (1 - 2/(9df) + z * sqrt(2/(9df)))^3
    """
    z = get_normal_quantile(1 - alpha)
    val = df * ((1 - 2/(9*df) + z * math.sqrt(2/(9*df)))**3)
    return val

# ==========================================
# 4. РЕАЛИЗАЦИЯ КРИТЕРИЕВ (По методичке)
# ==========================================

def pearson_test(sequence, k=20, alpha=0.05):
    """
    Критерий согласия Пирсона (Хи-квадрат).
    См. стр. 30, формулы (16) и (17).
    """
    n = len(sequence)
    
    # 1. Разбиваем на k интервалов и считаем частоты v_k
    counts = [0] * k
    for x in sequence:
        idx = int(x * k)
        if idx >= k: idx = k - 1
        counts[idx] += 1
    
    # 2. Теоретическая вероятность попадания в интервал p_k = 1/k
    # Ожидаемое количество: n * p_k = n / k
    expected = n / k 
    
    # 3. Вычисляем статистику (формула 16)
    # X^2 = Sum( (v_k - n*p_k)^2 / (n*p_k) )
    chi2_stat = 0.0
    for v in counts:
        chi2_stat += ((v - expected) ** 2) / expected
        
    # 4. Критическое значение Delta
    # Число степеней свободы = k - 1
    df = k - 1
    delta = get_chi2_critical(df, alpha)
    
    print(f"[Пирсон] Статистика X^2 = {chi2_stat:.4f}")
    print(f"[Пирсон] Критическое Delta = {delta:.4f} (для df={df})")
    
    # Решающее правило (17)
    if chi2_stat < delta:
        print("  -> Гипотеза H0 (равномерность) ПРИНИМАЕТСЯ")
        return True
    else:
        print("  -> Гипотеза H0 (равномерность) ОТВЕРГАЕТСЯ")
        return False

def kolmogorov_test(sequence, alpha=0.05):
    """
    Критерий согласия Колмогорова.
    См. стр. 31.
    """
    n = len(sequence)
    # 1. Вариационный ряд (сортировка)
    sorted_seq = sorted(sequence)
    
    # 2. Вычисление Dn = sup | F_n(x) - F_0(x) |
    # F_0(x) = x для равномерного распределения
    d_max = 0.0
    for i in range(1, n + 1):
        x = sorted_seq[i-1]
        # Отклонение "слева" и "справа" от ступеньки эмпирической функции
        d1 = abs((i / n) - x)
        d2 = abs(((i - 1) / n) - x)
        d_max = max(d_max, d1, d2)
        
    # Статистика критерия: sqrt(n) * Dn
    stat = math.sqrt(n) * d_max
    
    # Критическое значение Delta (квантиль распределения Колмогорова)
    # Для alpha=0.05 стандартное значение ~ 1.36
    # Можно вычислить точнее, но 1.36 - это стандарт для n > 40
    delta = 1.358 
    
    print(f"[Колмогоров] Статистика sqrt(n)*Dn = {stat:.4f}")
    print(f"[Колмогоров] Критическое Delta = {delta:.4f}")
    
    # Решающее правило
    if stat < delta:
        print("  -> Гипотеза H0 (равномерность) ПРИНИМАЕТСЯ")
        return True
    else:
        print("  -> Гипотеза H0 (равномерность) ОТВЕРГАЕТСЯ")
        return False

# ==========================================
# 5. ОСНОВНАЯ ПРОГРАММА
# ==========================================

if __name__ == "__main__":
    print("=== ЛАБОРАТОРНАЯ РАБОТА №1 (Вариант 13) ===")
    print(f"Параметры: a0={VAR_A0}, beta={VAR_BETA}, M=2^31, K={VAR_K}, N={N}")
    print("-" * 50)

    # 1. ГЕНЕРАЦИЯ ДАТЧИКОМ 1 (МКМ)
    gen1 = CongruentialGenerator(VAR_A0, VAR_BETA, VAR_M)
    seq1 = gen1.generate_sequence(N)
    
    print("\n>>> ПРОВЕРКА ДАТЧИКА 1 (МКМ)")
    pearson_test(seq1, k=20, alpha=ALPHA)
    kolmogorov_test(seq1, alpha=ALPHA)

    # 2. ПОДГОТОВКА ДЛЯ МЕТОДА МАКЛАРЕНА-МАРСАЛЬИ
    # Датчик 1 (значения) - пересоздаем или продолжаем (обычно берется тот же поток)
    # Для чистоты эксперимента создадим новый экземпляр того же типа
    gen_base = CongruentialGenerator(VAR_A0, VAR_BETA, VAR_M)
    
    # Датчик 2 (индексы) - должен быть другим. Возьмем классический MINSTD
    # a0 произвольное, beta = 16807, M = 2^31 - 1
    gen_aux = CongruentialGenerator(a0=12345, beta=16807, M=2**31 - 1)
    
    # 3. ГЕНЕРАЦИЯ ДАТЧИКОМ 2 (Макларен-Марсалья)
    gen_mm = MacLarenMarsagliaGenerator(gen_vals=gen_base, gen_index=gen_aux, K=VAR_K)
    seq2 = gen_mm.generate_sequence(N)
    
    print("\n>>> ПРОВЕРКА ДАТЧИКА 2 (Макларен-Марсалья)")
    pearson_test(seq2, k=20, alpha=ALPHA)
    kolmogorov_test(seq2, alpha=ALPHA)
    
    print("\n" + "-" * 50)
    print("Готово.")