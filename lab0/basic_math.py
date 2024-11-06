import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    matrix_a, matrix_b = np.array(matrix_a), np.array(matrix_b)
    if not matrix_a.shape[1] == matrix_b.shape[0]:
        raise ValueError('Размеры матриц неверные.')
    mult_result = np.ndarray(shape=(matrix_a.shape[0], matrix_b.shape[1]))
    m = matrix_a.shape[0]
    p =  matrix_b.shape[1]
    for j in range(m):
        for k in range(p):
            mult_result[j, k] = sum(np.multiply(matrix_a[j, :], matrix_b[:, k]))
    return mult_result.tolist()


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    def find_extremum(coeffs):
        a, b, c = coeffs
        if a == 0:
            return None if b == 0 else -c / b
        else:
            return -b / (2 * a)

    def evaluate_function(coeffs, x):
        a, b, c = coeffs
        return a * x**2 + b * x + c

    def find_common_solutions(a1, a2):
        # Переносим все в одну сторону
        a = a1[0] - a2[0]
        b = a1[1] - a2[1]
        c = a1[2] - a2[2]
        
        # Если все коэффициенты равны нулю, решений бесконечно много
        if a == 0 and b == 0 and c == 0:
            return None
        
        # Если уравнение стало линейным (a == 0), решаем линейное уравнение b*x + c = 0
        if a == 0:
            if b == 0:
                return []  # Нет решений, если b == 0 и c != 0
            else:
                x = -c / b
                return [(x, evaluate_function(a1, x))]  # Одно линейное решение (x, y)
        
        # Решение квадратного уравнения
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return []  # Нет решений
        elif discriminant == 0:
            x = -b / (2*a)
            return [(x, evaluate_function(a1, x))]  # Одно решение (x, y)
        else:
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            x2 = (-b - np.sqrt(discriminant)) / (2*a)
            return [(x1, evaluate_function(a1, x1)), (x2, evaluate_function(a1, x2))]

    # Преобразование строк в коэффициенты
    coeffs_1 = list(map(float, a_1.split()))
    coeffs_2 = list(map(float, a_2.split()))
    
    # Нахождение экстремумов
    extremum_1 = find_extremum(coeffs_1)
    extremum_2 = find_extremum(coeffs_2)
    
    # Поиск общих решений функций
    common_solutions = find_common_solutions(coeffs_1, coeffs_2)
    
    return common_solutions


def central_moment(x, order):
    """
    Функция для расчета центрального момента заданного порядка.
    """
    return np.sum((x - np.mean(x)) ** order) / len(x)


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    m2 = central_moment(x, 2)
    m3 = central_moment(x, 3)
    sigma = np.sqrt(m2)
    skewness = m3 / sigma**3
    return round(skewness, 2)


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    m2 = central_moment(x, 2)
    m4 = central_moment(x, 4)
    sigma = np.sqrt(m2)
    kurt = m4 / sigma**4 - 3
    return round(kurt, 2)
