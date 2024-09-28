import os
import math
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def my_func(x):
    """ Пользовательская функция. """
    # 1 / math.tan(x) + x ** 2 периодическая (0, pi)
    # x ** 2 * math.cos(2*x) + 1 непрерывная
    return x ** 2 * math.cos(2*x) + 1


def generate_nodes_and_values(x_nodes):
    """ Вычисления значений функции. """
    y_nodes = np.array([my_func(x) for x in x_nodes])
    return y_nodes


# Распределения ////////////////////////////////////////////////////////////////////////////////////////////////////// #
def chebyshev_distributed_nodes(a, b, n):
    """ Генерация узлов Чебышёва на интервале [a, b]. """
    i = np.arange(0, n + 1)
    nodes = 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * (n + 1))) + 0.5 * (a + b)
    return nodes


def evenly_distributed_nodes(a, b, n):
    """ Генерация равномерно распределенных узлов. """
    return np.linspace(a, b, n)


def distribution(a, b, n, dist):
    """ Общая функция для распределения с выбором метода. """
    match dist:
        case "evenly":
            return evenly_distributed_nodes(a, b, n)
        case "chebyshev":
            return chebyshev_distributed_nodes(a, b, n)
        case _:
            raise ValueError("Распределение должно быть 'evenly' или 'chebyshev'")


# Интерполяции /////////////////////////////////////////////////////////////////////////////////////////////////////// #
def lagrange_interpolation(x, y, x_test):
    """ Интерполяция по методу Лагранжа. """
    n = len(x)
    L = np.zeros_like(x_test)

    def lagrange_multiplier(i):
        result = 1
        for j in range(n):
            if i != j:
                result *= (x_test - x[j]) / (x[i] - x[j])
        return result

    for k in range(n):
        L += y[k] * lagrange_multiplier(k)

    return L


def newton_interpolation(x, y, x_test):
    """ Интерполяция по методу Ньютона. """
    n = len(x)
    divided_diff = np.zeros((n, n))
    divided_diff[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            divided_diff[i][j] = (divided_diff[i + 1][j - 1] - divided_diff[i][j - 1]) / (x[i + j] - x[i])

    def newton_polynomial(x_val):
        result = divided_diff[0, 0]
        w = 1
        for i in range(1, n):
            w *= (x_val - x[i - 1])
            result += divided_diff[0, i] * w
        return result

    return np.array([newton_polynomial(xi) for xi in x_test])


def spline_interpolation(x_nodes, y_nodes, x_test, degree=3):
    """ Интерполяция сплайнами с возможностью выбора степени. """
    # Проверим, что x_nodes строго возрастают
    if not np.all(np.diff(x_nodes) > 0):
        sorted_indices = np.argsort(x_nodes)
        x_nodes = x_nodes[sorted_indices]
        y_nodes = y_nodes[sorted_indices]
    spline = make_interp_spline(x_nodes, y_nodes, k=degree)
    return spline(x_test)


def linear_spline(x_nodes, y_nodes, x_test):
    """ Линейный сплайн S{1,0}"""
    a = np.diff(y_nodes) / np.diff(x_nodes)
    b = y_nodes[:-1]

    y_spline = np.zeros_like(x_test)
    for i in range(len(b)):
        # Индексируем, где x_test находится в текущем отрезке
        idx = (x_test >= x_nodes[i]) & (x_test <= x_nodes[i + 1])
        # Вычисляем значения сплайна для текущего отрезка
        dx = x_test[idx] - x_nodes[i]
        y_spline[idx] = a[i] * dx + b[i]

    return y_spline


def quadratic_spline(x_nodes, y_nodes, x_test):
    """ Квадратичный сплайн S{2,1}(x) """
    n = len(x_nodes) - 1  # Количество отрезков
    h = np.diff(x_nodes)  # Шаги между узлами

    # Инициализация матрицы A и вектора b
    A = np.zeros((n + 1, n + 1))  # Матрица размером (n+1, n+1)
    b_vec = np.zeros(n + 1)  # Вектор для системы уравнений

    # Заполнение матрицы A и вектора b
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    # Граничные условия: натуральный сплайн
    A[0, 0], A[n, n] = 1, 1  # Вторая производная на концах равна нулю

    # Решение системы для коэффициентов c (вторая производная)
    c = np.linalg.solve(A, b_vec)

    # Вычисление коэффициентов b и a для каждого отрезка
    b_coeff = np.diff(y_nodes) / h - h * (2 * c[:-1] + c[1:]) / 3
    a_coeff = (c[1:] - c[:-1]) / (3 * h)
    y_nodes_coeff = y_nodes[:-1]

    # Инициализация массива для значений сплайна
    y_spline = np.zeros_like(x_test)

    # Интерполяция значений сплайна для заданных x_test
    for i in range(len(y_nodes_coeff)):
        # Определяем индексы, где x_test находится в текущем отрезке
        idx = (x_test >= x_nodes[i]) & (x_test <= x_nodes[i + 1])
        dx = x_test[idx] - x_nodes[i]
        # Вычисляем значения сплайна на текущем отрезке
        y_spline[idx] = a_coeff[i] * dx ** 2 + b_coeff[i] * dx + y_nodes_coeff[i]

    return y_spline


def cubic_spline(x_nodes, y_nodes, x_test):
    """ Кубический сплайн S{3,2} """
    n = len(x_nodes) - 1  # Количество отрезков
    h = np.diff(x_nodes)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)

    # Заполнение системы для кубического сплайна
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h[i] - (y_nodes[i] - y_nodes[i - 1]) / h[i - 1])

    # Граничные условия: натуральный сплайн
    A[0, 0] = A[n, n] = 1
    c = np.linalg.solve(A, b)

    a = y_nodes[:-1]
    b = np.diff(y_nodes) / h - h * (2 * c[:-1] + c[1:]) / 3
    d = (c[1:] - c[:-1]) / (3 * h)

    y_spline = np.zeros_like(x_test)
    for i in range(len(a)):
        idx = (x_test >= x_nodes[i]) & (x_test <= x_nodes[i + 1])
        dx = x_test[idx] - x_nodes[i]
        y_spline[idx] = a[i] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return y_spline


def interpolate(x_nodes, y_nodes, x_fine, method):
    """ Общая функция для интерполяции с выбором метода. """
    match method:
        case "lagrange":
            return lagrange_interpolation(x_nodes, y_nodes, x_fine)
        case "newton":
            return newton_interpolation(x_nodes, y_nodes, x_fine)
        case "spline":
            return spline_interpolation(x_nodes, y_nodes, x_fine)
        case _:
            raise ValueError("Метод интерполяции должен быть из ['lagrange', 'newton', 'spline']")


# Основные расчёты /////////////////////////////////////////////////////////////////////////////////////////////////// #
def max_deviation(f, x_values, p):
    """ Максимальное отклонение. """
    deviations = np.abs(np.array([f(xi) for xi in x_values]) - p)
    return np.max(deviations)


def plot_and_save_interpolations(a, b, n_values, m_values, plot_dir, interpolation_method, distribution_method):
    """ Построение графиков для каждого n с выбранным методом интерполяции. """
    for i in range(len(n_values)):
        # Генерация узлов интерполирования (n)
        x_nodes = distribution(a, b, n_values[i], distribution_method)
        y_nodes = generate_nodes_and_values(x_nodes)

        # Генерация точек для вычисления отклонения (m)
        x_fine = distribution(a, b, m_values[i], "evenly")

        # Интерполяция
        interp_values = interpolate(x_nodes, y_nodes, x_fine, interpolation_method)
        max_dev = max_deviation(my_func, x_fine, interp_values)
        print(
            f"n = {n_values[i]}, m = {m_values[i]} ({distribution_method}): Макс. откл. ({interpolation_method}) = {max_dev:.10f}")

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(x_fine, [my_func(xi) for xi in x_fine], label='Исходная функция my_func(x)')
        plt.plot(x_fine, interp_values, linestyle='--',
                 label=f'Полином {interpolation_method}, n={n_values[i]}, m={m_values[i]}')

        # Настройки графика
        plt.legend()
        plt.title(f"Интерполяция методом {interpolation_method.capitalize()}, n = {n_values[i]}, m = {m_values[i]}")
        plt.xlabel("x")
        plt.ylabel("my_func(x)")
        plt.grid(True)

        # Сохранение графика как изображения
        filename = f"{plot_dir}/plot_{distribution_method}_{interpolation_method}_{n_values[i]}_{m_values[i]}.png"
        plt.savefig(filename)
        plt.close()


# Функции для демонстрации результатов /////////////////////////////////////////////////////////////////////////////// #
def create_directory(path):
    """ Создание директории для хранения графиков. """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////// #
def main():
    a, b = -10 + 0.1, 10 - 0.1  # Интервал [a, b]
    n_values = [5, 10, 15, 20]  # Количество узлов интерполирования
    m_values = [500, 1000, 1500, 2000]  # Количество точек для вычисления отклонения
    print(f"Исследуемый интервал: [{a}, {b}]")

    distribution_methods = ["evenly", "chebyshev"]
    interpolation_methods = ["lagrange", "newton", "spline"]

    # Путь для сохранения результатов
    plot_dir = 'plots'
    create_directory(plot_dir)

    for dist_method in distribution_methods:
        for inter_method in interpolation_methods:
            print()  # для отступа
            plot_and_save_interpolations(a, b, n_values, m_values, plot_dir,
                                         interpolation_method=inter_method,
                                         distribution_method=dist_method)


if __name__ == "__main__":
    main()
