import os
import shutil
import math

import numpy as np
import matplotlib.pyplot as plt


def my_func(x):
    """ Пользовательская функция. """
    return x * math.log(x + 2) ** 2


def generate_data(x_values, error_level=0.1, num_values=3):
    """Генерация данных с небольшими случайными ошибками."""
    x_data = []
    y_data = []

    for x in x_values:
        for _ in range(num_values):
            error = np.random.uniform(-error_level, error_level)
            x_data.append(x)
            y_data.append(my_func(x) + error)

    return np.array(x_data), np.array(y_data)


# Вычисление матриц ////////////////////////////////////////////////////////////////////////////////////////////////// #
def vandermonde_matrix(x_nodes, degree):
    """ Построение матрицы Вандермонда. """
    A = np.zeros((len(x_nodes), degree + 1))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = x_nodes[i] ** j

    return A


def legendre_matrix(x_nodes, degree):
    """ Вычисление многочленов Лежандра до заданной степени. """
    x_nodes = 2 * (x_nodes - x_nodes.min()) / (x_nodes.max() - x_nodes.min()) - 1

    n = len(x_nodes)
    P = np.zeros((n, degree + 1))

    # P_0(x) = 1
    P[:, 0] = 1

    if degree > 0:
        # P_1(x) = x
        P[:, 1] = x_nodes

    # Рекурсивное вычисление для P_n(x)
    for k in range(2, degree + 1):
        P[:, k] = ((2 * k - 1) * x_nodes * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k

    return P


def matrix(x_nodes, degree, matrix_type):
    """ Общая функция для вычисления матриц. """
    match matrix_type:
        case "vandermonde":
            return vandermonde_matrix(x_nodes, degree)
        case "legendre":
            return legendre_matrix(x_nodes, degree)
        case _:
            raise ValueError("Метод аппроксимации должен быть из\n"
                             "['vandermonde', 'legendre']")


# Метод наименьших квадратов ///////////////////////////////////////////////////////////////////////////////////////// #
def least_squares_method(x_nodes, y_nodes, degree, matrix_type):
    """ Метод наименьших квадратов. """
    A = matrix(x_nodes, degree, matrix_type)
    ATA = A.T @ A
    ATy = A.T @ y_nodes
    # Решение системы нормальных уравнений: (A^T * A) * c = A^T * y_nodes
    coffs = np.linalg.solve(ATA, ATy)
    return coffs


# Аппроксимация ////////////////////////////////////////////////////////////////////////////////////////////////////// #
def normal_polynomial_approximation(x_nodes, y_nodes, degree, x_points):
    """ Оценка значения степенного полинома с данными коэффициентами. """
    coffs = least_squares_method(x_nodes, y_nodes, degree, matrix_type="vandermonde")

    y_approx = np.zeros_like(x_points)
    for i, coffs in enumerate(coffs):
        y_approx += coffs * x_points ** i

    return y_approx


def orthogonal_polynomial_approximation(x_nodes, y_nodes, degree, x_points):
    """ Оценка значения полинома Лежандра с данными коэффициентами. """
    coffs = least_squares_method(x_nodes, y_nodes, degree, matrix_type="legendre")

    n = len(coffs) - 1
    y_approx = coffs[0] * np.ones_like(x_points)

    if n >= 1:
        P_prev = np.ones_like(x_points)  # P_0(x)
        P_curr = x_points                # P_1(x)
        y_approx += coffs[1] * P_curr   # Добавляем первый член

        # Используем рекуррентную формулу для вычисления P_n(x) для n >= 2
        for k in range(2, n + 1):
            P_next = ((2 * k - 1) * x_points * P_curr - (k - 1) * P_prev) / k
            y_approx += coffs[k] * P_next
            P_prev, P_curr = P_curr, P_next

    return y_approx


def approximate(x_nodes, y_nodes, degree, method, x_points):
    """ Общая функция для аппроксимации с выбором метода. """
    match method:
        case "normal":
            return normal_polynomial_approximation(x_nodes, y_nodes, degree, x_points)
        case "orthogonal":
            return orthogonal_polynomial_approximation(x_nodes, y_nodes, degree, x_points)
        case _:
            raise ValueError("Метод аппроксимации должен быть из\n"
                             "['normal', 'orthogonal']")


# Основные расчёты /////////////////////////////////////////////////////////////////////////////////////////////////// #
def plot_and_save_approximations(a, b, degrees, m_values, plot_dir, approximation_method):
    """ Построение графиков для каждой degree с выбранным методом аппроксимации. """
    x_points = np.linspace(a, b, m_values)
    y_points = np.array([my_func(xi) for xi in x_points])
    x_nodes, y_nodes = generate_data(x_points)

    for degree in degrees:
        # Вычисление аппроксимирующей функции
        y_approx = approximate(x_nodes, y_nodes, degree, approximation_method, x_points)

        print(f"({approximation_method.capitalize()}) degree={degree}, MSE: {np.mean((y_points - y_approx) ** 2)}")

        # Построение графика
        plt.figure(figsize=(12, 6))
        plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
        plt.plot(x_points, y_points, label='Исходная функция my_func(x)')
        plt.plot(x_points, y_approx, linestyle='--',
                 label=f'Полином {approximation_method}, degree={degree}')

        # Настройки графика
        plt.legend()
        plt.title(f"Аппроксимация методом {approximation_method.capitalize()}, degree = {degree}")
        plt.xlabel("x")
        plt.ylabel("my_func(x)")
        plt.grid(True)

        # Сохранение графика как изображения
        filename = f"{plot_dir}/plot_{approximation_method}_{degree}.png"
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
    a, b = -1, 1  # Интервал [a, b]
    degrees = [1, 2, 3, 4, 5]  # Степени аппроксимирующего полинома
    m_values = 100  # Количество семейств точек для вычисления аппроксимации
    print(f"Исследуемый интервал: [{a}, {b}]")

    approximation_methods = ["normal", "orthogonal"]

    # Путь для сохранения результатов
    plot_dir = 'plots/task6'
    create_directory(plot_dir)

    for method in approximation_methods:
        print()  # для отступа
        plot_and_save_approximations(a, b, degrees, m_values, plot_dir, approximation_method=method)


if __name__ == "__main__":
    main()
