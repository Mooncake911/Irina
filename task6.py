import os
import shutil
import math

import numpy as np
import matplotlib.pyplot as plt


def my_func(x):
    """ Пользовательская функция. """
    return x * math.log(x + 2) ** 2


def generate_data(f, x_values, error_level=0.1, num_values=3):
    """ Генерация данных с небольшими случайными ошибками. """
    x_data = []
    y_data = []

    for x in x_values:
        for _ in range(num_values):
            error = np.random.uniform(-error_level, error_level)
            x_data.append(x)
            y_data.append(f(x) + error)

    return np.array(x_data), np.array(y_data)


# Дополнительные вычисления ////////////////////////////////////////////////////////////////////////////////////////// #
def gaussian_elimination(A, b):
    """ Решение системы линейных уравнений Ax = b методом Гаусса. """
    n = len(b)

    # Прямой ход
    for i in range(n):
        # Нормализация текущей строки
        factor = A[i, i]
        A[i] = A[i] / factor
        b[i] /= factor

        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            b[j] -= factor * b[i]

    # Обратный ход
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i + 1:], x[i + 1:])

    return x


def vandermonde(x_nodes, degree):
    """ Построение матрицы Вандермонда. """
    A = np.zeros((len(x_nodes), degree + 1))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = x_nodes[i] ** j

    return A


def gram_schmidt(A):
    """ Ортогонализация по Грамму-Шмидту. """
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for i in range(m):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], Q[:, i])
            Q[:, i] -= R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(Q[:, i])

        # Проверка деления на ноль
        if R[i, i] > 1e-10:
            Q[:, i] /= R[i, i]
        else:
            # Если R[i, i] близок к нулю, то нормализуем вектор Q[:, i] и устанавливаем R[i, i] в 1
            Q[:, i] = np.zeros(n)
            Q[i, i] = 1.0

    return Q, R


# def legendre(x_nodes, degree):
#     """ Вычисление многочленов Лежандра до заданной степени. """
#     x_nodes = 2 * (x_nodes - x_nodes.min()) / (x_nodes.max() - x_nodes.min()) - 1
#
#     n = len(x_nodes)
#     P = np.zeros((n, degree + 1))
#
#     # P_0(x) = 1
#     P[:, 0] = 1
#
#     if degree > 0:
#         # P_1(x) = x
#         P[:, 1] = x_nodes
#
#     # Рекурсивное вычисление для P_n(x)
#     for k in range(2, degree + 1):
#         P[:, k] = ((2 * k - 1) * x_nodes * P[:, k - 1] - (k - 1) * P[:, k - 2]) / k
#
#     return P


# Аппроксимация ////////////////////////////////////////////////////////////////////////////////////////////////////// #
def normal_polynomial_approximation(x_nodes, y_nodes, degree):
    """ Оценка значения степенного полинома с данными коэффициентами. """
    # Создаем матрицу степеней узлов
    A = vandermonde(x_nodes, degree)
    # Метод наименьших квадратов.
    ATA = A.T @ A
    ATy = A.T @ y_nodes
    # Вычисление коэффициентов: (A^T * A) * c = A^T * y_nodes
    coffs = gaussian_elimination(ATA, ATy)
    return coffs


def orthogonal_polynomial_approximation(x_nodes, y_nodes, degree):
    """ Оценка значения степенного полинома с данными коэффициентами. """
    # Создаем матрицу степеней узлов
    A = vandermonde(x_nodes, degree)
    # Метод наименьших квадратов. Ортогонализация с помощью Грамма-Шмидта.
    Q, R = gram_schmidt(A)
    QTy = Q.T @ y_nodes
    # Вычисление коэффициентов: R * c = Q^T * y_nodes
    coffs = gaussian_elimination(R, QTy)
    return coffs


def approximate(x_nodes, y_nodes, degree, method):
    """ Общая функция для аппроксимации с выбором метода. """
    match method:
        case "normal":
            return normal_polynomial_approximation(x_nodes, y_nodes, degree)
        case "orthogonal":
            return orthogonal_polynomial_approximation(x_nodes, y_nodes, degree)
        case _:
            raise ValueError("Метод аппроксимации должен быть из\n"
                             "['normal', 'orthogonal']")


# Основные расчёты /////////////////////////////////////////////////////////////////////////////////////////////////// #
def plot_and_save_approximations(x_nodes, y_nodes, x_points, y_points, degrees, plot_dir, approximation_method):
    """ Построение графиков для каждой degree с выбранным методом аппроксимации. """
    for degree in degrees:
        # Вычисление коэффициентов аппроксимирующей функции
        coffs = approximate(x_nodes, y_nodes, degree, approximation_method)

        # Вычисление значений полинома
        y_approx = np.zeros_like(x_points)
        for i, coff in enumerate(coffs):
            y_approx += coff * x_points ** i

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

    # Генерация данных
    x_points = np.linspace(a, b, m_values)
    y_points = np.array([my_func(xi) for xi in x_points])
    x_nodes, y_nodes = generate_data(my_func, x_points)

    for method in approximation_methods:
        print()  # для отступа
        plot_and_save_approximations(x_nodes, y_nodes, x_points, y_points, degrees, plot_dir,
                                     approximation_method=method)


if __name__ == "__main__":
    main()
