import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from task4 import linear_spline, quadratic_spline, cubic_spline


def linear_spline_test(x_nodes, y_nodes, x_test):
    # Вызов функции для расчета значений сплайна (собственная)
    y_custom = linear_spline(x_nodes, y_nodes, x_test)

    # Сравнение с библиотекой SciPy
    spline_scipy = make_interp_spline(x_nodes, y_nodes, k=1)
    y_scipy = spline_scipy(x_test)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_test, y_custom, label='Линейный сплайн (собственный)')
    plt.plot(x_test, y_scipy, '--', label='Линейный сплайн (SciPy)')
    plt.legend()
    plt.show()


def quadratic_spline_test(x_nodes, y_nodes, x_test):
    # Вызов функции для расчета значений сплайна (собственная)
    y_custom = quadratic_spline(x_nodes, y_nodes, x_test)

    # Сравнение с библиотекой SciPy
    spline_scipy = make_interp_spline(x_nodes, y_nodes, k=2)
    y_scipy = spline_scipy(x_test)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_test, y_custom, label='Квадратичный сплайн (собственный)')
    plt.plot(x_test, y_scipy, '--', label='Квадратичный сплайн (SciPy)')
    plt.legend()
    plt.show()


def cubic_spline_test(x_nodes, y_nodes, x_test):
    # Вызов функции для расчета значений сплайна (собственная)
    y_custom = cubic_spline(x_nodes, y_nodes, x_test)

    # Сравнение с библиотекой SciPy
    spline_scipy = make_interp_spline(x_nodes, y_nodes, k=3, bc_type='natural')
    y_scipy = spline_scipy(x_test)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_test, y_custom, label='Кубический сплайн (собственный)')
    plt.plot(x_test, y_scipy, '--', label='Кубический сплайн (SciPy)')
    plt.legend()
    plt.show()


def main():
    # Пример узлов
    x_nodes = np.array([0, 1, 2, 3, 4])
    y_nodes = np.array([1, 2, 0, 2, 1])

    # Точки, для которых мы хотим найти значения сплайна
    x_test = np.linspace(0, 4, 100)

    # Тестирование сплайнов
    linear_spline_test(x_nodes, y_nodes, x_test)
    quadratic_spline_test(x_nodes, y_nodes, x_test)
    cubic_spline_test(x_nodes, y_nodes, x_test)


if __name__ == '__main__':
    main()
