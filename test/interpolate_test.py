import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange

from task4 import lagrange_interpolation, newton_interpolation


def lagrange_interpolation_test(x_nodes, y_nodes, x_test):
    # Вызов функции для расчета значений полинома Лагранжа (собственная)
    y_custom = lagrange_interpolation(x_nodes, y_nodes, x_test)

    # Сравнение с библиотекой SciPy
    lagrange_poly = lagrange(x_nodes, y_nodes)
    y_scipy = lagrange_poly(x_test)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_test, y_custom, label='Метод Лагранжа (собственный)')
    plt.plot(x_test, y_scipy, '--', label='Метод Лагранжа (SciPy)')
    plt.legend()
    plt.show()


def newton_interpolation_test(x_nodes, y_nodes, x_test):
    # Вызов функции для расчета значений полинома Ньютона (собственная)
    y_custom = newton_interpolation(x_nodes, y_nodes, x_test)

    # Сравнение с библиотекой SciPy
    lagrange_poly = lagrange(x_nodes, y_nodes)  # Использую Лагранжа, так как в scipy нет Ньютона
    y_scipy = lagrange_poly(x_test)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_test, y_custom, label='Метод Ньютона (собственный)')
    plt.plot(x_test, y_scipy, '--', label='Метод Ньютона (SciPy)')
    plt.legend()
    plt.show()


def main():
    # Пример узлов
    x_nodes = np.array([0, 1, 2, 3, 4])
    y_nodes = np.array([1, 2, 0, 2, 1])

    # Точки, для которых мы хотим найти значения сплайна
    x_test = np.linspace(0, 4, 100)

    # Тестирование интерполяции
    lagrange_interpolation_test(x_nodes, y_nodes, x_test)
    newton_interpolation_test(x_nodes, y_nodes, x_test)


if __name__ == '__main__':
    main()
