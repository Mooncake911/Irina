import numpy as np
from numpy.polynomial import Chebyshev
from matplotlib import pyplot as plt

from task6 import (generate_data,
                   normal_polynomial_approximation, orthogonal_polynomial_approximation)


def normal_polynomial_approximation_test(x_nodes, y_nodes, degree, x_points, y_points):
    # Собственная реализация
    coffs_custom = normal_polynomial_approximation(x_nodes, y_nodes, degree)
    y_custom = np.zeros_like(x_points)
    for i, coff in enumerate(coffs_custom):
        y_custom += coff * x_points ** i

    # Сравнение с библиотекой NumPy
    coffs_numpy = np.polyfit(x_nodes, y_nodes, degree)
    y_numpy = np.polyval(coffs_numpy, x_points)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_points, y_points, label='Исходная функция')
    plt.plot(x_points, y_custom, label='Метод МНК (собственный)')
    plt.plot(x_points, y_numpy, '--', label='Метод МНК (NumPy)')
    plt.legend()
    plt.title('Сравнение нормальной полиномиальной аппроксимации с NumPy')
    plt.show()


def orthogonal_polynomial_approximation_test(x_nodes, y_nodes, degree, x_points, y_points):
    # Собственная реализация
    coffs_custom = orthogonal_polynomial_approximation(x_nodes, y_nodes, degree)
    y_custom = np.zeros_like(x_points)
    for i, coff in enumerate(coffs_custom):
        y_custom += coff * x_points ** i

    # Сравнение с библиотекой NumPy (в numpy нет реализации с Граммом-Шмидтом поэтому Чебышев)
    coffs_numpy = Chebyshev.fit(x_nodes, y_nodes, degree)
    coffs_numpy = coffs_numpy.convert().coef
    y_numpy = Chebyshev(coffs_numpy)(x_points)

    # Визуализация
    plt.plot(x_nodes, y_nodes, 'ro', label='Узлы')
    plt.plot(x_points, y_points, label='Исходная функция')
    plt.plot(x_points, y_custom, label='Метод МНК (собственный)')
    plt.plot(x_points, y_numpy, '--', label='Метод МНК (NumPy)')
    plt.legend()
    plt.title('Сравнение ортогональной полиномиальной аппроксимации с NumPy')
    plt.show()


def main():
    a, b = -np.pi, np.pi
    degree = 3
    my_fun = np.sin  # Пример функции для аппроксимации

    # Генерация данных
    x_points = np.linspace(a, b, 50)
    y_points = my_fun(x_points)
    x_nodes, y_nodes = generate_data(my_fun, x_points)

    # Тестирование аппроксимаций
    normal_polynomial_approximation_test(x_nodes, y_nodes, degree, x_points, y_points)
    orthogonal_polynomial_approximation_test(x_nodes, y_nodes, degree, x_points, y_points)


if __name__ == '__main__':
    main()
