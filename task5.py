import numpy as np


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


def hessenberg(A):
    """ Приведение матрицы A к форме Хессенберга. """
    H = np.copy(A)
    n = H.shape[0]

    for k in range(n-2):
        x = H[k+1:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x + np.sign(x[0]) * e
        u = u / np.linalg.norm(u)
        H[k+1:, k:] -= 2.0 * np.outer(u, u @ H[k+1:, k:])
        H[:, k+1:] -= 2.0 * np.outer(H[:, k+1:] @ u, u)

    return H


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


# Вычисление собственных чисел матрицы /////////////////////////////////////////////////////////////////////////////// #
def power_method(A, delta=1e-8, tol=1e-6, max_iterations=1000):
    """ Степенной метод. """
    n = A.shape[0]

    lambda_k = None
    lambda_prev = np.inf  # Инициализируем переменную для хранения значения собственных чисел

    # Шаг 1. Инициализация случайного вектора и его нормализация
    y = np.random.rand(n)
    z = y / np.linalg.norm(y)

    for _ in range(max_iterations):
        # Шаг 2. Вычисление следующего вектора
        y_k = np.dot(A, z)

        # Нормализация вектора
        z = y_k / np.linalg.norm(y_k)

        # Шаг 3. Вычисление значений собственных чисел
        # найдём индексы множества I (где z[i] больше delta, иначе координата считается нулевой)
        I = [i for i in range(n) if abs(z[i]) > delta]
        lambda_k = np.array([y_k[i] / z[i] for i in I])

        # Шаг 4. Проверка сходимости
        if np.all(np.abs(lambda_k - lambda_prev)) < tol:
            # Если достигнута сходимость, выходим из цикла
            break
        # Сохраняем значения для следующей итерации
        lambda_prev = lambda_k

    return np.mean(lambda_k), z


def inverse_power_method(A, lambda_prev, delta=1e-8, tol=1e-6, max_iterations=1000):
    """ Обратный степенной метод со сдвигами. """
    n = A.shape[0]
    lambda_k = None

    # Шаг 1. Инициализация случайного вектора и его нормализация
    y = np.random.rand(n)
    z = y / np.linalg.norm(y)

    for _ in range(max_iterations):
        # Шаг 2. Решение системы линейных уравнений
        shifted_matrix = A - lambda_prev * np.eye(n)
        y_k = gaussian_elimination(shifted_matrix, z)

        # Нормализация вектора
        z = y_k / np.linalg.norm(y_k)

        # Шаг 3. Вычисление значений собственных чисел
        # найдём индексы множества I (где y[i] больше delta, иначе координата считается нулевой)
        I = [i for i in range(n) if abs(y[i]) > delta]
        lambda_k = lambda_prev + np.mean(np.array([(z[i] / y_k[i]) for i in I]))

        # Шаг 4. Проверка сходимости
        if np.abs(lambda_k - lambda_prev) < tol:
            # Если достигнута сходимость, выходим из цикла
            break
        # Сохраняем значения для следующей итерации
        lambda_prev = lambda_k

    return lambda_k, z


def qr_algorithm(A, tol=1e-8, max_iterations=1000):
    """ QR-алгоритм со сдвигами для нахождения собственных чисел """
    # Шаг 1: Приведение матрицы A к форме Хессенберга для ускорения вычислений
    H = hessenberg(A)
    n = H.shape[0]
    eigenvalues = []

    # Шаг 2: Основной цикл по уменьшению размерности матрицы
    for i in range(n - 1, 0, -1):
        iter_count = 0

        # Шаг 3: Итерации для нахождения одного собственного числа
        while iter_count < max_iterations:
            # Сдвиг по последнему элементу
            shift = H[i, i]

            # Шаг 4: QR-разложение с использованием сдвига
            # выполняем разложение матрицы H[:i+1, :i+1] - shift * I
            shifted_matrix = H[:i + 1, :i + 1] - shift * np.eye(i + 1)
            Q, R = gram_schmidt(shifted_matrix)

            # Шаг 5: Обновление матрицы H как R @ Q + shift * I
            H[:i + 1, :i + 1] = R @ Q + shift * np.eye(i + 1)

            # Шаг 6: Проверка на сходимость по элементу под главной диагональю
            if np.abs(H[i, i - 1]) < tol:
                # Если достигнута сходимость, выходим из цикла
                break
            iter_count += 1

        eigenvalues.append(H[i, i])  # Сохраняем найденное собственное число

        # Шаг 7: Понижение размерности матрицы
        H = H[:i, :i]  # Понижение размерности

    eigenvalues.append(H[0, 0])  # Последнее собственное число
    return np.array(eigenvalues)


# Основные расчёты /////////////////////////////////////////////////////////////////////////////////////////////////// #
def compute_eigenvalues(A, Lambda, method):
    """ Основная функция для запуска методов. """
    n, m = A.shape

    if n != m:
        raise ValueError(f"Матрица должна быть квадратной, но имеет размер {n}x{m}")

    match method:
        case "power":
            eig_val, eig_vec = power_method(A)
            print(f"Наибольшее по модулю собственное число ({method}): {eig_val}")
            print(f"Собственный вектор ({method}): {eig_vec}")

        case "inverse_power":
            for i in range(n):
                sigma0 = Lambda[i, i]
                eig_val, eig_vec = inverse_power_method(A, sigma0)
                print(f"Собственное число ({method}): {eig_val}")
                print(f"Собственный вектор ({method}): {eig_vec}")

        case "qr":
            eigenvalues = qr_algorithm(A)
            print(f"Собственные числа ({method}):\n {eigenvalues}")

        case _:
            raise ValueError("Неизвестный метод. Используйте: ['power', 'inverse_power', 'qr']")


def main():
    n = 5  # Размерность матрицы

    # Генерация случайной диагональной матрицы
    Lambda = np.diag(np.random.rand(n))
    # Генерация случайной матрицы C
    C = np.random.rand(n, n)
    # Матрица A
    A = np.linalg.inv(C) @ Lambda @ C
    print(f"Матрица A:\n {A}")

    # Нахождение собственных чисел матрицы A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(f"Собственные числа матрицы A:\n {eigenvalues}")
    print(f"Собственные векторы матрицы A:\n {eigenvectors}")

    methods = ["power", "inverse_power", "qr"]
    for method in methods:
        print()  # для отступа
        compute_eigenvalues(A=A, Lambda=Lambda, method=method)


if __name__ == "__main__":
    main()
