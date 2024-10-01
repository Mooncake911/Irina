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
    """ Приведение матрицы к форме Хессенберга. """
    return np.linalg.qr(A)[1]


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
    """ QR-алгоритм со сдвигами для нахождения всех собственных чисел матрицы A. """
    n = A.shape[0]
    A_hess = hessenberg(A)  # Приведение к форме Хессенберга
    eigenvalues = []

    while n > 1:
        for _ in range(max_iterations):
            # Сдвиг: используем нижний правый элемент в качестве сдвига
            sigma = A_hess[n - 1, n - 1]

            # QR-разложение для (A - sigma*I)
            Q, R = gram_schmidt(A_hess - sigma * np.eye(n))
            A_hess = np.dot(R, Q) + sigma * np.eye(n)

            # Проверяем сходимость для нижнего правого элемента
            if np.abs(A_hess[n - 1, n - 2]) < tol:
                # Найдено собственное число
                eigenvalues.append(A_hess[n - 1, n - 1])
                A_hess = A_hess[:n - 1, :n - 1]  # Уменьшаем размерность матрицы
                n -= 1
                break

    # Для матрицы 1x1
    if n == 1:
        eigenvalues.append(A_hess[0, 0])

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
