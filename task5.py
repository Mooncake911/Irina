import numpy as np


def power_method(A, tol=1e-8, max_iterations=1000):
    """ Степенной метод для нахождения наибольшего по модулю собственного числа. """
    n = A.shape[0]

    y = np.random.rand(n)
    z = y / np.linalg.norm(y)

    for k in range(max_iterations):
        yk = np.dot(A, z)
        zk = yk / np.linalg.norm(yk)

        # Проверка на сходимость
        if np.linalg.norm(zk - y) < tol:
            break

        y = zk

    z = np.dot(A, y)
    eigenvalue = np.dot(y.T, z)  # Наибольшее собственное число
    return eigenvalue, y


def inverse_power_method(A, mu, tol=1e-6, max_iterations=1000):
    """ Обратный степенной метод со сдвигом. """
    n = A.shape[0]

    I = np.eye(n)
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for i in range(max_iterations):
        try:
            w = np.linalg.solve(A - mu * I, v)
        except np.linalg.LinAlgError:
            break

        v_new = w / np.linalg.norm(w)

        # Проверка на сходимость
        if np.linalg.norm(v_new - v) < tol:
            break

        v = v_new

    eigenvalue = np.dot(v.T, np.dot(A, v)) + mu  # Собственное число
    return eigenvalue, v


def hessenberg(A):
    """ Приведение матрицы к форме Хессенберга. """
    return np.linalg.qr(A)[1]


def qr_method(A, tol=1e-8, max_iterations=1000):
    """ QR-алгоритм со сдвигами. """
    A_hessenberg = hessenberg(A)
    n = A.shape[0]

    for i in range(max_iterations):
        Q, R = np.linalg.qr(A_hessenberg)
        A_hessenberg = R @ Q

        # Проверка малости поддиагональных элементов
        if np.all(np.abs(A_hessenberg[np.tril_indices(n, -1)]) < tol):
            break

    eigenvalues = np.diag(A_hessenberg)  # Собственные числа
    return eigenvalues


def compute_eigenvalues(A, Lambda, method):
    """ Основная функция для запуска методов. """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Матрица должна быть квадратной, но имеет размер {n}x{m}")

    match method:
        case "power":
            eig_val, eig_vec = power_method(A)
            print(f"Наибольшее собственное число (степенной метод):\n {eig_val}")
            print(f"Собственный вектор (степенной метод):\n {eig_vec}")

        case "inverse_power":
            eigenvalues = []
            eigenvectors = []
            for i in range(n):
                mu = Lambda[i, i]  # Используем известные собственные числа
                eig_val, eig_vec = inverse_power_method(A, mu)
                eigenvalues.append(eig_val)
                eigenvectors.append(eig_vec)

            print(f"Собственные числа (обратный степенной метод):\n {eigenvalues}")
            print(f"Собственные вектора (обратный степенной метод):\n {eigenvectors}")

        case "qr":
            eigenvalues = qr_method(A)
            print(f"Собственные числа (QR-алгоритм):\n {eigenvalues}")

        case _:
            raise ValueError("Неизвестный метод. Используйте: ['power', 'inverse', 'qr']")


def main():
    n = 5  # Размерность матрицы

    # Генерация случайной диагональной матрицы
    Lambda = np.diag(np.random.rand(n))
    # Генерация случайной матрицы C
    C = np.random.rand(n, n)
    # Матрица A
    A = np.linalg.inv(C) @ Lambda @ C
    print(A)

    methods = ["power", "inverse_power", "qr"]
    for method in methods:
        print()  # для отступа
        compute_eigenvalues(A=A, Lambda=Lambda, method=method)


if __name__ == "__main__":
    main()
