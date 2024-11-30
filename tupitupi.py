import numpy as np


# Функция для загрузки данных из файла
def load_data(filename):
    data = np.loadtxt(filename, delimiter=' ')
    roughness_values = data[:, 0]  # Первый столбец — шероховатость
    errors = data[:, 1]  # Второй столбец — погрешности
    return roughness_values, errors


# Функция для вычисления средней арифметической шероховатости и погрешности
def calculate_average_roughness(roughness_values, errors):
    # Среднее арифметическое шероховатости
    avg_roughness = np.mean(roughness_values)

    # Средняя погрешность через сумму квадратов
    avg_error = np.sqrt(np.sum(errors ** 2)) / len(errors)

    return avg_roughness, avg_error


# Функция для оценки достаточного количества снимков
def estimate_sufficient_samples(roughness_values, target_error):
    current_mean = np.mean(roughness_values)
    current_std = np.std(roughness_values, ddof=1)  # стандартное отклонение выборки
    n_required = (1.96 * current_std / target_error) ** 2  # оценка достаточного числа снимков
    return int(np.ceil(n_required))


# Основная функция программы
def main(filename, target_error):
    # Загружаем данные
    roughness_values, errors = load_data(filename)

    # Вычисляем средние значения
    avg_roughness, avg_error = calculate_average_roughness(roughness_values, errors)

    # Оцениваем, сколько снимков достаточно для целевой погрешности
    sufficient_samples = estimate_sufficient_samples(roughness_values, target_error)

    # Вывод результатов
    print(f"Средняя арифметическая шероховатость: {avg_roughness:.4f} ± {avg_error:.4f}")
    print(f"Для достижения целевой погрешности ±{target_error} потребуется минимум {sufficient_samples} снимков.")


if __name__ == "__main__":
    filename = "result_снизу.txt"
    target_error = 0.02
    main(filename, target_error)
