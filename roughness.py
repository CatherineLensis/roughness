import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, measure
from scipy.ndimage import gaussian_filter


# Функция бутстрэп для вычисления погрешности
def bootstrap_std(data, num_samples=1000):
    data_1d = data.flatten()  # Преобразуем массив в одномерный
    if len(data_1d) == 0:  # Проверяем, что данные не пустые
        raise ValueError("Пустой массив данных для бутстрэпирования.")

    # Применяем бутстрэп к одномерному массиву
    bootstrap_samples = np.random.choice(data_1d, size=(num_samples, len(data_1d)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    return np.std(bootstrap_means)
# Функция для расчета среднеквадратичной шероховатости (Rq)
def calculate_rms_roughness(edge_image, pixel_size):
    """
    Вычисление среднеквадратичной шероховатости (Rq) по краевому изображению.
    :param edge_image: 2D массив изображения краевой части
    :param pixel_size: размер пикселя (в микрометрах)
    :return: значение Rq (в микрометрах)
    """
    if edge_image.size == 0:  # Проверяем, что изображение не пустое
        raise ValueError("Пустое изображение!")

    smoothed_image = gaussian_filter(edge_image.astype(np.float64), sigma=1)
    mean_height = np.mean(smoothed_image)
    squared_deviations = (smoothed_image - mean_height) ** 2

    rms_roughness = np.sqrt(np.mean(squared_deviations)) * pixel_size

    # Используем одномерные данные для бутстрэпирования

    rms_error1 = bootstrap_std(np.sqrt(squared_deviations)) * pixel_size

    return rms_roughness, rms_error1

# Функция для расчета средней арифметической шероховатости (Ra)
def calculate_arithmetic_roughness(edge_image, pixel_size):
    """
    Вычисление средней арифметической шероховатости (Ra).
    :param edge_image: 2D массив изображения краевой части
    :param pixel_size: размер пикселя (в микрометрах)
    :return: значение Ra (в микрометрах)
    """
    smoothed_image = gaussian_filter(edge_image, sigma=1)
    mean_height = np.mean(smoothed_image)
    deviations = np.abs(smoothed_image - mean_height)
    ra_roughness = np.mean(deviations) * pixel_size
    # Погрешность как стандартное отклонение отклонений
    error = np.std(deviations) * pixel_size
    return ra_roughness, error
# Функция для расчета максимальной высоты профиля (Rz)
def calculate_max_height_roughness(edge_image, pixel_size):
    """
    Вычисление максимальной высоты профиля (Rz).
    :param edge_image: 2D массив изображения краевой части
    :param pixel_size: размер пикселя (в микрометрах)
    :return: значение Rz (в микрометрах)
    """
    heights = np.argmax(edge_image, axis=0)
    min_height = np.min(heights)
    max_height = np.max(heights)
    rz = (max_height - min_height) * pixel_size
    # Погрешность как разница стандартных отклонений высот
    error = bootstrap_std(heights) * pixel_size
    return rz, error

# Функция для удаления шума
def remove_noise(image_data):
    # Применение медианный фильтр для снижения шумов
    from scipy.ndimage import median_filter
    return median_filter(image_data, size=3)
def extract_edge(image_data):
    """
    Выделение краев объекта на изображении (например, пластинки).
    :param image_data: 2D массив изображения
    :return: Изображение с выделенным краем
    """
    # Применяем фильтр Собеля для выделения границ
    edges = filters.sobel(image_data)

    # Применяем бинаризацию для выделения ярко выраженного края
    threshold = filters.threshold_otsu(edges)
    edge_binary = edges > threshold * 0.25
    return edge_binary

# Загружаем изображение в формате .jpg с помощью PIL
def load_image_jpg(image_path):
    """
    Загрузка изображения в формате .jpg и преобразование в 2D массив
    :param image_path: путь к изображению .jpg
    :return: 2D массив изображения
    """
    # Открываем изображение и преобразуем в градации серого
    image = Image.open(image_path).convert('L')
    image_data = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image_data
# Функция бутстрэп для вычисления погрешности

# Путь к изображению
image_path = "C:\\Users\\Кутенок\\Desktop\\ИИ изображения\\№5\\снизу\\SNAP-130659-0049.jpg"
#

# Загружаем изображение
image_data = load_image_jpg(image_path)

# Задаем размер пикселя (в микрометрах)
pixel_size = 3.5 # 3.5 микрон на пиксель

# Выделяем край пластинки
edge_image = extract_edge(image_data)

# Рассчитываем Rq, Ra, Rz с их погрешностями
rms_roughness_value, rms_error = calculate_rms_roughness(edge_image, pixel_size)
arithmetic_roughness_value, ra_error = calculate_arithmetic_roughness(edge_image, pixel_size)
max_height_roughness_value, rz_error = calculate_max_height_roughness(edge_image, pixel_size)

# Выводим результаты с погрешностями

# Погрешность!!!
# Суммирование среднего шероховатостей всех снимков

# Выводим результаты с погрешностями
print(f"Среднеквадратичная шероховатость (Rq): {rms_roughness_value:.10f} ± {rms_error:.4f} микрометров")
print(f"Средняя арифметическая шероховатость (Ra): {arithmetic_roughness_value:.10f} ± {ra_error:.4f} микрометров")
print(f"Максимальная высота профиля (Rz): {max_height_roughness_value:.10f} ± {rz_error:.4f} микрометров")
with open('result_снизу.txt', 'a') as file:
    file.write(str(rms_roughness_value) + " ")
    file.write(str(rms_error))

# Визуализируем исходное изображение и край
plt.figure(figsize=(12, 6))

# Исходное изображение
plt.subplot(1, 2, 1)
plt.imshow(image_data, cmap='gray')
plt.title('Изображение поверхности (РЭМ)')

# Выводим значение шероховатости на изображении с выделенным краем
plt.subplot(1, 2, 2)
plt.imshow(edge_image, cmap='gray')
plt.title('Выделенный край пластинки')

# Выводим значения шероховатостей с погрешностями на изображении
plt.text(15, 90, f'Rq: {rms_roughness_value:.4f} ± {rms_error:.4f} µm', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(15, 210, f'Ra: {arithmetic_roughness_value:.4f} ± {ra_error:.4f} µm', color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(15, 330, f'Rz: {max_height_roughness_value:.4f} ± {rz_error:.4f} µm', color='blue', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.show()