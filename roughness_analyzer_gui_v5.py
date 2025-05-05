import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure as mplfig
import threading
import csv
import cv2
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

class RoughnessAnalyzer:
    def __init__(self, master):
        self.master = master
        self.master.title("Анализатор шероховатости")
        self.master.geometry("1200x800")
        self.master.minsize(1000, 700)

        # Переменные
        self.image_files = []
        self.result_data = []  # Результаты текущего анализа
        self.all_results = []  # Все результаты всех анализов
        self.pixel_size = tk.DoubleVar(value=3.5)
        self.target_error = tk.DoubleVar(value=0.02)
        self.current_image_path = None
        self.current_image_data = None
        self.current_coordinates = None
        self.coordinates_csv_path = None
        self.output_dir = os.getcwd()

        self._create_ui()

        if not HAS_OPENPYXL:
            messagebox.showwarning(
                "Предупреждение",
                "Пакет openpyxl не установлен. Данные будут сохраняться в формате CSV вместо Excel.\n"
                "Для сохранения в Excel используйте команду: pip install openpyxl"
            )

    def _create_ui(self):
        # --- Фрейм управления ---
        control_frame = ttk.LabelFrame(self.master, text="Управление")
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # --- Фрейм визуализации ---
        self.image_frame = ttk.LabelFrame(self.master, text="Визуализация")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # --- Фрейм результатов ---
        results_frame = ttk.LabelFrame(self.master, text="Результаты")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Элементы управления
        ttk.Label(control_frame, text="Размер пикселя (мкм):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.pixel_size, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(control_frame, text="Целевая погрешность:").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(control_frame, textvariable=self.target_error, width=10).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)

        # Кнопки управления
        ttk.Button(control_frame, text="Загрузить изображения", command=self.load_images).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(control_frame, text="Рассчитать координаты", command=self.calculate_and_save_coordinates).grid(row=0, column=5, padx=5, pady=5)
        ttk.Button(control_frame, text="Выбрать CSV", command=self.select_coordinates_csv).grid(row=0, column=6, padx=5, pady=5)
        ttk.Button(control_frame, text="Анализировать", command=self.analyze_images).grid(row=1, column=4, padx=5, pady=5)
        ttk.Button(control_frame, text="Сохранить результаты", command=self.save_results).grid(row=1, column=5, padx=5, pady=5, columnspan=2)

        control_frame.columnconfigure(8, weight=1)

        # Визуализация
        self.fig = mplfig.Figure(figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, self.image_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.fig.add_subplot(121)
        self.fig.add_subplot(122)
        self.fig.tight_layout()

        # Таблица результатов
        results_subframe = ttk.Frame(results_frame)
        results_subframe.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        columns = ("Файл", "Rq (мкм)", "Rq погрешность", "Ra (мкм)", "Ra погрешность", "Rz (мкм)", "Rz погрешность")
        self.results_tree = ttk.Treeview(results_subframe, columns=columns, show="headings")

        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(results_subframe, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Статистика
        self.stats_frame = ttk.Frame(results_frame)
        self.stats_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        ttk.Label(self.stats_frame, text="Средние значения:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.avg_roughness = tk.StringVar(value="")
        ttk.Label(self.stats_frame, textvariable=self.avg_roughness).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Прогресс и статус бар
        status_progress_frame = ttk.Frame(self.master)
        status_progress_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.progress_frame = ttk.Frame(status_progress_frame)
        self.progress_frame.pack(fill=tk.X, expand=True)

        ttk.Label(self.progress_frame, text="Прогресс:").pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL,
                                            length=300, mode='determinate',
                                            variable=self.progress_var)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.status_var = tk.StringVar(value="Готов к работе")
        self.statusbar = ttk.Label(status_progress_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(fill=tk.X, pady=(5,0))

    def load_images(self):
        files = filedialog.askopenfilenames(
            title="Выберите изображения",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")]
        )

        if files:
            self.image_files = list(files)
            self.status_var.set(f"Загружено {len(files)} изображений")
            logging.info(f"Загружено {len(files)} изображений.")

            # Не очищаем all_results, только текущие результаты
            for i in self.results_tree.get_children():
                self.results_tree.delete(i)
            self.result_data = []
            self.current_coordinates = None
            self.coordinates_csv_path = None
            # Перерисовываем таблицу с сохраненными результатами
            for result in self.all_results:
                self._update_tree(result)
            self.calculate_statistics()

            if len(files) > 0:
                self.display_image_with_coordinates(files[0], None)

            self.progress_var.set(0)

    def select_coordinates_csv(self):
        if not self.image_files:
            messagebox.showwarning("Предупреждение", "Сначала загрузите хотя бы одно изображение!")
            return

        filepath = filedialog.askopenfilename(
            title="Выберите CSV файл с координатами",
            filetypes=[("CSV files", "*.csv")]
        )
        if filepath:
            self.coordinates_csv_path = filepath
            self.current_coordinates = None
            self.status_var.set(f"Выбран CSV файл: {os.path.basename(filepath)}")
            logging.info(f"Выбран CSV файл с координатами: {filepath}")
            try:
                coords = self.load_and_process_coordinates(source_csv=filepath)
                if self.current_image_path:
                    self.display_image_with_coordinates(self.current_image_path, coords['raw'])
                else:
                    messagebox.showinfo("Информация", "CSV файл выбран. Отображение координат будет после загрузки изображения.")
            except Exception as e:
                messagebox.showerror("Ошибка CSV", f"Не удалось загрузить или обработать CSV файл: {e}")
                self.coordinates_csv_path = None
                logging.error(f"Ошибка загрузки/обработки CSV '{filepath}': {e}")

    def load_image_grayscale(self, image_path):
        try:
            with open(image_path, 'rb') as f:
                image_data = np.frombuffer(f.read(), np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось декодировать изображение: {os.path.basename(image_path)}")
            return image
        except Exception as e:
            logging.error(f"Ошибка при загрузке изображения '{image_path}': {e}")
            messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить изображение: {os.path.basename(image_path)}\n{e}")
            return None

    def roll(self, a, b, dx=1, dy=1):
        shape = a.shape[:-2] + ((a.shape[-2] - b.shape[-2]) // dy + 1,) + ((a.shape[-1] - b.shape[-1]) // dx + 1,) + b.shape
        strides = a.strides[:-2] + (a.strides[-2] * dy,) + (a.strides[-1] * dx,) + a.strides[-2:]
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def calculate_edge_coordinates(self, edge_image, window_size=(1, 1)):
        if edge_image.ndim > 2:
            edge_image = edge_image[:, :, 0]
        if edge_image.size == 0 or window_size[0] * window_size[1] == 0:
            logging.warning("calculate_edge_coordinates: edge_image or window is empty.")
            return np.empty((0, 2))
        if edge_image.shape[0] < window_size[0] or edge_image.shape[1] < window_size[1]:
            logging.warning("calculate_edge_coordinates: edge_image is smaller than window size.")
            return np.empty((0, 2))

        window = np.ones(window_size)
        rolled_edges = self.roll(edge_image, window, dx=1, dy=1)
        coordinates = []
        for i in range(rolled_edges.shape[0]):
            for j in range(rolled_edges.shape[1]):
                window_data = rolled_edges[i, j]
                if np.any(window_data):
                    coordinates.append((j * window_size[1], i * window_size[0]))
        return np.array(coordinates)

    def save_coordinates_to_csv(self, coordinates, filename):
        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["X", "Y"])
                if coordinates is not None and coordinates.shape[0] > 0:
                    if coordinates.ndim == 2 and coordinates.shape[1] == 2:
                        writer.writerows(coordinates)
                    else:
                        logging.warning(f"Unexpected coordinate format for saving: {coordinates.shape}")
                        for coord in coordinates:
                            writer.writerow(coord)
            logging.info(f"Координаты сохранены в {filename}")
        except Exception as e:
            logging.error(f"Ошибка сохранения координат в CSV '{filename}': {e}")
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить CSV файл: {os.path.basename(filename)}\n{e}")

    def calculate_and_save_coordinates(self):
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и выберите изображение!")
            return

        image_path = self.current_image_path
        try:
            logging.info(f"Расчет координат для: {os.path.basename(image_path)}")
            image_gray = self.load_image_grayscale(image_path)
            if image_gray is None:
                return

            edges = cv2.Canny(image_gray, threshold1=50, threshold2=150)
            logging.info(f"Найдены края Canny для {os.path.basename(image_path)}")

            coordinates = self.calculate_edge_coordinates(edges, window_size=(1, 1))
            logging.info(f"Рассчитано {len(coordinates)} координат для {os.path.basename(image_path)}")

            if coordinates.shape[0] == 0:
                messagebox.showwarning("Расчет координат", "Не удалось найти точки края на изображении.")
                self.current_coordinates = None
                self.display_image_with_coordinates(image_path, None)
                return

            self.current_coordinates = coordinates
            self.coordinates_csv_path = None

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_dir = os.path.dirname(image_path)
            csv_path = os.path.join(output_dir, f"coordinates_{base_name}.csv")
            self.save_coordinates_to_csv(coordinates, csv_path)

            self.display_image_with_coordinates(image_path, coordinates)
            self.status_var.set(f"Координаты рассчитаны и сохранены в {os.path.basename(csv_path)}")
            logging.info(f"Координаты для {os.path.basename(image_path)} сохранены в {csv_path}")

        except Exception as e:
            logging.error(f"Не удалось рассчитать/сохранить координаты для {os.path.basename(image_path)}: {e}", exc_info=True)
            messagebox.showerror("Ошибка расчета", f"Не удалось рассчитать/сохранить координаты:\n{e}")
            self.current_coordinates = None
            if self.current_image_path:
                self.display_image_with_coordinates(self.current_image_path, None)

    def load_and_process_coordinates(self, source_csv=None, internal_coords=None):
        coordinates = None
        source = ""

        if source_csv and os.path.exists(source_csv):
            try:
                df = pd.read_csv(source_csv)
                if 'X' not in df.columns or 'Y' not in df.columns:
                    raise ValueError("CSV файл должен содержать колонки 'X' и 'Y'")
                coordinates = df[['X', 'Y']].values
                source = f"CSV: {os.path.basename(source_csv)}"
                logging.info(f"Загружены координаты из {source_csv}")
            except Exception as e:
                logging.error(f"Ошибка чтения CSV '{source_csv}': {e}")
                raise ValueError(f"Ошибка чтения CSV файла '{os.path.basename(source_csv)}': {e}") from e
        elif internal_coords is not None and len(internal_coords) > 0:
            if internal_coords.ndim == 2 and internal_coords.shape[1] == 2:
                coordinates = internal_coords
                source = "внутреннего расчета"
                logging.info("Используются внутренне рассчитанные координаты.")
            else:
                raise ValueError(f"Неверный формат внутренних координат: {internal_coords.shape}")
        else:
            raise ValueError("Нет доступных координат для обработки (ни CSV, ни внутренних).")

        if coordinates is None or len(coordinates) == 0:
            raise ValueError(f"Координаты из {source} пусты.")

        try:
            df_coords = pd.DataFrame(coordinates, columns=['X', 'Y'])
            df_coords['X'] = pd.to_numeric(df_coords['X'], errors='coerce')
            df_coords['Y'] = pd.to_numeric(df_coords['Y'], errors='coerce')
            df_coords.dropna(subset=['X', 'Y'], inplace=True)

            if df_coords.empty:
                raise ValueError("Нет валидных числовых координат после очистки.")

            profile_df = df_coords.loc[df_coords.groupby('X')['Y'].idxmin()]
            profile_df = profile_df.sort_values(by='X').reset_index(drop=True)

            profile_y_coords = profile_df['Y'].values
            profile_x_coords = profile_df['X'].values

            if len(profile_y_coords) == 0:
                raise ValueError("Не удалось извлечь профиль края из координат.")

            logging.info(f"Профиль края (min Y для каждого X) извлечен, {len(profile_y_coords)} точек.")
            return {'raw': coordinates, 'profile_y': profile_y_coords, 'profile_x': profile_x_coords}

        except Exception as e:
            logging.error(f"Ошибка обработки координат из {source}: {e}", exc_info=True)
            raise ValueError(f"Ошибка обработки координат из {source}: {e}") from e

    def display_image_with_coordinates(self, image_path, coordinates):
        try:
            logging.debug(f"Отображение: {os.path.basename(image_path)}")
            img_data = self.load_image_grayscale(image_path)
            if img_data is None:
                self.fig.clear()
                self.canvas.draw()
                self.status_var.set(f"Ошибка загрузки: {os.path.basename(image_path)}")
                return

            self.current_image_path = image_path
            self.current_image_data = img_data

            self.fig.clear()
            ax1 = self.fig.add_subplot(121)
            ax2 = self.fig.add_subplot(122)

            ax1.imshow(img_data, cmap='gray', aspect='auto')
            ax1.set_title(f'Изображение ({os.path.basename(image_path)})')
            ax1.set_xlabel("Пиксели (X)")
            ax1.set_ylabel("Пиксели (Y)")

            if coordinates is not None and len(coordinates) > 0:
                if coordinates.ndim == 2 and coordinates.shape[1] == 2:
                    x_coords = coordinates[:, 0]
                    y_coords = coordinates[:, 1]
                    ax2.scatter(x_coords, y_coords, color='black', s=1, alpha=0.7)
                    ax2.set_title('Точки координат')
                    ax2.set_xlabel("Пиксели (X)")
                    ax2.set_ylabel("Пиксели (Y)")
                    ax2.set_xlim(ax1.get_xlim())
                    ax2.set_ylim(ax1.get_ylim())
                else:
                    logging.warning(f"Неверный формат координат для отображения: {coordinates.shape}")
                    ax2.set_title('Ошибка формата координат')
            else:
                ax2.set_title('Координаты не загружены/рассчитаны')
                ax2.set_xlim(ax1.get_xlim())
                ax2.set_ylim(ax1.get_ylim())
                ax2.invert_yaxis()

            self.fig.tight_layout()
            self.canvas.draw()

            self.status_var.set(f"Отображается: {os.path.basename(image_path)}")
            logging.debug(f"Изображение {os.path.basename(image_path)} отображено.")

        except Exception as e:
            logging.error(f"Не удалось отобразить изображение/координаты '{os.path.basename(image_path)}': {e}", exc_info=True)
            messagebox.showerror("Ошибка отображения", f"Не удалось отобразить данные:\n{e}")
            try:
                self.fig.clear()
                self.canvas.draw()
            except:
                pass

    def analyze_images(self):
        if not self.image_files:
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображения!")
            return

        if self.coordinates_csv_path is None and self.current_coordinates is None:
            messagebox.showwarning("Предупреждение", "Координаты не рассчитаны и CSV файл не выбран. Нечего анализировать.")
            return
        elif self.coordinates_csv_path and not os.path.exists(self.coordinates_csv_path):
            messagebox.showerror("Ошибка", f"Выбранный CSV файл не найден:\n{self.coordinates_csv_path}")
            return

        # Очищаем только текущие результаты и таблицу
        for i in self.results_tree.get_children():
            self.results_tree.delete(i)
        self.result_data = []

        # Перерисовываем таблицу с сохраненными результатами
        for result in self.all_results:
            self._update_tree(result)

        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        self.status_var.set("Анализ изображений...")
        pixel_size = self.pixel_size.get()
        total_files = len(self.image_files)
        logging.info(f"Начало анализа {total_files} файлов. Размер пикселя: {pixel_size} мкм.")

        processed_coords = None
        source_description = ""

        try:
            if self.coordinates_csv_path:
                processed_coords = self.load_and_process_coordinates(source_csv=self.coordinates_csv_path)
                source_description = f"из файла {os.path.basename(self.coordinates_csv_path)}"
            elif self.current_coordinates is not None:
                processed_coords = self.load_and_process_coordinates(internal_coords=self.current_coordinates)
                source_description = "по внутреннему расчету"
            else:
                raise ValueError("Нет источника координат для анализа.")

            y_profile = processed_coords['profile_y']
            logging.info(f"Координаты для анализа успешно загружены/обработаны {source_description}. Длина профиля: {len(y_profile)}")

            for i, img_path in enumerate(self.image_files):
                filename = os.path.basename(img_path)
                current_status = f"Анализ {i+1}/{total_files}: {filename} (коорд. {source_description})"
                self.master.after(0, lambda s=current_status: self.status_var.set(s))
                self.master.after(0, lambda value=(i / total_files) * 100: self.progress_var.set(value))

                try:
                    rq, rq_err = self.calculate_rms_roughness(y_profile, pixel_size)
                    ra, ra_err = self.calculate_arithmetic_roughness(y_profile, pixel_size)
                    rz, rz_err = self.calculate_max_height_roughness(y_profile, pixel_size)

                    result = {
                        "file": filename,
                        "rq": rq, "rq_err": rq_err,
                        "ra": ra, "ra_err": ra_err,
                        "rz": rz, "rz_err": rz_err
                    }
                    self.result_data.append(result)
                    self.all_results.append(result)

                    self.master.after(0, self._update_tree, result)

                except Exception as e_calc:
                    logging.error(f"Ошибка расчета шероховатости для {filename}: {e_calc}", exc_info=True)
                    self.master.after(0, messagebox.showerror, "Ошибка расчета", f"Ошибка при расчете для {filename}:\n{e_calc}")
                    error_result = {"file": filename, "rq": "Ошибка", "rq_err": "", "ra": "", "ra_err": "", "rz": "", "rz_err": ""}
                    self.result_data.append(error_result)
                    self.all_results.append(error_result)
                    self.master.after(0, self._update_tree, error_result)

        except Exception as e_coords:
            logging.error(f"Критическая ошибка загрузки/обработки координат: {e_coords}", exc_info=True)
            self.master.after(0, messagebox.showerror, "Ошибка координат", f"Не удалось загрузить или обработать координаты:\n{e_coords}")
            self.master.after(0, lambda: self.status_var.set("Ошибка анализа координат"))
            self.master.after(0, lambda: self.progress_var.set(0))
            return

        self.master.after(0, lambda: self.progress_var.set(100))
        self.master.after(0, self.calculate_statistics)
        final_status = f"Анализ завершен ({total_files} файлов, коорд. {source_description})."
        self.master.after(0, lambda s=final_status: self.status_var.set(s))
        logging.info("Анализ завершен.")

    def _update_tree(self, result):
        try:
            values = (
                result.get("file", "N/A"),
                f"{result.get('rq', 0):.4f}" if isinstance(result.get('rq'), (int, float)) else str(result.get('rq', 'N/A')),
                f"{result.get('rq_err', 0):.4f}" if isinstance(result.get('rq_err'), (int, float)) else str(result.get('rq_err', '')),
                f"{result.get('ra', 0):.4f}" if isinstance(result.get('ra'), (int, float)) else str(result.get('ra', '')),
                f"{result.get('ra_err', 0):.4f}" if isinstance(result.get('ra_err'), (int, float)) else str(result.get('ra_err', '')),
                f"{result.get('rz', 0):.4f}" if isinstance(result.get('rz'), (int, float)) else str(result.get('rz', '')),
                f"{result.get('rz_err', 0):.4f}" if isinstance(result.get('rz_err'), (int, float)) else str(result.get('rz_err', ''))
            )
            self.results_tree.insert("", tk.END, values=values)
        except Exception as e:
            logging.error(f"Ошибка обновления таблицы результатов: {e}", exc_info=True)

    def bootstrap_std(self, data, num_samples=500):
        data_1d = np.asarray(data).flatten()
        n = len(data_1d)
        if n < 2 or np.all(data_1d == data_1d[0]):
            return 0.0
        try:
            bootstrap_indices = np.random.randint(0, n, size=(num_samples, n))
            bootstrap_samples = data_1d[bootstrap_indices]
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            std_err = np.std(bootstrap_means, ddof=1)
            return std_err if np.isfinite(std_err) else 0.0
        except Exception as e:
            logging.warning(f"Ошибка в bootstrap_std: {e}")
            return 0.0

    def calculate_rms_roughness(self, y_coords, pixel_size):
        y_coords = np.asarray(y_coords)
        if len(y_coords) < 2:
            return 0.0, 0.0
        try:
            mean_height = np.mean(y_coords)
            squared_deviations = (y_coords - mean_height) ** 2
            rms_roughness_px = np.sqrt(np.mean(squared_deviations))
            y_coords_err_px = self.bootstrap_std(y_coords)
            rms_error = y_coords_err_px * pixel_size
            return rms_roughness_px * pixel_size, rms_error
        except Exception as e:
            logging.warning(f"Ошибка расчета Rq: {e}")
            return 0.0, 0.0

    def calculate_arithmetic_roughness(self, y_coords, pixel_size):
        y_coords = np.asarray(y_coords)
        if len(y_coords) < 2:
            return 0.0, 0.0
        try:
            mean_height = np.mean(y_coords)
            abs_deviations = np.abs(y_coords - mean_height)
            ra_roughness_px = np.mean(abs_deviations)
            ra_error_px = self.bootstrap_std(abs_deviations)
            return ra_roughness_px * pixel_size, ra_error_px * pixel_size
        except Exception as e:
            logging.warning(f"Ошибка расчета Ra: {e}")
            return 0.0, 0.0

    def calculate_max_height_roughness(self, y_coords, pixel_size):
        y_coords = np.asarray(y_coords)
        if len(y_coords) < 2:
            return 0.0, 0.0
        try:
            min_height = np.min(y_coords)
            max_height = np.max(y_coords)
            rz_px = max_height - min_height
            error_px = self.bootstrap_std(y_coords)
            return rz_px * pixel_size, error_px * pixel_size
        except Exception as e:
            logging.warning(f"Ошибка расчета Rz: {e}")
            return 0.0, 0.0

    def calculate_statistics(self):
        if not self.all_results:
            self.avg_roughness.set("")
            return

        try:
            rq_values = np.array([r["rq"] for r in self.all_results if isinstance(r.get("rq"), (int, float))])
            ra_values = np.array([r["ra"] for r in self.all_results if isinstance(r.get("ra"), (int, float))])
            rz_values = np.array([r["rz"] for r in self.all_results if isinstance(r.get("rz"), (int, float))])

            if len(rq_values) == 0:
                self.avg_roughness.set("Нет валидных данных для статистики.")
                return

            avg_rq = np.mean(rq_values)
            avg_ra = np.mean(ra_values)
            avg_rz = np.mean(rz_values)

            std_rq = np.std(rq_values, ddof=1) if len(rq_values) > 1 else 0.0
            std_ra = np.std(ra_values, ddof=1) if len(ra_values) > 1 else 0.0
            std_rz = np.std(rz_values, ddof=1) if len(rz_values) > 1 else 0.0

            stats_text = (f"Rq: {avg_rq:.4f} ± {std_rq:.4f} мкм | "
                          f"Ra: {avg_ra:.4f} ± {std_ra:.4f} мкм | "
                          f"Rz: {avg_rz:.4f} ± {std_rz:.4f} мкм")
            self.avg_roughness.set(stats_text)
            logging.info(f"Статистика рассчитана: {stats_text}")

        except Exception as e:
            logging.error(f"Ошибка при расчете итоговой статистики: {e}", exc_info=True)
            self.avg_roughness.set(f"Ошибка при расчете статистики: {e}")

    def save_results(self):
        if not self.all_results:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения!")
            return

        save_dir = filedialog.askdirectory(title="Выберите директорию для сохранения")
        if not save_dir:
            return

        try:
            df = pd.DataFrame(self.all_results)

            base_filename = "roughness_results"
            if HAS_OPENPYXL:
                data_file_path = os.path.join(save_dir, f"{base_filename}.xlsx")
                df.to_excel(data_file_path, index=False, engine='openpyxl')
                format_type = "Excel"
            else:
                data_file_path_csv = os.path.join(save_dir, f"{base_filename}.csv")
                df.to_csv(data_file_path_csv, index=False, encoding='utf-8', sep=',')
                data_file_path_excel_csv = os.path.join(save_dir, f"{base_filename}_excel_ru.csv")
                df.to_csv(data_file_path_excel_csv, index=False, encoding='utf-8-sig', sep=';')
                data_file_path = data_file_path_csv
                format_type = "CSV"

            logging.info(f"Таблица результатов сохранена в {data_file_path}")

            summary_path = os.path.join(save_dir, "roughness_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Итоговая статистика анализа шероховатости\n")
                f.write("=" * 40 + "\n")
                rq_values = np.array([r["rq"] for r in self.all_results if isinstance(r.get("rq"), (int, float))])
                ra_values = np.array([r["ra"] for r in self.all_results if isinstance(r.get("ra"), (int, float))])
                rz_values = np.array([r["rz"] for r in self.all_results if isinstance(r.get("rz"), (int, float))])

                if len(rq_values) > 0:
                    avg_rq = np.mean(rq_values)
                    std_rq = np.std(rq_values, ddof=1) if len(rq_values) > 1 else 0.0
                    avg_ra = np.mean(ra_values)
                    std_ra = np.std(ra_values, ddof=1) if len(ra_values) > 1 else 0.0
                    avg_rz = np.mean(rz_values)
                    std_rz = np.std(rz_values, ddof=1) if len(rz_values) > 1 else 0.0

                    f.write(f"Среднеквадратичная шероховатость (Rq):\n")
                    f.write(f"  Среднее: {avg_rq:.4f} мкм\n")
                    f.write(f"  Стандартное отклонение (разброс): {std_rq:.4f} мкм\n\n")
                    f.write(f"Средняя арифметическая шероховатость (Ra):\n")
                    f.write(f"  Среднее: {avg_ra:.4f} мкм\n")
                    f.write(f"  Стандартное отклонение (разброс): {std_ra:.4f} мкм\n\n")
                    f.write(f"Максимальная высота профиля (Rz):\n")
                    f.write(f"  Среднее: {avg_rz:.4f} мкм\n")
                    f.write(f"  Стандартное отклонение (разброс): {std_rz:.4f} мкм\n\n")
                else:
                    f.write("Нет валидных данных для расчета статистики.\n")

            logging.info(f"Итоговая статистика сохранена в {summary_path}")

            if self.current_image_path and self.current_image_data is not None:
                try:
                    coords_to_plot = None
                    if self.coordinates_csv_path:
                        processed_coords = self.load_and_process_coordinates(source_csv=self.coordinates_csv_path)
                        coords_to_plot = processed_coords['raw']
                    elif self.current_coordinates is not None:
                        coords_to_plot = self.current_coordinates

                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.imshow(self.current_image_data, cmap='gray', aspect='auto')
                    plt.title(f'Изображение ({os.path.basename(self.current_image_path)})')
                    plt.xlabel("Пиксели (X)")
                    plt.ylabel("Пиксели (Y)")

                    plt.subplot(1, 2, 2)
                    if coords_to_plot is not None and len(coords_to_plot) > 0:
                        if coords_to_plot.ndim == 2 and coords_to_plot.shape[1] == 2:
                            plt.scatter(coords_to_plot[:, 0], coords_to_plot[:, 1], color='black', s=1, alpha=0.7)
                            plt.title('Точки координат')
                            plt.xlim(plt.subplot(1, 2, 1).get_xlim())
                            plt.ylim(plt.subplot(1, 2, 1).get_ylim())
                            plt.gca().invert_yaxis()
                        else:
                            plt.title('Ошибка формата координат')
                    else:
                        plt.title('Координаты не отображены')
                        plt.xlim(plt.subplot(1, 2, 1).get_xlim())
                        plt.ylim(plt.subplot(1, 2, 1).get_ylim())
                        plt.gca().invert_yaxis()

                    plt.xlabel("Пиксели (X)")
                    plt.ylabel("Пиксели (Y)")
                    plt.tight_layout()
                    plot_path = os.path.join(save_dir, "last_visualization.png")
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    logging.info(f"Последняя визуализация сохранена в {plot_path}")

                except Exception as e_plot:
                    logging.error(f"Не удалось сохранить изображение последней визуализации: {e_plot}", exc_info=True)

            message = (f"Результаты сохранены в {format_type}-формате: {os.path.basename(data_file_path)}\n"
                       f"Статистика сохранена в: {os.path.basename(summary_path)}\n"
                       f"Путь: {save_dir}")
            if not HAS_OPENPYXL:
                message += f"\n(Также сохранен CSV для Excel RU: {os.path.basename(data_file_path_excel_csv)})"
                message += "\n\nДля сохранения напрямую в Excel установите: pip install openpyxl"

            messagebox.showinfo("Успех", message)

        except Exception as e:
            logging.error(f"Не удалось сохранить результаты: {e}", exc_info=True)
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить результаты:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RoughnessAnalyzer(root)
    root.mainloop()