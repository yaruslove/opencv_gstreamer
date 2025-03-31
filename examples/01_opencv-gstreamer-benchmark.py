#!/usr/bin/env python3
import cv2
import time
import sys
import os
import argparse
import psutil
import threading
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Глобальные переменные для хранения статистики
cpu_usage_stats = {'cpu': [], 'ram': [], 'timestamp': []}
resources_monitor_active = False
fps_monitor_active = False

# Глобальные переменные для отслеживания FPS в динамике
cpu_fps_data = {'timestamps': [], 'fps': [], 'start_time': None, 'frames_count': 0, 'last_check_time': None}
gpu_fps_data = {'timestamps': [], 'fps': [], 'start_time': None, 'frames_count': 0, 'last_check_time': None}

def monitor_resources(stats_dict, interval=0.5):
    """Функция для мониторинга ресурсов системы"""
    global resources_monitor_active
    process = psutil.Process(os.getpid())
    
    while resources_monitor_active:
        # Получение загрузки CPU для текущего процесса
        cpu_percent = process.cpu_percent(interval=0)
        # Получение использования RAM
        ram_usage = process.memory_info().rss / (1024 * 1024)  # В МБ
        
        # Сохранение статистики
        stats_dict['cpu'].append(cpu_percent)
        stats_dict['ram'].append(ram_usage)
        stats_dict['timestamp'].append(time.time())
        
        time.sleep(interval)

def start_resource_monitoring(stats_dict):
    """Запуск мониторинга ресурсов в отдельном потоке"""
    global resources_monitor_active
    resources_monitor_active = True
    monitor_thread = threading.Thread(target=monitor_resources, args=(stats_dict,))
    monitor_thread.daemon = True
    monitor_thread.start()
    return monitor_thread

def stop_resource_monitoring():
    """Остановка мониторинга ресурсов"""
    global resources_monitor_active
    resources_monitor_active = False

def reset_stats(stats_dict):
    """Сброс статистики"""
    stats_dict['cpu'] = []
    stats_dict['ram'] = []
    stats_dict['timestamp'] = []

def init_fps_data(data_dict):
    """Инициализация данных FPS"""
    data_dict['timestamps'] = []
    data_dict['fps'] = []
    data_dict['start_time'] = time.time()
    data_dict['frames_count'] = 0
    data_dict['last_check_time'] = time.time()
    # Инициализируем первые точки данных сразу чтобы избежать пустого графика
    data_dict['timestamps'].append(0.0)
    data_dict['fps'].append(0.0)

def update_fps_data(data_dict, frames_increment=1, interval=1.0):
    """Обновление данных FPS"""
    current_time = time.time()
    
    # Увеличиваем счетчик кадров
    data_dict['frames_count'] += frames_increment
    
    # Проверяем, прошел ли интервал для обновления FPS
    elapsed = current_time - data_dict['last_check_time']
    if elapsed >= interval:
        # Вычисляем реальный FPS - общее количество кадров / общее прошедшее время
        total_elapsed = current_time - data_dict['start_time']
        current_fps = data_dict['frames_count'] / total_elapsed if total_elapsed > 0 else 0
        
        # Сохраняем текущее значение FPS и временную метку
        data_dict['fps'].append(current_fps)
        data_dict['timestamps'].append(total_elapsed)
        data_dict['last_check_time'] = current_time
        return current_fps
    return None

def ensure_dir_exists(file_path):
    """Создает директорию для файла, если она не существует"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Создана директория: {directory}")

def process_video_cpu(input_path, output_path, frames_to_process=None, display=False):
    """Обрабатывает видео с использованием CPU"""
    print("\n--- Обработка через CPU ---")
    
    # Сбрасываем и запускаем мониторинг ресурсов
    reset_stats(cpu_usage_stats)
    start_resource_monitoring(cpu_usage_stats)
    
    # Инициализируем отслеживание FPS
    global cpu_fps_data
    init_fps_data(cpu_fps_data)
    
    if not os.path.exists(input_path):
        print(f"ОШИБКА: Файл не найден: {input_path}")
        stop_resource_monitoring()
        return 0, 0, 0, 0
    
    if output_path:
        ensure_dir_exists(output_path)
    
    # Открываем видео для чтения через OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ОШИБКА: Не удалось открыть видео: {input_path}")
        stop_resource_monitoring()
        return 0, 0, 0, 0
    
    # Получаем свойства видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Свойства видео: {width}x{height}, {fps} FPS")
    
    # Настраиваем запись, если нужно
    writer = None
    if output_path:
        # Используем mp4v кодек
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            print("mp4v кодек не удался, пробуем XVID...")
            writer.release()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print(f"ОШИБКА: Не удалось создать writer для {output_path}")
                stop_resource_monitoring()
                return 0, 0, 0, 0
            else:
                print(f"CPU запись настроена с XVID")
        else:
            print(f"CPU запись настроена с mp4v")
    
    # Обработка видео
    start_time = time.time()
    processed_frames = 0
    fps_update_interval = 0.2  # Обновляем данные FPS каждые 0.2 секунды
    last_fps_update = 0
    frame_batch_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обработка кадра (простое размытие)
        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Запись кадра
        if writer:
            writer.write(processed)
        
        # Отображение, если нужно
        if display:
            cv2.imshow('CPU Processing', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processed_frames += 1
        frame_batch_count += 1
        
        # Обновляем статистику FPS каждые fps_update_interval секунд
        current_time = time.time()
        elapsed_since_update = current_time - last_fps_update
        
        if elapsed_since_update >= fps_update_interval:
            # Рассчитать текущий FPS
            fps_current = processed_frames / (current_time - start_time)
            
            # Сохраняем текущее значение FPS и время
            cpu_fps_data['fps'].append(fps_current)
            cpu_fps_data['timestamps'].append(current_time - start_time)
            
            # Сбрасываем счетчики для следующего обновления
            last_fps_update = current_time
            frame_batch_count = 0
            
        # Вывод прогресса
        if processed_frames % 500 == 0:
            print(f"CPU: Обработано {processed_frames} кадров за {current_time - start_time:.2f}с, FPS: {fps_current:.2f}")
        
        if frames_to_process and processed_frames >= frames_to_process:
            break
    
    # Освобождаем ресурсы
    total_time = time.time() - start_time
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Останавливаем мониторинг ресурсов
    stop_resource_monitoring()
    
    # Проверяем результат и вычисляем средний FPS
    avg_fps = processed_frames / total_time if total_time > 0 else 0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if output_path and os.path.exists(output_path) else 0
    
    if output_path and os.path.exists(output_path):
        print(f"CPU видео сохранено: {output_path} ({file_size_mb:.2f} МБ)")
        print(f"Обработано {processed_frames} кадров за {total_time:.2f}с, FPS: {avg_fps:.2f}")
    
    # Убедимся, что есть хотя бы одна запись FPS для построения графика
    if not cpu_fps_data['fps']:
        cpu_fps_data['fps'].append(avg_fps)
        cpu_fps_data['timestamps'].append(0.0)
    
    return processed_frames, avg_fps, total_time, file_size_mb

def process_video_gpu(input_path, output_path, frames_to_process=None, display=False):
    """Обрабатывает видео с использованием GPU"""
    print("\n--- Обработка через GPU ---")
    
    # Сбрасываем и запускаем мониторинг ресурсов
    reset_stats(cpu_usage_stats)
    start_resource_monitoring(cpu_usage_stats)
    
    # Инициализируем отслеживание FPS
    global gpu_fps_data
    init_fps_data(gpu_fps_data)
    
    if not os.path.exists(input_path):
        print(f"ОШИБКА: Файл не найден: {input_path}")
        stop_resource_monitoring()
        return 0, 0, 0, 0
    
    if output_path:
        ensure_dir_exists(output_path)
    
    # # GStreamer пайплайн для GPU декодирования mp4 h264
    gst_pipeline = (
        f"filesrc location={input_path} ! "
        f"qtdemux ! "
        f"h264parse ! "
        f"nvh264dec ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink sync=false drop=true"
    )

    # # GStreamer пайплайн для GPU декодирования mp4 Видео использует AV1 Работает
    # gst_pipeline = (
    #     f"filesrc location={input_path} ! "
    #     f"qtdemux ! "  # Демультиплексер для MP4
    #     f"av1parse ! "  # Парсер для AV1 потока
    #     f"nvav1dec ! "  # Декодер AV1 через NVIDIA NVDEC
    #     f"videoconvert ! " 
    #     f"video/x-raw,format=BGR ! " 
    #     f"appsink sync=false drop=true"
    # )

    # ##  AV1 Работает  но пропускает кадры
    # gst_pipeline = (
    #     f"filesrc location={input_path} ! "
    #     f"qtdemux ! "
    #     f"av1parse ! "
    #     f"nvav1dec ! "  # Декодирование AV1 через NVDEC
    #     f"cudaconvert ! "  # Конвертация цветового пространства на GPU
    #     f"video/x-raw(memory:CUDAMemory),format=BGR ! "  # Явное указание формата
    #     f"cudadownload ! "  # Копирование данных из CUDA в CPU
    #     f"video/x-raw,format=BGR ! "  # Формат для OpenCV
    #     f"appsink sync=false drop=true"
    # )

    ### Работает но медленно AV1 
    # gst_pipeline = (
    #     f"filesrc location={input_path} ! "
    #     f"qtdemux ! "
    #     f"av1parse ! "
    #     f"nvav1dec ! "  # Декодирование AV1 через NVDEC
    #     f"cudaconvert ! "  # Конвертация цветового пространства на GPU
    #     f"video/x-raw(memory:CUDAMemory),format=BGR ! "  # Явное указание формата
    #     f"cudadownload ! "  # Копирование из GPU в CPU
    #     f"queue max-size-buffers=3 ! "  # Буфер для стабилизации потока
    #     f"video/x-raw,format=BGR ! "
    #     f"appsink sync=false name=sink"
    # )

    ###  Работает медленно AV1 
    # gst_pipeline = (
    #     f"filesrc location={input_path} ! "
    #     f"qtdemux ! "
    #     f"av1parse ! "
    #     f"nvav1dec ! "  # Декодирование через NVDEC
    #     f"cudacompositor name=comp ! "  # Синхронизация потоков
    #     f"cudaconvertscale ! "  # Объединенная конвертация+масштабирование на GPU
    #     f"video/x-raw(memory:CUDAMemory),format=BGR ! "
    #     f"cudadownload ! "
    #     f"queue max-size-buffers=5 leaky=downstream ! "  # Оптимальный буфер
    #     f"video/x-raw,format=BGR ! "
    #     f"appsink sync=false name=sink"
    # )

    #### Работает медленно но быстрее остальных оптимизированно  AV1 
    # gst_pipeline = (
    #     f"filesrc location={input_path} ! "
    #     f"qtdemux ! "
    #     f"av1parse ! "
    #     f"nvav1dec ! "  # Hardware AV1 decoding
    #     f"video/x-raw(memory:CUDAMemory), format=NV12 ! "  # Explicitly set format after decoder
    #     f"cudaconvertscale ! "  # Combined conversion and scaling for better performance
    #     f"video/x-raw(memory:CUDAMemory), format=RGBA ! "  # RGBA works better than BGR with CUDA
    #     f"cudadownload ! "  # Transfer from GPU to CPU memory
    #     f"videoconvert ! "  # Standard converter to ensure format compatibility
    #     f"video/x-raw, format=BGR ! "  # Format for OpenCV
    #     f"queue max-size-buffers=3 leaky=downstream ! "  # Small queue to prevent memory buildup
    #     f"appsink sync=false drop=true max-buffers=1 name=sink"  # Only keep latest frame
    # )




        
    print(f"GPU пайплайн декодирования: {gst_pipeline}")
    
    # Открываем видео через GStreamer
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    # Если не удалось открыть через GStreamer, пробуем обычное открытие
    if not cap.isOpened():
        print("Не удалось открыть через GStreamer, пробуем стандартный метод...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"ОШИБКА: Не удалось открыть видео: {input_path}")
            stop_resource_monitoring()
            return 0, 0, 0, 0
    
    # Получаем свойства видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Свойства видео: {width}x{height}, {fps} FPS")
    
    # Настраиваем запись, если нужно
    writer = None
    if output_path:
        # Используем GStreamer пайплайн для GPU-ускоренного кодирования
        gst_out_pipeline = (
            f"appsrc ! "
            f"video/x-raw,format=BGR ! "
            f"videoconvert ! "
            f"video/x-raw,format=BGRA ! "  # Альтернативный поддерживаемый формат
            f"cudaupload ! "
            f"cudaconvertscale ! "  # Используем cudaconvertscale
            f"video/x-raw(memory:CUDAMemory),format=NV12,width={width},height={height} ! "
            f"nvh265enc " 
            f"preset=4 "
            f"tune=1 "
            f"rc-mode=3 "
            f"repeat-sequence-header=1 "
            f"aud=1 "
            f"! h265parse ! "
            f"qtmux ! "
            f"filesink location={output_path}"
        )
        
        print(f"GPU пайплайн кодирования: {gst_out_pipeline}")
        
        writer = cv2.VideoWriter(gst_out_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        
        if not writer.isOpened():
            print("GStreamer-кодирование не удалось, пробуем XVID...")
            writer.release()
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print(f"ОШИБКА: Не удалось создать writer для {output_path}")
                stop_resource_monitoring()
                return 0, 0, 0, 0
            else:
                print(f"GPU запись настроена с XVID (запасной вариант)")
        else:
            print(f"GPU запись настроена с NVIDIA H.265 аппаратным ускорением")
    
    # Обработка видео
    start_time = time.time()
    processed_frames = 0
    fps_update_interval = 0.2  # Обновляем данные FPS каждые 0.2 секунды
    last_fps_update = 0
    frame_batch_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Обработка кадра (простое размытие)
        processed = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Запись кадра
        if writer:
            writer.write(processed)
        
        # Отображение, если нужно
        if display:
            cv2.imshow('GPU Processing', processed)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        processed_frames += 1
        frame_batch_count += 1
        
        # Обновляем статистику FPS каждые fps_update_interval секунд
        current_time = time.time()
        elapsed_since_update = current_time - last_fps_update
        
        if elapsed_since_update >= fps_update_interval:
            # Рассчитать текущий FPS
            fps_current = processed_frames / (current_time - start_time)
            
            # Сохраняем текущее значение FPS и время
            gpu_fps_data['fps'].append(fps_current)
            gpu_fps_data['timestamps'].append(current_time - start_time)
            
            # Сбрасываем счетчики для следующего обновления
            last_fps_update = current_time
            frame_batch_count = 0
            
        # Вывод прогресса
        if processed_frames % 500 == 0:
            print(f"GPU: Обработано {processed_frames} кадров за {current_time - start_time:.2f}с, FPS: {fps_current:.2f}")
        
        if frames_to_process and processed_frames >= frames_to_process:
            break
    
    # Освобождаем ресурсы
    total_time = time.time() - start_time
    cap.release()
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()
    
    # Останавливаем мониторинг ресурсов
    stop_resource_monitoring()
    
    # Проверяем результат и вычисляем средний FPS
    avg_fps = processed_frames / total_time if total_time > 0 else 0
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024) if output_path and os.path.exists(output_path) else 0
    
    if output_path and os.path.exists(output_path):
        print(f"GPU видео сохранено: {output_path} ({file_size_mb:.2f} МБ)")
        print(f"Обработано {processed_frames} кадров за {total_time:.2f}с, FPS: {avg_fps:.2f}")
    
    # Убедимся, что есть хотя бы одна запись FPS для построения графика
    if not gpu_fps_data['fps']:
        gpu_fps_data['fps'].append(avg_fps)
        gpu_fps_data['timestamps'].append(0.0)
    
    return processed_frames, avg_fps, total_time, file_size_mb

def generate_cpu_usage_report(cpu_stats, prefix=""):
    """Генерирует отчет о загрузке CPU"""
    if not cpu_stats['cpu']:
        return "Статистика загрузки CPU не собрана"
    
    avg_cpu = np.mean(cpu_stats['cpu'])
    max_cpu = np.max(cpu_stats['cpu'])
    min_cpu = np.min(cpu_stats['cpu'])
    
    avg_ram = np.mean(cpu_stats['ram'])
    max_ram = np.max(cpu_stats['ram'])
    
    report = [
        f"{prefix}Статистика загрузки CPU:",
        f"   Средняя загрузка CPU: {avg_cpu:.2f}%",
        f"   Максимальная загрузка CPU: {max_cpu:.2f}%",
        f"   Минимальная загрузка CPU: {min_cpu:.2f}%",
        f"   Среднее использование RAM: {avg_ram:.2f} МБ",
        f"   Максимальное использование RAM: {max_ram:.2f} МБ"
    ]
    
    return "\n".join(report)

def save_resource_stats_to_file(stats, filename, test_type):
    """Сохраняет статистику ресурсов в CSV-файл"""
    if not stats['cpu']:
        return
    
    ensure_dir_exists(filename)
    
    with open(filename, 'w') as f:
        f.write("timestamp,cpu_percent,ram_mb,test_type\n")
        
        base_time = stats['timestamp'][0] if stats['timestamp'] else 0
        
        for t, cpu, ram in zip(stats['timestamp'], stats['cpu'], stats['ram']):
            f.write(f"{t-base_time:.3f},{cpu:.2f},{ram:.2f},{test_type}\n")

def plot_fps_comparison(cpu_fps_data, gpu_fps_data, output_path):
    """Создает график сравнения FPS для CPU и GPU"""
    plt.figure(figsize=(12, 6))
    
    # Отображаем данные FPS, если они есть
    if cpu_fps_data['timestamps']:
        plt.plot(cpu_fps_data['timestamps'], cpu_fps_data['fps'], label='OpenCV (CPU)', color='blue', linewidth=2)
    
    if gpu_fps_data['timestamps']:
        plt.plot(gpu_fps_data['timestamps'], gpu_fps_data['fps'], label='GStreamer+OpenCV (GPU)', color='green', linewidth=2)
    
    # Настраиваем график
    plt.title('Сравнение FPS: CPU vs GPU', fontsize=16)
    plt.xlabel('Время (сек)', fontsize=14)
    plt.ylabel('Кадров в секунду (FPS)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Автоматически настраиваем диапазон осей
    y_max = 0
    if cpu_fps_data['fps']:
        y_max = max(y_max, max(cpu_fps_data['fps']) * 1.1)  # Добавляем 10% для лучшей видимости
    if gpu_fps_data['fps']:
        y_max = max(y_max, max(gpu_fps_data['fps']) * 1.1)
    
    if y_max > 0:
        plt.ylim(0, y_max)
    
    # Добавляем границы для лучшей видимости
    plt.tight_layout()
    
    # Сохраняем график
    ensure_dir_exists(output_path)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"График сравнения FPS сохранен: {output_path}")

def plot_resource_usage(cpu_stats, gpu_stats, output_dir, timestamp):
    """Создает графики использования ресурсов для CPU и GPU"""
    # График использования CPU
    plt.figure(figsize=(12, 6))
    
    has_cpu_data = False
    has_gpu_data = False
    
    if cpu_stats['cpu']:
        base_time = cpu_stats['timestamp'][0]
        cpu_times = [t - base_time for t in cpu_stats['timestamp']]
        plt.plot(cpu_times, cpu_stats['cpu'], label='OpenCV (CPU)', color='blue', linewidth=2)
        has_cpu_data = True
    
    if gpu_stats['cpu']:
        base_time = gpu_stats['timestamp'][0]
        gpu_times = [t - base_time for t in gpu_stats['timestamp']]
        plt.plot(gpu_times, gpu_stats['cpu'], label='GStreamer+OpenCV (GPU)', color='green', linewidth=2)
        has_gpu_data = True
    
    if has_cpu_data or has_gpu_data:
        plt.title('Использование CPU при обработке видео', fontsize=16)
        plt.xlabel('Время (сек)', fontsize=14)
        plt.ylabel('Использование CPU (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Сохраняем график использования CPU
        cpu_usage_path = f"{output_dir}/cpu_usage_{timestamp}.png"
        ensure_dir_exists(cpu_usage_path)
        plt.savefig(cpu_usage_path, dpi=300)
        plt.close()
        print(f"График использования CPU сохранен: {cpu_usage_path}")
    
    # График использования RAM
    plt.figure(figsize=(12, 6))
    
    has_cpu_data = False
    has_gpu_data = False
    
    if cpu_stats['ram']:
        base_time = cpu_stats['timestamp'][0]
        cpu_times = [t - base_time for t in cpu_stats['timestamp']]
        plt.plot(cpu_times, cpu_stats['ram'], label='OpenCV (CPU)', color='blue', linewidth=2)
        has_cpu_data = True
    
    if gpu_stats['ram']:
        base_time = gpu_stats['timestamp'][0]
        gpu_times = [t - base_time for t in gpu_stats['timestamp']]
        plt.plot(gpu_times, gpu_stats['ram'], label='GStreamer+OpenCV (GPU)', color='green', linewidth=2)
        has_gpu_data = True
    
    if has_cpu_data or has_gpu_data:
        plt.title('Использование RAM при обработке видео', fontsize=16)
        plt.xlabel('Время (сек)', fontsize=14)
        plt.ylabel('Использование RAM (МБ)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Сохраняем график использования RAM
        ram_usage_path = f"{output_dir}/ram_usage_{timestamp}.png"
        ensure_dir_exists(ram_usage_path)
        plt.savefig(ram_usage_path, dpi=300)
        plt.close()
        print(f"График использования RAM сохранен: {ram_usage_path}")

def create_comparison_table(cpu_data, gpu_data, output_path):
    """Создает сводную таблицу сравнения CPU и GPU"""
    # Создаем DataFrame
    comparison_data = {
        'Метрика': [
            'Средний FPS', 
            'Размер файла (МБ)', 
            'Время обработки (сек)', 
            'Среднее использование RAM (МБ)', 
            'Среднее использование CPU (%)'
        ],
        'OpenCV (CPU)': [
            f"{cpu_data.get('avg_fps', 'N/A'):.2f}" if cpu_data.get('avg_fps') is not None else 'N/A',
            f"{cpu_data.get('file_size', 'N/A'):.2f}" if cpu_data.get('file_size') is not None else 'N/A',
            f"{cpu_data.get('processing_time', 'N/A'):.2f}" if cpu_data.get('processing_time') is not None else 'N/A',
            f"{cpu_data.get('avg_ram', 'N/A'):.2f}" if cpu_data.get('avg_ram') is not None else 'N/A',
            f"{cpu_data.get('avg_cpu', 'N/A'):.2f}" if cpu_data.get('avg_cpu') is not None else 'N/A'
        ],
        'GStreamer+OpenCV (GPU)': [
            f"{gpu_data.get('avg_fps', 'N/A'):.2f}" if gpu_data.get('avg_fps') is not None else 'N/A',
            f"{gpu_data.get('file_size', 'N/A'):.2f}" if gpu_data.get('file_size') is not None else 'N/A',
            f"{gpu_data.get('processing_time', 'N/A'):.2f}" if gpu_data.get('processing_time') is not None else 'N/A',
            f"{gpu_data.get('avg_ram', 'N/A'):.2f}" if gpu_data.get('avg_ram') is not None else 'N/A',
            f"{gpu_data.get('avg_cpu', 'N/A'):.2f}" if gpu_data.get('avg_cpu') is not None else 'N/A'
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Создаем HTML-таблицу с красивым форматированием
    html_table = """
    <html>
    <head>
        <style>
            table {
                width: 100%;
                border-collapse: collapse;
                font-family: Arial, sans-serif;
            }
            th, td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #4CAF50;
                color: white;
            }
            tr:nth-child(even) {
                background-color: #f2f2f2;
            }
            tr:hover {
                background-color: #ddd;
            }
            .table-title {
                text-align: center;
                font-size: 20px;
                margin-bottom: 10px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="table-title">Сравнение GPU и CPU обработки видео</div>
    """
    
    html_table += df.to_html(index=False)
    html_table += """
    </body>
    </html>
    """
    
    # Сохраняем таблицу в HTML-файл
    ensure_dir_exists(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_table)
    
    # Также сохраняем в CSV
    csv_path = output_path.replace('.html', '.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Сводная таблица сохранена в HTML: {output_path}")
    print(f"Сводная таблица сохранена в CSV: {csv_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Сравнительное тестирование обработки видео: CPU vs GPU с NVIDIA HW Acceleration')
    parser.add_argument('--input', required=True, help='Путь к входному видео')
    parser.add_argument('--output-cpu', help='Путь для сохранения обработанного видео (CPU)')
    parser.add_argument('--output-gpu', help='Путь для сохранения обработанного видео (GPU)')
    parser.add_argument('--frames', type=int, help='Количество кадров для обработки (по умолчанию: все)')
    parser.add_argument('--display', action='store_true', help='Отображать обработку в реальном времени')
    parser.add_argument('--skip-cpu', action='store_true', help='Пропустить тест на CPU')
    parser.add_argument('--skip-gpu', action='store_true', help='Пропустить тест на GPU')
    parser.add_argument('--output-dir', default='benchmark_results', help='Директория для сохранения результатов')
    parser.add_argument('--encoding-quality', default='high', choices=['basic', 'high'], 
                      help='Качество кодирования: basic - без настроек, high - оптимизированное')
    parser.add_argument('--save-stats', action='store_true', help='Сохранить детальную статистику в CSV-файл')
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    if not os.path.exists(args.input):
        print(f"ОШИБКА: Входной файл не найден: {args.input}")
        return
    
    # Исправляем расширения файлов для лучшей совместимости с GPU кодированием
    cpu_output = args.output_cpu
    gpu_output = args.output_gpu
    
    if cpu_output and cpu_output.lower().endswith('.mov'):
        cpu_output_base = os.path.splitext(cpu_output)[0]
        cpu_output = f"{cpu_output_base}.avi"
        print(f"Изменено расширение CPU выхода на .avi: {cpu_output}")
    
    if gpu_output and gpu_output.lower().endswith('.mov'):
        gpu_output_base = os.path.splitext(gpu_output)[0]
        gpu_output = f"{gpu_output_base}.mp4"  # Используем .mp4 для HEVC/H.265
        print(f"Изменено расширение GPU выхода на .mp4: {gpu_output}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Создаем директорию для сохранения результатов
    output_dir = args.output_dir
    ensure_dir_exists(output_dir)
    
    print(f"{'='*60}")
    print(f"ЗАПУСК ОБРАБОТКИ ВИДЕО - {datetime.now()}")
    print(f"{'='*60}")
    print(f"Входное видео: {args.input}")
    print(f"CPU выход: {cpu_output if not args.skip_cpu else 'Пропущено'}")
    print(f"GPU выход: {gpu_output if not args.skip_gpu else 'Пропущено'}")
    print(f"Макс. кадров: {args.frames if args.frames else 'Все'}")
    print(f"{'='*60}")
    
    cpu_frames = 0
    gpu_frames = 0
    cpu_stats = {'cpu': [], 'ram': [], 'timestamp': []}
    gpu_stats = {'cpu': [], 'ram': [], 'timestamp': []}
    
    cpu_results = {
        'frames': 0,
        'avg_fps': None,
        'processing_time': None,
        'file_size': None,
        'avg_cpu': None,
        'avg_ram': None
    }
    
    gpu_results = {
        'frames': 0,
        'avg_fps': None,
        'processing_time': None,
        'file_size': None,
        'avg_cpu': None,
        'avg_ram': None
    }
    
    # Запускаем обработку GPU
    if not args.skip_gpu:
        gpu_start = time.time()
        gpu_frames, gpu_avg_fps, gpu_time, gpu_file_size = process_video_gpu(args.input, gpu_output, args.frames, args.display)
        gpu_stats = cpu_usage_stats.copy()
        print(f"GPU обработка завершена: {gpu_frames} кадров за {gpu_time:.2f}с")
        
        # Сохраняем результаты
        gpu_results['frames'] = gpu_frames
        gpu_results['avg_fps'] = gpu_avg_fps
        gpu_results['processing_time'] = gpu_time
        gpu_results['file_size'] = gpu_file_size
        gpu_results['avg_cpu'] = np.mean(gpu_stats['cpu']) if gpu_stats['cpu'] else None
        gpu_results['avg_ram'] = np.mean(gpu_stats['ram']) if gpu_stats['ram'] else None
        
        # Сохраняем статистику если нужно
        if args.save_stats:
            save_resource_stats_to_file(
                gpu_stats, 
                f"{args.output_dir}/gpu_stats_{timestamp}.csv",
                "GPU"
            )
    
    time.sleep(3)
    # Запускаем обработку CPU
    if not args.skip_cpu:
        cpu_start = time.time()
        cpu_frames, cpu_avg_fps, cpu_time, cpu_file_size = process_video_cpu(args.input, cpu_output, args.frames, args.display)
        cpu_stats = cpu_usage_stats.copy()
        print(f"CPU обработка завершена: {cpu_frames} кадров за {cpu_time:.2f}с")
        
        # Сохраняем результаты
        cpu_results['frames'] = cpu_frames
        cpu_results['avg_fps'] = cpu_avg_fps
        cpu_results['processing_time'] = cpu_time
        cpu_results['file_size'] = cpu_file_size
        cpu_results['avg_cpu'] = np.mean(cpu_stats['cpu']) if cpu_stats['cpu'] else None
        cpu_results['avg_ram'] = np.mean(cpu_stats['ram']) if cpu_stats['ram'] else None
        
        # Сохраняем статистику если нужно
        if args.save_stats:
            save_resource_stats_to_file(
                cpu_stats, 
                f"{args.output_dir}/cpu_stats_{timestamp}.csv",
                "CPU"
            )
    
    # Создаем графики и таблицы сравнения
    # График FPS
    fps_chart_path = f"{output_dir}/fps_comparison_{timestamp}.png"
    plot_fps_comparison(cpu_fps_data, gpu_fps_data, fps_chart_path)
    
    # Графики использования ресурсов
    plot_resource_usage(cpu_stats, gpu_stats, output_dir, timestamp)
    
    # Таблица сравнения
    table_path = f"{output_dir}/comparison_table_{timestamp}.html"
    comparison_table = create_comparison_table(cpu_results, gpu_results, table_path)
    
    # Итоговый отчет
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЙ ОТЧЕТ - {datetime.now()}")
    print(f"{'='*60}")
    
    # Проверяем CPU файл
    if cpu_output and os.path.exists(cpu_output):
        size_mb = os.path.getsize(cpu_output) / (1024 * 1024)
        print(f"✅ CPU видео: {cpu_output}")
        print(f"   Размер: {size_mb:.2f} МБ")
        print(f"   Кадров: {cpu_frames}")
        if cpu_frames > 0 and 'cpu_time' in locals():
            print(f"   FPS: {cpu_frames/cpu_time:.2f}")
    elif not args.skip_cpu and cpu_output:
        print(f"❌ CPU видео не создано: {cpu_output}")
    
    # Проверяем GPU файл
    if gpu_output and os.path.exists(gpu_output):
        size_mb = os.path.getsize(gpu_output) / (1024 * 1024)
        print(f"✅ GPU видео: {gpu_output}")
        print(f"   Размер: {size_mb:.2f} МБ")
        print(f"   Кадров: {gpu_frames}")
        if gpu_frames > 0 and 'gpu_time' in locals():
            print(f"   FPS: {gpu_frames/gpu_time:.2f}")
    elif not args.skip_gpu and gpu_output:
        print(f"❌ GPU видео не создано: {gpu_output}")
    
    # Сравнение производительности
    if cpu_frames > 0 and gpu_frames > 0 and 'cpu_time' in locals() and 'gpu_time' in locals():
        cpu_fps = cpu_frames / cpu_time
        gpu_fps = gpu_frames / gpu_time
        speedup = gpu_fps / cpu_fps
        print(f"\nСравнение производительности:")
        print(f"CPU FPS: {cpu_fps:.2f}")
        print(f"GPU FPS: {gpu_fps:.2f}")
        print(f"Ускорение GPU vs CPU: {speedup:.2f}x")
    
    # Вывод статистики загрузки CPU
    print(f"\n{'='*60}")
    print(f"СТАТИСТИКА ИСПОЛЬЗОВАНИЯ РЕСУРСОВ")
    print(f"{'='*60}")
    
    if not args.skip_cpu and cpu_stats['cpu']:
        print(generate_cpu_usage_report(cpu_stats, "CPU обработка: "))
    
    if not args.skip_gpu and gpu_stats['cpu']:
        print("\n" + generate_cpu_usage_report(gpu_stats, "GPU обработка: "))
    
    # Сравнение загрузки CPU при GPU и CPU обработке
    if not args.skip_cpu and not args.skip_gpu and cpu_stats['cpu'] and gpu_stats['cpu']:
        avg_cpu_load = np.mean(cpu_stats['cpu'])
        avg_gpu_load = np.mean(gpu_stats['cpu'])
        cpu_load_reduction = 100 * (1 - avg_gpu_load / avg_cpu_load) if avg_cpu_load > 0 else 0
        
        print(f"\nСнижение нагрузки на CPU при использовании GPU: {cpu_load_reduction:.2f}%")
    
    print(f"{'='*60}")
    
    # Выводим информацию о созданных отчетах и графиках
    print(f"\nОтчеты и графики сохранены в директорию: {output_dir}")
    print(f"График сравнения FPS: {fps_chart_path}")
    print(f"Сводная таблица: {table_path}")
    
    # Сохранение отчета в файл
    report_file = f"{args.output_dir}/benchmark_report_{timestamp}.txt"
    ensure_dir_exists(report_file)
    
    with open(report_file, 'w') as f:
        f.write(f"ОТЧЕТ О ТЕСТИРОВАНИИ ПРОИЗВОДИТЕЛЬНОСТИ ОБРАБОТКИ ВИДЕО\n")
        f.write(f"Дата и время: {datetime.now()}\n")
        f.write(f"Входное видео: {args.input}\n")
        f.write(f"Кадров обработано: CPU={cpu_frames}, GPU={gpu_frames}\n\n")
        
        if cpu_frames > 0 and 'cpu_time' in locals():
            f.write(f"CPU обработка:\n")
            f.write(f"  Время: {cpu_time:.2f}с\n")
            f.write(f"  FPS: {cpu_frames/cpu_time:.2f}\n")
            if cpu_stats['cpu']:
                f.write(f"  {generate_cpu_usage_report(cpu_stats)}\n\n")
        
        if gpu_frames > 0 and 'gpu_time' in locals():
            f.write(f"GPU обработка:\n")
            f.write(f"  Время: {gpu_time:.2f}с\n")
            f.write(f"  FPS: {gpu_frames/gpu_time:.2f}\n")
            if gpu_stats['cpu']:
                f.write(f"  {generate_cpu_usage_report(gpu_stats)}\n\n")
        
        if cpu_frames > 0 and gpu_frames > 0 and 'cpu_time' in locals() and 'gpu_time' in locals():
            f.write(f"Сравнение производительности:\n")
            f.write(f"  Ускорение GPU vs CPU: {speedup:.2f}x\n")
            
            if cpu_stats['cpu'] and gpu_stats['cpu']:
                avg_cpu_load = np.mean(cpu_stats['cpu'])
                avg_gpu_load = np.mean(gpu_stats['cpu'])
                cpu_load_reduction = 100 * (1 - avg_gpu_load / avg_cpu_load) if avg_cpu_load > 0 else 0
                f.write(f"  Снижение нагрузки на CPU: {cpu_load_reduction:.2f}%\n")
        
        f.write(f"\nСозданные отчеты и графики:\n")
        f.write(f"  График сравнения FPS: {fps_chart_path}\n")
        f.write(f"  Графики использования ресурсов: {output_dir}/cpu_usage_{timestamp}.png, {output_dir}/ram_usage_{timestamp}.png\n")
        f.write(f"  Сводная таблица: {table_path}\n")
    
    print(f"Отчет сохранен в файл: {report_file}")

if __name__ == "__main__":
    main()