#!/usr/bin/env python3
import cv2
import time
import sys
import argparse

# Настройка аргументов командной строки
parser = argparse.ArgumentParser(description='Декодирование и кодирование видео с GPU-ускорением')
parser.add_argument('-i', '--input', default="/fanxiangssd/yaroslav/projects/001_inference_system/003_data/in/IMG_3682rotated.MOV",
                    help='Путь к входному видеофайлу')
parser.add_argument('-o', '--output', default="output.mp4",
                    help='Путь к выходному видеофайлу')
parser.add_argument('-b', '--bitrate', type=int, default=5000000,
                    help='Битрейт для кодирования (в битах/сек, по умолчанию: 5000000)')
args = parser.parse_args()

# Пути к видеофайлам
video_path = args.input
output_path = args.output


bitrate=int(args.bitrate)
# bitrate = 2000000  # Безопасное значение битрейта

print(f"Входное видео: {video_path}")
print(f"Выходное видео: {output_path}")
print(f"Битрейт: {bitrate}")

# GStreamer пайплайн для GPU декодирования
gst_pipeline = (
    f"filesrc location={video_path} ! "
    f"qtdemux ! "
    f"h264parse ! "
    f"nvh264dec ! "
    f"videoconvert ! "
    f"video/x-raw,format=BGR ! "
    f"appsink sync=false drop=true"
)

print(f"Пайплайн декодирования: {gst_pipeline}")

# Открываем видео через GStreamer
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть видео")
    sys.exit(1)

# Получаем свойства видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Разрешение: {width}x{height}, FPS: {fps}")

# GStreamer пайплайн для GPU кодирования


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



print(f"Пайплайн кодирования: {gst_out_pipeline}")

# Открываем видеозапись через GStreamer
out = cv2.VideoWriter(gst_out_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))

if not out.isOpened():
    print("Ошибка: Не удалось создать файл выходного видео")
    cap.release()
    sys.exit(1)

# Измеряем производительность
start_time = time.time()
frame_count = 0

# Обрабатываем все кадры
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Записываем кадр в выходной файл
    is_true = out.write(frame)
    print(f"is_true {is_true}")
    
    frame_count += 1
    
    # Вывод прогресса каждые 100 кадров
    if frame_count % 100 == 0:
        elapsed = time.time() - start_time
        print(f"Кадров: {frame_count}, Время: {elapsed:.2f}с, FPS: {frame_count/elapsed:.2f}")

# Закрываем видео и освобождаем ресурсы
cap.release()
out.release()

# Итоговая статистика
total_time = time.time() - start_time
avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\nГотово: {frame_count} кадров за {total_time:.2f}с")
print(f"Средний FPS: {avg_fps:.2f}")
print(f"Видео сохранено: {output_path}")