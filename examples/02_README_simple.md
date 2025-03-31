# Пример использование open-cv +gstreamer + gpu nvh264dec => nvh265enc


```bash
python3 simple_decode_encode.py -i /fanxiangssd/yaroslav/projects/001_inference_system/003_data/in/IMG_3682rotated_15sec.MOV \
-o /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/video_2/IMG_3682rotated_15sec.mp4 \
-b 2000000
```




# GPU-ускоренное транскодирование видео

## Описание
Скрипт выполняет транскодирование видео из H.264 в H.265 (HEVC) с использованием аппаратного ускорения NVIDIA GPU. Позволяет значительно ускорить процесс конвертации по сравнению с CPU-обработкой.

## Технические особенности
- Аппаратное декодирование (nvh264dec) и кодирование (nvh265enc) видео
- Интеграция OpenCV с GStreamer и CUDA для оптимальной производительности
- Мониторинг скорости обработки (FPS) в реальном времени

## Пайплайны
- **Декодирование**: `filesrc → qtdemux → h264parse → nvh264dec → videoconvert → appsink`
- **Кодирование**: `appsrc → videoconvert → cudaupload → cudaconvertscale → nvh265enc → h265parse → qtmux → filesink`

## Использование
```bash
python3 gpu_transcoder.py --input входное_видео.mp4 --output выходное_видео.mp4 --bitrate 5000000
```

### Аргументы
- `-i, --input`: входное видео
- `-o, --output`: выходное видео (по умолчанию: "output.mp4")
- `-b, --bitrate`: битрейт в битах/сек (по умолчанию: 5000000)

## Требования
- NVIDIA GPU с поддержкой NVENC/NVDEC
- Драйверы NVIDIA и CUDA
- OpenCV с поддержкой GStreamer
- GStreamer с плагинами NVIDIA



