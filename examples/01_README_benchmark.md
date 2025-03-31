# Video Processing Benchmark (CPU vs GPU)

Данный скрипт предназначен для сравнительного тестирования производительности обработки видео, используя CPU (OpenCV) и GPU (OpenCV+GStreamer с NVIDIA аппаратным ускорением).

## Функциональные возможности

- Обработка видео через CPU (стандартный OpenCV) и GPU (OpenCV+GStreamer)
- Измерение FPS, времени обработки, загрузки CPU и использования RAM
- Генерация графиков сравнения FPS
- Отображение использования CPU и RAM на графиках
- Создание сводной таблицы для сравнения производительности
- Формирование подробных отчетов в разных форматах

## Требования

- Python 3.6+
- OpenCV с поддержкой GStreamer
- NVIDIA GPU с установленными драйверами
- GStreamer с плагинами NVIDIA
- Дополнительные пакеты Python:
  - matplotlib
  - pandas
  - numpy
  - psutil

### Установка зависимостей

```bash
pip install opencv-python matplotlib pandas numpy psutil
```

Для поддержки GStreamer необходимо установить системные зависимости:

```bash
# Ubuntu/Debian
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
```

Для поддержки NVIDIA требуется установка драйверов NVIDIA, CUDA и соответствующих плагинов GStreamer.

## Использование

```
python video_benchmark.py --input <путь_к_видео> [опции]
```

### Аргументы командной строки

| Аргумент | Описание |
|----------|----------|
| `--input` | **Обязательный**. Путь к входному видео |
| `--output-cpu` | Путь для сохранения обработанного видео (CPU) |
| `--output-gpu` | Путь для сохранения обработанного видео (GPU) |
| `--frames` | Количество кадров для обработки (по умолчанию: все) |
| `--display` | Отображать обработку в реальном времени |
| `--skip-cpu` | Пропустить тест на CPU |
| `--skip-gpu` | Пропустить тест на GPU |
| `--output-dir` | Директория для сохранения результатов (по умолчанию: benchmark_results) |
| `--encoding-quality` | Качество кодирования: basic - без настроек, high - оптимизированное (по умолчанию: high) |
| `--save-stats` | Сохранить детальную статистику в CSV-файл |

## Примеры использования

### Базовое сравнение CPU и GPU

```bash
python opencv-gstreamer-benchmark.py --input /fanxiangssd/yaroslav/projects/001_inference_system/003_data/in/IMG_3682rotated_15sec.MOV --output-dir /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/output \
--output-cpu /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/video/IMG_3682rotated_15sec_CPU.MOV \
--output-gpu /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/video/IMG_3682rotated_15sec_GPU.mp4 \
--frames 2000



python opencv-gstreamer-benchmark.py --input /fanxiangssd/yaroslav/projects/001_inference_system/003_data/in_mock/moscow_video.mp4 --output-dir /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/output \
--output-cpu /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/video_2/moscow_video.avi \
--output-gpu /fanxiangssd/yaroslav/projects/001_inference_system/004_gstreamer_opencv_docker/python_scripts/video_2/moscow_video.mp4
```

```bash
python video_benchmark.py --input sample.mp4 --output-cpu processed_cpu.mp4 --output-gpu processed_gpu.mp4
```

### Только GPU тест (пропуск CPU)

```bash
python video_benchmark.py --input sample.mp4 --output-gpu processed_gpu.mp4 --skip-cpu
```

### Только CPU тест (пропуск GPU)

```bash
python video_benchmark.py --input sample.mp4 --output-cpu processed_cpu.mp4 --skip-gpu
```

### Обработка ограниченного количества кадров

```bash
python video_benchmark.py --input sample.mp4 --output-cpu processed_cpu.mp4 --output-gpu processed_gpu.mp4 --frames 1000
```

### Сохранение результатов в другую директорию

```bash
python video_benchmark.py --input sample.mp4 --output-dir my_benchmark_results
```

### Запуск с отображением процесса обработки

```bash
python video_benchmark.py --input sample.mp4 --display
```

### Сохранение детальной статистики в CSV

```bash
python video_benchmark.py --input sample.mp4 --save-stats
```

### Комплексный пример

```bash
python video_benchmark.py --input sample.mp4 --output-cpu results/cpu_processed.mp4 --output-gpu results/gpu_processed.mp4 --frames 2000 --output-dir benchmark_results/test1 --save-stats --display
```

## Выходные данные

Скрипт создает следующие файлы в директории `--output-dir`:

1. **Графики**:
   - `fps_comparison_***.png` - Сравнение FPS между CPU и GPU
   - `cpu_usage_***.png` - График использования CPU
   - `ram_usage_***.png` - График использования RAM

2. **Таблицы сравнения**:
   - `comparison_table_***.html` - HTML таблица с результатами
   - `comparison_table_***.csv` - CSV таблица с результатами

3. **Отчеты**:
   - `benchmark_report_***.txt` - Текстовый отчет с результатами
   - При указании `--save-stats` также создаются `cpu_stats_***.csv` и `gpu_stats_***.csv`

## Особенности работы

- Скрипт использует простую обработку (размытие Гаусса) в качестве тестовой операции.
- GPU-версия пытается использовать аппаратное ускорение NVIDIA для декодирования и кодирования видео.
- Если GStreamer не может быть использован, скрипт автоматически переключается на стандартный OpenCV.
- Метрики ресурсов собираются в течение всего процесса обработки.

## Анализ результатов

После запуска скрипта вы получите полный отчет, который поможет определить:

1. Разницу в скорости обработки (FPS) между CPU и GPU
2. Снижение нагрузки на CPU при использовании GPU
3. Разницу в размере выходных файлов
4. Использование ресурсов системы
5. Общее время обработки

Эта информация полезна для оптимизации процессов обработки видео и выбора наиболее эффективного метода для вашего конкретного сценария.