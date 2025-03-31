# Подробная инструкция по сборке OpenCV 4.9.0 с поддержкой NVIDIA GPU и GStreamer для Python 3.10

В этой инструкции описан процесс сборки OpenCV 4.9.0 из исходников с поддержкой GStreamer и аппаратного ускорения NVIDIA (NVENC/NVDEC) для Python 3.10 на Ubuntu 22.04.

## 1. Подготовка системы и предварительные требования

### Характеристики системы
- Ubuntu 22.04
- NVIDIA RTX 4090
- Драйвер NVIDIA: 570.124.06
- CUDA Version: 12.8

### Установка драйверов NVIDIA
```bash
# Проверка текущего драйвера
nvidia-smi
 # Проверка версии драйвера GPU 
cat /proc/driver/nvidia/version
# Проверка версии CUDA 
nvcc --version 

# Если драйвер не установлен или требуется обновление
sudo apt update
sudo apt install nvidia-driver-570
# После установки может потребоваться перезагрузка
sudo reboot
```

### Установка CUDA Toolkit
Правильнее всего зайти и установить по инструкции сайта:
https://developer.nvidia.com/cuda-toolkit
CUDA Toolkit необходим для компиляции с поддержкой GPU:
```bash
# Скачивание и установка CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_540.32.01_linux.run
sudo sh cuda_12.8.0_540.32.01_linux.run
```

Опционально добавьте пути CUDA в ~/.bashrc:
```bash
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 2. Установка зависимостей

### Основные инструменты разработки
```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config git unzip
```

### Установка старых компиляторов
Новые компиляторы gcc-12 или 13 имеютдругие стандарты, на них при сборке OpenCV происходят ошибки.  
```bash
sudo apt install -y gcc-11 g++-11
```

### Установка GStreamer с поддержкой NVIDIA
```bash
# Установка базовых компонентов GStreamer
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-ugly gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-tools

# Добавление репозитория NVIDIA для GStreamer плагинов
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Установка NVIDIA GStreamer плагинов
sudo apt install -y gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl gstreamer1.0-rtsp gstreamer1.0-x
```

### Установка NVIDIA Video SDK (если требуется)
Для полной поддержки NVENC/NVDEC рекомендуется установить NVIDIA Video SDK:
```bash
# Скачайте SDK с сайта NVIDIA и установите
# Официальная страница: https://developer.nvidia.com/nvidia-video-codec-sdk
wget https://developer.download.nvidia.com/compute/video-sdk/redist/v12.1/NVIDIA_Video_Codec_SDK_12.1.14.zip
unzip NVIDIA_Video_Codec_SDK_12.1.14.zip
sudo cp -a Video_Codec_SDK_12.1.14/Interface/. /usr/local/include/
sudo cp -a Video_Codec_SDK_12.1.14/Lib/linux/stubs/x86_64/. /usr/local/lib/
```

### Дополнительные библиотеки для OpenCV
```bash
sudo apt install -y libgtk-3-dev libtbb-dev libatlas-base-dev libdc1394-dev libxine2-dev libv4l-dev
sudo apt install -y libtesseract-dev libleptonica-dev
sudo apt install -y libopenblas-dev liblapack-dev libeigen3-dev
```

## 3. Создание и настройка Python окружения
Жалательно делать в отдельном окружении, особенно на Ubuntu чтоб не испортить python интерпритатор.

```bash
# Установка Python 3.10 (если еще не установлен)
sudo apt install -y python3.10 python3.10-dev python3.10-venv python3-pip

# Создание виртуального окружения
mkdir -p /home/000_venv
python3.10 -m venv /home/000_venv/venv_opencvCuda
source /home/000_venv/venv_opencvCuda/bin/activate

# Установка необходимых пакетов Python
pip install numpy
```

!!! Дальше все действия выполняйте в установленном venv python3.10

## 4. Проверка работоспособности GStreamer

После установки GStreamer и плагинов NVIDIA, проверьте их доступность и работоспособность:

```bash
# Проверка доступных плагинов NVIDIA
gst-inspect-1.0 | grep -i nv
gst-inspect-1.0 nvh264enc

### Запустите gst pipe
# Базовый тест кодирования (создает H.264 файл)
gst-launch-1.0 videotestsrc num-buffers=300 ! \
    'video/x-raw, width=1920, height=1080' ! \
    nvh264enc ! \
    h264parse ! \
    mp4mux ! \
    filesink location=test_nvenc.mp4 -v

# Тест декодирования (требует H.264 файл)
gst-launch-1.0 filesrc location=test_nvenc.mp4 ! \
    qtdemux ! \
    h264parse ! \
    nvh264dec ! \
    videoconvert ! \
    fakesink sync=false -v
```

## 5. Настройка компиляторов GCC-11 и G++-11
Переключаемся на старый С++ интерпритатор
```bash
# Установка переменных окружения для использования GCC-11
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

# Проверка версий компиляторов

# Должно показать gcc-11.x.x
$CC --version
# Должно показать g++-11.x.x
$CXX --version  
```

## 6. Загрузка исходного кода OpenCV

Обосгование версии
**Почему OpenCV 4.9.0 лучше новых версий в вашем случае:**

1. **Стабильность NVENC/NVDEC**  
   В 4.9.0 поддержка NVIDIA кодеков проверена с драйверами (570.x) и CUDA 12.8. В новых версиях:
   - Меняются флаги сборки (например, `WITH_NVCUVID` → `WITH_CUDAVID`).  
   - Требуются обновленные Video SDK, которые могут конфликтовать с CUDA 12.8.

2. **Минимизация конфликтов**  
   Отключил FFmpeg чтоб собрать без ошибок. В новых версиях:  
   - Усиливается зависимость от FFmpeg для `videoio`, даже с флагом `WITH_FFMPEG=OFF`.  
   - GStream-пайплайны могут ломаться из-за изменений в бэкендах.

3. **Совместимость с RTX 4090**  
   Флаг `CUDA_ARCH_BIN=8.9` для Ada Lovelace работает без ошибок. В новых версиях:  
   - Добавляют поддержку будущих архитектур (например, Blackwell), но ломают обратную совместимость.  
   - Часты баги определения SM-версии GPU.

4. **Проверенная сборка**  
   Вы уже настроили 4.9.0 под Python 3.10. Новые версии:  
   - Меняют систему генерации Python-биндингов (например, на `scikit-build`).  
   - Требуют пересборки с новыми флагами (риск ошибок линковки).

**Резюме:**  
Оставайтесь на 4.9.0, пока:  
- GStreamer + NVENC/NVDEC работают.  
- Не требуется интеграция с новыми API (TensorRT 9+).  
- CUDA 12.8 поддерживается драйверами. Обновляйтесь только при явной необходимости в фичах из ChangeLog.


```bash
# Создайте рабочую директорию куда временно скачается opencv
mkdir -p ./opencv_build && cd ./opencv_build

## Если есть скаченная репа можно сделать git checkout на ветку 4.9.0

# Загрузка OpenCV 4.9.0 и модулей contrib
OPENCV_VERSION=4.9.0 && \
wget -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip && \
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip && \
unzip opencv.zip && \
unzip opencv_contrib.zip && \
rm opencv.zip opencv_contrib.zip && \
mv opencv-$OPENCV_VERSION opencv && \
mv opencv_contrib-$OPENCV_VERSION opencv_contrib && \
mkdir build && cd build
```

## 7. Конфигурация и сборка OpenCV

```bash
# Убедитесь, что вы активировали нужное python3.10 виртуальное окружение
source /home/../venv_opencvCuda/bin/activate

# Задайте правильный путь к вашему виртуальному окружению
VENV_PATH="/home/../venv_opencvCuda"
VENV_PATH="/home/yaroslav/Documents/001_Projects/000_venv/venv_opencvCudaTest"

# Конфигурация CMake
cmake -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
-D CMAKE_C_COMPILER=/usr/bin/gcc-11 \
-D CMAKE_CXX_COMPILER=/usr/bin/g++-11 \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=${VENV_PATH} \
-D PYTHON3_EXECUTABLE=${VENV_PATH}/bin/python3 \
-D PYTHON3_INCLUDE_DIR=$(${VENV_PATH}/bin/python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=${VENV_PATH}/lib/python3.10/site-packages \
-D WITH_CUDA=ON \
-D WITH_CUDNN=OFF \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN=8.9 \
-D CUDA_ARCH_PTX="" \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D WITH_GSTREAMER=ON \
-D WITH_GSTREAMER_0_10=OFF \
-D WITH_LIBV4L=ON \
-D WITH_V4L=ON \
-D WITH_FFMPEG=OFF \
-D OPENCV_FFMPEG_USE_FIND_PACKAGE=ON \
-D OPENCV_FFMPEG_LINK_DYNAMICALLY=ON \
-D WITH_TBB=ON \
-D WITH_NVCUVID=ON \
-D WITH_NVCUVENC=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=ON \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_EXAMPLES=OFF \
-D BUILD_SHARED_LIBS=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_opencv_highgui=OFF \
-D BUILD_opencv_apps=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D BUILD_DOCS=OFF \
-D WITH_GTK=OFF \
-D WITH_QT=OFF \
-D WITH_X11=OFF \
-D WITH_OPENGL=OFF \
-D BUILD_opencv_cudaarithm=OFF \
-D BUILD_opencv_cudabgsegm=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D BUILD_opencv_cudafeatures2d=OFF \
-D BUILD_opencv_cudafilters=OFF \
-D BUILD_opencv_cudaimgproc=OFF \
-D BUILD_opencv_cudalegacy=OFF \
-D BUILD_opencv_cudaobjdetect=OFF \
-D BUILD_opencv_cudaoptflow=OFF \
-D BUILD_opencv_cudastereo=OFF \
-D BUILD_opencv_cudawarping=OFF \
-D BUILD_opencv_photo=OFF \
../opencv
```

### Пояснения к ключевым параметрам конфигурации:

- `WITH_GSTREAMER=ON`: Включает поддержку GStreamer
- `CMAKE_INSTALL_PREFIX`: Устанавливает OpenCV в директорию виртуального окружения
- `WITH_CUDA=ON`: Включает поддержку CUDA
- `WITH_NVCUVID=ON` и `WITH_NVCUVENC=ON`: Включает поддержку NVIDIA декодера и кодировщика
- `WITH_FFMPEG=OFF`: Отключает поддержку FFmpeg
- `CUDA_ARCH_BIN=8.9`: Оптимизировано для архитектуры Ada Lovelace (RTX 4090)
- `BUILD_opencv_*=OFF`: Отключает сборку ненужных модулей для экономии времени и места

### Компиляция и установка

```bash
# Компиляция с использованием всех доступных ядер
make -j$(nproc)

# Установка в виртуальное окружение
make install

```

## 8. Проверка установки

```bash
# Активация виртуального окружения
source /home/000_venv/venv_opencvCuda/bin/activate

# Проверка установки OpenCV в Python
python -c "import cv2; print(cv2.__version__); print(cv2.getBuildInformation())"
```

## 9. Примеры использования OpenCV с GStreamer и NVIDIA GPU

### Пример декодирования видео с использованием GPU

```python
import cv2
import numpy as np

# Путь к видеофайлу
video_path = "test_nvenc.mp4"

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

# Открываем видео через GStreamer
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Обработка кадра...
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Пример кодирования видео с использованием GPU

```python
import cv2
import numpy as np

# Параметры выходного видео
output_path = "output_video.mp4"
width, height = 1920, 1080
fps = 30.0

# GStreamer пайплайн для GPU кодирования
gst_out_pipeline = (
    f"appsrc ! "
    f"video/x-raw,format=BGR ! "
    f"videoconvert ! "
    f"video/x-raw,format=BGRA ! "
    f"cudaupload ! "
    f"cudaconvertscale ! "
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

# Создаем VideoWriter для записи с использованием GPU
out = cv2.VideoWriter(gst_out_pipeline, cv2.CAP_GSTREAMER, 0, fps, (width, height))

# Создаем тестовое видео
for i in range(300):
    # Создаем тестовый кадр
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Рисуем что-то на кадре
    cv2.circle(frame, (width//2, height//2), 100 + i, (0, 0, 255), -1)
    cv2.putText(frame, f"Frame: {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Записываем кадр
    out.write(frame)

# Закрываем VideoWriter
out.release()
```

## Советы и решение проблем

1. **Проблемы с компиляцией**:
   - Убедитесь, что используете GCC-11, так как более новые версии компилятора могут вызывать проблемы.
   - Проверьте настройки CUDA_ARCH_BIN для вашей видеокарты.

2. **Проверка поддержки CUDA**:
   ```python
   import cv2
   print(cv2.cuda.getCudaEnabledDeviceCount())  # Должно быть больше 0
   ```

3. **Оптимизация для вашей видеокарты**:
   - Для RTX 4090 архитектура 8.9 (Ada Lovelace).
   - Проверьте архитектуру вашей карты на [NVIDIA CUDA Wiki](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

4. **Отладка проблем GStreamer**:
   ```bash
   # Добавьте -v для подробного вывода
   GST_DEBUG=3 gst-launch-1.0 videotestsrc ! nvh264enc ! fakesink -v
   ```

Надеюсь, эта инструкция поможет вам успешно скомпилировать OpenCV с поддержкой GPU NVIDIA и GStreamer для Python 3.10. В случае возникновения проблем, обратитесь к официальной документации OpenCV или форумам NVIDIA для получения дополнительной информации.

```bash
docker run  --gpus all -it --name cuda_opencv 05621a898111
```