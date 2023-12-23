import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import time
import cv2
import numpy as np

# CUDA kernel для уменьшения размера изображения (аналог pyrDown)
PYRDOWN_KERNEL = SourceModule("""
__global__ void pyr_down(float *src, float *dest, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int new_width = width / 2;
    int new_height = height / 2;

    if (x < new_width && y < new_height)
    {
        int idx = (y * 2 * width + x * 2) * 3;  // Индекс для RGB
        float r = (src[idx] + src[idx + 3] + src[idx + width * 3] + src[idx + width * 3 + 3]) / 4.0;
        float g = (src[idx + 1] + src[idx + 4] + src[idx + width * 3 + 1] + src[idx + width * 3 + 4]) / 4.0;
        float b = (src[idx + 2] + src[idx + 5] + src[idx + width * 3 + 2] + src[idx + width * 3 + 5]) / 4.0;

        dest[(y * new_width + x) * 3] = r;
        dest[(y * new_width + x) * 3 + 1] = g;
        dest[(y * new_width + x) * 3 + 2] = b;
    }
}
""").get_function("pyr_down")


def gpu_pyr_down(image):
    # Функция для уменьшения размера изображения на GPU
    height, width, _ = image.shape
    new_width = width // 2
    new_height = height // 2
    output = np.zeros((new_height, new_width, 3), dtype=np.float32)

    block = (16, 16, 1)
    grid = (new_width // block[0], new_height // block[1])
    PYRDOWN_KERNEL(cuda.In(image), cuda.Out(output), np.int32(width), np.int32(height), block=block, grid=grid)

    return output


def measure_gpu_processing_time(image_path, output_path):
    # Основная функция с расчётом времени прохода
    start_time = time.time()

    image = cv2.imread(image_path).astype(np.float32)
    reduced_image = gpu_pyr_down(image)
    cv2.imwrite(output_path, reduced_image)

    end_time = time.time()
    return end_time - start_time  # Время выполнения в секундах


def average_gpu_processing_time(image_path, output_path, runs=3):
    # Функция для запуска проходов
    total_time = 0
    for run in range(runs):
        run_time = measure_gpu_processing_time(image_path, output_path)
        print(f"Время обработки #{run + 1}: {run_time} секунд")
        total_time += run_time
    return total_time / runs  # Среднее время выполнения


if __name__ == "__main__":
    images = ["img_10280x7680.jpg", "img_12800x9600.jpg", "img_20480x15360.jpg"]
    for image in images:
        average_time = average_gpu_processing_time(image, f"pyramid_{image}")
        print(f"Среднее время обработки для изображения {image}: {average_time} секунд\n")
