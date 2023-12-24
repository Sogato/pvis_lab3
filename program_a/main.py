import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import time
import cv2
import numpy as np

# CUDA kernel для преобразования в градации серого
GRAYSCALE_KERNEL = SourceModule("""
__global__ void rgb_to_grayscale(float *input, float *output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = (y * width + x) * 3;  // Индекс для RGB
        float r = input[idx];
        float g = input[idx + 1];
        float b = input[idx + 2];
        output[y * width + x] = (r + g + b) / 3.0;
    }
}
""").get_function("rgb_to_grayscale")

# CUDA kernel для пороговой обработки
THRESHOLD_KERNEL = SourceModule("""
__global__ void apply_threshold(float *input, float *output, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        output[idx] = input[idx] > threshold ? 1.0 : 0.0;
    }
}
""").get_function("apply_threshold")


EROSION_KERNEL = SourceModule("""
__global__ void erosion_kernel(float *input, float *output, int width, int height, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_kernel = kernel_size / 2;

    if (x >= half_kernel && y >= half_kernel && x < (width - half_kernel) && y < (height - half_kernel)) 
    {
        bool erode = true;
        for (int ky = -half_kernel; ky <= half_kernel; ky++)
        {
            for (int kx = -half_kernel; kx <= half_kernel; kx++)
            {
                if (input[(y + ky) * width + (x + kx)] == 0) 
                {
                    erode = false;
                    break;
                }
            }
            if (!erode) break;
        }
        output[y * width + x] = erode ? 1.0f : 0.0f;
    }
}
""").get_function("erosion_kernel")


def load_image(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path).astype(np.float32)
    return image


def gpu_rgb_to_grayscale(image):
    # Преобразование в градации серого на GPU
    height, width = image.shape[:2]
    output = np.zeros((height, width), dtype=np.float32)
    GRAYSCALE_KERNEL(cuda.In(image), cuda.Out(output), np.int32(width), np.int32(height), block=(16, 16, 1),
                     grid=(width // 16, height // 16))
    return output


def gpu_apply_threshold(grayscale, threshold):
    # Применение пороговой обработки на GPU
    height, width = grayscale.shape
    output = np.zeros_like(grayscale)
    THRESHOLD_KERNEL(cuda.In(grayscale), cuda.Out(output), np.int32(width), np.int32(height), np.float32(threshold),
                     block=(16, 16, 1), grid=(width // 16, height // 16))
    return output


def gpu_erosion(binary, kernel_size):
    # Применения эрозии на GPU
    height, width = binary.shape
    output = np.zeros_like(binary)
    EROSION_KERNEL(cuda.In(binary), cuda.Out(output), np.int32(width), np.int32(height), np.int32(kernel_size), block=(16, 16, 1), grid=(width // 16, height // 16))
    return output


def convert_to_black_and_white_and_save(eroded, output_path):
    # Преобразование в черно-белое изображение и сохранение
    black_and_white = eroded * 255
    cv2.imwrite(output_path, black_and_white)


def measure_gpu_processing_time(image_path, output_path, threshold, kernel_size):
    # Основная функция с расчётом времени прохода
    start_time = time.time()

    image = load_image(image_path)
    grayscale = gpu_rgb_to_grayscale(image)
    binary = gpu_apply_threshold(grayscale, threshold)
    eroded = gpu_erosion(binary, kernel_size)
    convert_to_black_and_white_and_save(eroded, output_path)

    end_time = time.time()
    return end_time - start_time  # Время выполнения в секундах


def average_gpu_processing_time(image_path, output_path, threshold, kernel_size, runs=3):
    # Функция для запуска проходов
    total_time = 0
    for run in range(runs):
        run_time = measure_gpu_processing_time(image_path, output_path, threshold, kernel_size)
        print(f"Время обработки #{run + 1}: {run_time} секунд")
        total_time += run_time
    return total_time / runs  # Среднее время выполнения


if __name__ == "__main__":
    images = ["img_10280x7680.jpg", "img_12800x9600.jpg", "img_20480x15360.jpg"]
    print('Тестирование программы А\n')
    for image in images:
        average_time = average_gpu_processing_time(image, f"convert_{image}", 150, 2)
        print(f"Среднее время обработки для изображения {image}: {average_time} секунд\n")
