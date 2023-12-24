import pyopencl as cl
import time
import cv2
import numpy as np

# OpenCL kernel
OPENCL_KERNEL = """
__kernel void pyr_down(__global float *src, __global float *dest, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int new_width = width / 2;
    int new_height = height / 2;

    if (x < new_width && y < new_height) {
        int idx = (y * 2 * width + x * 2) * 3;  // Индекс для RGB
        float r = (src[idx] + src[idx + 3] + src[idx + width * 3] + src[idx + width * 3 + 3]) / 4.0;
        float g = (src[idx + 1] + src[idx + 4] + src[idx + width * 3 + 1] + src[idx + width * 3 + 4]) / 4.0;
        float b = (src[idx + 2] + src[idx + 5] + src[idx + width * 3 + 2] + src[idx + width * 3 + 5]) / 4.0;

        dest[(y * new_width + x) * 3] = r;
        dest[(y * new_width + x) * 3 + 1] = g;
        dest[(y * new_width + x) * 3 + 2] = b;
    }
}
"""


def gpu_pyr_down(image):
    # Инициализация OpenCL
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)

    mf = cl.mem_flags
    src_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    dest_buf = cl.Buffer(context, mf.WRITE_ONLY, image.nbytes // 4)

    # Компиляция ядра
    program = cl.Program(context, OPENCL_KERNEL).build()

    # Размеры изображения
    height, width, _ = image.shape
    new_width = width // 2
    new_height = height // 2
    output = np.zeros((new_height, new_width, 3), dtype=np.float32)

    # Выполнение ядра
    program.pyr_down(queue, (new_width, new_height), None, src_buf, dest_buf, np.int32(width), np.int32(height))

    # Чтение результата
    cl.enqueue_copy(queue, output, dest_buf).wait()

    return output


def measure_gpu_processing_time(image_path, output_path):
    start_time = time.time()

    image = cv2.imread(image_path).astype(np.float32)
    reduced_image = gpu_pyr_down(image)
    cv2.imwrite(output_path, reduced_image)

    end_time = time.time()
    return end_time - start_time  # Время выполнения в секундах


def average_gpu_processing_time(image_path, output_path, runs=3):
    total_time = 0
    for run in range(runs):
        run_time = measure_gpu_processing_time(image_path, output_path)
        print(f"Время обработки #{run + 1}: {run_time} секунд")
        total_time += run_time
    return total_time / runs  # Среднее время выполнения


if __name__ == "__main__":
    images = ["img_10280x7680.jpg", "img_12800x9600.jpg", "img_20480x15360.jpg"]
    print('Тестирование программы B\n')
    for image in images:
        average_time = average_gpu_processing_time(image, f"pyramid_{image}")
        print(f"Среднее время обработки для изображения {image}: {average_time} секунд\n")
