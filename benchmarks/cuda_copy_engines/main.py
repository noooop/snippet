import ctypes
import torch


def get_cuda_copy_engines(device_id=0):
    """
    通过底层 CUDA Driver API 查询指定 GPU 的 Copy Engine (DMA) 数量
    """
    if not torch.cuda.is_available():
        print("未检测到可用的 CUDA 环境！")
        return None

    # 1. 加载底层 CUDA 驱动库
    try:
        # Windows 系统为 nvcuda.dll，Linux 系统为 libcuda.so
        cuda = ctypes.CDLL('libcuda.so') if ctypes.sizeof(ctypes.c_void_p) == 8 and not torch.sys.platform.startswith(
            'win') else ctypes.CDLL('nvcuda.dll')
    except Exception as e:
        print(f"加载底层 CUDA 驱动失败，请检查驱动是否安装正确: {e}")
        return None

    # 2. 定义 CUDA 常量 (对应底层 C++ 的 CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)
    # 该属性用于获取设备是否支持并发内存复制以及有多少个异步引擎
    # https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html
    # cuda-samples/cpp/1_Utilities/deviceQuery
    # ./deviceQuery
    # Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40

    # 3. 初始化 CUDA 驱动
    cuda.cuInit(0)

    # 4. 获取设备句柄
    device = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(device), device_id)

    # 5. 查询属性
    count = ctypes.c_int()
    cuda.cuDeviceGetAttribute(ctypes.byref(count), CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, device)

    return count.value


if __name__ == "__main__":
    # 获取当前激活的显卡信息
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    engine_count = get_cuda_copy_engines(device_id)

    print("=" * 50)
    print(f"当前 GPU 设备: [{device_id}] {gpu_name}")
    print(f"底层硬件报告的 Copy Engines (DMA) 数量: {engine_count}")
    print("=" * 50)

    if engine_count and engine_count >= 2:
        print(f"🎉 您的显卡拥有 {engine_count} 个 Copy Engines，支持【双向全双工】或【计算与通信完美重叠】！")
    else:
        print("⚠️ 您的显卡仅报告 1 个 Copy Engine。如果在 Windows 下，请确认是否被 WDDM 驱动限制。")
