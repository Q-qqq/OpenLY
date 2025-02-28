import cv2
import numpy as np
from collections import deque
import time


def fast_selection(image_path, seed_point, tolerance=30, color_space='RGB'):
    """
    快速选择算法实现

    参数：
    image_path: 输入图像路径
    seed_point: 种子点坐标 (x, y)
    tolerance: 颜色容差阈值（0-100）
    color_space: 使用的颜色空间（'RGB' 或 'LAB'）

    返回：
    mask: 生成的选区掩膜（二值图像）
    """
    # 读取图像并转换颜色空间
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")

    # 将容差转换为0-442范围（RGB颜色空间最大欧式距离）
    scaled_tolerance = tolerance * 4.418  # 100 -> 441.8（约等于√3*255）

    # 颜色空间转换
    if color_space.upper() == 'LAB':
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:  # 默认使用RGB
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取图像尺寸
    height, width = processed_image.shape[:2]

    # 初始化变量
    mask = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=bool)
    queue = deque([seed_point])

    # 验证种子点有效性
    x, y = seed_point
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("种子点超出图像范围")

    # 获取种子颜色
    seed_color = processed_image[y, x]

    # 定义八邻域偏移量
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    while queue:
        x, y = queue.popleft()

        # 跳过已访问或越界的像素
        if not (0 <= x < width and 0 <= y < height) or visited[y, x]:
            continue

        visited[y, x] = True

        # 计算颜色差异
        current_color = processed_image[y, x]
        color_diff = np.linalg.norm(current_color - seed_color)

        if color_diff <= scaled_tolerance:
            mask[y, x] = 255
            # 将相邻像素加入队列
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if not visited[ny, nx]:
                        queue.append((nx, ny))

    # 使用形态学操作优化选区边界
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# 使用示例
if __name__ == "__main__":
    # 参数设置
    image_path = "G:\\pro\\new_nut\\data\\images\\train\\inclusion_92.jpg"
    image_path2 = "G:\\project\\p6\\data\\images\\train\\01_17_12_3.bmp"
    seed_point = (400, 400)  # (x, y) 格式
    tolerance = 50  # 0-100之间的值
    color_space = 'LAB'  # 可选 'RGB' 或 'LAB'

    # 生成掩膜
    t1 = time.time()
    #mask = fast_selection(image_path2, seed_point, tolerance, color_space)
    t2 = time.time()
    t3 = t2- t1
    print(t3)
    image = cv2.imread(image_path2)
    h, w = image.shape[:2]
    mask = np.zeros([h+2,w+2], dtype=np.uint8)
    t4 = time.time()
    mask = cv2.floodFill(image,mask, seed_point, [255,255,255],  [12,12,12],[12,12,12])
    t5 = time.time()
    print(t5-t4)
    # 显示结果
    cv2.imshow("Selection Mask", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()