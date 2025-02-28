import cv2
import numpy as np

# 全局变量
drawing = False         # 是否正在绘制
mode_add = True         # True=添加选区，False=减去选区
ix, iy = -1, -1         # 初始坐标
brush_size = 40         # 画笔大小
mask = None             # 蒙版（0=背景，1=前景，2=可能的前景/背景）
output = None           # 最终输出图像

# 初始化GrabCut参数
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

def on_mouse(event, x, y, flags, param):
    global ix, iy, drawing, mask, output, img

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        draw(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        update_grabcut()

def draw(x, y):
    global mask
    # 根据模式绘制蒙版
    if mode_add:
        cv2.circle(mask, (x, y), brush_size, 1, -1)  # 添加选区（前景）
    else:
        cv2.circle(mask, (x, y), brush_size, 0, -1)  # 减去选区（背景）

def update_grabcut():
    global mask, output, img
    # 运行GrabCut算法优化选区
    temp_mask = mask.copy()
    cv2.grabCut(img, temp_mask, [10,10,600,600], bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    # 生成最终蒙版（前景和可能的前景）
    final_mask = np.where((temp_mask == 1) | (temp_mask == 3), 255, 0).astype('uint8')
    # 应用蒙版到原图
    output = cv2.bitwise_and(img, img, mask=final_mask)

# 主程序
if __name__ == "__main__":
    img = cv2.imread("G:\\project\\p6\\data\\images\\train\\01_18_34_2.bmp")
    if img is None:
        print("Error: Image not found!")
        exit()

    # 初始化蒙版（全背景）
    mask = np.zeros(img.shape[:2], np.uint8)
    output = img.copy()

    cv2.namedWindow("Quick Select Tool")
    cv2.setMouseCallback("Quick Select Tool", on_mouse)

    while True:
        # 显示原图和选区效果
        display = cv2.addWeighted(img, 0.7, output, 0.3, 0)
        cv2.putText(display, f"Mode: {'Add (+)' if mode_add else 'Subtract (-)'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Quick Select Tool", display)

        key = cv2.waitKey(1)
        if key == ord('m'):  # 切换模式
            mode_add = not mode_add
        elif key == ord('r'):  # 重置选区
            mask = np.zeros(img.shape[:2], np.uint8)
            output = img.copy()
        elif key == 27:  # ESC退出
            break

    cv2.destroyAllWindows()