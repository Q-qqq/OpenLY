import sys
import cv2  # 新增OpenCV依赖
import numpy as np
from PySide2.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QVBoxLayout, QPushButton
from PySide2.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QPaintEvent, QMouseEvent
from PySide2.QtCore import Qt, QPoint, QSize


class MaskedImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = QPixmap()
        self.mask_pixmap = QPixmap()
        self.drawing = False
        self.last_original_point = QPoint()
        self.mask_color = QColor(255, 0, 0, 100)
        self.brush_size = 20



    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        # 绘制原图（缩放填充控件）
        if not self.original_pixmap.isNull():
            painter.drawPixmap(self.rect(), self.original_pixmap)
            # 绘制掩膜（同样缩放）
            painter.drawPixmap(self.rect(), self.mask_pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and not self.original_pixmap.isNull():
            self.drawing = True
            pos = self._convert_pos_to_original(event.pos())
            self.last_original_point = pos
            self._draw_mask(pos)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and event.buttons() & Qt.LeftButton:
            pos = self._convert_pos_to_original(event.pos())
            self._draw_mask(pos)
            self.last_original_point = pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _convert_pos_to_original(self, widget_pos: QPoint) -> QPoint:
        """将控件坐标转换为原图坐标"""
        if self.original_pixmap.isNull():
            return QPoint()
        # 计算缩放后的图像区域
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        widget_width = self.width()
        widget_height = self.height()

        scale = min(widget_width / img_width, widget_height / img_height)
        scaled_w = img_width * scale
        scaled_h = img_height * scale
        x_offset = (widget_width - scaled_w) / 2
        y_offset = (widget_height - scaled_h) / 2

        # 转换坐标
        original_x = (widget_pos.x() - x_offset) / scale
        original_y = (widget_pos.y() - y_offset) / scale

        # 限制在图像范围内
        original_x = max(0, min(img_width - 1, original_x))
        original_y = max(0, min(img_height - 1, original_y))
        return QPoint(int(original_x), int(original_y))

    def _draw_mask(self, pos: QPoint):
        """在掩膜上绘制"""
        painter = QPainter(self.mask_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.setPen(QPen(self.mask_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.last_original_point, pos)
        painter.end()
        self.update()

    def set_opencv_mask(self, cv_mask: np.ndarray):
        """从OpenCV图像设置掩膜（必须为单通道或BGRA格式）"""
        if cv_mask.ndim == 2:  # 单通道灰度图
            qimage = QImage(
                cv_mask.data,
                cv_mask.shape[1],
                cv_mask.shape[0],
                cv_mask.strides[0],
                QImage.Format_Grayscale8
            )
        elif cv_mask.shape[2] == 4:  # BGRA格式
            qimage = QImage(
                cv_mask.data,
                cv_mask.shape[1],
                cv_mask.shape[0],
                cv_mask.strides[0],
                QImage.Format_ARGB32
            )
        else:
            raise ValueError("Unsupported OpenCV image format")

        self.mask_pixmap = QPixmap.fromImage(qimage)
        self.update()

    def set_opencv_mask2(self, cv_mask: np.ndarray):
        """从OpenCV图像设置掩膜（必须为单通道或BGRA格式）"""
        if cv_mask.ndim == 2:
            cv2.merge([cv_mask, cv_mask, cv_mask], cv_mask)
        if cv_mask.shape[2] == 3:
            if cv_mask.max() > 1:
                cv_mask = cv_mask / 255
            cv_mask[:, :, 2] = cv_mask[:, :, 2] * 255
            cv_mask[:, :, 0] = cv_mask[:, :, 0] * 0
            cv_mask[:, :, 1] = cv_mask[:, :, 1] * 0
            cv_mask = cv_mask.astype(np.uint8)
            b, g, r = cv2.split(cv_mask)
            a = np.ones_like(b) * 60
            cv_mask = cv2.merge([b, g, r, a])
        qimage = QImage(
            cv_mask.data,
            cv_mask.shape[1],
            cv_mask.shape[0],
            cv_mask.strides[0],
            QImage.Format_ARGB32
        )

        self.mask = cv_mask
        self.mask_pixmap = QPixmap.fromImage(qimage)
        self.update()

    def get_opencv_mask(self) -> np.ndarray:
        """获取当前掩膜的OpenCV格式（BGRA四通道）"""
        qimage = self.mask_pixmap.toImage().convertToFormat(QImage.Format_ARGB32)
        ptr = qimage.bits()
        ptr.setsize(qimage.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape(
            qimage.height(),
            qimage.width(),
            4
        )  # shape (H, W, 4)
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)  # 转换为OpenCV的BGRA格式

    def set_image(self, pixmap: QPixmap):
        self.original_pixmap = pixmap
        # 初始化透明掩膜（与图像同尺寸）
        self.mask_pixmap = QPixmap(pixmap.size())
        self.mask_pixmap.fill(Qt.transparent)
        self.setMinimumSize(pixmap.size())
        self.update()

    # 以下方法保持与原始版本相同（paintEvent/mouse事件处理等）...


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("半透明掩膜绘制示例")
        self.setGeometry(100, 100, 800, 600)

        # 主控件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # 图像显示控件
        self.image_widget = MaskedImageWidget()
        layout.addWidget(self.image_widget)

        # 按钮
        btn_load = QPushButton("加载图片")
        btn_load.clicked.connect(self.load_image)
        layout.addWidget(btn_load)
        # 新增测试按钮
        btn_test_mask = QPushButton("生成测试掩膜")
        btn_test_mask.clicked.connect(self.generate_test_mask)
        layout.addWidget(btn_test_mask)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "打开图片", "", "图片文件 (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_widget.set_image(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def generate_test_mask(self):
        """使用OpenCV生成测试掩膜"""
        if self.image_widget.original_pixmap.isNull():
            return

        # 创建OpenCV格式的随机圆形掩膜
        h = self.image_widget.original_pixmap.height()
        w = self.image_widget.original_pixmap.width()
        mask = np.zeros((h, w, 4), dtype=np.uint8)
        #mask = cv2.imread("G://final_mask.jpg")

        # 在OpenCV中绘制半透明绿色圆形
        center = (w // 2, h // 2)
        radius = w+10
        cv2.circle(
            mask,
            center,
            radius,
            (0, 255, 0, 128),  # BGRA颜色（OpenCV格式）
            -1
        )

        # 应用掩膜到控件
        self.image_widget.set_opencv_mask2(mask)

    # ... 其他代码保持相同 ...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())