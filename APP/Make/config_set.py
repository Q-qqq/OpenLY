from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from ultralytics.cfg import get_cfg, CFG_FLOAT_KEYS, CFG_BOOL_KEYS, CFG_FRACTION_KEYS,CFG_INT_KEYS,CFG_OTHER_KEYS
from ultralytics.utils import yaml_save, ROOT

from APP.Designer import config_setUI

class Configset(config_setUI.Ui_MainWindow):
    def __init__(self, cfg, overrides=None, mainwindow=None):
        super(Configset, self).__init__()
        self.setupUi(mainwindow)
        self.cfg_path = cfg
        self.args = get_cfg(cfg, overrides)
        self.assign()
        self.Aave_a.triggered.connect(self.saveToFile)
        self.tooltipSave()
        self.cbbItemSave()

    def getWeight(self, arg):
        """根据参数名称获取对应的组件"""
        w_names = dir(self)
        for w_name in w_names:
            a_name = self.getArg(w_name)
            if (a_name == arg):
                return getattr(self, w_name)

    def tooltipSave(self):

        base_args = ["task",
                     "model", "data", "epoches", "batch","imgsz","device","name", "optimizer","resume",
                     "val","conf","iou",
                     "source",
                     "format",
                     "lr0","lrf","hsv_h", "hsv_s", "hsv_v", "degrees","translate","shear","perspective","flipud","fliplr","mosaic","mixup", "copy_paste" ,"auto_augment", "erasing", "crop_fraction"]
        tips = {"基础参数": base_args, "全局参数": {}, "训练参数": {}, "验证/测试参数": {}, "预测参数": {}, "导出参数": {}, "可视化参数": {},
                "超参数": {}}
        l_names = dir(self)
        for i in range(102):
            l = getattr(self, f"l{i+1}")
            if isinstance(l, QLabel):
                ta = self.getTip(l.toolTip())
                sp = "：" if "：" in ta else ":"
                a_name = ta.split(sp)[0]
                tip = ta.split(sp)[1]
                w = self.getWeight(a_name)
                t = self.getTip(w.statusTip().lower()) if w!=None else ""
                d = self.getTip(w.toolTip().lower()) if w!=None else ""
                page_name = w.parent().parent().parent().parent().objectName() if w != None else ""
                g = "训练参数" if page_name == "train_p" \
                    else "全局参数" if page_name == "global_p" \
                    else "验证/测试参数" if page_name == "val_p" \
                    else "预测参数" if page_name == "predict_p" \
                    else "导出参数" if page_name == "export_p" \
                    else "超参数" if page_name == "hyp_p" \
                    else "可视化参数" if page_name == "visualize_p" \
                    else ""
                wt = "cbb" if isinstance(w, QComboBox) \
                    else "le" if isinstance(w, QLineEdit) \
                    else "dsb" if isinstance(w, QDoubleSpinBox) \
                    else "sb" if isinstance(w, QSpinBox) \
                    else ""
                meg = f"{a_name}：{tip}$widgetType~{wt}$type~{t}$default~{d}"
                if isinstance(w, (QDoubleSpinBox, QSpinBox)):
                    mi = w.minimum()
                    ma = w.maximum()
                    dem = w.decimals() if isinstance(w, QDoubleSpinBox) else 0
                    meg += f"$min~{mi}$max~{ma}"
                    meg += f"$decimal~{dem}" if isinstance(w, QDoubleSpinBox) else ""
                if isinstance(w, QComboBox):
                    edit = w.isEditable()
                    meg += f"$edit~{edit}"
                tips[g].update({l.text().replace("：", "") : meg})
        yaml_save(ROOT / "ultralytics"/"cfg"/"cfg_status.yaml", tips)

    def cbbItemSave(self):
        w_names = dir(self)
        cbb_items = {}
        for w_name in w_names:
            w = getattr(self,w_name)
            if isinstance(w, QComboBox):
                a_name = self.getArg(w_name)
                items = [w.itemText(i) for i in range(w.count())]
                cbb_items.update({a_name:items})
        yaml_save(ROOT / "ultralytics"/"cfg"/"selected.yaml", cbb_items)





    def getTip(self, tip):
        ts = tip.split(">")
        ta = [s for s in ts if s != "" and not s.startswith("<")]
        if len(ta) == 0: return ""
        ta[0] = ta[0].split("<")[0]
        return ta[0]

    def getArg(self, wname):
        """更具组件名称获取对应的参数名"""
        a_name = wname.lower().split("_")
        a_name.pop(-1)
        a_name = "_".join(a_name)
        return a_name

    def assign(self):
        """对参数组件进行赋值"""
        for arg in self.args:
            w = self.getWeight(arg[0])
            if isinstance(w, QComboBox):
                w.setCurrentText(str(arg[1]))
            elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                w.setValue(arg[1] if arg[1] else 0)
            elif isinstance(w, QLineEdit):
                w.setText(str(arg[1]))

    def saveToFile(self):
        self.save(True)
        QMessageBox.information(None, "cfg save to file", "保存成功")

    def save(self, to_file=False):
        for arg in self.args:
            w = self.getWeight(arg[0])
            if isinstance(w, QComboBox):
                setattr(self.args, arg[0], eval(w.currentText()) if arg[0] in CFG_BOOL_KEYS else w.currentText())
            elif isinstance(w, (QDoubleSpinBox, QSpinBox)):
                setattr(self.args, arg[0],w.value())
            elif isinstance(w, QLineEdit):
                setattr(self.args, arg[0], eval(w.text()))
        if to_file:
            yaml_save(self.cfg_path, vars(self.args))





if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Configset("G://源码//源码//ultralytics//cfg//default.yaml", mainwindow=MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())