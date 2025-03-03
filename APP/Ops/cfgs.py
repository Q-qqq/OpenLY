import copy


from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from pathlib import Path
from ultralytics.utils import yaml_load, yaml_save, LOGGER,ROOT, DEFAULT_CFG_DICT
from ultralytics.cfg import get_cfg, cfg2dict, CFG_FLOAT_KEYS, CFG_BOOL_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_OTHER_KEYS
from APP import APP_SETTINGS, EXPERIMENT_SETTINGS, PROJ_SETTINGS,getExistDirectory, getOpenFileName, APP_ROOT
from APP.Utils import get_widget
import glob

class CfgsTreeWidget(QTreeWidget):
    """
    Attribute:
        args: 参数字典{name:value}
        tips: 从参数信息cfg_status.yaml文件加载出来的字典，含有参数注释，参数控件类型，上下限等
        base_args: 默认显示的基础参数列表
        browse_args: 可以从文件夹中浏览赋值的参数列表
        item_color: 参数数子项颜色
        last_click_item: 上一次点击的子项
        last_click_color: 上一次点击的子项的颜色
        click_widget: 点击子项显示的控件
        roots: 参数树根列表
    """
    Task_Change_Signal = Signal(str)
    def __init__(self,parent):
        super().__init__(parent)
        self.item_color = (QColor(221, 255, 238, 255), QColor(239, 255, 247, 255))
        self.last_click_item = None
        self.last_click_color =None
        self.click_widget = False
        self.roots = []
        self.setAlternatingRowColors(True)

        self.eventConnect()
        self.args = DEFAULT_CFG_DICT
        tips = yaml_load(ROOT / "cfg" / "cfg_status.yaml")
        if isinstance(tips, (str, Path)):
            tips = yaml_load(tips)
        self.base_args = tips.pop("基础参数")
        self.browse_args = tips.pop("可浏览参数")
        self.getWidgetMegs(tips)
        self.initTrees()
        self.setRootColor()
        self.showArgs(False)
        self.expandAll()
        #self.setStyleSheet(u"QTreeWidget::branch:!has-children{background-color: rgb(221, 255, 238);}")



    def eventConnect(self):
        self.itemClicked.connect(self.clickedSlot)


        
        

    def initTrees(self):
        """对参数树进行初始化设置"""
        assert len(self.roots), f"未获得参数树根目录"
        root_font = QFont("幼圆", 12)
        child_font = QFont("幼圆", 10)

        self.clear()
        self.setColumnCount(2)
        head_items = self.headerItem()
        head_items.setText(0, "属性")
        head_items.setText(1, "值")


        self.setColumnWidth(0,140)
        for r in self.roots:
            root = QTreeWidgetItem(self)
            root.setText(0, r)
            root.setFont(0, root_font)
            root.setSizeHint(0, QSize(0, 30))
            root.setSizeHint(1, QSize(0, 30))
            self.addTopLevelItem(root)
        for name, v in self.widgets.items():
            child_a = QTreeWidgetItem(self.topLevelItem(self.roots.index(v["root"])))
            child_a.setText(0, v["cn"])
            child_a.setToolTip(0, v["tooltip"])
            child_a.setStatusTip(0, name)
            child_a.setFont(0, child_font)
            child_a.setSizeHint(0, QSize(0, 25))

            child_a.setToolTip(1, v["default"])
            child_a.setFont(1, child_font)
            child_a.setSizeHint(1, QSize(0, 25))
            value = self.args[name] if name in self.args.keys() else v["default"]
            value = self.checkValue(name, value)
            if v["widgetType"] == "cb":
                w = self.createWidget(name, value)
                self.setItemWidget(child_a,1, w)
            else:
                child_a.setText(1, str(value))

    def setRootColor(self):
        if APP_SETTINGS["style"] == "cute":
            color = QColor(205, 127, 50, 255)
        elif APP_SETTINGS["style"] == "technology":
            color = QColor(138, 43, 224, 255)
        for i in range(self.topLevelItemCount()):
            root = self.topLevelItem(i)
            root.setBackground(0, color)
            root.setBackground(1, color)

    def updateTrees(self, args, overrides=None):
        """初始化参数"""
        self.opt = args
        if overrides:
            overrides = yaml_load(overrides)
        else:
            overrides = {}
        self.updateArgs(args, overrides)

    def initArgs(self):
        """初始化参数"""
        default_args = copy.deepcopy(DEFAULT_CFG_DICT)
        self.updateArgs(default_args)

    def updateArgs(self, args, overrides=None):
        """更新所有参数"""
        args = cfg2dict(get_cfg(args, overrides))
        exp = Path(PROJ_SETTINGS["current_experiment"])
        proj = str(exp.parent)
        name = str(exp.name)
        for k,  v  in zip(["project", "name"],[proj, name]):
            if k in args.keys():
                args[k] = v
        for name, tip in self.widgets.items():
            value = args.get(name) if name in args.keys() else tip["default"]
            value = self.checkValue(name, value)
            self.setValue(name, value)


    def getWidgetValue(self,widget):
        """获取HBox容器内参数控件的值"""
        name = widget.objectName()
        w = get_widget(widget, name+"_value")
        if isinstance(w, QComboBox):
            return w.currentText()
        elif isinstance(w, QCheckBox):
            return w.isChecked()
        elif isinstance(w, (QSpinBox,QDoubleSpinBox)):
            return w.value()
        elif isinstance(w, QLineEdit):
            return w.text()
        else:
            QMessageBox.warning(self.parent(), "警告", f"参数{name}获取值失败")


    def getDict(self, megs):
        """
        将输出的参数信息转换成字典
        Args:
            megs(lsit):[widget-dsb, type-float, default-0, min-0.0, max-0.001, decimal-5, edit-bool,items-list]
        Returns:
            （dict）:将-左右两边分别作为键和值的字典
        """
        dt = {}
        for meg in megs:
            dt.update({meg.split("~")[0]: meg.split("~")[1]})
        return dt

    def getWidgetMegs(self, tips):
        """根据tips参数创建参数组件，并存储其参数至字典"""
        self.widgets = {}
        self.roots.clear()
        for root, tip in tips.items():
            self.roots.append(root)
            for cn, ws in tip.items():
                megs = ws.split("$")
                tooltip = megs.pop(0)
                name = tooltip.split("：")[0]
                t = self.getDict(megs)
                t.update({"name": name})
                t.update({"tooltip": tooltip})
                t.update({"cn": cn})
                t.update({"root": root})
                self.widgets.update({name: t})

    def getName(self, object_name:str):
        """根据输入的对象名去除_value 或者 _browse得到参数名"""
        if object_name.endswith("_value"):
            return object_name.replace("_value", "")
        elif object_name.endswith("_browse"):
            return object_name.replace("_browse", "")
        else:
            return object_name
    
    def checkValue(self, name, value):
        """检查并转换参数值至对应的类型"""
        if value is None:
            value = self.widgets[name]["default"]
        try:
            if name in CFG_FRACTION_KEYS:
                if not isinstance(value, float):
                    value = float(value)
                if not (0.0 <= value <= 1.0):
                    LOGGER.warning(f"属性{name}的值类型为fraction，范围在0-1，但现在为{value}")
                    value = 0 if value < 0 else 1
            elif name in CFG_INT_KEYS:
                if not isinstance(value, int):
                    value = eval(value)
            elif name in CFG_FLOAT_KEYS or name in ["cls_pw", "obj", "obj_pw", "iou_t", "anchor_t", "fl_gamma", "gr", "v5_box", "v5_cls"]:
                if not isinstance(value, float):
                    value = float(value)
            elif name in CFG_BOOL_KEYS  or name in ["noautoanchor"]:
                if not isinstance(value, bool):
                    value = True if value.lower() == "true" else False
            elif name in CFG_OTHER_KEYS:
                if isinstance(value, str):
                    value = value.replace("，", ",")
                    try:
                        value = eval(value)
                    except:
                        pass
        except Exception as ex:
            QMessageBox.warning(self.parent(),"警告", f"属性类型转换错误:{name}：{value}\n{str(ex)}")
        return value

    def setValue(self, name, value):
        """设置对应控件参数值"""
        self.setTreeItemText(name, value)
        self.args[name] = self.checkValue(name,value)


    def removeInvalidKey(self, args):
        invalid_keys = ["tracker", "save_dir", "cfg", "mode"]
        for key in invalid_keys:
            if key in args.keys():
                args.pop(key)
        return args


    def createWidget(self,name, value):
        """
        创建参数控件
        """
        widget_args = self.widgets[name]
        widget_type = widget_args["widgetType"]
        font = QFont("宋体", 10)

        #水平容器
        widget = QWidget(self)
        widget.setObjectName(name)
        widget.setStatusTip(widget_args["type"])
        hl = QHBoxLayout(widget)
        hl.setContentsMargins(0,0,0,0)
        hl.setSpacing(1)

        #浏览文件按钮
        browse_bp = None
        if name in self.browse_args:
            browse_bp = QPushButton(widget)
            browse_bp.setText("...")
            browse_bp.setObjectName(name + "_browse")
            browse_bp.setMaximumWidth(20)
            browse_bp.clicked.connect(lambda: self.browsePbClicked(widget))
        #创建
        if  widget_type == "cbb":
            cbb = QComboBox(widget)
            items = widget_args.get("items")
            if items:
                cbb.addItems(items.split(","))
            if name == "model":
                models = glob.glob(str(APP_ROOT / "ultralytics" /"cfg" / "models" / "**" / "**"), recursive=False)
                cbb.clear()
                cbb.addItems([Path(m).name for m in models if Path(m).suffix in [".yaml", ".yml"]])
            current_text = str(value)
            cbb_items = [cbb.itemText(i) for i in range(cbb.count())]
            if current_text not in cbb_items:
                cbb.addItem(current_text)
            cbb.setCurrentText(current_text)
            cbb.setEditable(eval(widget_args["edit"]))
            cbb.setObjectName(name + "_value")
            cbb.setFont(font)
            cbb.currentTextChanged.connect(lambda: self.changeEvents(cbb))
            hl.addWidget(cbb)
            if browse_bp:
                hl.addWidget(browse_bp)
        elif widget_type == "cb":
            cb = QCheckBox(widget)
            cb.setFont(font)
            cb.setObjectName(name + "_value")
            cb.setText("")
            cb.setChecked(bool(value))
            cb.stateChanged.connect(lambda :self.changeEvents(cb))
            hl.addWidget(cb)
            if browse_bp:
                hl.addWidget(browse_bp)
        elif widget_type == "sb":
            sb = QSpinBox(widget)
            sb.setMinimum(int(float(widget_args["min"])))
            sb.setMaximum(int(float(widget_args["max"])))
            sb.setFont(font)
            sb.setObjectName(name+"_value")
            sb.setValue(int(value))
            sb.valueChanged.connect(lambda: self.changeEvents(sb))
            hl.addWidget(sb)
            if browse_bp:
                hl.addWidget(browse_bp)
        elif widget_type == "dsb":
            dsb = QDoubleSpinBox(widget)
            dsb.setMinimum(float(widget_args["min"]))
            dsb.setMaximum(float(widget_args["max"]))
            dsb.setDecimals(int(float(widget_args["decimal"])))
            dsb.setFont(font)
            dsb.setObjectName(name+"_value")
            dsb.setValue(float(value))
            dsb.valueChanged.connect(lambda: self.changeEvents(dsb))
            hl.addWidget(dsb)
            if browse_bp:
                hl.addWidget(browse_bp)
        elif widget_type == "le":
            le = QLineEdit(widget)
            le.setFont(font)
            le.setObjectName(name + "_value")
            le.setText(str(value))
            le.textChanged.connect(lambda: self.changeEvents(le))
            hl.addWidget(le)
            if browse_bp:
                hl.addWidget(browse_bp)
        return widget


    def changeEvents(self, widget):
        """修改参数后将参数同步至self.args"""
        name = self.getName(widget.objectName())
        if isinstance(widget, QComboBox):
            self.args[name] = self.checkValue(name, widget.currentText())
            if name == "task":
                self.Task_Change_Signal.emit(self.args[name])
        elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            self.args[name] = self.checkValue(name, widget.value())
        elif isinstance(widget, QLineEdit):
            self.args[name] = self.checkValue(name, widget.text())
        elif isinstance(widget, QCheckBox):
            self.args[name] = self.checkValue(name, str(widget.isChecked()))
        self.setTreeItemText(name, self.args[name])
        self.save()

    def browsePbClicked(self, widget):
        name = self.getName(widget.objectName())
        filters = {"model": "model(*.yaml | *.pt)", "data": "data(*.yaml | *.zip | *.rar)", "pretrained": "model(*.pt)", "source":"source(*)"}
        if name == "data" and self.args["task"] == "classify":
            file_name = getExistDirectory(self, "分类数据集文件夹")
        else:
            file_name = getOpenFileName(self, name, filter=filters[name])[0]
        if file_name != "":
            value_w = get_widget(widget, name+"_value")
            if isinstance(value_w, QComboBox):
                value_w.setCurrentText(file_name)
            elif isinstance(value_w, QLineEdit):
                value_w.setText(file_name)
            self.setValue(name, file_name)


    def clickedSlot(self, item:QTreeWidgetItem, column):
        if item == self.last_click_item:  #此次点击与上一次点击属于同一个item
            if column == 0:  #此次点击为列0  取消widget选中状态
                w = self.itemWidget(item, 1)
                if w:
                    if self.widgets[w.objectName()]["widgetType"] != "cb":  #不是checkBox的删除
                        item.setText(1, str(self.getWidgetValue(w)))
                        self.removeItemWidget(item, 1)
                        w.deleteLater()
            return
        if self.last_click_item:  #去除上次点击的widget选中状态
            self.last_click_item.setBackground(0, self.last_click_color)
            self.last_click_item.setBackground(1, self.last_click_color)
            w = self.itemWidget(self.last_click_item, 1)
            if w:
                if self.widgets[w.objectName()]["widgetType"] != "cb":
                    self.last_click_item.setText(1, str(self.getWidgetValue(w)))
                    self.removeItemWidget(self.last_click_item,1)
                    w.deleteLater()
        self.last_click_color = item.backgroundColor(0)
        self.last_click_item = item
        selected_color = QColor(9, 38, 229, 100)
        item.setBackground(0, selected_color)
        item.setBackground(1, selected_color)
        top_level = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        if column == 1 and item not in top_level:
            w = self.itemWidget(self.last_click_item, 1)
            if w and self.widgets[w.objectName()]["widgetType"] == "cb":
                pass
                #w.setStyleSheet(u"background-color: rgba(255, 255, 255, 255);")
            else:
                widget = self.createWidget(item.statusTip(0), item.text(1))
                self.setItemWidget(item, 1, widget)
                item.setText(1, "")



    def save(self):
        """保存参数至yaml文件"""
        args = copy.deepcopy(self.args)
        if self.args["source"] == "选中图像":
            args["source"] = None
        yaml_save(self.opt, args)



    def setTreeItemText(self, name, value):
        """根据提供的参数英文名称和对应的值。设置参数树上对应的值"""
        items = self.findItems(self.widgets[name]["cn"], Qt.MatchContains | Qt.MatchRecursive, column=0)
        for item in items:
            if item.text(0) == self.widgets[name]["cn"]:
                if self.widgets[name]["widgetType"] != "cb":
                    item.setText(1, str(value))
                else:
                    w = self.itemWidget(item, 1)
                    cb = get_widget(w, name+"_value")
                    cb.setChecked(value)

    def getTreeItemText(self, name):
        """根据参数英文名称获取参数树上的值"""
        items = self.findItems(self.widgets[name]["cn"], Qt.MatchContains | Qt.MatchRecursive, column=0)
        if items:
            return items[0].text(1)
        else:
            return ""



    def showArgs(self, more):
        """显示参数到界面的cfg_tw(tree widget)上
        Args:
            more(bool): 是否显示所有参数"""
        brush = QBrush()
        brush.setStyle(Qt.SolidPattern)
        if APP_SETTINGS["style"] == "cute":
            color = ( QColor(216, 191, 216, 255), QColor(230, 224, 255, 255))
        elif APP_SETTINGS["style"] == "technology":
            color = (QColor(14, 19, 33, 255), QColor(26,72,95, 255))
        self.setRootColor()
        ci = 1
        for i in range(self.topLevelItemCount()):
            child = self.topLevelItem(i)
            for j in range(child.childCount()):
                brush.setColor(color[ci])
                if not more:  # 只显示基础参数
                    if child.child(j).statusTip(0) in self.base_args:
                        child.child(j).setHidden(False)
                        child.child(j).setBackground(0, brush)
                        child.child(j).setBackground(1, brush)
                        ci = 0 if ci == 1 else 1
                    else:
                        child.child(j).setHidden(True)
                else:  # 显示所有参数
                    child.child(j).setHidden(False)
                    child.child(j).setBackground(0, brush)
                    child.child(j).setBackground(1, brush)
                    ci = 0 if ci == 1 else 1