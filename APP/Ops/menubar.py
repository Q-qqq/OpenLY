import copy
import shutil

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from pathlib import Path


from APP  import PROJ_SETTINGS, getExistDirectory, getOpenFileName, APP_SETTINGS, loadQssStyleSheet
from APP.Utils.filters import MenuFilter
from APP.Make import VocToYolo, CocoToYolo, PngToYolo


class MenuTool(QObject):
    def __init__(self, parent):
        """parent: train"""
        super().__init__(parent)
        self.eventConnect()
        if APP_SETTINGS["style"] == "cute":
            self.parent().Style_cute_a.setChecked(True)
            self.parent().Style_technology_a.setChecked(False)
            self.parent().Style_light_a.setChecked(False)


    def eventConnect(self):
        self.parent().New_project_a.triggered.connect(self.file_newProject)
        self.parent().Open_project_a.triggered.connect(self.file_openProject)
        self.parent().New_experiment_a.triggered.connect(self.file_newExperiment)
        self.parent().Open_experiment_a.triggered.connect(self.file_openExperiment)
        self.parent().Load_dataset_a.triggered.connect(self.edit_loadDataset)
        self.parent().menubar.installEventFilter(MenuFilter(self))
        self.parent().Show_allargs_a.triggered.connect(lambda :self.parent().cfgs_widget.showArgs(self.parent().Show_allargs_a.isChecked()))
        self.parent().Cfgs_init_a.triggered.connect(self.edit_initArgsAction)
        self.parent().Save_a.triggered.connect(self.file_save)
        self.parent().Save_as_a.triggered.connect(self.file_saveAs)
        self.parent().Exit_a.triggered.connect(self.file_exit)
        self.parent().Load_model_a.triggered.connect(self.edit_loadModelAction)
        self.parent().Back_start_a.triggered.connect(self.file_showStart)
        self.parent().Voc_to_yolo_a.triggered.connect(self.tool_vocToYolo)
        self.parent().Coco_to_yolo_a.triggered.connect(self.tool_cocoToYolo)
        self.parent().Png_to_yolo_a.triggered.connect(self.tool_pngToTolo)
        self.parent().Style_cute_a.triggered.connect(self.file_loadCuteStyle)
        self.parent().Style_technology_a.triggered.connect(self.file_loadTechnologyStyle)
        self.parent().Style_light_a.triggered.connect(self.file_loadLightStyle)

    def file_showStart(self):
        """显示开始界面"""
        self.parent().start_w.show()

    def file_newProject(self):
        """新建项目"""
        proj = getExistDirectory(self.parent(), "项目文件")
        if proj != "":
            if (Path(proj) / "SETTINGS.yamml").exists():
                QMessageBox.warning(self.parent(), "提示","项目已存在，请重新选择一个空白文件夹")
                return
            self.parent().start_w.addNewProject(proj)

    def file_openProject(self):
        """打开项目"""
        proj = getExistDirectory(self.parent(), "项目文件")
        if proj != "":
            self.parent().start_w.addOldProject(proj)

    def file_newExperiment(self):
        """新建实验"""
        name, ok = QInputDialog.getText(self.parent(), "新建实验", "实验名称：", text="")
        if ok and name != "":
            new_experiment = Path(PROJ_SETTINGS["name"]) / "experiments" / name
            if new_experiment.exists():
                QMessageBox.warning(self.parent(), "提示", "实验已存在，请重新命名")
                self.file_newExperiment()
                return
            self.parent().openExperiment(new_experiment)
        elif ok and name == "":
            QMessageBox.warning(self.parent(), "提示", "实验名称不能为空, 创建失败")

    def file_openExperiment(self):
        """打开实验"""
        dir = getExistDirectory(self.parent(), "打开实验", dir=str(Path(PROJ_SETTINGS["name"])))
        if dir != "":
            self.parent().openExperiment(dir)

    def file_save(self):
        """保存实验"""
        self.parent().cfgs_widget.save()
        if Path(self.parent().experiment).parent.name == "expcache":   #实验未命名，存储于缓存区
            name, ok = QInputDialog.getText(self.parent(), "保存实验", "实验名称：", text=Path(PROJ_SETTINGS['current_experiment']).name)
            if ok and name != "":
                exp_p = Path(f"{PROJ_SETTINGS['name']}//experiments//{name}")
                if exp_p.exists():
                    QMessageBox.information(self.parent(), "提示", "实验已存在，请重新命名")
                    self.file_save()
                    return
                shutil.copytree(PROJ_SETTINGS['current_experiment'], str(exp_p))
                shutil.rmtree(PROJ_SETTINGS['current_experiment'])
                self.parent().openExperiment(exp_p)
                PROJ_SETTINGS.updateExperiment(str(Path(f"{PROJ_SETTINGS['name']}//experiments//{name}")))
            elif ok and name == "":
                QMessageBox.information(self.parent(), "提示", "保存失败，实验名称不能为空")


    def file_saveAs(self):
        """实验另存为"""
        name, ok = QInputDialog.getText(self.parent(), "实验另存为", "实验名称：", text="")
        if ok and name != "":
            new_experiment = Path(PROJ_SETTINGS["name"]) / "experiments" / name
            shutil.copytree(PROJ_SETTINGS["current_experiment"], new_experiment)
            self.parent().openExperoment(new_experiment)
        elif ok and name == "":
            QMessageBox.warning(self.parent(), "提示", "实验名称不能为空, 创建失败")

    def file_exit(self):
        """退出"""
        self.parent().close()
    
    def file_loadCuteStyle(self):
        """加载cute主题"""
        if self.parent().Style_cute_a.isChecked():
            APP_SETTINGS.update({"style": "cute"})
            loadQssStyleSheet(self.parent().app, self.parent())
            self.parent().Style_technology_a.setChecked(False)
            self.parent().Style_light_a.setChecked(False)
            self.parent().cfgs_widget.showArgs(self.parent().Show_allargs_a.isChecked())
            
    
    def file_loadTechnologyStyle(self):
        """加载科技主题"""
        if self.parent().Style_technology_a.isChecked():
            APP_SETTINGS.update({"style": "technology"})
            loadQssStyleSheet(self.parent().app, self.parent())
            self.parent().Style_cute_a.setChecked(False)
            self.parent().Style_light_a.setChecked(False)
            self.parent().cfgs_widget.showArgs(self.parent().Show_allargs_a.isChecked())
    
    def file_loadLightStyle(self):
        """加载light主题"""
        if self.parent().Style_light_a.isChecked():
            APP_SETTINGS.update({"style": "light"})
            loadQssStyleSheet(self.parent().app, self.parent())
            self.parent().Style_cute_a.setChecked(False)
            self.parent().Style_technology_a.setChecked(False)
            self.parent().cfgs_widget.showArgs(self.parent().Show_allargs_a.isChecked())
            

    def edit_loadDataset(self):
        """加载数据集"""
        data = self.parent().cfgs_widget.args["data"]
        if data != "":
            self.parent().buildDataset(data)

    def edit_showClasses(self):
        if not self.parent().sift_dataset:
            return
        names = self.parent().sift_dataset.getNames()
        self.parent().Classes_menu.clear()
        for num,cls in names.items():
            cm = self.edit_createAction(self.parent().menubar, cls)
            self.parent().Classes_menu.addMenu(cm)
        a3 = self.parent().Classes_menu.addAction("添加")
        a3.triggered.connect(self.edit_addClassAction)

    def edit_createAction(self, parent, cls):
        cm = QMenu(parent=parent, title=cls)
        a1 = QAction("重命名", cm)
        a1.setObjectName(cls)
        a2 = QAction("删除", cm)
        a1.triggered.connect(lambda: self.edit_renameClassAction(a1))
        a2.triggered.connect(lambda: self.edit_deleteClassAction(a1))
        cm.addActions([a1, a2])
        return cm

    def edit_renameClassAction(self, cls):
        name = cls.objectName()
        new_name, ok = QInputDialog.getText(self.parent(), "重命名", "新种类名称：", text=name)
        if ok and new_name != "":
            self.parent().label_ops.renameClass(name, new_name)
        elif ok and new_name == "":
            QMessageBox.warning(self.parent(), "提示", "种类名称不能为空, 重命名失败")

    def edit_deleteClassAction(self, action):
        name = action.parent().title()
        req = QMessageBox.information(self.parent(), "提示", f"是否确定删除种类{name},将同时删除种类{name}的标签或种类文件", QMessageBox.Yes | QMessageBox.No)
        if req == QMessageBox.Yes:
            self.parent().label_ops.deleteClass(name)

    def edit_addClassAction(self):
        name, ok = QInputDialog.getText(self.parent(), "重命名", "新种类名称：", text="")
        if ok and name != "":
            self.parent().label_ops.addClass(name)
        elif ok and name == "":
            QMessageBox.warning(self.parent(), "提示", "种类名称不能为空, 新增种类失败")

    def edit_initArgsAction(self):
        """参数初始化"""
        self.parent().cfgs_widget.initArgs()

    def edit_loadModelAction(self):
        file = getOpenFileName(self.parent(), "训练模型文件", filter="data file(*.yaml | *.pt)")
        if file[0] != "":
            self.parent().cfgs_widget.setValue("model", file[0])

    def tool_vocToYolo(self):
        vy = VocToYolo(self.parent())
        vy.show()

    def tool_cocoToYolo(self):
        cy = CocoToYolo(self.parent())
        cy.show()

    def tool_pngToTolo(self):
        py = PngToYolo(self.parent())
        py.show()






