from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from APP.Designer.DesignerPy import startUI

from pathlib import Path
import glob

from ultralytics.utils.files import increment_path

from APP import getExistDirectory, checkProject, APP_SETTINGS, PROJ_SETTINGS, EXPERIMENT_SETTINGS


class Start(QWidget, startUI.Ui_Form):
    Start_Signal = Signal(str)
    def __init__(self, parent=None, f=Qt.Dialog):
        super().__init__(parent,f)
        self.setupUi(self)
        self.start = False
        projects= APP_SETTINGS["projects"]
        if projects != []:
            self.Projs_lw.addItems(projects)
        self.eventConnect()

    def eventConnect(self):
        self.Projs_lw.doubleClicked.connect(self.projsLwDoubleClicked)
        self.Create_new_pro_pb.clicked.connect(lambda :self.addNewProject(self.New_pro_dir_le.text() + "\\" + self.New_pro_name_le.text()))
        self.Add_exist_project_pb.clicked.connect(lambda :self.addOldProject(self.Exist_pro_path_le.text()))
        self.Browse_new_project_dir_pb.clicked.connect(self.browseNewDir)
        self.Browse_exist_project_dir_pb.clicked.connect(self.browseExistDir)

    def browseNewDir(self):
        project_p = getExistDirectory(self,"新项目存储文件夹")
        if project_p != "":
            self.New_pro_dir_le.setText(project_p)

    def browseExistDir(self):
        project_p = getExistDirectory(self, "项目文件夹")
        if project_p != "":
            self.Exist_pro_path_le.setText(project_p)

    def projsLwDoubleClicked(self, selected_r):
        item = self.Projs_lw.item(selected_r.row())
        proj =item.text()
        if not checkProject(proj):
            ans = QMessageBox.information(self, "提示", f"{proj}不是一个完整的项目，请问是否将其从列表中删移除？",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ans is QMessageBox.Yes:
                self.delectProject(item)
            return
        self.openProject(proj)

    def addItem(self, project):
        items = [Path(self.Projs_lw.item(i).text()) for i in range(self.Projs_lw.count())]
        items = [str(p) for p in items if p.exists()]
        project = str(Path(project))
        if project in items:
            return
        else:
            self.Projs_lw.addItem(project)


    def addNewProject(self, project):
        self.createProject(project)
        self.addItem(project)
        self.openProject(project)

    def addOldProject(self, project):
        if not checkProject(project):
            QMessageBox.warning(self,"警告", f"{project}不是一个完整的项目，无法添加")
            return
        self.createProject(project)
        self.addItem(project)
        self.openProject(project)

    def createProject(self, project):
        project_path = Path(project)
        if not project_path.exists():
            project_path.mkdir()
        APP_SETTINGS.updateProject(str(project_path))  # 添加新项目至系统列表
        PROJ_SETTINGS.load(str(project_path))

    def openProject(self, project):
        project = str(Path(project))
        APP_SETTINGS.updateProject(project)
        PROJ_SETTINGS.load(project)
        experiment = PROJ_SETTINGS["current_experiment"]
        exp_cache_p = f"{project}\\experiments\\expcache"
        experiments = [str(Path(f)) for f in glob.glob(f"{project}\\experiments\\**", recursive=False)]
        cache_experiments = [str(Path(f)) for f in glob.glob(f"{project}\\experiments\\expcache\\**")]
        if str(Path(experiment)) not in experiments and str(experiment) not in cache_experiments:  # current实验不存在， 新建默认实验
            experiment = increment_path(f"{exp_cache_p}\\untitled", mkdir=True)
        no_label_path = Path(project) / "data" / "no_label"
        if not no_label_path.exists():
            no_label_path.mkdir(parents=True, exist_ok=True)
        self.start = True
        self.Start_Signal.emit(str(experiment))  # 打开项目
        self.close()

    def delectProject(self, item):
        """从列表中移除项目"""
        r  = self.Projs_lw.row(item)
        self.Projs_lw.takeItem(r)
        APP_SETTINGS["projects"].pop(APP_SETTINGS["projects"].index(item.text()))

    def showEvent(self, event:QShowEvent) -> None:
        self.start = False

    def closeEvent(self, event:QCloseEvent) -> None:
        if not self.start:
            self.parent().close()




