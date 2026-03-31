import ast
import csv
import functools
import json
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from ultralytics import YOLO

# 处理打包环境
if getattr(sys, 'frozen', False):
    # 打包环境
    base_dir = Path(sys._MEIPASS)
else:
    # 开发环境
    base_dir = Path(__file__).parent

from theme_icons import ThemeIcons


class BatchDetectionThread(QThread):
    """批量检测线程"""
    result_ready = Signal(str, object, object, float,
                          object, list)  # 文件路径, 原图, 结果图, 耗时, 检测结果, 类别名称
    progress_updated = Signal(int)
    current_file_changed = Signal(str)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    finished = Signal()

    def __init__(self, model, folder_path, confidence_threshold=0.25, supported_formats=None):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.confidence_threshold = confidence_threshold
        self.supported_formats = supported_formats or [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.tif']
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0

    def run(self):
        self.is_running = True

        try:
            # 收集所有支持的图片文件
            image_files = []
            for fmt in self.supported_formats:
                image_files.extend(Path(self.folder_path).rglob(f'*{fmt}'))
                # image_files.extend(Path(self.folder_path).rglob(f'*{fmt.upper()}'))

            total_files = len(image_files)
            if total_files == 0:
                self.status_changed.emit("文件夹中没有找到支持的图片格式")
                self.finished.emit()
                return

            self.status_changed.emit(f"开始批量处理 {total_files} 个文件...")

            # 获取类别名称
            class_names = list(self.model.names.values())

            for i, img_path in enumerate(image_files):
                if not self.is_running:
                    break

                self.current_file_changed.emit(str(img_path))

                try:
                    # 处理单个图片
                    start_time = time.time()
                    results = self.model(
                        str(img_path), conf=self.confidence_threshold, verbose=False)
                    end_time = time.time()

                    # 获取原图
                    original_img = cv2.imread(str(img_path))
                    if original_img is not None:
                        original_img = cv2.cvtColor(
                            original_img, cv2.COLOR_BGR2RGB)

                        # 获取结果图
                        result_img = results[0].plot()
                        result_img = cv2.cvtColor(
                            result_img, cv2.COLOR_BGR2RGB)

                        self.result_ready.emit(str(img_path), original_img, result_img,
                                               end_time - start_time, results, class_names)
                        self.processed_count += 1

                except Exception as e:
                    self.error_occurred.emit(
                        f"处理文件 {img_path.name} 时发生错误: {str(e)}")
                    self.error_count += 1

                # 更新进度
                progress = int(((i + 1) / total_files) * 100)
                self.progress_updated.emit(progress)

                # 状态更新
                if (i + 1) % 10 == 0 or i == total_files - 1:
                    self.status_changed.emit(
                        f"处理进度: {i + 1}/{total_files} (成功: {self.processed_count}, 错误: {self.error_count})")

        except Exception as e:
            self.error_occurred.emit(f"批量处理发生错误: {str(e)}")
        finally:
            self.is_running = False
            # self.finished.emit()

    def stop(self):
        """停止批量检测"""
        self.is_running = False


class MultiCameraMonitorThread(QThread):
    camera_result_ready = Signal(int, object, object, float, object, list)
    camera_error = Signal(int, str)
    camera_status = Signal(int, str)
    finished = Signal()

    def __init__(self, model, camera_ids, conf=0.25, fps=10):
        super().__init__()
        self.model = model
        self.cam_ids = camera_ids
        self.conf = conf
        self.period = 1.0 / fps  # 帧间隔
        self.caps = {}  # {id: cv2.VideoCapture}
        self.active = {}  # {id: bool} 是否在线
        self.last_t = {}  # {id: float}

        # 线程同步
        self._run_flag = True
        self._pause_cond = QWaitCondition()
        self._pause_mutex = QMutex()
        self._paused_flag = False

    # ----------------- 生命周期 -----------------
    def run(self):
        self._open_all()
        if not self.caps:
            self.finished.emit()
            return

        cls_names = list(self.model.names.values())

        while self._run_flag:
            self._pause_mutex.lock()
            if self._paused_flag:
                self._pause_cond.wait(self._pause_mutex)
            self._pause_mutex.unlock()

            for cid in list(self.caps.keys()):
                if not self._run_flag:
                    break
                if not self._grab_and_infer(cid, cls_names):
                    self._reconnect_later(cid)  # 断线后异步重连
            self.msleep(10)

        self._close_all()
        self.finished.emit()

    def stop(self):
        self._run_flag = False
        self.resume()  # 确保等待线程被唤醒
        self.wait()

    def pause(self):
        self._pause_mutex.lock()
        self._paused_flag = True
        self._pause_mutex.unlock()

    def resume(self):
        self._pause_mutex.lock()
        self._paused_flag = False
        self._pause_mutex.unlock()
        self._pause_cond.wakeAll()

    # ----------------- 私有工具 -----------------
    def _open_all(self):
        for cid in self.cam_ids:
            cap = cv2.VideoCapture(cid, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                self.caps[cid] = cap
                self.active[cid] = True
                self.last_t[cid] = 0.0
                self.camera_status.emit(cid, "已连接")
            else:
                self.camera_error.emit(cid, "无法打开")
                cap.release()

    def _close_all(self):
        for cap in self.caps.values():
            cap.release()
        self.caps.clear()

    def _grab_and_infer(self, cid, cls_names):
        cap = self.caps.get(cid)
        if not cap or not cap.isOpened():
            return False

        # 读帧非阻塞：先 grab 再 retrieve
        if not cap.grab():
            return False

        now = time.time()
        if now - self.last_t[cid] < self.period:
            return True  # 未超时，但帧已 grab，避免堆积
        self.last_t[cid] = now

        ret, frame = cap.retrieve()
        if not ret:
            return False

        try:
            t0 = time.time()
            results = self.model(frame, conf=self.conf, verbose=False)
            infer_ms = (time.time() - t0) * 1000
            out_img = results[0].plot()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            self.camera_result_ready.emit(cid, rgb_frame, rgb_out,
                                          infer_ms / 1000.0, results, cls_names)
            return True
        except Exception as e:
            self.camera_error.emit(cid, f"推理异常: {e}")
            return False

    def _reconnect_later(self, cid):
        # 简单策略：5 秒后重试
        if self.active.get(cid) is False:
            return
        self.active[cid] = False
        self.camera_status.emit(cid, "正在重连中…")
        threading.Timer(5.0, lambda: self._try_reopen(cid)).start()

    def _try_reopen(self, cid):
        if cid in self.caps:
            self.caps[cid].release()
        cap = cv2.VideoCapture(cid)
        if cap.isOpened():
            self.caps[cid] = cap
            self.active[cid] = True
            self.camera_status.emit(cid, "已重连")
        else:
            cap.release()
            self._reconnect_later(cid)


class ModelSelectionDialog(QDialog):
    """模型选择对话框"""

    # 常量定义
    LOCAL_TAB_INDEX = 0
    NETWORK_TAB_INDEX = 1
    OFFICIAL_NETWORK_TAB_INDEX = 2
    MODEL_NAME_COL = 0
    SIZE_COL = 1
    MODIFIED_COL = 2
    PATH_COL = 3
    STATUS_COL = 4
    ACTION_COL = 5

    COLUMN_HEADERS_LOCAL = ["模型名称", "大小", "修改时间", "路径"]
    COLUMN_HEADERS_NETWORK = ["模型名称", "大小(MB)", "修改时间", "类别数量", "状态", "操作"]
    COLUMN_HEADERS_OFFICIAL_NETWORK = [
        "模型名称", "大小(MB)", "修改时间", "类别数量", "状态", "操作"]

    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.selected_model = None
        self.network_models = []
        self.official_network_models = []
        self.init_ui()
        self.load_network_models()
        self.load_official_network_models()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("高级模型选择")
        self.setModal(True)
        self.resize(900, 700)

        layout = QVBoxLayout(self)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.tabBar().setObjectName("dialogTabBar")
        layout.addWidget(self.tab_widget)

        # 本地模型标签页
        self.local_tab = QWidget()
        self.setup_local_tab()
        self.tab_widget.addTab(
            self.local_tab,
            ThemeIcons.icon("folder_open", 17, "#6366f1"),
            "本地资源模型",
        )

        # 私有网络模型标签页
        self.network_tab = QWidget()
        self.setup_network_tab()
        self.tab_widget.addTab(
            self.network_tab,
            ThemeIcons.icon("globe", 17, "#6366f1"),
            "私有网络资源模型",
        )

        # 官方网络模型标签页
        self.official_network_tab = QWidget()
        self.setup_official_network_tab()
        self.tab_widget.addTab(
            self.official_network_tab,
            ThemeIcons.icon("building", 17, "#6366f1"),
            "官方网络资源模型",
        )

        # 帮助标签页
        self.help_tab = QWidget()
        self.setup_help_tab()
        self.tab_widget.addTab(
            self.help_tab,
            ThemeIcons.icon("help", 17, "#6366f1"),
            "使用帮助",
        )

        # 按钮区域
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setStyleSheet(StyleManager.get_main_stylesheet(1.0))

    def setup_local_tab(self):
        """设置本地模型标签页"""
        layout = QVBoxLayout(self.local_tab)

        # 路径选择组
        path_group = QGroupBox("自定义模型路径")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("输入自定义模型目录路径...")
        path_layout.addWidget(self.path_edit)

        browse_btn = QPushButton("浏览")
        browse_btn.setIcon(ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_btn.setIconSize(QSize(18, 18))
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setIcon(ThemeIcons.icon("refresh", 18, "#ffffff"))
        refresh_btn.setIconSize(QSize(18, 18))
        refresh_btn.clicked.connect(self.refresh_models)
        path_layout.addWidget(refresh_btn)

        layout.addWidget(path_group)

        # 模型列表组
        models_group = QGroupBox("可用模型")
        models_layout = QVBoxLayout(models_group)

        self.model_table = self._create_table(self.COLUMN_HEADERS_LOCAL, 4)
        self.model_table.doubleClicked.connect(self.accept)
        self.model_table.setMinimumHeight(450)
        models_layout.addWidget(self.model_table)

        layout.addWidget(models_group)
        self.refresh_models()

    def setup_network_tab(self):
        """设置网络模型标签页"""
        layout = QVBoxLayout(self.network_tab)

        # 下载路径组
        path_group = QGroupBox("路径设置")
        path_layout = QHBoxLayout(path_group)

        self.download_path_edit = QLineEdit()
        self.download_path_edit.setText(
            str((base_dir / "pt_models").absolute()))
        self.download_path_edit.setPlaceholderText("模型下载目录路径...")
        path_layout.addWidget(self.download_path_edit)

        browse_download_btn = QPushButton("浏览")
        browse_download_btn.setIcon(
            ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_download_btn.setIconSize(QSize(18, 18))
        browse_download_btn.clicked.connect(self.browse_download_path)
        path_layout.addWidget(browse_download_btn)

        layout.addWidget(path_group)

        # 网络模型组
        models_group = QGroupBox("网络模型资源")
        models_layout = QVBoxLayout(models_group)

        self.network_table = self._create_table(self.COLUMN_HEADERS_NETWORK, 6)
        self.network_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.network_table.customContextMenuRequested.connect(
            self.show_network_context_menu)
        self.network_table.doubleClicked.connect(self.show_network_model_info)
        self.network_table.setMinimumHeight(450)
        models_layout.addWidget(self.network_table)

        layout.addWidget(models_group)

    def setup_official_network_tab(self):
        """设置官方网络模型标签页"""
        layout = QVBoxLayout(self.official_network_tab)

        # 下载路径组
        path_group = QGroupBox("路径设置")
        path_layout = QHBoxLayout(path_group)

        self.official_download_path_edit = QLineEdit()
        self.official_download_path_edit.setText(
            str((base_dir / "YOLO_pt").absolute()))
        self.official_download_path_edit.setPlaceholderText("官方模型下载目录路径...")
        path_layout.addWidget(self.official_download_path_edit)

        browse_official_btn = QPushButton("浏览")
        browse_official_btn.setIcon(
            ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_official_btn.setIconSize(QSize(18, 18))
        browse_official_btn.clicked.connect(self.browse_official_download_path)
        path_layout.addWidget(browse_official_btn)

        layout.addWidget(path_group)

        # 官方网络模型组
        models_group = QGroupBox("官方YOLO模型资源")
        models_layout = QVBoxLayout(models_group)

        self.official_network_table = self._create_table(
            self.COLUMN_HEADERS_OFFICIAL_NETWORK, 6)
        self.official_network_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.official_network_table.customContextMenuRequested.connect(
            self.show_official_network_context_menu)
        self.official_network_table.doubleClicked.connect(
            self.show_official_network_model_info)
        self.official_network_table.setMinimumHeight(450)
        models_layout.addWidget(self.official_network_table)

        layout.addWidget(models_group)

    def setup_help_tab(self):
        """设置帮助标签页"""
        layout = QVBoxLayout(self.help_tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel("模型加载使用指南")
        title_label.setStyleSheet("""
            font-size: 20px;
            font-weight: 800;
            color: #312e81;
            padding: 10px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)

        # 1. 本地资源模型
        local_group = QGroupBox("💻 本地资源模型")
        local_layout = QVBoxLayout(local_group)
        local_text = QTextEdit()
        local_text.setReadOnly(True)
        local_text.setHtml("""
        <h3 style="color: #3498db;">什么是本地资源模型？</h3>
        <p>本地资源模型是指您已经下载并存储在计算机上的YOLO模型文件（.pt格式）。</p>
        
        <h3 style="color: #3498db;">如何使用？</h3>
        <ol>
            <li><b>自动扫描：</b>程序启动时会自动扫描默认目录下的模型文件</li>
            <li><b>自定义路径：</b>在"自定义模型路径"输入框中输入您的模型所在目录，点击"浏览"按钮选择文件夹</li>
            <li><b>刷新列表：</b>点击"🔄 刷新"按钮更新模型列表</li>
            <li><b>选择模型：</b>在表格中单击选中您需要的模型</li>
            <li><b>确认选择：</b>点击"确定"按钮加载选中的模型</li>
        </ol>
        
        <h3 style="color: #3498db;">支持的模型格式</h3>
        <ul>
            <li>YOLOv8 模型 (.pt文件)</li>
            <li>YOLOv11 模型 (.pt文件)</li>
            <li>YOLOv26 模型 (.pt文件)</li>
        </ul>
        
        <h3 style="color: #e74c3c;">注意事项</h3>
        <ul>
            <li>确保模型文件完整且未损坏</li>
            <li>模型文件需要有相应的读取权限</li>
            <li>建议使用官方或可信来源的模型文件</li>
        </ul>
        """)
        local_text.setMaximumHeight(600)
        local_text.setMinimumHeight(350)
        local_layout.addWidget(local_text)
        scroll_layout.addWidget(local_group)

        # 2. 私有网络资源模型
        network_group = QGroupBox("🌐 私有网络资源模型")
        network_layout = QVBoxLayout(network_group)
        network_text = QTextEdit()
        network_text.setReadOnly(True)
        network_text.setHtml("""
        <h3 style="color: #9b59b6;">什么是私有网络资源模型？</h3>
        <p>私有网络资源模型是指存储在私有服务器或GitHub Releases上的模型文件，
        您可以直接从网络下载使用，无需预先保存在本地。</p>
        
        <h3 style="color: #9b59b6;">如何使用？</h3>
        <ol>
            <li><b>设置下载路径：</b>在"📥 路径设置"中指定模型下载保存的位置（默认：pt_models文件夹）</li>
            <li><b>浏览模型列表：</b>表格中显示了所有可用的网络模型资源，包括：
                <ul>
                    <li>模型名称和版本</li>
                    <li>文件大小</li>
                    <li>更新日期</li>
                    <li>支持的类别数量</li>
                    <li>当前下载状态</li>
                </ul>
            </li>
            <li><b>下载模型：</b>
                <ul>
                    <li>点击"📥 下载"按钮下载选中的模型</li>
                    <li>或右键点击模型行选择"下载模型"</li>
                    <li>下载过程中状态会显示"下载中..."</li>
                </ul>
            </li>
            <li><b>复制下载链接：</b>点击"🔗 复制"按钮可复制模型的下载链接，用于其他下载工具</li>
            <li><b>查看详情：</b>双击模型行可查看详细信息，包括所有支持的检测类别</li>
            <li><b>选择模型：</b>下载完成后，选中模型并点击"确定"按钮加载</li>
        </ol>
        
        <h3 style="color: #9b59b6;">下载状态说明</h3>
        <ul>
            <li><span style="color: #e74c3c;">未下载</span> - 模型尚未下载到本地</li>
            <li><span style="color: #f39c12;">下载中...</span> - 正在从网络下载模型</li>
            <li><span style="color: #27ae60;">已下载</span> - 模型已存在于本地，可以直接使用</li>
        </ul>
        
        <h3 style="color: #e74c3c;">注意事项</h3>
        <ul>
            <li>下载模型需要网络连接</li>
            <li>模型文件较大，请确保有足够的磁盘空间</li>
            <li>下载时间取决于网络速度和文件大小</li>
            <li>如果下载失败，可以复制链接使用其他下载工具</li>
        </ul>
        """)
        network_text.setMaximumHeight(600)
        network_text.setMinimumHeight(450)
        network_layout.addWidget(network_text)
        scroll_layout.addWidget(network_group)

        # 3. 官方网络资源模型
        official_group = QGroupBox("🏢 官方网络资源模型")
        official_layout = QVBoxLayout(official_group)
        official_text = QTextEdit()
        official_text.setReadOnly(True)
        official_text.setHtml("""
        <h3 style="color: #27ae60;">什么是官方网络资源模型？</h3>
        <p>官方网络资源模型是指由Ultralytics官方发布的YOLO模型，
        包括YOLOv11系列（n/s/m/l/x）和YOLOv26系列（n/s/m/l）等多种规格，
        以及支持分割任务的seg版本。</p>
        
        <h3 style="color: #27ae60;">模型规格说明</h3>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr style="background-color: #ecf0f1;">
                <th>规格</th>
                <th>说明</th>
                <th>适用场景</th>
            </tr>
            <tr>
                <td><b>n (nano)</b></td>
                <td>最轻量级，约5-6MB</td>
                <td>边缘设备、移动端、实时性要求高的场景</td>
            </tr>
            <tr>
                <td><b>s (small)</b></td>
                <td>轻量级，约18-20MB</td>
                <td>平衡速度和精度</td>
            </tr>
            <tr>
                <td><b>m (medium)</b></td>
                <td>中等，约38-43MB</td>
                <td>一般桌面应用</td>
            </tr>
            <tr>
                <td><b>l (large)</b></td>
                <td>大型，约49-54MB</td>
                <td>精度优先的场景</td>
            </tr>
            <tr>
                <td><b>x (xlarge)</b></td>
                <td>超大型，约109-119MB</td>
                <td>最高精度，计算资源充足</td>
            </tr>
        </table>
        
        <h3 style="color: #27ae60;">如何使用？</h3>
        <ol>
            <li><b>设置下载路径：</b>在"📥 路径设置"中指定模型下载保存的位置（默认：YOLO_pt文件夹）</li>
            <li><b>浏览模型列表：</b>表格中显示了所有官方可用的YOLO模型</li>
            <li><b>下载模型：</b>
                <ul>
                    <li>点击"📥 下载"按钮从Ultralytics官方仓库下载</li>
                    <li>支持断点续传，如果下载中断可以重新下载</li>
                </ul>
            </li>
            <li><b>复制下载链接：</b>点击"🔗 复制"按钮可复制官方下载链接</li>
            <li><b>查看详情：</b>双击模型行可查看模型支持的80个COCO数据集类别</li>
            <li><b>选择模型：</b>下载完成后，选中模型并点击"确定"按钮加载</li>
        </ol>
        
        <h3 style="color: #27ae60;">模型类别信息</h3>
        <p>所有官方模型都基于COCO数据集训练，支持检测以下80个类别：</p>
        <p style="font-size: 12px; color: #7f8c8d;">
        人、自行车、汽车、摩托车、飞机、公交车、火车、卡车、船、红绿灯、消防栓、
        停车标志、停车计时器、长椅、鸟、猫、狗、马、羊、牛、大象、熊、斑马、长颈鹿、
        背包、雨伞、手提包、领带、行李箱、飞盘、滑雪板、 snowboard、运动球、风筝、
        棒球棒、棒球手套、滑板、冲浪板、网球拍、瓶子、酒杯、杯子、叉子、刀、勺子、
        碗、香蕉、苹果、三明治、橙子、西兰花、胡萝卜、热狗、披萨、甜甜圈、蛋糕、
        椅子、沙发、盆栽植物、床、餐桌、马桶、电视、笔记本电脑、鼠标、遥控器、
        键盘、手机、微波炉、烤箱、烤面包机、水槽、冰箱、书、时钟、花瓶、剪刀、
        泰迪熊、吹风机、牙刷
        </p>
        
        <h3 style="color: #e74c3c;">注意事项</h3>
        <ul>
            <li>官方模型需要从GitHub下载，请确保网络可以访问github.com</li>
            <li>较大的模型（x版本）下载时间较长，请耐心等待</li>
            <li>seg版本模型支持实例分割任务，文件比普通版本稍大</li>
            <li>建议根据您的硬件配置选择合适的模型规格</li>
            <li>首次下载后，模型会缓存在本地，下次使用无需重新下载</li>
        </ul>
        """)
        official_text.setMaximumHeight(600)
        official_text.setMinimumHeight(450)
        official_layout.addWidget(official_text)
        scroll_layout.addWidget(official_group)

        # 通用提示
        tips_group = QGroupBox("💡 通用提示")
        tips_layout = QVBoxLayout(tips_group)
        tips_text = QTextEdit()
        tips_text.setReadOnly(True)
        tips_text.setHtml("""
        <h3 style="color: #f39c12;">快速开始建议</h3>
        <ol>
            <li><b>新手推荐：</b>如果您是第一次使用，建议从"官方网络资源模型"中下载YOLOv11n.pt（最小最快）</li>
            <li><b>精度优先：</b>如果需要更高的检测精度，可以选择YOLOv11x.pt或YOLOv26l.pt</li>
            <li><b>已有模型：</b>如果您已经有下载好的模型文件，使用"本地资源模型"直接加载</li>
            <li><b>网络不好：</b>如果网络下载慢，可以复制链接使用下载工具，然后放到对应目录使用本地加载</li>
        </ol>
        
        <h3 style="color: #f39c12;">常见问题</h3>
        <ul>
            <li><b>Q: 下载失败怎么办？</b><br>
            A: 检查网络连接，或复制链接使用其他下载工具手动下载后放到对应目录。</li>
            <li><b>Q: 模型加载失败？</b><br>
            A: 确保模型文件完整，尝试重新下载或使用其他模型。</li>
            <li><b>Q: 如何选择模型规格？</b><br>
            A: 根据您的硬件配置选择，配置低选n/s，配置高选l/x。</li>
            <li><b>Q: seg版本和普通版本有什么区别？</b><br>
            A: seg版本支持实例分割（像素级物体轮廓），普通版本只支持目标检测（矩形框）。</li>
        </ul>
        """)
        tips_text.setMaximumHeight(280)
        tips_text.setMinimumHeight(200)
        tips_layout.addWidget(tips_text)
        scroll_layout.addWidget(tips_group)

        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

    def _create_table(self, headers, column_count):
        """创建标准表格控件"""
        table = QTableWidget()
        table.setColumnCount(column_count)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        return table

    def browse_path(self):
        """浏览自定义路径"""
        path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if path:
            self.path_edit.setText(path)
            self.refresh_models()

    def browse_download_path(self):
        """浏览下载路径"""
        path = QFileDialog.getExistingDirectory(self, "选择下载目录")
        if path:
            self.download_path_edit.setText(path)

    def browse_official_download_path(self):
        """浏览官方模型下载路径"""
        path = QFileDialog.getExistingDirectory(self, "选择官方模型下载目录")
        if path:
            self.official_download_path_edit.setText(path)

    def refresh_models(self):
        """刷新本地模型列表"""
        try:
            custom_path = self.path_edit.text() or None
            models = self.model_manager.scan_models(custom_path)

            self.model_table.setRowCount(len(models))
            for row, model in enumerate(models):
                self.model_table.setItem(
                    row, self.MODEL_NAME_COL, QTableWidgetItem(model['name']))
                self.model_table.setItem(
                    row, self.SIZE_COL, QTableWidgetItem(model['size']))
                self.model_table.setItem(
                    row, self.MODIFIED_COL, QTableWidgetItem(model['modified']))
                self.model_table.setItem(
                    row, self.PATH_COL, QTableWidgetItem(model['path']))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")

    def load_network_models(self):
        """加载网络模型数据"""
        try:
            csv_path = base_dir / "csv_reports" / "pt_files_report.csv"
            if not csv_path.exists():
                QMessageBox.warning(self, "警告", f"未找到网络模型数据文件 {csv_path}")
                return

            models_data = self._read_csv_with_encodings(csv_path)
            if not models_data:
                QMessageBox.warning(self, "警告", "无法正确读取网络模型数据文件")
                return

            self.network_models = models_data
            self.refresh_network_models()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载网络模型数据失败: {str(e)}")

    def load_official_network_models(self):
        """加载官方网络模型数据"""
        try:
            csv_path = base_dir / "csv_reports" / "YOLO_pt_files_report.csv"
            if not csv_path.exists():
                QMessageBox.warning(self, "警告", f"未找到官方网络模型数据文件 {csv_path}")
                return

            models_data = self._read_csv_with_encodings(csv_path)
            if not models_data:
                QMessageBox.warning(self, "警告", "无法正确读取官方网络模型数据文件")
                return

            self.official_network_models = models_data
            self.refresh_official_network_models()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载官方网络模型数据失败: {str(e)}")

    def _read_csv_with_encodings(self, file_path):
        """尝试多种编码读取CSV文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                    if data and '文件名' in data[0]:
                        return data
            except UnicodeDecodeError:
                continue
        return []

    def refresh_network_models(self):
        """刷新网络模型列表"""
        self.network_table.setRowCount(len(self.network_models))

        for row, model in enumerate(self.network_models):
            # 基本信息列
            self.network_table.setItem(
                row, self.MODEL_NAME_COL, QTableWidgetItem(model['文件名']))
            self.network_table.setItem(
                row, self.SIZE_COL, QTableWidgetItem(f"{model['大小(MB)']} MB"))
            self.network_table.setItem(
                row, self.MODIFIED_COL, QTableWidgetItem(model['修改日期']))
            self.network_table.setItem(
                row, self.STATUS_COL - 1, QTableWidgetItem(model['类别数量']))  # 类别数量列

            # 状态列 - 根据本地文件存在情况判断
            download_path = Path(self.download_path_edit.text())
            local_path = download_path / model['文件名']
            is_downloaded = local_path.exists()

            status_text = "已下载" if is_downloaded else "未下载"
            status_color = QColor(
                "#27ae60") if is_downloaded else QColor("#e74c3c")

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(status_color)
            self.network_table.setItem(row, self.STATUS_COL, status_item)

            # 操作列
            self._create_operation_buttons(row, model)

    def refresh_official_network_models(self):
        """刷新官方网络模型列表"""
        self.official_network_table.setRowCount(
            len(self.official_network_models))

        for row, model in enumerate(self.official_network_models):
            # 基本信息列
            self.official_network_table.setItem(
                row, self.MODEL_NAME_COL, QTableWidgetItem(model['文件名']))
            self.official_network_table.setItem(
                row, self.SIZE_COL, QTableWidgetItem(f"{model['大小(MB)']} MB"))
            self.official_network_table.setItem(
                row, self.MODIFIED_COL, QTableWidgetItem(model['修改日期']))
            self.official_network_table.setItem(
                row, self.STATUS_COL - 1, QTableWidgetItem(model['类别数量']))  # 类别数量列

            # 状态列 - 根据本地文件存在情况判断
            download_path = Path(self.official_download_path_edit.text())
            local_path = download_path / model['文件名']
            is_downloaded = local_path.exists()

            status_text = "已下载" if is_downloaded else "未下载"
            status_color = QColor(
                "#27ae60") if is_downloaded else QColor("#e74c3c")

            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(status_color)
            self.official_network_table.setItem(
                row, self.STATUS_COL, status_item)

            # 操作列
            self._create_official_operation_buttons(row, model)

    def _create_operation_buttons(self, row, model):
        """创建操作按钮"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # 下载按钮
        download_btn = QPushButton("下载")
        download_btn.setIcon(ThemeIcons.icon("download", 14, "#ffffff"))
        download_btn.setIconSize(QSize(14, 14))
        download_btn.setFixedSize(76, 30)
        # 使用partial解决闭包问题
        download_btn.clicked.connect(
            functools.partial(self.download_network_model, row))

        # 复制链接按钮
        copy_btn = QPushButton("复制")
        copy_btn.setIcon(ThemeIcons.icon("link", 14, "#ffffff"))
        copy_btn.setIconSize(QSize(14, 14))
        copy_btn.setFixedSize(76, 30)
        # 使用partial解决闭包问题
        copy_btn.clicked.connect(functools.partial(
            self.copy_download_link, model))

        # 检查是否已下载
        download_path = Path(self.download_path_edit.text())

        local_path = download_path / model['文件名']
        is_downloaded = local_path.exists()

        if is_downloaded:
            download_btn.setText("已下载")
            download_btn.setIcon(QIcon())
            download_btn.setEnabled(False)

        layout.addWidget(download_btn)
        layout.addWidget(copy_btn)
        layout.addStretch()

        self.network_table.setCellWidget(row, self.ACTION_COL, widget)

    def _create_official_operation_buttons(self, row, model):
        """创建官方模型操作按钮"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        # 下载按钮
        download_btn = QPushButton("下载")
        download_btn.setIcon(ThemeIcons.icon("download", 14, "#ffffff"))
        download_btn.setIconSize(QSize(14, 14))
        download_btn.setFixedSize(76, 30)
        # 使用partial解决闭包问题
        download_btn.clicked.connect(functools.partial(
            self.download_official_network_model, row))

        # 复制链接按钮
        copy_btn = QPushButton("复制")
        copy_btn.setIcon(ThemeIcons.icon("link", 14, "#ffffff"))
        copy_btn.setIconSize(QSize(14, 14))
        copy_btn.setFixedSize(76, 30)
        # 使用partial解决闭包问题
        copy_btn.clicked.connect(functools.partial(
            self.copy_official_download_link, model))

        # 检查是否已下载
        download_path = Path(self.official_download_path_edit.text())
        local_path = download_path / model['文件名']
        is_downloaded = local_path.exists()

        if is_downloaded:
            download_btn.setText("已下载")
            download_btn.setIcon(QIcon())
            download_btn.setEnabled(False)

        layout.addWidget(download_btn)
        layout.addWidget(copy_btn)
        layout.addStretch()

        self.official_network_table.setCellWidget(row, self.ACTION_COL, widget)

    def show_network_model_info(self):
        """显示网络模型详细信息"""
        row = self.network_table.currentRow()
        if row < 0 or row >= len(self.network_models):
            return

        model = self.network_models[row]
        try:
            class_info = ast.literal_eval(model['类别信息'])
            class_text = "\n".join(
                [f"{k}: {v}" for k, v in class_info.items()])
        except:
            class_text = model['类别信息']

        info = (
            f"模型名称: {model['文件名']}\n"
            f"大小: {model['大小(MB)']} MB\n"
            f"修改时间: {model['修改日期']}\n"
            f"类别数量: {model['类别数量']}\n\n"
            f"类别信息:\n{class_text}"
        )
        # 使用自定义对话框替代 QMessageBox，以支持最大高度与滚动显示
        dlg = QDialog(self)
        dlg.setWindowTitle("模型详细信息")
        dlg_layout = QVBoxLayout(dlg)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(info)
        text_edit.setMinimumWidth(480)
        text_edit.setMaximumHeight(400)  # 最大高度，超过则显示滚动条
        dlg_layout.addWidget(text_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        dlg_layout.addWidget(btn_box)

        dlg.exec()

    def show_official_network_model_info(self):
        """显示官方网络模型详细信息"""
        row = self.official_network_table.currentRow()
        if row < 0 or row >= len(self.official_network_models):
            return

        model = self.official_network_models[row]
        try:
            class_info = ast.literal_eval(model['类别信息'])
            class_text = "\n".join(
                [f"{k}: {v}" for k, v in class_info.items()])
        except:
            class_text = model['类别信息']

        info = (
            f"模型名称: {model['文件名']}\n"
            f"大小: {model['大小(MB)']} MB\n"
            f"修改时间: {model['修改日期']}\n"
            f"类别数量: {model['类别数量']}\n\n"
            f"类别信息:\n{class_text}"
        )
        # 使用自定义对话框替代 QMessageBox，以支持最大高度与滚动显示
        dlg = QDialog(self)
        dlg.setWindowTitle("官方模型详细信息")
        dlg_layout = QVBoxLayout(dlg)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(info)
        text_edit.setMinimumWidth(480)
        text_edit.setMaximumHeight(400)  # 最大高度，超过则显示滚动条
        dlg_layout.addWidget(text_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(dlg.accept)
        dlg_layout.addWidget(btn_box)

        dlg.exec()

    def show_network_context_menu(self, pos):
        """显示网络模型右键菜单"""
        row = self.network_table.currentRow()
        if row < 0:
            return

        menu = QMenu(self)
        download_action = menu.addAction("📥 下载模型")
        download_action.triggered.connect(
            lambda: self.download_network_model(row))
        menu.exec(self.network_table.viewport().mapToGlobal(pos))

    def show_official_network_context_menu(self, pos):
        """显示官方网络模型右键菜单"""
        row = self.official_network_table.currentRow()
        if row < 0:
            return

        menu = QMenu(self)
        download_action = menu.addAction("📥 下载模型")
        download_action.triggered.connect(
            lambda: self.download_official_network_model(row))
        menu.exec(self.official_network_table.viewport().mapToGlobal(pos))

    def download_network_model(self, row):
        """下载网络模型"""
        if row >= len(self.network_models):
            return

        model = self.network_models[row]
        model_name = model['文件名']
        download_dir = Path(self.download_path_edit.text())

        try:
            # 准备下载目录
            download_dir.mkdir(parents=True, exist_ok=True)
            local_path = download_dir / model_name

            # 检查文件存在
            if local_path.exists():
                reply = QMessageBox.question(
                    self, "确认覆盖",
                    f"模型文件 {model_name} 已存在，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # 更新状态
            status_item = self.network_table.item(row, self.STATUS_COL)
            status_item.setText("下载中...")
            status_item.setForeground(QColor("#f39c12"))

            # 执行下载
            self._perform_download(model_name, local_path)

            # 更新状态
            status_item.setText("已下载")
            status_item.setForeground(QColor("#27ae60"))

            # 更新按钮状态
            widget = self.network_table.cellWidget(row, self.ACTION_COL)
            for btn in widget.findChildren(QPushButton):
                if "下载" in btn.text():
                    btn.setText("✅ 已下载")
                    btn.setEnabled(False)

            QMessageBox.information(
                self, "下载完成",
                f"模型 {model_name} 下载完成！\n保存路径: {local_path}"
            )

        except Exception as e:
            # 恢复状态
            status_item = self.network_table.item(row, self.STATUS_COL)
            status_item.setText("下载失败")
            status_item.setForeground(QColor("#e74c3c"))
            QMessageBox.critical(self, "下载失败", f"错误: {str(e)}")

    def download_official_network_model(self, row):
        """下载官方网络模型"""
        if row >= len(self.official_network_models):
            return

        model = self.official_network_models[row]
        model_name = model['文件名']
        download_dir = Path(self.official_download_path_edit.text())

        try:
            # 准备下载目录
            download_dir.mkdir(parents=True, exist_ok=True)
            local_path = download_dir / model_name

            # 检查文件存在
            if local_path.exists():
                reply = QMessageBox.question(
                    self, "确认覆盖",
                    f"模型文件 {model_name} 已存在，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # 更新状态
            status_item = self.official_network_table.item(
                row, self.STATUS_COL)
            status_item.setText("下载中...")
            status_item.setForeground(QColor("#f39c12"))

            # 执行下载
            self._perform_official_download(model_name, local_path)

            # 更新状态
            status_item.setText("已下载")
            status_item.setForeground(QColor("#27ae60"))

            # 更新按钮状态
            widget = self.official_network_table.cellWidget(
                row, self.ACTION_COL)
            for btn in widget.findChildren(QPushButton):
                if "下载" in btn.text():
                    btn.setText("✅ 已下载")
                    btn.setEnabled(False)

            QMessageBox.information(
                self, "下载完成",
                f"官方模型 {model_name} 下载完成！\n保存路径: {local_path}"
            )

        except Exception as e:
            # 恢复状态
            status_item = self.official_network_table.item(
                row, self.STATUS_COL)
            status_item.setText("下载失败")
            status_item.setForeground(QColor("#e74c3c"))
            QMessageBox.critical(self, "下载失败", f"错误: {str(e)}")

    def _perform_download(self, model_name, save_path):
        """执行实际的下载操作"""
        url = f"https://github.com/JingW-ui/PI-MAPP/releases/download/pt_download/{model_name}"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def _perform_official_download(self, model_name, save_path):
        """执行官方模型的实际下载操作"""
        # 官方YOLO模型从Ultralytics官方下载
        url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}"

        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def copy_download_link(self, model):
        """复制下载链接到剪贴板"""
        try:
            if not model or '文件名' not in model:
                raise ValueError("模型数据无效")

            url = f"https://github.com/JingW-ui/PI-MAPP/releases/download/pt_download/{model['文件名']}"
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "复制成功", "下载链接已复制到剪贴板")
        except Exception as e:
            QMessageBox.critical(self, "复制失败", f"错误: {str(e)}")

    def copy_official_download_link(self, model):
        """复制官方模型下载链接到剪贴板"""
        try:
            if not model or '文件名' not in model:
                raise ValueError("模型数据无效")

            url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model['文件名']}"
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "复制成功", "官方模型下载链接已复制到剪贴板")
        except Exception as e:
            QMessageBox.critical(self, "复制失败", f"错误: {str(e)}")

    def accept(self):
        """确认选择模型"""
        current_tab = self.tab_widget.currentIndex()

        if current_tab == self.LOCAL_TAB_INDEX:
            self._handle_local_selection()
        elif current_tab == self.NETWORK_TAB_INDEX:
            self._handle_network_selection()
        elif current_tab == self.OFFICIAL_NETWORK_TAB_INDEX:
            self._handle_official_network_selection()
        else:
            super().accept()

    def _handle_local_selection(self):
        """处理本地模型选择"""
        row = self.model_table.currentRow()
        if row >= 0:
            self.selected_model = self.model_table.item(
                row, self.PATH_COL).text()
            super().accept()

    def _handle_network_selection(self):
        """处理网络模型选择"""
        row = self.network_table.currentRow()
        if row < 0:
            return

        model = self.network_models[row]
        model_name = model['文件名']
        local_path = Path(self.download_path_edit.text()) / model_name

        if not local_path.exists():
            QMessageBox.warning(self, "警告", "请先下载选中的网络模型！")
            return

        self.selected_model = str(local_path)
        super().accept()

    def _handle_official_network_selection(self):
        """处理官方网络模型选择"""
        row = self.official_network_table.currentRow()
        if row < 0:
            return

        model = self.official_network_models[row]
        model_name = model['文件名']
        local_path = Path(self.official_download_path_edit.text()) / model_name

        if not local_path.exists():
            QMessageBox.warning(self, "警告", "请先下载选中的官方网络模型！")
            return

        self.selected_model = str(local_path)
        super().accept()


class DetectionResultWidget(QWidget):
    """检测结果显示组件"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sheet = QFrame()
        sheet.setObjectName("wfResultSheet")
        sl = QVBoxLayout(sheet)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(0)

        # 五列同表：序号/类别最窄 → 置信度中等 → 坐标较宽 → 尺寸最宽（比例随表格宽度变化）
        self.result_table = QTableWidget()
        self.result_table.setObjectName("wfResultTable")
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(
            ["序号", "类别", "置信度", "坐标", "尺寸"])
        _hdr = self.result_table.horizontalHeader()
        _hdr.setDefaultAlignment(Qt.AlignCenter)
        _hdr.setHighlightSections(False)
        for i in range(5):
            _hdr.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.result_table.setMinimumHeight(200)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_table.verticalScrollBar().rangeChanged.connect(
            lambda *_: QTimer.singleShot(0,
                                         self._sync_result_table_column_widths)
        )
        self.result_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        sl.addWidget(self.result_table, 1)
        layout.addWidget(sheet, 1)

        # 统计信息
        self.stats_label = QLabel("等待检测结果...")
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # 与多行统计文案高度对齐，避免单行/多行切换时底部区域上下跳
        self.stats_label.setMinimumHeight(72)
        self.stats_label.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #eef2ff, stop:1 #e0e7ff);
            border: 1px solid #c7d2fe;
            padding: 14px 16px;
            border-radius: 12px;
            font-size: 12px;
            color: #312e81;
            font-weight: 600;
        """)
        layout.addWidget(self.stats_label)
        layout.setSpacing(10)

    def _sync_result_table_column_widths(self):
        """与底部 Tab 同宽铺满；序号/类别/置信度收窄，坐标/尺寸略左移（比例上多占宽）。"""
        t = self.result_table
        vw = t.viewport().width()
        if vw < 80:
            return
        budget = max(260, vw - 2)
        fr = (0.035, 0.075, 0.09, 0.34, 0.46)
        mins = (32, 48, 56, 88, 96)
        widths = [max(mins[i], int(budget * fr[i])) for i in range(5)]
        s = sum(widths)
        if s > budget:
            scale = budget / s
            widths = [max(mins[i], int(widths[i] * scale)) for i in range(5)]
        diff = budget - sum(widths)
        if diff != 0:
            widths[-1] += diff
        for i, w in enumerate(widths):
            t.setColumnWidth(i, w)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_result_table_column_widths()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_result_table_column_widths)

    def update_results(self, results, class_names, inference_time):
        """更新检测结果"""
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            self.result_table.setRowCount(0)
            self.stats_label.setText("未检测到目标")
            return

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        # 更新表格
        self.result_table.setRowCount(len(confidences))

        class_counts = {}
        for i, (conf, cls, box) in enumerate(zip(confidences, classes, xyxy)):
            class_name = class_names[cls] if cls < len(
                class_names) else f"类别{cls}"

            # 统计类别数量
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            self.result_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.result_table.setItem(i, 1, QTableWidgetItem(class_name))

            # 置信度带颜色
            conf_item = QTableWidgetItem(f"{conf:.3f}")
            if conf > 0.8:
                conf_item.setBackground(QColor(16, 185, 129, 55))
            elif conf > 0.5:
                conf_item.setBackground(QColor(245, 158, 11, 55))
            else:
                conf_item.setBackground(QColor(248, 113, 113, 55))
            self.result_table.setItem(i, 2, conf_item)

            self.result_table.setItem(
                i, 3, QTableWidgetItem(f"({box[0]:.0f},{box[1]:.0f})"))
            self.result_table.setItem(i, 4, QTableWidgetItem(
                f"{box[2] - box[0]:.0f}×{box[3] - box[1]:.0f}"))

        self._sync_result_table_column_widths()

        # 更新统计信息
        total_objects = len(confidences)
        avg_confidence = np.mean(confidences)

        stats_text = f"检测到 {total_objects} 个目标 | "
        stats_text += f"平均置信度: {avg_confidence:.3f} | "
        stats_text += f"耗时: {inference_time:.3f} 秒\n"
        stats_text += "类别统计: " + \
            " | ".join([f"{name}: {count}" for name,
                       count in class_counts.items()])

        self.stats_label.setText(stats_text)


class MonitoringWidget(QWidget):
    """监控页面组件（使用主界面统一模型与运行控制）。"""

    def __init__(self, model_manager, camera_manager):
        super().__init__()
        self.model_manager = model_manager
        self.camera_manager = camera_manager
        self.monitoring_thread = None
        self.camera_labels = {}
        self.current_model = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        hint = QLabel("监控使用右侧“输入源”的摄像头设置；开始/暂停/停止使用顶部统一按钮。")
        hint.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        hint.setContentsMargins(8, 0, 8, 0)
        hint.setStyleSheet("color:#64748b; font-size:12px; padding:4px 6px;")
        layout.addWidget(hint, 0)

        # 监控显示区域
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setStyleSheet("""
            QScrollArea {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
            }
            QScrollArea > QWidget > QWidget {   /* viewport */
                background: transparent;
            }
            QScrollArea::corner {               /* 右下角空白三角 */
                background: transparent;
            }
        """)
        self.monitor_widget = QWidget()
        self.monitor_layout = QGridLayout(self.monitor_widget)
        self.monitor_scroll.setWidget(self.monitor_widget)
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_scroll.setMinimumHeight(220)

        layout.addWidget(self.monitor_scroll, 1)

    def set_shared_model(self, model):
        """由主界面注入当前已加载模型。"""
        self.current_model = model

    def start_monitoring(self, shared_model=None, camera_ids=None):
        """开始监控（优先使用主界面已加载模型）。"""
        model = shared_model if shared_model is not None else self.current_model
        if not model:
            QMessageBox.warning(self, "警告", "请先在右侧任务配置中加载模型")
            return

        if camera_ids is None:
            QMessageBox.warning(self, "警告", "请先在右侧输入源选择摄像头")
            return
        camera_ids = [
            cid for cid in camera_ids if cid is not None and cid >= 0]
        if not camera_ids:
            QMessageBox.warning(self, "警告", "请选择至少一个有效摄像头")
            return

        # 清空之前的显示
        self.clear_monitor_display()

        # 创建显示标签
        self.create_camera_labels(camera_ids)
        # 设置等高宽
        self.set_equal_column_stretch()
        # 启动监控线程
        self.monitoring_thread = MultiCameraMonitorThread(
            model, camera_ids)
        self.monitoring_thread.camera_result_ready.connect(
            self.update_camera_display)
        self.monitoring_thread.camera_error.connect(self.handle_camera_error)
        self.monitoring_thread.finished.connect(self.on_monitoring_finished)

        self.monitoring_thread.start()

    def stop_monitoring(self):
        """暂停/继续监控"""
        if self.monitoring_thread and self.monitoring_thread._run_flag:
            if self.monitoring_thread._paused_flag:  # 监测是否已暂停
                self.monitoring_thread.resume()  # 恢复
            else:
                self.monitoring_thread.pause()  # 暂停

    def clear_monitoring(self):
        """停止监控"""
        if self.monitoring_thread:
            self.monitoring_thread.stop()
        self.clear_monitor_display()

    def is_monitoring_active(self):
        return bool(self.monitoring_thread and self.monitoring_thread._run_flag)

    def is_monitoring_paused(self):
        return bool(self.monitoring_thread and self.monitoring_thread._paused_flag)

    def create_camera_labels(self, camera_ids):
        """创建摄像头显示标签"""
        self.camera_labels = {}

        cols = 2  # 每行2个摄像头
        for i, camera_id in enumerate(camera_ids):
            row = i // cols
            col = i % cols

            # 创建摄像头组
            camera_group = QGroupBox(f"摄像头 {camera_id}")
            camera_group.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(248, 249, 250, 0.9), stop:1 rgba(233, 236, 239, 0.9));
                color: #7f8c8d;
                font-weight: bold;
                font-size: 14px;
                border-radius: 10px;

            """)
            # camera_group.setMaximumHeight(350)
            camera_layout = QVBoxLayout(camera_group)

            # 图像显示标签
            image_label = QLabel("等待连接...")
            image_label.setMinimumSize(300, 240)
            image_label.setStyleSheet("""
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(248, 249, 250, 0.9), stop:1 rgba(233, 236, 239, 0.9));
                color: #7f8c8d;
                font-weight: bold;
                font-size: 14px;
                border-radius: 10px;

            """)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setScaledContents(True)

            camera_layout.addWidget(image_label, stretch=6)

            # 状态标签
            status_label = QLabel("状态: 初始化中...")
            status_label.setStyleSheet("color: #7f8c8d; font-size: 10px;")
            camera_layout.addWidget(status_label)
            camera_layout.addStretch()

            self.camera_labels[camera_id] = {
                'image': image_label,
                'status': status_label,
                'group': camera_group
            }

            self.monitor_layout.addWidget(camera_group, row, col)

    def set_equal_column_stretch(self):
        for c in range(self.monitor_layout.columnCount()):
            self.monitor_layout.setColumnStretch(c, 1)
        for r in range(self.monitor_layout.rowCount()):
            self.monitor_layout.setRowStretch(r, 1)

    def clear_monitor_display(self):
        """清空监控显示"""
        for camera_id in list(self.camera_labels.keys()):
            self.camera_labels[camera_id]['group'].deleteLater()
        self.camera_labels.clear()

    def update_camera_display(self, camera_id, original_img, result_img, inference_time, results, class_names):
        """更新摄像头显示"""
        if camera_id not in self.camera_labels:
            return

        # 显示结果图
        self.display_image(result_img, self.camera_labels[camera_id]['image'])

        # 更新状态
        if results and results[0].boxes and len(results[0].boxes) > 0:
            object_count = len(results[0].boxes)
            self.camera_labels[camera_id]['status'].setText(
                f"状态: 检测到 {object_count} 个目标 | 耗时: {inference_time:.3f}s"
            )
        else:
            self.camera_labels[camera_id]['status'].setText(
                f"状态: 无目标 | 耗时: {inference_time:.3f}s"
            )

    def handle_camera_error(self, camera_id, error_msg):
        """处理摄像头错误"""
        if camera_id in self.camera_labels:
            self.camera_labels[camera_id]['status'].setText(f"错误: {error_msg}")
            self.camera_labels[camera_id]['status'].setStyleSheet(
                "color: red; font-size: 10px;")

    def on_monitoring_finished(self):
        """监控结束"""
        for camera_id in self.camera_labels:
            self.camera_labels[camera_id]['status'].setText("状态: 已停止")

    def display_image(self, img_array, label):
        """显示图像"""
        if img_array is None:
            return

        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)


class StyleManager:
    """样式管理器 - 提供渐变和现代化UI样式"""

    # 界面正文：Win11 可变字体 + 微软雅黑 UI（中文更清晰），无则回退 Segoe UI
    UI_FONT_FAMILY = (
        '"Segoe UI Variable Text", "Segoe UI Variable", "Segoe UI", '
        '"Microsoft YaHei UI", "PingFang SC", "Hiragino Sans GB", sans-serif'
    )
    # 日志等宽：优先 Cascadia（Win10+ 常见），再 Consolas
    MONO_FONT_FAMILY = (
        '"Cascadia Mono", "Cascadia Code", "JetBrains Mono", '
        '"Consolas", "Courier New", monospace'
    )

    @staticmethod
    def application_ui_font():
        """QApplication 默认字体，与样式表一致。"""
        f = QFont()
        f.setFamilies([
            "Segoe UI Variable Text",
            "Segoe UI Variable",
            "Segoe UI",
            "Microsoft YaHei UI",
            "PingFang SC",
        ])
        f.setPointSizeF(10.0)
        f.setStyleHint(QFont.StyleHint.SansSerif)
        return f

    @staticmethod
    def log_mono_font(pixel_size: int):
        """运行日志等宽字体。"""
        f = QFont()
        f.setFamilies([
            "Cascadia Mono",
            "Cascadia Code",
            "JetBrains Mono",
            "Consolas",
            "Courier New",
        ])
        f.setPixelSize(max(8, int(pixel_size)))
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setFixedPitch(True)
        return f

    @staticmethod
    def _scale_css_font_sizes(css, font_scale):
        """按 font_scale 缩放样式表中的 font-size（仅 px），避免窗口全屏字过小、缩小窗体字过大。"""
        s = max(0.76, min(1.38, float(font_scale)))

        def px(n):
            return max(7, int(round(n * s)))

        return re.sub(
            r'font-size:\s*(\d+)px',
            lambda m: f'font-size: {px(int(m.group(1)))}px',
            css,
        )

    @staticmethod
    def get_main_stylesheet(font_scale=1.0):
        css = """
            /* 竞赛级：靛蓝 #6366f1 主色 · 大留白 · 轻拟物卡片 */
            QMainWindow {
                background: #e2e8f0;
                color: #334155;
                font-family: __UI_FONT__;
                font-size: 13px;
            }

            QWidget {
                font-family: __UI_FONT__;
            }

            QWidget#wireframePage {
                background: #f8fafc;
            }

            /* 线框稿顶栏：slate-700 → blue-500 → sky-400 */
            QFrame#appHeader {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #334155, stop:0.45 #3b82f6, stop:1 #38bdf8);
                border: none;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
                min-height: 80px;
            }
            QWidget#headerStatsStrip {
                background: transparent;
            }
            QFrame#headerStatChip {
                background: rgba(255, 255, 255, 0.11);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                min-width: 88px;
                max-width: 132px;
            }
            QLabel#headerStatTitle {
                color: rgba(241, 245, 249, 0.88);
                font-size: 11px;
                font-weight: 600;
                background: transparent;
            }
            QLabel#headerStatValue {
                color: #ffffff;
                font-size: 13px;
                font-weight: 700;
                background: transparent;
            }

            QLabel#appTitle {
                font-size: 22px;
                font-weight: 600;
                color: #ffffff;
                letter-spacing: 0.5px;
                background: transparent;
            }

            QLabel#appSubtitle {
                font-size: 13px;
                color: rgba(255, 255, 255, 0.82);
                font-weight: 500;
                background: transparent;
            }

            QFrame#headerLogoBadge {
                background: rgba(255, 255, 255, 0.14);
                border: 1px solid rgba(255, 255, 255, 0.22);
                border-radius: 16px;
            }

            QPushButton#headerQuickBtn {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 16px;
                padding: 0px;
            }
            QPushButton#headerQuickBtn:hover {
                background: rgba(255, 255, 255, 0.18);
            }
            QPushButton#headerToolBtn {
                background: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(255, 255, 255, 0.35);
                border-radius: 16px;
                padding: 8px 14px;
                color: #e5e7eb;
                font-size: 11px;
                font-weight: 500;
                min-height: 32px;
            }
            QPushButton#headerToolBtn:hover {
                background: rgba(255, 255, 255, 0.18);
                border-color: rgba(255, 255, 255, 0.65);
            }
            QPushButton#headerToolBtn:pressed,
            QPushButton#headerToolBtn:checked {
                background: rgba(15, 23, 42, 0.40);
                border-color: rgba(148, 163, 184, 0.90);
            }
            QPushButton#headerToolBtn:focus {
                outline: none;
            }
            QToolButton#headerToolMenuBtn {
                background: rgba(15, 23, 42, 0.22);
                border: 1px solid rgba(148, 163, 184, 0.55);
                border-radius: 16px;
                padding: 8px 14px;
                color: #e5e7eb;
                font-size: 11px;
                font-weight: 500;
                min-height: 32px;
            }
            QToolButton#headerToolMenuBtn:hover {
                background: rgba(148, 163, 184, 0.28);
                border-color: rgba(191, 219, 254, 0.95);
            }
            QToolButton#headerToolMenuBtn::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 10px;
                margin-right: 6px;
            }

            /* 主操作与全局 QPushButton 渐变一致，统一靛蓝 #6366f1 */
            QPushButton[variant="skyPrimary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #818cf8, stop:1 #6366f1);
                border: none;
                color: #ffffff;
                border-radius: 12px;
            }
            QPushButton[variant="skyPrimary"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
            }
            QPushButton[variant="skyPrimary"]:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
            }
            QPushButton[variant="skyPrimary"]:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }

            QSlider#confThresholdSlider::groove:horizontal {
                border: 1px solid #e2e8f0;
                height: 8px;
                background: #f1f5f9;
                border-radius: 4px;
            }
            QSlider#confThresholdSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #6366f1);
                border: 1px solid #4f46e5;
                width: 18px;
                height: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QSlider#confThresholdSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #c7d2fe, stop:1 #818cf8);
            }

            QFrame#wireframeToolbar {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 24px;
            }

            QFrame#toolbarSection {
                background: #ffffff;
                border: 1px solid #e5e7eb;
                border-radius: 14px;
            }

            QLabel#toolbarSectionTitle {
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                letter-spacing: 0.5px;
                text-transform: none;
                padding: 0 0 2px 0;
            }

            QLabel#toolbarFieldLabel {
                color: #475569;
                font-size: 12px;
                font-weight: 600;
            }

            QLabel#toolbarFormLabel {
                color: #475569;
                font-size: 12px;
                font-weight: 700;
                min-width: 72px;
                padding-top: 2px;
            }

            QLabel#currentFileValue {
                color: #0f172a;
                font-size: 12px;
                font-weight: 600;
            }
            QLineEdit#pathReadonlyField,
            QLineEdit#pathEditableField {
                min-height: 34px;
                padding: 0 10px;
                border-radius: 10px;
                border: 1px solid #dbe3ef;
                color: #0f172a;
                font-size: 12px;
            }
            QLineEdit#pathReadonlyField {
                background: #f8fafc;
            }
            QLineEdit#pathEditableField {
                background: #ffffff;
            }
            QLineEdit#pathReadonlyField:focus,
            QLineEdit#pathEditableField:focus {
                border-color: #818cf8;
            }

            QScrollArea#analysisSidebarScroll {
                background: transparent;
                border: none;
            }
            QFrame#wireframeCanvasCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }

            QFrame#wireframeCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 24px;
            }
            QFrame#sourceTopCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-top-left-radius: 0px;
                border-top-right-radius: 0px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }

            QFrame#wireframeCardHeader {
                border-bottom: 2px solid #e2e8f0;
            }

            /* 底部检测结果表 / 侧栏运行日志：统一纯蓝顶栏 + 白字 */
            QFrame#wfInsetHeaderBar {
                background: #3b82f6;
                border: none;
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                min-height: 44px;
            }
            QLabel#wfInsetHeaderTitle {
                color: #ffffff;
                font-size: 14px;
                font-weight: 700;
                background: transparent;
            }

            QWidget#wfLogSheetBody {
                background: transparent;
            }
            QFrame#wireframeCard QPlainTextEdit#wfLogText {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                outline: none;
                background: #f8fafc;
                color: #0f172a;
                padding: 10px 12px;
                font-size: 12px;
                selection-background-color: #bfdbfe;
                selection-color: #0f172a;
            }
            QFrame#wireframeCard QWidget#wfLogFooter {
                background: transparent;
                border: none;
            }

            QFrame#wfResultSheet {
                background: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 8px;
            }

            QWidget#runButtonsCorner {
                background: transparent;
            }
            QWidget#canvasProgressCorner {
                background: transparent;
            }

            QFrame#wfTitleBar {
                background: #cbd5e1;
                border-radius: 4px;
            }

            QLabel#wfCardTitle {
                font-size: 14px;
                font-weight: 700;
                color: #0f172a;
            }

            QLabel#wfCanvasTitle {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }

            QLabel#wfMutedHint {
                color: #64748b;
                font-size: 12px;
            }
            QFrame#overviewMetricCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
            }
            QLabel#overviewMetricTitle {
                color: #64748b;
                font-size: 12px;
                font-weight: 600;
                background: transparent;
            }
            QLabel#overviewMetricValue {
                color: #0f172a;
                font-size: 16px;
                font-weight: 700;
                background: transparent;
            }

            QLabel#wfStatTitle {
                font-size: 11px;
                color: #64748b;
                font-weight: 600;
            }

            QLabel#wfStatValue {
                font-size: 15px;
                font-weight: 700;
                color: #0f172a;
            }

            QFrame#wfSmallStat {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
            }

            QLabel#wfClassName {
                font-size: 11px;
                color: #64748b;
            }
            QLabel#wfClassPct {
                font-size: 11px;
                font-weight: 700;
                color: #0f172a;
            }

            QProgressBar#wfClassBar0::chunk {
                background: #6366f1;
                border-radius: 4px;
            }
            QProgressBar#wfClassBar1::chunk {
                background: #818cf8;
                border-radius: 4px;
            }
            QProgressBar#wfClassBar2::chunk {
                background: #a5b4fc;
                border-radius: 4px;
            }

            QLabel#wfTargetThumb {
                background: #f1f5f9;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                color: #94a3b8;
                font-size: 12px;
            }

            QLabel#wfTargetLine {
                font-size: 12px;
                color: #475569;
            }

            QFrame#canvasProgressCard {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
            }

            QLabel#wfProgressLabel {
                font-size: 12px;
                font-weight: 600;
                color: #64748b;
            }

            QProgressBar#canvasProgressBar {
                border: none;
                border-radius: 999px;
                background: #e2e8f0;
                max-height: 8px;
                min-height: 8px;
            }
            QProgressBar#canvasProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #6366f1);
                border-radius: 999px;
            }
            QProgressBar#canvasProgressBar[progressState="done"]::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34d399, stop:1 #10b981);
                border-radius: 999px;
            }

            QPushButton#modePill {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 10px 22px;
                color: #334155;
                font-weight: 600;
            }
            QPushButton#modePill:checked {
                background: #6366f1;
                border-color: #6366f1;
                color: #ffffff;
            }
            QPushButton#modePill:hover:!checked {
                border-color: #c7d2fe;
                background: #f8fafc;
            }

            QTabWidget#bottomDrawer::pane {
                border: 1px solid #e2e8f0;
                border-radius: 24px;
                background: #ffffff;
                top: 0px;
                padding: 0px;
                margin-top: 0px;
            }

            QTabBar#bottomDrawerTabBar::tab {
                background: transparent;
                border: none;
                border-radius: 12px;
                padding: 12px 20px;
                margin-right: 8px;
                color: #64748b;
                font-weight: 600;
                font-size: 13px;
            }
            QTabBar#bottomDrawerTabBar::tab:selected {
                background: #e0f2fe;
                color: #0369a1;
                font-weight: 700;
            }
            QTabBar#bottomDrawerTabBar::tab:hover:!selected {
                background: #f8fafc;
                color: #475569;
            }

            QLabel#wfPlaceholder {
                color: #94a3b8;
                font-size: 13px;
                padding: 40px;
            }

            QTableWidget#wfResultTable {
                border: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
                background: #ffffff;
                gridline-color: #f1f5f9;
            }
            QTableWidget#wfResultTable QHeaderView::section {
                background: #4a86e8;
                color: #ffffff;
                padding: 8px 6px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                border-right: 1px solid rgba(255, 255, 255, 0.28);
            }

            QFrame#headerPill {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 999px;
                min-width: 132px;
                max-width: 180px;
            }

            QLabel#headerPillLabel {
                color: #f8fafc;
                font-size: 12px;
                font-weight: 600;
                background: transparent;
            }

            QProgressBar#headerPillBarInfer,
            QProgressBar#headerPillBarModel,
            QProgressBar#headerPillBarPipe {
                border: none;
                border-radius: 3px;
                background: rgba(255, 255, 255, 0.14);
                max-height: 5px;
                min-height: 5px;
            }

            QProgressBar#headerPillBarInfer::chunk {
                background: #a855f7;
                border-radius: 3px;
            }
            QProgressBar#headerPillBarModel::chunk {
                background: #2dd4bf;
                border-radius: 3px;
            }
            QProgressBar#headerPillBarPipe::chunk {
                background: #a5b4fc;
                border-radius: 3px;
            }

            QFrame#sidePanel {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 16px;
            }

            QLabel#panelTitle {
                font-size: 13px;
                font-weight: 700;
                color: #0f172a;
                padding: 8px 4px 8px 4px;
                border: none;
                border-bottom: 2px solid #e0e7ff;
                background: transparent;
            }

            QLabel#sideFormLabel {
                color: #475569;
                font-size: 13px;
                min-width: 104px;
                max-width: 104px;
                padding-right: 6px;
            }

            QLabel#previewPlaceholder {
                border: 2px dashed #cbd5e1;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ffffff, stop:1 #f8fafc);
                color: #64748b;
                font-weight: 600;
                font-size: 13px;
                border-radius: 18px;
                padding: 28px;
            }

            QLabel#batchPreviewPlaceholder {
                border: 2px dashed #cbd5e1;
                background: #fafafa;
                color: #64748b;
                font-weight: 600;
                font-size: 12px;
                border-radius: 16px;
                padding: 20px 16px;
            }

            QLabel#batchInfoCard {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 14px 16px;
                color: #334155;
                font-size: 12px;
            }

            QFrame#statusStrip {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }

            QLabel#statusKey {
                color: #64748b;
                font-size: 11px;
            }

            QLabel#statusValue {
                color: #0f172a;
                font-size: 11px;
                font-weight: 600;
            }

            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 18px;
                background: #fafafa;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                color: #334155;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.2px;
                background: #fafafa;
            }

            QPushButton {
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                border-radius: 10px;
                color: #ffffff;
                min-height: 32px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #818cf8, stop:1 #6366f1);
            }

            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
            }

            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
            }

            QPushButton:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }

            QPushButton[variant="primary"] {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #818cf8, stop:1 #6366f1);
                border: none;
                color: #ffffff;
            }
            QPushButton[variant="primary"]:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
            }
            QPushButton[variant="primary"]:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
            }

            QPushButton[variant="secondary"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                color: #475569;
            }
            QPushButton[variant="secondary"]:hover {
                background: #f8fafc;
                border-color: #c7d2fe;
                color: #312e81;
            }
            QPushButton[variant="secondary"]:pressed {
                background: #eef2ff;
            }

            QPushButton[variant="stop"] {
                background: #64748b;
                border: none;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton[variant="stop"]:hover {
                background: #475569;
            }
            QPushButton[variant="stop"]:pressed {
                background: #334155;
            }
            QPushButton[variant="stop"]:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }

            QPushButton#toolBtn {
                min-width: 76px;
                max-width: 96px;
                padding: 6px 8px;
                font-size: 11px;
                font-weight: 600;
                min-height: 30px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                color: #475569;
            }
            QPushButton#toolBtn:hover {
                background: #f8fafc;
                border-color: #c7d2fe;
                color: #4338ca;
            }
            QPushButton#toolBtn:pressed {
                background: #eef2ff;
            }

            QFrame#runControlPanel,
            QFrame#presetActionPanel {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
            QFrame#runControlPanel QLabel#runControlHint {
                color: #64748b;
                font-size: 11px;
            }

            /* 侧栏滚动区：无框，避免再套一层边线 */
            QScrollArea#sideScroll {
                border: none;
                background: transparent;
            }

            QComboBox {
                padding: 6px 10px;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                font-size: 12px;
                min-width: 72px;
                min-height: 28px;
                color: #0f172a;
            }

            QComboBox:focus {
                border-color: #818cf8;
                background: #ffffff;
            }

            QProgressBar {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                font-size: 11px;
                max-height: 22px;
                background: #f1f5f9;
                color: #475569;
            }

            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #818cf8, stop:1 #6366f1);
                border-radius: 7px;
                margin: 1px;
            }

            QTextEdit {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: #ffffff;
                font-family: __MONO_FONT__;
                font-size: 11px;
                padding: 10px 12px;
                color: #334155;
                selection-background-color: rgba(99, 102, 241, 0.35);
            }

            QSpinBox, QDoubleSpinBox {
                padding: 6px 10px;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                min-width: 80px;
                font-size: 12px;
                color: #0f172a;
            }

            QTabWidget::pane {
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                background: #ffffff;
                margin-top: 0px;
                padding: 10px;
            }

            QTabBar#mainTabBar::tab {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 10px;
                padding: 8px 16px;
                margin-right: 8px;
                margin-bottom: 0px;
                margin-top: 0px;
                font-weight: 600;
                font-size: 12px;
                color: #64748b;
                min-height: 34px;
            }

            QTabBar#mainTabBar::tab:selected {
                background: #f8fafc;
                border: 1px solid #dbe3ef;
                border-bottom: 2px solid #6366f1;
                color: #0f172a;
            }

            QTabBar#mainTabBar::tab:hover:!selected {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #475569;
            }

            QTabBar#dialogTabBar::tab {
                background: transparent;
                border: none;
                border-bottom: 3px solid transparent;
                padding: 12px 20px;
                margin-right: 4px;
                font-weight: 600;
                font-size: 12px;
                color: #64748b;
            }

            QTabBar#dialogTabBar::tab:selected {
                color: #4338ca;
                border-bottom: 3px solid #6366f1;
                background: rgba(99, 102, 241, 0.06);
            }

            QTabBar#dialogTabBar::tab:hover:!selected {
                color: #475569;
                background: #f8fafc;
            }

            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                background: #ffffff;
                gridline-color: #f1f5f9;
                selection-background-color: rgba(99, 102, 241, 0.12);
                alternate-background-color: #fafafa;
            }

            QTableWidget::item {
                padding: 12px 14px;
                border: none;
            }

            QTableWidget::item:selected {
                background: rgba(99, 102, 241, 0.18);
                color: #0f172a;
            }

            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4f46e5, stop:1 #4338ca);
                color: #f8fafc;
                padding: 12px 14px;
                border: none;
                font-weight: 700;
            }

            QListWidget {
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                selection-background-color: rgba(99, 102, 241, 0.12);
            }

            QListWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #f1f5f9;
            }

            QListWidget::item:selected {
                background: rgba(99, 102, 241, 0.18);
                color: #0f172a;
            }

            QScrollBar:vertical {
                background: #f1f5f9;
                width: 10px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #cbd5e1;
                border-radius: 5px;
                min-height: 24px;
            }

            QScrollBar::handle:vertical:hover {
                background: #94a3b8;
            }

            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #818cf8;
            }

            QSplitter::handle:horizontal {
                background: #e2e8f0;
                width: 5px;
                border-radius: 2px;
                margin: 0 2px;
            }
            QSplitter::handle:horizontal:hover {
                background: #c7d2fe;
            }
        """
        css = css.replace("__UI_FONT__", StyleManager.UI_FONT_FAMILY)
        css = css.replace("__MONO_FONT__", StyleManager.MONO_FONT_FAMILY)
        return StyleManager._scale_css_font_sizes(css, font_scale)

    @staticmethod
    def get_image_label_style():
        return """
            border: 2px dashed #cbd5e1;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #f8fafc);
            color: #64748b;
            font-weight: 600;
            font-size: 13px;
            border-radius: 18px;
            padding: 24px;
        """


class CameraManager:
    """摄像头管理器 - 处理多摄像头检测和管理"""

    def __init__(self):
        self.cameras = []
        self.scan_cameras()

    def scan_cameras(self):
        """扫描可用摄像头"""
        self.cameras = []

        # 检测摄像头（检测前8个索引）
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # 获取摄像头信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)

                    camera_info = {
                        'id': i,
                        'name': f"摄像头 {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps if fps > 0 else 30,
                        'available': True
                    }
                    self.cameras.append(camera_info)
                cap.release()

        # 如果没有摄像头，添加虚拟摄像头用于测试
        if not self.cameras:
            self.cameras.append({
                'id': -1,
                'name': "未检测到摄像头",
                'resolution': "N/A",
                'fps': 0,
                'available': False
            })

    def get_available_cameras(self):
        """获取可用摄像头列表"""
        return [cam for cam in self.cameras if cam['available']]

    def get_camera_info(self, camera_id):
        """获取摄像头信息"""
        for cam in self.cameras:
            if cam['id'] == camera_id:
                return cam
        return None


class ModelManager:
    """模型管理器 - 处理模型扫描和加载"""

    def __init__(self):
        self.models_paths = [
            base_dir / "pt_models",
            base_dir / "models",
            base_dir / "weights",
        ]
        self.current_model = None
        self.class_names = []

    def scan_models(self, custom_path=None):
        """扫描模型文件"""
        models = []
        search_paths = self.models_paths.copy()

        if custom_path and Path(custom_path).exists():
            search_paths.insert(0, Path(custom_path))

        for model_dir in search_paths:
            if model_dir.exists():
                try:
                    pt_files = sorted(model_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        models.append({
                            'name': pt_file.name,
                            'path': str(pt_file),
                            'size': self._get_file_size(pt_file),
                            'modified': self._get_modification_time(pt_file)
                        })
                except Exception as e:
                    print(f"扫描目录 {model_dir} 时出错: {e}")

        return models

    def load_model(self, model_path):
        """加载模型"""
        try:
            self.current_model = YOLO(model_path)
            self.class_names = list(self.current_model.names.values())
            return True
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False

    def get_class_names(self):
        """获取类别名称"""
        return self.class_names

    def _get_file_size(self, file_path):
        """获取文件大小"""
        try:
            size = file_path.stat().st_size
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"

    def _get_modification_time(self, file_path):
        """获取修改时间"""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
        except:
            return "Unknown"


class DetectionThread(QThread):
    """增强的检测线程"""
    result_ready = Signal(object, object, float, object,
                          list)  # 原图, 结果图, 耗时, 检测结果, 类别名称
    progress_updated = Signal(int)
    status_changed = Signal(str)
    error_occurred = Signal(str)
    fps_updated = Signal(float)
    finished = Signal()

    def __init__(self, model, source_type, source_path=None, camera_id=0, confidence_threshold=0.25):
        super().__init__()
        self.model = model
        self.source_type = source_type
        self.source_path = source_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

    def run(self):
        self.is_running = True
        try:
            if self.source_type == 'image':
                self._process_image()
            elif self.source_type == 'video':
                self._process_video()
            elif self.source_type == 'camera':
                self._process_camera()
        except Exception as e:
            self.error_occurred.emit(f"检测过程发生错误: {str(e)}")
        finally:
            self.is_running = False
            self.finished.emit()

    def _process_image(self):
        """处理单张图片"""
        if not self.source_path or not Path(self.source_path).exists():
            self.error_occurred.emit("图片文件不存在")
            return

        self.status_changed.emit("正在处理图片...")

        start_time = time.time()
        results = self.model(
            self.source_path, conf=self.confidence_threshold, verbose=False)
        end_time = time.time()

        original_img = cv2.imread(self.source_path)
        if original_img is None:
            self.error_occurred.emit("无法读取图片文件")
            return

        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        class_names = list(self.model.names.values())

        self.result_ready.emit(original_img, result_img,
                               end_time - start_time, results, class_names)
        self.progress_updated.emit(100)

    def _process_video(self):
        """处理视频文件"""
        if not self.source_path or not Path(self.source_path).exists():
            self.error_occurred.emit("视频文件不存在")
            return

        cap = cv2.VideoCapture(self.source_path)
        if not cap.isOpened():
            self.error_occurred.emit("无法打开视频文件")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        class_names = list(self.model.names.values())

        self.status_changed.emit(f"开始处理视频 (共{total_frames}帧)...")

        while cap.isOpened() and self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = self.model(
                frame, conf=self.confidence_threshold, verbose=False)
            end_time = time.time()

            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            self.result_ready.emit(
                original_img, result_img, end_time - start_time, results, class_names)

            frame_count += 1
            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)

            # 更新FPS
            self._update_fps()

            # 状态更新（每30帧更新一次）
            if frame_count % 30 == 0:
                current_fps = self._get_current_fps()
                self.status_changed.emit(
                    f"处理中... {frame_count}/{total_frames} 帧 (FPS: {current_fps:.1f})")

            time.sleep(0.033)  # 约30fps

        cap.release()

    def _process_camera(self):
        """处理摄像头"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self.error_occurred.emit(f"无法打开摄像头 {self.camera_id}")
            return

        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        class_names = list(self.model.names.values())
        self.status_changed.emit(f"摄像头 {self.camera_id} 已启动...")

        while cap.isOpened() and self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            start_time = time.time()
            results = self.model(
                frame, conf=self.confidence_threshold, verbose=False)
            end_time = time.time()

            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            self.result_ready.emit(
                original_img, result_img, end_time - start_time, results, class_names)

            # 更新FPS
            self._update_fps()

            # 状态更新（每60帧更新一次）
            if self.frame_count % 60 == 0:
                current_fps = self._get_current_fps()
                self.status_changed.emit(f"摄像头运行中 (FPS: {current_fps:.1f})")

            time.sleep(0.033)  # 约30fps

        cap.release()

    def _update_fps(self):
        """更新FPS计算"""
        self.frame_count += 1
        self.fps_counter += 1

        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_updated.emit(fps)
            self.fps_counter = 0
            self.last_fps_time = current_time

    def _get_current_fps(self):
        """获取当前FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time > 0:
            return self.fps_counter / (current_time - self.last_fps_time)
        return 0

    def pause(self):
        self.is_paused = True
        self.status_changed.emit(f"暂停中...")

    def resume(self):
        self.is_paused = False
        self.status_changed.emit(f"恢复检测")

    def stop(self):
        self.is_running = False
        self.status_changed.emit(f"检测结束!")


class EnhancedDetectionUI(QMainWindow):
    """增强的检测UI主窗口"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.detection_thread = None
        self.batch_detection_thread = None
        self.current_source_type = 'image'
        self.current_source_path = None
        self.default_save_dir = str((base_dir / "results").absolute())
        self.confidence_threshold = 0.25
        self.batch_results = []
        self.current_batch_index = 0
        self.preset_data = {}
        self._delete_confirm_target = None
        self.task_preset_file = base_dir / "task_presets.json"

        # 管理器
        self.camera_manager = CameraManager()
        self.model_manager = ModelManager()
        self.log_text = QPlainTextEdit()
        self._status_model = QLabel()
        self._status_mode = QLabel()
        self._status_source = QLabel()
        self._status_fps = QLabel()
        self._status_latency = QLabel()
        self._status_objects = QLabel()
        # 实时预览最后一帧（用于窗口缩放时按等比例重绘）
        self._last_preview_original = None
        self._last_preview_result = None
        self._current_file_full_path = None
        self.init_ui()
        self.setWindowIcon(self.create_enhanced_icon())

        self._last_applied_font_scale = None
        self._font_resize_timer = QTimer(self)
        self._font_resize_timer.setSingleShot(True)
        self._font_resize_timer.timeout.connect(self._apply_ui_font_scale)
        self._apply_ui_font_scale()

        # 全局点击事件：用于在“确认删除”高亮状态下，点击其它区域时自动还原按钮
        self.installEventFilter(self)

    def _compute_font_scale(self):
        """按窗口短边相对参考尺寸缩放字号，全屏偏大、缩小窗口略小。"""
        w = max(self.width(), 320)
        h = max(self.height(), 240)
        m = min(w, h)
        ref = 880.0
        s = m / ref
        return max(0.78, min(1.36, s))

    def _apply_ui_font_scale(self):
        s = self._compute_font_scale()
        prev = getattr(self, "_last_applied_font_scale", None)
        if prev is not None and abs(s - prev) < 0.012:
            return
        self._last_applied_font_scale = s
        self.setStyleSheet(StyleManager.get_main_stylesheet(font_scale=s))
        if hasattr(self, "log_text"):
            self.log_text.setFont(
                StyleManager.log_mono_font(10 * s))
        if hasattr(self, "conf_spinbox"):
            self.conf_spinbox.setFixedWidth(max(56, int(round(72 * s))))
        adv = getattr(self, "advanced_model_btn", None)
        if adv is not None:
            adv.setFixedWidth(max(72, int(round(90 * s))))
        # 顶部角标运行区：随字号缩放同步高度，防止按钮被 Tab 顶栏裁切
        compact_h = max(34, int(round(34 * min(s, 1.15))))
        corner_h = max(40, int(round(40 * min(s, 1.15))))
        for btn in (getattr(self, "start_btn", None), getattr(self, "pause_btn", None), getattr(self, "stop_btn", None)):
            if btn is not None:
                btn.setMinimumHeight(compact_h)
        if hasattr(self, "tab_widget"):
            tab_bar = self.tab_widget.tabBar()
            if tab_bar is not None:
                tab_bar.setMinimumHeight(corner_h)
            corner = self.tab_widget.cornerWidget(Qt.Corner.TopRightCorner)
            if corner is not None:
                corner.setMinimumHeight(corner_h)
        # 顶部运行按钮宽度与「新建预设」一致，保证上下视觉统一
        preset_btn = getattr(self, "new_preset_btn", None)
        if preset_btn is not None:
            preset_w = preset_btn.sizeHint().width()
            run_btns = [getattr(self, "start_btn", None), getattr(
                self, "pause_btn", None), getattr(self, "stop_btn", None)]
            run_btns = [b for b in run_btns if b is not None]
            need_w = max((b.sizeHint().width() for b in run_btns), default=0)
            # 与预设三按钮统一：保持文本不截断，同时允许在行内等分铺满
            btn_w = max(88, int(preset_w), int(need_w))
            for btn in run_btns:
                btn.setMinimumWidth(0)
                btn.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                btn.setMaximumWidth(16777215)
            top_run_row = getattr(self, "_top_run_row", None)
            if top_run_row is not None:
                top_run_row.setMinimumWidth(0)
                top_run_row.setMaximumWidth(16777215)
                top_run_row.setSizePolicy(
                    QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        rdw = getattr(self, "result_detail_widget", None)
        if rdw is not None:
            rdw.stats_label.setMinimumHeight(max(56, int(round(72 * s))))
            spx = max(10, int(round(12 * s)))
            pdp = max(12, int(round(14 * s)))
            rdw.stats_label.setStyleSheet(
                f"""
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #eef2ff, stop:1 #e0e7ff);
                border: 1px solid #c7d2fe;
                padding: {pdp}px 16px;
                border-radius: 12px;
                font-size: {spx}px;
                color: #312e81;
                font-weight: 600;
                """
            )

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("Dimension 目标检测系统")
        # 默认尺寸（非最大化恢复时使用）；首次启动在 main() 中 showMaximized
        self.resize(1400, 900)
        self.setMinimumSize(1024, 680)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_wrap = QWidget()
        content_wrap.setObjectName("wireframePage")
        self._content_wrap = content_wrap
        wrap_layout = QVBoxLayout(content_wrap)
        # 去掉外层留白，取消主内容区四周间隙
        wrap_layout.setContentsMargins(0, 0, 0, 0)
        # 头部卡片与主工作区无断层连接
        wrap_layout.setSpacing(0)

        app_header = QFrame()
        app_header.setObjectName("appHeader")
        self._app_header = app_header
        header_layout = QHBoxLayout(app_header)
        header_layout.setContentsMargins(28, 18, 28, 18)
        header_layout.setSpacing(16)
        title_row = QHBoxLayout()
        title_row.setSpacing(16)
        logo_badge = QFrame()
        logo_badge.setObjectName("headerLogoBadge")
        logo_badge.setFixedSize(56, 56)
        logo_lay = QVBoxLayout(logo_badge)
        logo_lay.setContentsMargins(0, 0, 0, 0)
        logo_ic = QLabel()
        logo_ic.setPixmap(ThemeIcons.pixmap("sparkles", 26, "#ffffff"))
        logo_ic.setAlignment(Qt.AlignCenter)
        logo_lay.addWidget(logo_ic)
        title_block = QVBoxLayout()
        title_block.setSpacing(4)
        app_title = QLabel("Dimension")
        app_title.setObjectName("appTitle")
        app_sub = QLabel("目标检测工作台 · 实时推理 · 批量分析 · 多路监控")
        app_sub.setObjectName("appSubtitle")
        title_block.addWidget(app_title)
        title_block.addWidget(app_sub)
        title_row.addWidget(logo_badge, 0, Qt.AlignVCenter)
        title_row.addLayout(title_block, 1)
        header_layout.addLayout(title_row, 0)
        header_layout.addStretch(1)
        wrap_layout.addWidget(app_header)
        self.result_detail_widget = DetectionResultWidget()
        self.result_detail_widget.stats_label.hide()

        center = self._build_main_workspace()
        wrap_layout.addWidget(center, 1)

        self.bottom_drawer = QTabWidget()
        self.bottom_drawer.setObjectName("bottomDrawer")
        btb = self.bottom_drawer.tabBar()
        btb.setObjectName("bottomDrawerTabBar")
        self.log_text.setFont(StyleManager.log_mono_font(10))

        self.bottom_drawer.addTab(self.result_detail_widget, "检测明细")
        hist_tab = QLabel("历史任务记录将显示在此处。")
        hist_tab.setAlignment(Qt.AlignCenter)
        hist_tab.setObjectName("wfPlaceholder")
        self.bottom_drawer.addTab(hist_tab, "历史任务")
        q_tab = QLabel("批处理队列与任务编排将显示在此处。")
        q_tab.setAlignment(Qt.AlignCenter)
        q_tab.setObjectName("wfPlaceholder")
        self.bottom_drawer.addTab(q_tab, "批处理队列")

        self._main_left_column_layout.addWidget(self.bottom_drawer, 0)
        self.tab_widget.currentChanged.connect(self._on_main_tab_changed)
        self._on_main_tab_changed(self.tab_widget.currentIndex())

        main_layout.addWidget(content_wrap, 1)

        # 状态栏
        self.statusBar().showMessage("就绪 — 请选择模型与检测源")
        # 首次启动时同步：主 Tab 与输入源下拉保持单一状态来源
        if hasattr(self, "tab_widget"):
            self._sync_source_options_for_tab(self.tab_widget.currentIndex())

        # 尝试加载默认模型
        self.try_load_default_model()
        self._update_header_pills()
        self._update_current_file_display()

    def _on_main_tab_changed(self, index: int):
        """仅「实时检测」显示底部检测明细/历史/队列；批量与设备监控占满主画布。"""
        if hasattr(self, "bottom_drawer"):
            self.bottom_drawer.setVisible(index == 0)
        self._sync_source_options_for_tab(index)

    def _sync_source_options_for_tab(self, tab_index: int):
        """让右侧输入源选项与主 Tab 语义一致，避免双导航冲突。"""
        if not hasattr(self, "source_combo"):
            return
        tab_source_map = {
            0: ["单张图片", "视频文件", "摄像头"],  # 实时检测
            1: ["文件夹批量"],                  # 批量分析
            2: ["摄像头"],                      # 设备监控
        }
        allowed = tab_source_map.get(tab_index, ["单张图片", "视频文件", "摄像头"])
        previous = self.source_combo.currentText()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(allowed)
        self.source_combo.blockSignals(False)
        target = previous if previous in allowed else allowed[0]
        self.source_combo.setCurrentText(target)
        # 强制同步内部 source_type，避免下拉文本与实际选择逻辑不一致
        self.on_source_changed(target)
        # 仅当有多个可选输入源时显示「模式」行；单一模式时隐藏以减少冗余
        if hasattr(self, "source_mode_row"):
            self.source_mode_row.setVisible(len(allowed) > 1)

    def _get_monitor_camera_ids(self):
        """设备监控从右侧输入源读取摄像头。"""
        camera_id = self.camera_combo.currentData() if hasattr(self, "camera_combo") else -1
        if camera_id is None or camera_id == -1:
            return []
        return [int(camera_id)]

    def _form_label(self, text):
        """侧栏表单左列标签：固定宽度、右对齐，与其它行对齐。"""
        lab = QLabel(text)
        lab.setObjectName("sideFormLabel")
        lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        return lab

    def _toolbar_label(self, text, width=56):
        """工具栏内短标签：左对齐、与输入框垂直居中。"""
        lab = QLabel(text)
        lab.setObjectName("toolbarFieldLabel")
        lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lab.setFixedWidth(width)
        return lab

    def _toolbar_section_title(self, text: str) -> QLabel:
        t = QLabel(text)
        t.setObjectName("toolbarSectionTitle")
        return t

    def _update_current_file_display(self):
        """根据当前输入源显示文件路径或设备名称。"""
        if not hasattr(self, "current_file_label"):
            return
        if getattr(self, "current_source_type", None) == "camera":
            cam_text = self.camera_combo.currentText() if hasattr(
                self, "camera_combo") else ""
            if cam_text and self.camera_combo.currentData() != -1:
                self.current_file_label.setText(cam_text)
                self.current_file_label.setToolTip(cam_text)
            else:
                self.current_file_label.setText("未选择摄像头")
                self.current_file_label.setToolTip("请先选择可用摄像头")
            self.current_file_label.setCursorPosition(0)
            self._refresh_right_overview()
            return
        if self._current_file_full_path:
            p = Path(self._current_file_full_path)
            self.current_file_label.setText(str(p.resolve()))
            self.current_file_label.setToolTip(str(p.resolve()))
            self.current_file_label.setCursorPosition(0)
        else:
            self.current_file_label.setText("未选择文件")
            self.current_file_label.setToolTip("请使用「选择文件」或工具栏中的「打开文件」")
            self.current_file_label.setCursorPosition(0)
        self._refresh_right_overview()

    def _panel_title_row(self, text, icon_name: str):
        """面板标题：Lucide 风格矢量图标 + 标题。"""
        row = QWidget()
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(ThemeIcons.pixmap(icon_name, 20, "#6366f1"))
        text_lbl = QLabel(text)
        text_lbl.setObjectName("panelTitle")
        lay.addWidget(icon_lbl, 0, Qt.AlignVCenter)
        lay.addWidget(text_lbl, 0, Qt.AlignVCenter)
        lay.addStretch()
        return row

    @staticmethod
    def _set_btn_icon(btn, name: str, color: str = "#ffffff", size: int = 18):
        btn.setIcon(ThemeIcons.icon(name, size, color))
        btn.setIconSize(QSize(size, size))

    def _apply_card_shadow(self, widget, blur=32, dy=10, alpha=42):
        """设计稿：卡片轻悬浮阴影（靛色光晕）。"""
        eff = QGraphicsDropShadowEffect(widget)
        eff.setBlurRadius(blur)
        eff.setOffset(0, dy)
        eff.setColor(QColor(79, 70, 229, alpha))
        widget.setGraphicsEffect(eff)

    def _build_header_pill(self, icon_name: str, title: str, bar_object_name: str):
        """顶栏右侧胶囊：图标 + 标题 + 细进度条。"""
        pill = QFrame()
        pill.setObjectName("headerPill")
        v = QVBoxLayout(pill)
        v.setContentsMargins(14, 10, 14, 12)
        v.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        ic = QLabel()
        ic.setPixmap(ThemeIcons.pixmap(icon_name, 16, "#e0e7ff"))
        lab = QLabel(title)
        lab.setObjectName("headerPillLabel")
        top.addWidget(ic, 0, Qt.AlignVCenter)
        top.addWidget(lab, 0, Qt.AlignVCenter)
        top.addStretch()
        bar = QProgressBar()
        bar.setObjectName(bar_object_name)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)
        v.addLayout(top)
        v.addWidget(bar)
        return pill, bar

    def _on_main_progress(self, value: int):
        self._set_progress_state("running")
        self.progress_bar.setValue(value)
        self._update_header_pills()

    def _set_progress_state(self, state: str):
        """任务进度视觉状态：idle/running/done。"""
        if not hasattr(self, "progress_bar"):
            return
        self.progress_bar.setProperty("progressState", state)
        style = self.progress_bar.style()
        style.unpolish(self.progress_bar)
        style.polish(self.progress_bar)
        self.progress_bar.update()
        if hasattr(self, "progress_title_label"):
            title_map = {
                "idle": "任务进度",
                "running": "任务进行中",
                "done": "任务完成",
            }
            self.progress_title_label.setText(title_map.get(state, "任务进度"))

    def _update_header_pills(self):
        """同步顶栏三胶囊：模型就绪 / 推理负载 / 任务管线。"""
        b_model = getattr(self, "_pill_model_bar", None)
        b_infer = getattr(self, "_pill_infer_bar", None)
        b_pipe = getattr(self, "_pill_pipe_bar", None)
        if not b_model or not b_infer or not b_pipe:
            return
        b_model.setValue(100 if getattr(
            self, "model", None) is not None else 0)
        det = getattr(self, "detection_thread",
                      None) and self.detection_thread.is_running
        bat = getattr(self, "batch_detection_thread",
                      None) and self.batch_detection_thread.is_running
        pb = getattr(self, "progress_bar", None)
        pv = pb.value() if pb else 0
        if det or bat:
            b_infer.setValue(max(12, pv) if pv > 0 else 42)
        else:
            b_infer.setValue(14)
        if bat:
            b_pipe.setValue(max(4, pv))
        else:
            b_pipe.setValue(28 if pv == 0 else min(96, pv + 24))

    def _on_header_model_convert_clicked(self):
        """顶部模型转换入口（当前版本先引导到模型管理）。"""
        self.log_message("模型转换入口：请在模型管理中执行格式转换/导出。")
        self.show_model_selection_dialog()

    def _focus_task_preset(self):
        """顶部任务预设入口：定位到右侧任务配置区域。"""
        if hasattr(self, "preset_combo"):
            self.preset_combo.setFocus()
            self.log_message("已定位到任务预设，请在右侧进行新建/修改/删除。")

    def _place_dialog_near_widget(self, dialog, anchor_widget, margin=8):
        """将对话框摆在锚点控件附近（默认紧贴下方），并限制在屏幕工作区内。"""
        dialog.adjustSize()
        scr = QGuiApplication.screenAt(
            anchor_widget.mapToGlobal(anchor_widget.rect().center()))
        if scr is None:
            scr = QGuiApplication.primaryScreen()
        avail = scr.availableGeometry()
        sh = dialog.sizeHint()
        w = max(sh.width(), dialog.width())
        h = max(sh.height(), dialog.height())
        top_left = anchor_widget.mapToGlobal(
            QPoint(0, anchor_widget.height() + margin))
        x, y = top_left.x(), top_left.y()
        if x + w > avail.right():
            x = max(avail.left() + 8, avail.right() - w - 8)
        if y + h > avail.bottom():
            y = anchor_widget.mapToGlobal(QPoint(0, -h - margin)).y()
        y = max(avail.top() + 8, min(y, avail.bottom() - h - 8))
        x = max(avail.left() + 8, min(x, avail.right() - w - 8))
        dialog.move(x, y)

    def _create_run_buttons_corner(self, compact=False):
        """运行控制按钮组；compact=True 用于右侧任务配置卡片。"""
        scale = self._compute_font_scale() if hasattr(
            self, "_compute_font_scale") else 1.0
        compact_h = max(34, int(round(34 * min(scale, 1.15))))
        row = QWidget()
        row.setObjectName("runButtonsCorner")
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)

        self.start_btn = QPushButton("开始检测")
        self.start_btn.setObjectName("runStartBtn")
        self._set_btn_icon(self.start_btn, "play", "#ffffff")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setProperty("variant", "skyPrimary")
        self.start_btn.setMinimumHeight(compact_h if compact else 38)
        if compact:
            self.start_btn.setMinimumWidth(0)
            self.start_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.start_btn.setMinimumWidth(120)
        self.start_btn.setToolTip("开始检测后可用暂停/停止；批量模式不支持暂停。")
        h.addWidget(self.start_btn, 1 if compact else 0)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setObjectName("runPauseBtn")
        self._set_btn_icon(self.pause_btn, "pause", "#6366f1")
        self.pause_btn.clicked.connect(self.pause_detection)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setProperty("variant", "secondary")
        self.pause_btn.setMinimumHeight(compact_h if compact else 38)
        if compact:
            self.pause_btn.setMinimumWidth(0)
            self.pause_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.pause_btn.setMinimumWidth(100)
        h.addWidget(self.pause_btn, 1 if compact else 0)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("runStopBtn")
        self._set_btn_icon(self.stop_btn, "square", "#f8fafc")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setProperty("variant", "stop")
        self.stop_btn.setMinimumHeight(compact_h if compact else 38)
        if compact:
            self.stop_btn.setMinimumWidth(0)
            self.stop_btn.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.stop_btn.setMinimumWidth(100)
        h.addWidget(self.stop_btn, 1 if compact else 0)
        return row

    def _create_progress_corner(self):
        """任务进度行：与右侧表单行风格统一。"""
        w = QWidget()
        w.setObjectName("canvasProgressCorner")
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)
        self.progress_title_label = self._toolbar_label("任务进度", 64)
        h.addWidget(self.progress_title_label, 0, Qt.AlignVCenter)
        self.progress_bar.setMinimumWidth(0)
        self.progress_bar.setMaximumWidth(16777215)
        h.addWidget(self.progress_bar, 1, Qt.AlignVCenter)
        return w

    def _create_top_controls_corner(self):
        """任务控制区：操作按钮 + 进度反馈，风格与下方配置统一。"""
        w = QWidget()
        w.setObjectName("canvasTopControlsCorner")
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        w.setMinimumHeight(0)
        h = QVBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        run_row = self._create_run_buttons_corner(compact=True)
        self._top_run_row = run_row
        run_row.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Fixed)
        h.addWidget(run_row, 0)
        h.addWidget(self._create_progress_corner(), 0)
        return w

    def _build_right_sidebar(self):
        """右侧栏：任务概览 + 运行日志；配置入口统一放到顶部浮层。"""
        def align_row(r: QHBoxLayout):
            r.setSpacing(12)
            r.setContentsMargins(0, 0, 0, 0)

        col = QWidget()
        col.setObjectName("analysisSidebar")
        col.setMinimumWidth(312)
        col.setMaximumWidth(448)
        col.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        main_v = QVBoxLayout(col)
        main_v.setContentsMargins(0, 0, 0, 0)
        main_v.setSpacing(0)

        sidebar_top = QWidget()
        sidebar_top_lay = QVBoxLayout(sidebar_top)
        sidebar_top_lay.setSpacing(12)
        sidebar_top_lay.setContentsMargins(0, 0, 0, 0)

        # —— 任务配置 ——
        task = QFrame()
        self.task_card = task
        task.setObjectName("wireframeCard")
        tv = QVBoxLayout(task)
        tv.setContentsMargins(0, 0, 0, 0)
        tv.setSpacing(0)
        tv.addWidget(self._wireframe_card_header("任务控制", "settings"))
        task_body = QWidget()
        tb = QVBoxLayout(task_body)
        tb.setContentsMargins(10, 12, 10, 12)
        tb.setSpacing(10)
        tb.addWidget(self._create_top_controls_corner(), 0)
        r_model = QHBoxLayout()
        align_row(r_model)
        r_model.addWidget(self._toolbar_label("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(96)
        self.model_combo.setMinimumHeight(32)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.init_model_combo()
        r_model.addWidget(self.model_combo, 1)
        self.advanced_model_btn = QPushButton("高级")
        self._set_btn_icon(self.advanced_model_btn, "settings", "#ffffff")
        self.advanced_model_btn.clicked.connect(
            self.show_model_selection_dialog)
        self.advanced_model_btn.setFixedWidth(90)
        self.advanced_model_btn.setMinimumHeight(32)
        self.advanced_model_btn.setProperty("variant", "skyPrimary")
        r_model.addWidget(self.advanced_model_btn)
        tb.addLayout(r_model)

        r_conf = QHBoxLayout()
        align_row(r_conf)
        r_conf.addWidget(self._toolbar_label("置信度", 56))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setObjectName("confThresholdSlider")
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setMinimumHeight(28)
        self.conf_slider.setMaximumHeight(28)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        r_conf.addWidget(self.conf_slider, 1)
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setSingleStep(0.01)
        self.conf_spinbox.setValue(0.25)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.valueChanged.connect(
            self.on_confidence_spinbox_changed)
        self.conf_spinbox.setFixedWidth(80)
        self.conf_spinbox.setMinimumHeight(32)
        r_conf.addWidget(self.conf_spinbox)
        tb.addLayout(r_conf)

        r_pre_top = QHBoxLayout()
        align_row(r_pre_top)
        r_pre_top.addWidget(self._toolbar_label("任务预设", 64))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumHeight(32)
        self.preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.currentTextChanged.connect(
            self.on_preset_selection_changed)
        r_pre_top.addWidget(self.preset_combo, 1)
        tb.addLayout(r_pre_top)
        r_pre_btn = QHBoxLayout()
        r_pre_btn.setSpacing(8)
        r_pre_btn.setContentsMargins(0, 0, 0, 0)
        self.new_preset_btn = QPushButton("新建预设")
        self._set_btn_icon(self.new_preset_btn, "folder_plus", "#ffffff")
        self.new_preset_btn.clicked.connect(self.create_new_preset)
        self.new_preset_btn.setProperty("variant", "skyPrimary")
        self.new_preset_btn.setMinimumHeight(34)
        self.new_preset_btn.setMinimumWidth(0)
        self.new_preset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.new_preset_btn.setToolTip("基于当前配置创建一个新任务预设")
        r_pre_btn.addWidget(self.new_preset_btn, 1)
        self.save_preset_btn = QPushButton("修改预设")
        self._set_btn_icon(self.save_preset_btn, "save", "#6366f1")
        self.save_preset_btn.clicked.connect(self.save_current_preset)
        self.save_preset_btn.setEnabled(False)
        self.save_preset_btn.setProperty("variant", "secondary")
        self.save_preset_btn.setMinimumHeight(34)
        self.save_preset_btn.setMinimumWidth(0)
        self.save_preset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.save_preset_btn.setToolTip("将当前配置保存到已选预设")
        r_pre_btn.addWidget(self.save_preset_btn, 1)
        self.delete_preset_btn = QPushButton("删除预设")
        self._set_btn_icon(self.delete_preset_btn, "trash", "#f8fafc")
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        self.delete_preset_btn.setEnabled(False)
        self.delete_preset_btn.setProperty("variant", "stop")
        self.delete_preset_btn.setMinimumHeight(34)
        self.delete_preset_btn.setMinimumWidth(0)
        self.delete_preset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.delete_preset_btn.setToolTip("删除当前选中的任务预设")
        r_pre_btn.addWidget(self.delete_preset_btn, 1)
        tb.addLayout(r_pre_btn)

        # —— 输入源 ——
        self.source_card = QFrame()
        self.source_card.setObjectName("sourceTopCard")
        sv = QVBoxLayout(self.source_card)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(0)
        sv.addWidget(self._wireframe_card_header("输入源", "link"))
        sec2 = QWidget()
        s2 = QVBoxLayout(sec2)
        s2.setSpacing(8)
        s2.setContentsMargins(10, 12, 10, 12)
        self.source_mode_row = QWidget()
        r2_mode = QHBoxLayout(self.source_mode_row)
        align_row(r2_mode)
        r2_mode.addWidget(self._toolbar_label("模式", 56))
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(88)
        self.source_combo.setMinimumHeight(32)
        self.source_combo.addItems(
            ["单张图片", "视频文件", "摄像头", "文件夹批量"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        r2_mode.addWidget(self.source_combo, 1)
        self.select_file_btn = QPushButton("浏览…")
        self._set_btn_icon(self.select_file_btn, "folder_open", "#6366f1")
        self.select_file_btn.clicked.connect(self.select_file)
        self.select_file_btn.setObjectName("toolBtn")
        self.select_file_btn.setMinimumHeight(34)
        self.select_file_btn.setMinimumWidth(80)
        self.select_file_btn.setToolTip("根据上方模式选择图片、视频、文件夹或批量目录")
        s2.addWidget(self.source_mode_row)

        self.camera_bar = QWidget()
        self.camera_select_layout = QHBoxLayout(self.camera_bar)
        self.camera_select_layout.setSpacing(8)
        self.camera_select_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_select_layout.addWidget(self._toolbar_label("摄像头", 56))
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(80)
        self.camera_combo.setMinimumHeight(32)
        self.refresh_camera_list()
        self.camera_combo.currentIndexChanged.connect(
            self._on_camera_selection_changed)
        self.camera_select_layout.addWidget(self.camera_combo, 1)
        refresh_camera_btn = QPushButton("刷新")
        self._set_btn_icon(refresh_camera_btn, "refresh", "#6366f1")
        refresh_camera_btn.setObjectName("toolBtn")
        refresh_camera_btn.setToolTip("重新扫描本机摄像头列表")
        refresh_camera_btn.setMinimumHeight(32)
        refresh_camera_btn.setMinimumWidth(64)
        refresh_camera_btn.clicked.connect(self.refresh_camera_list)
        self.camera_select_layout.addWidget(refresh_camera_btn)
        s2.addWidget(self.camera_bar)

        r2b = QHBoxLayout()
        align_row(r2b)
        self.current_file_title_label = QLabel("当前文件")
        self.current_file_title_label.setObjectName("toolbarFieldLabel")
        self.current_file_title_label.setAlignment(
            Qt.AlignLeft | Qt.AlignVCenter)
        self.current_file_title_label.setFixedWidth(64)
        r2b.addWidget(self.current_file_title_label)
        self.current_file_label = QLineEdit("未选择文件")
        self.current_file_label.setObjectName("pathReadonlyField")
        self.current_file_label.setReadOnly(True)
        self.current_file_label.setMinimumHeight(34)
        self.current_file_label.setCursorPosition(0)
        r2b.addWidget(self.current_file_label, 1)
        r2b.addWidget(self.select_file_btn, 0)
        s2.addLayout(r2b)

        # 输出目录：扁平化为单行，去掉额外边框与长说明文字
        r_sv = QHBoxLayout()
        align_row(r_sv)
        r_sv.addWidget(self._toolbar_label("输出目录", 64))
        self.save_dir_edit = QLineEdit(self.default_save_dir)
        self.save_dir_edit.setObjectName("pathEditableField")
        self.save_dir_edit.setPlaceholderText("检测结果保存目录")
        self.save_dir_edit.setMinimumHeight(34)
        self.save_dir_edit.setToolTip("批量分析与监控截图结果的默认保存位置")
        self.save_dir_edit.textChanged.connect(self.on_save_dir_changed)
        self.save_dir_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        r_sv.addWidget(self.save_dir_edit, 1)
        save_dir_btn = QPushButton("浏览…")
        self._set_btn_icon(save_dir_btn, "folder_open", "#6366f1")
        save_dir_btn.setObjectName("toolBtn")
        save_dir_btn.setToolTip("选择默认结果保存目录")
        save_dir_btn.setMinimumHeight(34)
        save_dir_btn.setMinimumWidth(80)
        save_dir_btn.clicked.connect(self.select_save_directory)
        r_sv.addWidget(save_dir_btn)
        s2.addLayout(r_sv)
        sv.addWidget(sec2)
        # —— 任务配置（恢复为右侧常驻，便于高频切换模型）——
        tv.addWidget(task_body)
        self.source_card.setVisible(False)
        sidebar_top_lay.addWidget(task)
        self._apply_card_shadow(task)

        # —— 任务概览（右侧新主区）——
        overview_card = QFrame()
        overview_card.setObjectName("wireframeCard")
        ov = QVBoxLayout(overview_card)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(0)
        ov.addWidget(self._wireframe_card_header("任务概览", "list"))
        overview_body = QWidget()
        ob = QVBoxLayout(overview_body)
        ob.setContentsMargins(10, 12, 10, 12)
        ob.setSpacing(8)

        mode_row = QHBoxLayout()
        align_row(mode_row)
        mode_row.addWidget(self._toolbar_label("当前模式", 64))
        self.overview_mode_value = QComboBox()
        self.overview_mode_value.setMinimumHeight(34)
        self.overview_mode_value.setToolTip("直接切换当前任务模式")
        self.overview_mode_value.currentTextChanged.connect(
            lambda t: self.source_combo.setCurrentText(t))
        mode_row.addWidget(self.overview_mode_value, 1)
        self.overview_mode_row = QWidget()
        self.overview_mode_row.setLayout(mode_row)
        ob.addWidget(self.overview_mode_row)
        camera_row = QHBoxLayout()
        align_row(camera_row)
        camera_row.addWidget(self._toolbar_label("摄像头", 64))
        self.overview_camera_combo = QComboBox()
        self.overview_camera_combo.setMinimumHeight(34)
        self.overview_camera_combo.currentIndexChanged.connect(
            lambda i: self.camera_combo.setCurrentIndex(i))
        camera_row.addWidget(self.overview_camera_combo, 1)
        self.overview_camera_refresh_btn = QPushButton("刷新")
        self._set_btn_icon(self.overview_camera_refresh_btn,
                           "refresh", "#6366f1")
        self.overview_camera_refresh_btn.setObjectName("toolBtn")
        self.overview_camera_refresh_btn.setMinimumHeight(32)
        self.overview_camera_refresh_btn.setMinimumWidth(64)
        self.overview_camera_refresh_btn.setToolTip("重新扫描本机摄像头列表")
        self.overview_camera_refresh_btn.clicked.connect(
            self.refresh_camera_list)
        camera_row.addWidget(self.overview_camera_refresh_btn, 0)
        self.overview_camera_row = QWidget()
        self.overview_camera_row.setLayout(camera_row)
        self.overview_camera_row.setVisible(False)
        ob.addWidget(self.overview_camera_row)
        file_row = QHBoxLayout()
        align_row(file_row)
        self.overview_file_title_label = self._toolbar_label("当前文件", 64)
        file_row.addWidget(self.overview_file_title_label)
        self.overview_file_value = QLineEdit("未选择文件")
        self.overview_file_value.setObjectName("pathReadonlyField")
        self.overview_file_value.setReadOnly(True)
        self.overview_file_value.setMinimumHeight(34)
        file_row.addWidget(self.overview_file_value, 1)
        self.overview_pick_btn = QPushButton("浏览…")
        self._set_btn_icon(self.overview_pick_btn, "folder_open", "#6366f1")
        self.overview_pick_btn.setObjectName("toolBtn")
        self.overview_pick_btn.setMinimumHeight(34)
        self.overview_pick_btn.setMinimumWidth(80)
        self.overview_pick_btn.clicked.connect(self.select_file)
        file_row.addWidget(self.overview_pick_btn, 0)
        ob.addLayout(file_row)

        out_row = QHBoxLayout()
        align_row(out_row)
        out_row.addWidget(self._toolbar_label("输出目录", 64))
        self.overview_outdir_edit = QLineEdit()
        self.overview_outdir_edit.setObjectName("pathEditableField")
        self.overview_outdir_edit.setMinimumHeight(34)
        self.overview_outdir_edit.textChanged.connect(
            lambda t: self.save_dir_edit.setText(t))
        out_row.addWidget(self.overview_outdir_edit, 1)
        self.overview_outdir_btn = QPushButton("浏览…")
        self._set_btn_icon(self.overview_outdir_btn, "folder_open", "#6366f1")
        self.overview_outdir_btn.setObjectName("toolBtn")
        self.overview_outdir_btn.setMinimumHeight(34)
        self.overview_outdir_btn.setMinimumWidth(80)
        self.overview_outdir_btn.clicked.connect(self.select_save_directory)
        out_row.addWidget(self.overview_outdir_btn, 0)
        ob.addLayout(out_row)
        metric_grid = QGridLayout()
        metric_grid.setContentsMargins(0, 4, 0, 0)
        metric_grid.setHorizontalSpacing(8)
        metric_grid.setVerticalSpacing(8)

        def metric_card(title: str):
            card = QFrame()
            card.setObjectName("overviewMetricCard")
            lay = QVBoxLayout(card)
            lay.setContentsMargins(10, 8, 10, 8)
            lay.setSpacing(4)
            t = QLabel(title)
            t.setObjectName("overviewMetricTitle")
            v = QLabel("—")
            v.setObjectName("overviewMetricValue")
            lay.addWidget(t)
            lay.addWidget(v)
            return card, v

        obj_card, self.overview_objects_value = metric_card("目标数")
        fps_card, self.overview_fps_value = metric_card("FPS")
        lat_card, self.overview_latency_value = metric_card("推理耗时")
        model_card, self.overview_model_metric_value = metric_card("当前模型")
        metric_grid.addWidget(obj_card, 0, 0)
        metric_grid.addWidget(fps_card, 0, 1)
        metric_grid.addWidget(lat_card, 1, 0)
        metric_grid.addWidget(model_card, 1, 1)
        ob.addLayout(metric_grid)

        # 将状态同步映射到右侧指标卡（替代顶部指标条）
        self._wf_stat_labels = {
            "objects": self.overview_objects_value,
            "fps": self.overview_fps_value,
            "latency": self.overview_latency_value,
            "model": self.overview_model_metric_value,
        }

        ov_btns = QHBoxLayout()
        ov_btns.setContentsMargins(0, 2, 0, 0)
        ov_btns.setSpacing(8)
        self.open_result_dir_btn = QPushButton("打开结果目录")
        self._set_btn_icon(self.open_result_dir_btn, "folder_open", "#6366f1")
        self.open_result_dir_btn.setProperty("variant", "secondary")
        self.open_result_dir_btn.setMinimumHeight(34)
        self.open_result_dir_btn.clicked.connect(self._open_output_dir)
        ov_btns.addWidget(self.open_result_dir_btn, 1)
        self.export_summary_btn = QPushButton("导出摘要")
        self._set_btn_icon(self.export_summary_btn, "save", "#6366f1")
        self.export_summary_btn.setProperty("variant", "secondary")
        self.export_summary_btn.setMinimumHeight(34)
        self.export_summary_btn.clicked.connect(self._export_task_summary)
        ov_btns.addWidget(self.export_summary_btn, 1)
        ob.addLayout(ov_btns)

        ov.addWidget(overview_body)
        sidebar_top_lay.addWidget(overview_card)
        self._apply_card_shadow(overview_card)

        # —— 运行日志：与任务配置、输入源同列、同滚动区域（右侧一整块）——
        log_card = QFrame()
        log_card.setObjectName("wireframeCard")
        log_card.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        lv = QVBoxLayout(log_card)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self._wireframe_card_header("运行日志", "list"))
        log_body = QWidget()
        log_body.setObjectName("wfLogSheetBody")
        lbl = QVBoxLayout(log_body)
        lbl.setContentsMargins(12, 12, 12, 12)
        lbl.setSpacing(8)
        self.log_text.setObjectName("wfLogText")
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("")
        self.log_text.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.log_text.setMinimumHeight(120)
        self.log_text.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.log_text.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_text.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        lbl.addWidget(self.log_text, 1)
        log_foot = QWidget()
        log_foot.setObjectName("wfLogFooter")
        lf = QHBoxLayout(log_foot)
        lf.setContentsMargins(0, 0, 0, 0)
        lf.addStretch()
        self.clear_log_btn = QPushButton("清除日志")
        self._set_btn_icon(self.clear_log_btn, "eraser", "#6366f1")
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.clear_log_btn.setProperty("variant", "secondary")
        self.clear_log_btn.setMinimumHeight(32)
        self.clear_log_btn.setMinimumWidth(88)
        lf.addWidget(self.clear_log_btn)
        lbl.addWidget(log_foot)
        lv.addWidget(log_body, 1)
        sidebar_top_lay.addWidget(log_card, 1)

        top_scroll = QScrollArea()
        top_scroll.setObjectName("analysisSidebarScroll")
        top_scroll.setWidgetResizable(True)
        top_scroll.setFrameShape(QFrame.Shape.NoFrame)
        top_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        top_scroll.setWidget(sidebar_top)

        main_v.addWidget(top_scroll, 1)
        return col

    def _wireframe_card_header(self, title: str, icon_name: str):
        """线框稿卡片顶栏：标题占位条 + 文案 + 右侧图标。"""
        head = QFrame()
        head.setObjectName("wireframeCardHeader")
        hl = QHBoxLayout(head)
        hl.setContentsMargins(20, 16, 20, 16)
        bar = QFrame()
        bar.setObjectName("wfTitleBar")
        bar.setFixedHeight(6)
        bar.setFixedWidth(96)
        hl.addWidget(bar, 0, Qt.AlignVCenter)
        tit = QLabel(title)
        tit.setObjectName("wfCardTitle")
        hl.addWidget(tit, 0, Qt.AlignVCenter)
        hl.addStretch()
        ic = QLabel()
        ic.setPixmap(ThemeIcons.pixmap(icon_name, 18, "#64748b"))
        hl.addWidget(ic, 0, Qt.AlignVCenter)
        return head

    def _init_status_defaults(self):
        self._status_model.setText("-")
        self._status_mode.setText("-")
        self._status_source.setText("-")
        self._status_fps.setText("-")
        self._status_latency.setText("-")
        self._status_objects.setText("-")

    def _refresh_wireframe_sidebar(self, results, class_names, inference_time):
        """根据检测结果刷新顶栏四项检测概要。"""
        if not hasattr(self, "_wf_stat_labels"):
            return
        try:
            self._wf_stat_labels["latency"].setText(
                f"{inference_time * 1000:.0f} ms")
            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                n = len(results[0].boxes)
                self._wf_stat_labels["objects"].setText(str(n))
            else:
                self._wf_stat_labels["objects"].setText("0")
        except Exception:
            pass
        try:
            mn = self.model_combo.currentText() if hasattr(
                self, "model_combo") else "-"
            self._wf_stat_labels["model"].setText(
                mn[:14] + ("…" if len(mn) > 14 else ""))
            self._wf_stat_labels["model"].setToolTip(mn)
        except Exception:
            pass

    def _sync_wireframe_overview_from_status(self):
        """将内部状态栏同步到顶栏四项概要。"""
        if not hasattr(self, "_wf_stat_labels"):
            return
        try:
            mn = self._status_model.text()
            self._wf_stat_labels["model"].setText(
                mn[:14] + ("…" if len(mn) > 14 else ""))
            self._wf_stat_labels["model"].setToolTip(mn)
            self._wf_stat_labels["objects"].setText(
                self._status_objects.text())
            self._wf_stat_labels["fps"].setText(self._status_fps.text())
            self._wf_stat_labels["latency"].setText(
                self._status_latency.text())
        except Exception:
            pass
        self._refresh_right_overview()

    def _refresh_right_overview(self):
        """同步右侧任务概览（只读信息），突出运行态，而非重复配置。"""
        if not hasattr(self, "overview_mode_value"):
            return
        try:
            # 模式与来源：使用状态栏里的短语义，避免与顶部配置重复
            mode_text = self._status_mode.text() if hasattr(self, "_status_mode") else "-"
            source_text = self._status_source.text() if hasattr(
                self, "_status_source") else "-"
            model_text = self._status_model.text() if hasattr(self, "_status_model") else "-"

            if hasattr(self, "source_combo"):
                self.overview_mode_value.blockSignals(True)
                self.overview_mode_value.clear()
                for i in range(self.source_combo.count()):
                    self.overview_mode_value.addItem(
                        self.source_combo.itemText(i))
                self.overview_mode_value.setCurrentText(
                    self.source_combo.currentText())
                self.overview_mode_value.blockSignals(False)
                # 单一模式时仍展示，但禁用下拉，避免冗余操作
                self.overview_mode_value.setEnabled(
                    self.source_combo.count() > 1 and self.source_combo.isEnabled()
                )
                self.overview_mode_value.setToolTip(
                    self.source_combo.currentText() or mode_text or "-")
                if hasattr(self, "overview_camera_combo"):
                    self.overview_camera_combo.blockSignals(True)
                    self.overview_camera_combo.clear()
                    for i in range(self.camera_combo.count()):
                        self.overview_camera_combo.addItem(
                            self.camera_combo.itemText(i), self.camera_combo.itemData(i))
                    self.overview_camera_combo.setCurrentIndex(
                        self.camera_combo.currentIndex())
                    self.overview_camera_combo.blockSignals(False)
                    self.overview_camera_combo.setEnabled(
                        self.source_combo.isEnabled())
                if hasattr(self, "overview_camera_refresh_btn"):
                    self.overview_camera_refresh_btn.setEnabled(
                        self.source_combo.isEnabled())
                if hasattr(self, "overview_camera_row"):
                    self.overview_camera_row.setVisible(
                        self.source_combo.currentText() == "摄像头")
                if hasattr(self, "overview_mode_row"):
                    # 设备监控模式下固定为摄像头，隐藏“当前模式”以减少冗余
                    in_monitor_tab = hasattr(
                        self, "tab_widget") and self.tab_widget.currentIndex() == 2
                    self.overview_mode_row.setVisible(not in_monitor_tab)
                if hasattr(self, "overview_file_title_label") and hasattr(self, "current_file_title_label"):
                    self.overview_file_title_label.setText(
                        self.current_file_title_label.text())
                if hasattr(self, "overview_file_value"):
                    self.overview_file_value.setText(
                        self.current_file_label.text() if hasattr(self, "current_file_label") else "-"
                    )
                    self.overview_file_value.setToolTip(
                        self.current_file_label.toolTip() if hasattr(
                            self, "current_file_label") else "-"
                    )
                if hasattr(self, "overview_pick_btn"):
                    self.overview_pick_btn.setVisible(
                        self.source_combo.currentText() != "摄像头")
                    can_pick = self.source_combo.isEnabled() and self.source_combo.currentText() != "摄像头"
                    self.overview_pick_btn.setEnabled(can_pick)
                if hasattr(self, "overview_outdir_edit") and hasattr(self, "save_dir_edit"):
                    self.overview_outdir_edit.blockSignals(True)
                    self.overview_outdir_edit.setText(
                        self.save_dir_edit.text())
                    self.overview_outdir_edit.blockSignals(False)
                if hasattr(self, "overview_outdir_btn"):
                    self.overview_outdir_btn.setEnabled(True)
            self.overview_model_metric_value.setText(model_text or "-")
            self.overview_model_metric_value.setToolTip(model_text or "-")
        except Exception:
            pass

    def _open_output_dir(self):
        """打开当前输出目录。"""
        target = self.save_dir_edit.text().strip() if hasattr(self, "save_dir_edit") else ""
        if not target:
            QMessageBox.information(self, "提示", "当前没有可打开的输出目录")
            return
        path = Path(target)
        if not path.exists():
            QMessageBox.warning(self, "提示", "输出目录不存在，请先设置有效路径")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _export_task_summary(self):
        """导出右侧任务概览为文本。"""
        default_name = f"task_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出任务摘要",
            str((base_dir / default_name).absolute()),
            "文本文件 (*.txt)",
        )
        if not save_path:
            return
        try:
            lines = [
                f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"当前模式: {self.overview_mode_value.currentText()}",
                f"当前来源: {self._status_source.text() if hasattr(self, '_status_source') else '-'}",
                f"当前模型: {self.overview_model_metric_value.text()}",
                f"目标数: {self.overview_objects_value.text() if hasattr(self, 'overview_objects_value') else '-'}",
                f"FPS: {self.overview_fps_value.text() if hasattr(self, 'overview_fps_value') else '-'}",
                f"推理耗时: {self.overview_latency_value.text() if hasattr(self, 'overview_latency_value') else '-'}",
            ]
            Path(save_path).write_text("\n".join(lines), encoding="utf-8")
            self.log_message(f"任务摘要已导出: {save_path}")
        except Exception as e:
            QMessageBox.warning(self, "导出失败", f"导出任务摘要失败：{e}")

    def _build_main_workspace(self):
        """左 — Tab 与运行按钮/任务进度同一行（角标），无「检测画布/运行控制」标题；右 — 配置侧栏。"""
        self._init_status_defaults()

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setObjectName("canvasProgressBar")

        self.tab_widget = QTabWidget()
        mtb = self.tab_widget.tabBar()
        mtb.setObjectName("mainTabBar")
        mtb.show()

        realtime_tab = self.create_realtime_tab()
        batch_tab = self.create_batch_tab()
        self.monitor_tab = MonitoringWidget(
            self.model_manager, self.camera_manager)
        self.tab_widget.addTab(
            realtime_tab, ThemeIcons.icon("radio", 17, "#0ea5e9"), "实时检测")
        self.tab_widget.addTab(
            batch_tab, ThemeIcons.icon("folders", 17, "#0ea5e9"), "批量分析")
        self.tab_widget.addTab(
            self.monitor_tab, ThemeIcons.icon("monitor", 17, "#0ea5e9"), "设备监控")

        canvas = QFrame()
        canvas.setObjectName("wireframeCanvasCard")
        cv = QVBoxLayout(canvas)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)
        cv.addWidget(self.tab_widget, 1)

        right = self._build_right_sidebar()
        self._sync_wireframe_overview_from_status()

        left_col = QWidget()
        left_col.setObjectName("mainLeftColumn")
        lv = QVBoxLayout(left_col)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(20)
        lv.addWidget(canvas, 1)
        self._main_left_column_layout = lv

        row = QWidget()
        rh = QHBoxLayout(row)
        rh.setContentsMargins(0, 0, 0, 0)
        rh.setSpacing(20)
        rh.addWidget(left_col, 16)
        rh.addWidget(right, 7)
        return row

    def create_realtime_tab(self):
        """创建实时检测标签页"""
        widget = QWidget()
        layout_top = QVBoxLayout(widget)
        layout_top.setContentsMargins(0, 0, 0, 0)
        layout_top.setSpacing(8)

        # 双画面行：占满剩余高度，左右 1:1
        image_row = QHBoxLayout()
        image_row.setSpacing(10)

        # 原图显示
        original_container = QWidget()
        original_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(0, 0, 0, 0)
        original_layout.setSpacing(6)

        original_layout.addWidget(
            self._panel_title_row("源画面", "image"))

        self.original_label = QLabel("暂无输入源\n请选择图片、视频或摄像头以开始预览")
        self.original_label.setObjectName("previewPlaceholder")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(240, 180)
        self.original_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.original_label.setScaledContents(False)
        original_layout.addWidget(self.original_label, 1)

        # 结果图显示
        result_container = QWidget()
        result_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout = QVBoxLayout(result_container)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(6)

        result_layout.addWidget(
            self._panel_title_row("检测结果", "chart"))

        self.result_label = QLabel("暂无结果\n开始检测后将显示框与类别标签")
        self.result_label.setObjectName("previewPlaceholder")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(240, 180)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.result_label.setScaledContents(False)
        result_layout.addWidget(self.result_label, 1)

        image_row.addWidget(original_container, 1)
        image_row.addWidget(result_container, 1)
        layout_top.addLayout(image_row, stretch=1)

        return widget

    def create_batch_tab(self):
        """创建批量结果标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 控制栏
        control_bar = QHBoxLayout()
        control_bar.addWidget(
            self._panel_title_row("批量检测结果", "folders"))
        control_bar.addStretch()

        # 导航按钮
        self.prev_result_btn = QPushButton("上一张")
        self._set_btn_icon(self.prev_result_btn, "chevron_left", "#6366f1")
        self.prev_result_btn.setProperty("variant", "secondary")
        self.prev_result_btn.clicked.connect(self.show_prev_result)
        self.prev_result_btn.setEnabled(False)
        control_bar.addWidget(self.prev_result_btn)

        self.result_index_label = QLabel("0/0")
        self.result_index_label.setStyleSheet(
            "font-weight: bold; margin: 0 10px;")
        control_bar.addWidget(self.result_index_label)

        self.next_result_btn = QPushButton("下一张")
        self._set_btn_icon(self.next_result_btn, "chevron_right", "#6366f1")
        self.next_result_btn.setProperty("variant", "secondary")
        self.next_result_btn.clicked.connect(self.show_next_result)
        self.next_result_btn.setEnabled(False)
        control_bar.addWidget(self.next_result_btn)

        # 保存按钮
        self.save_results_btn = QPushButton("保存结果")
        self._set_btn_icon(self.save_results_btn, "save", "#ffffff")
        self.save_results_btn.clicked.connect(self.save_batch_results)
        self.save_results_btn.setEnabled(False)
        control_bar.addWidget(self.save_results_btn)

        # 清空按钮
        self.clear_results_btn = QPushButton("清空结果")
        self._set_btn_icon(self.clear_results_btn, "trash", "#f8fafc")
        self.clear_results_btn.setProperty("variant", "stop")
        self.clear_results_btn.clicked.connect(self.clear_batch_results)
        self.clear_results_btn.setEnabled(False)
        control_bar.addWidget(self.clear_results_btn)

        layout.addLayout(control_bar)

        # 图像显示
        image_layout = QHBoxLayout()

        self.batch_original_label = QLabel("原图预览\n选择文件夹后运行检测")
        self.batch_original_label.setObjectName("batchPreviewPlaceholder")
        self.batch_original_label.setAlignment(Qt.AlignCenter)
        self.batch_original_label.setWordWrap(True)
        self.batch_original_label.setMinimumSize(280, 200)
        self.batch_original_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.batch_original_label.setScaledContents(False)

        self.batch_result_label = QLabel("结果预览\n检测完成后显示")
        self.batch_result_label.setObjectName("batchPreviewPlaceholder")
        self.batch_result_label.setAlignment(Qt.AlignCenter)
        self.batch_result_label.setWordWrap(True)
        self.batch_result_label.setMinimumSize(280, 200)
        self.batch_result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.batch_result_label.setScaledContents(False)

        image_layout.addWidget(self.batch_original_label, 1)
        image_layout.addWidget(self.batch_result_label, 1)
        layout.addLayout(image_layout, stretch=1)

        # 结果信息卡片
        self.batch_info_label = QLabel("选择文件夹开始批量检测。")
        self.batch_info_label.setObjectName("batchInfoCard")
        self.batch_info_label.setTextFormat(Qt.RichText)
        self.batch_info_label.setWordWrap(True)
        self.batch_info_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.batch_info_label)

        return widget

    def init_model_combo(self):
        """初始化模型下拉框"""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = self.model_manager.scan_models()

        if not models:
            self.model_combo.addItem("无可用模型")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.addItems([model['name'] for model in models])
            self.model_combo.setEnabled(True)
        self.model_combo.blockSignals(False)

    def try_load_default_model(self):
        """尝试加载默认模型"""
        if self.model_combo.count() > 0 and self.model_combo.itemText(0) != "无可用模型":
            first_model = self.model_combo.itemText(0)
            self.load_model_by_name(first_model)

    def load_model_by_name(self, model_name):
        """根据名称加载模型"""
        models = self.model_manager.scan_models()
        for model in models:
            if model['name'] == model_name:
                self.load_model(model['path'])
                break

    def load_model(self, model_path):
        """加载模型"""
        try:
            self.model = YOLO(model_path)
            self.log_message(f"模型加载成功: {Path(model_path).name}")
            # self.update_button_states()
            self._update_header_pills()
            return True
        except Exception as e:
            self.log_message(f"模型加载失败: {str(e)}")
            self.model = None
            self._update_header_pills()
            return False

    def show_model_selection_dialog(self):
        """显示模型选择对话框"""
        dialog = ModelSelectionDialog(self.model_manager, self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_model:
            model_name = Path(dialog.selected_model).name
            # 先更新下拉框，避免触发重复加载
            self.model_combo.blockSignals(True)
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                self.model_combo.addItem(model_name)
                self.model_combo.setCurrentText(model_name)
            self.model_combo.blockSignals(False)

            # 再加载模型
            if self.load_model(dialog.selected_model):
                pass  # 加载成功，已在load_model中记录日志

    def refresh_camera_list(self):
        """刷新摄像头列表"""
        self.camera_manager.scan_cameras()
        self.camera_combo.clear()

        cameras = self.camera_manager.get_available_cameras()
        if cameras:
            for camera in cameras:
                self.camera_combo.addItem(
                    f"{camera['name']} ({camera['resolution']})", camera['id'])
        else:
            self.camera_combo.addItem("无摄像头", -1)
            idx = self.camera_combo.count() - 1
            self.camera_combo.setItemData(
                idx, "未检测到可用摄像头，请检查设备或点击刷新", Qt.ToolTipRole)
        self._update_current_file_display()

    def _on_camera_selection_changed(self):
        """摄像头切换时同步当前设备显示。"""
        if getattr(self, "current_source_type", None) == "camera":
            if self.camera_combo.currentData() != -1:
                self._status_source.setText(self.camera_combo.currentText())
            else:
                self._status_source.setText("摄像头")
        self._update_current_file_display()

    def on_model_changed(self, model_text):
        """模型选择改变"""
        if model_text != "无可用模型":
            self.load_model_by_name(model_text)
            self._status_model.setText(model_text)
            self._sync_wireframe_overview_from_status()

    def on_confidence_changed(self, value):
        """置信度滑块改变"""
        conf_value = value / 100.0
        self.confidence_threshold = conf_value
        self.conf_spinbox.blockSignals(True)
        self.conf_spinbox.setValue(conf_value)
        self.conf_spinbox.blockSignals(False)

    def on_confidence_spinbox_changed(self, value):
        """置信度数值框改变"""
        self.confidence_threshold = value
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(int(value * 100))
        self.conf_slider.blockSignals(False)

    def on_source_changed(self, source_text):
        """检测源改变"""
        source_map = {
            "单张图片": "image",
            "视频文件": "video",
            "摄像头": "camera",
            "文件夹批量": "batch",
        }
        self.current_source_type = source_map.get(source_text)

        # 显示/隐藏摄像头选择
        is_camera = self.current_source_type == "camera"
        self.camera_bar.setVisible(is_camera)
        if hasattr(self, "select_file_btn"):
            # 摄像头模式不需要文件浏览按钮，避免语义冲突
            self.select_file_btn.setVisible(not is_camera)
        if hasattr(self, "current_file_title_label"):
            title_map = {
                "image": "当前文件",
                "video": "当前文件",
                "batch": "当前目录",
                "camera": "当前设备",
            }
            self.current_file_title_label.setText(
                title_map.get(self.current_source_type, "当前文件"))

        self.current_source_path = None
        self._current_file_full_path = None
        self._update_current_file_display()
        self.clear_display_windows()
        self.update_button_states()
        self._status_mode.setText(source_text)
        if self.current_source_type == "camera":
            self._status_source.setText("摄像头")
        else:
            self._status_source.setText("-")
        self._sync_wireframe_overview_from_status()

    def on_save_dir_changed(self, value):
        """更新默认保存目录"""
        path_text = value.strip()
        if path_text:
            self.default_save_dir = path_text
        self._refresh_right_overview()

    def select_save_directory(self):
        """选择默认结果保存目录"""
        start_dir = self.default_save_dir if Path(
            self.default_save_dir).exists() else str(base_dir)
        directory = QFileDialog.getExistingDirectory(
            self, "选择默认结果保存目录", start_dir)
        if directory:
            self.save_dir_edit.setText(directory)
            self.log_message(f"默认结果保存目录已设置: {directory}")

    def _source_type_to_label(self, source_type):
        label_map = {
            "image": "单张图片",
            "video": "视频文件",
            "camera": "摄像头",
            "batch": "文件夹批量",
        }
        return label_map.get(source_type, "单张图片")

    def _collect_current_task_config(self):
        """收集当前任务配置"""
        return {
            "model_name": self.model_combo.currentText(),
            "confidence_threshold": float(self.confidence_threshold),
            "source_type": self.current_source_type,
            "source_path": self.current_source_path,
            "save_dir": self.default_save_dir,
            "camera_id": self.camera_combo.currentData()
        }

    def _write_task_presets_to_file(self):
        """将预设写入本地文件"""
        try:
            with open(self.task_preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.preset_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.log_message(f"保存预设文件失败: {str(e)}")
            return False

    def _load_task_presets_from_file(self):
        """从本地文件加载预设"""
        if not self.task_preset_file.exists():
            self.preset_data = {}
            return

        try:
            with open(self.task_preset_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                self.preset_data = loaded_data if isinstance(
                    loaded_data, dict) else {}
        except Exception as e:
            self.preset_data = {}
            self.log_message(f"预设文件读取失败，已重置: {str(e)}")

    def _refresh_preset_combo(self):
        """刷新预设下拉列表"""
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("请选择预设")
        for preset_name in sorted(self.preset_data.keys()):
            self.preset_combo.addItem(preset_name)
        self.preset_combo.blockSignals(False)
        self._update_preset_action_buttons()

    def _update_preset_action_buttons(self):
        """根据当前预设选择更新按钮状态"""
        current_name = self.preset_combo.currentText().strip()
        has_valid_selection = (
            current_name
            and current_name != "请选择预设"
            and current_name in self.preset_data
        )
        self.save_preset_btn.setEnabled(has_valid_selection)
        self.delete_preset_btn.setEnabled(has_valid_selection)
        # 切换预设或失去有效选择时，退出“确认删除”状态
        if not has_valid_selection or self._delete_confirm_target != current_name:
            self._delete_confirm_target = None
            self.delete_preset_btn.setText("删除预设")
            self.delete_preset_btn.setToolTip("删除当前选中的任务预设")
            self.delete_preset_btn.setStyleSheet("")

    def create_new_preset(self):
        """快速新建预设"""
        dialog = QInputDialog(self)
        dialog.setWindowTitle("新建任务预设")
        dialog.setLabelText("请输入新预设名称:")
        dialog.setInputMode(QInputDialog.InputMode.TextInput)
        dialog.setTextValue("")
        dialog.setOkButtonText("确定")
        dialog.setCancelButtonText("取消")
        self._place_dialog_near_widget(dialog, self.new_preset_btn)
        if dialog.exec() != QDialog.Accepted:
            return
        preset_name = dialog.textValue().strip()
        if not preset_name:
            return

        if preset_name in self.preset_data:
            QMessageBox.information(
                self, "提示", f"预设 '{preset_name}' 已存在，请直接修改后点击“更新预设”覆盖。")
            self.preset_combo.setCurrentText(preset_name)
            return

        config = self._collect_current_task_config()
        self.preset_data[preset_name] = config
        if self._write_task_presets_to_file():
            self._refresh_preset_combo()
            self.preset_combo.setCurrentText(preset_name)
            self.log_message(f"新任务预设已创建: {preset_name}")

    def save_current_preset(self):
        """更新当前选中的预设"""
        current_name = self.preset_combo.currentText().strip()
        is_existing_selection = (
            current_name
            and current_name != "请选择预设"
            and current_name in self.preset_data
        )
        if not is_existing_selection:
            QMessageBox.information(self, "提示", "请先选择一个已有预设，再执行更新。")
            self._update_preset_action_buttons()
            return

        config = self._collect_current_task_config()
        preset_name = current_name
        self.preset_data[preset_name] = config
        if self._write_task_presets_to_file():
            self._refresh_preset_combo()
            self.preset_combo.setCurrentText(preset_name)
            self.log_message(f"任务预设已更新: {preset_name}")

    def on_preset_selection_changed(self, preset_name):
        """选择预设后自动应用"""
        preset_name = preset_name.strip()
        self._update_preset_action_buttons()
        if not preset_name or preset_name == "请选择预设":
            self._refresh_right_overview()
            return
        if preset_name not in self.preset_data:
            self._refresh_right_overview()
            return
        self.apply_selected_preset(show_message=False)
        self._refresh_right_overview()

    def apply_selected_preset(self, show_message=True):
        """应用当前选中的任务预设"""
        preset_name = self.preset_combo.currentText()
        if not preset_name or preset_name == "请选择预设":
            if show_message:
                QMessageBox.information(self, "提示", "请先选择一个预设")
            return

        config = self.preset_data.get(preset_name)
        if not config:
            if show_message:
                QMessageBox.warning(self, "警告", "预设内容无效")
            return

        model_name = config.get("model_name", "")
        if model_name:
            index = self.model_combo.findText(model_name)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            else:
                self.log_message(f"预设模型未在本地列表中找到: {model_name}")

        confidence = float(config.get("confidence_threshold", 0.25))
        confidence = max(0.01, min(1.0, confidence))
        self.conf_spinbox.setValue(confidence)

        source_type = config.get("source_type", "image")
        if hasattr(self, "tab_widget"):
            if source_type == "batch":
                self.tab_widget.setCurrentIndex(1)
            elif source_type in ("image", "video"):
                self.tab_widget.setCurrentIndex(0)
        source_label = self._source_type_to_label(source_type)
        source_index = self.source_combo.findText(source_label)
        if source_index >= 0:
            self.source_combo.setCurrentIndex(source_index)

        source_path = config.get("source_path")
        if source_path:
            path_obj = Path(source_path)
            if path_obj.exists():
                self.current_source_path = source_path
                self._current_file_full_path = source_path
                self._update_current_file_display()
                if self.current_source_type in ["image", "video"]:
                    self.preview_file(source_path)
            else:
                self.log_message(f"预设源路径不存在: {source_path}")

        save_dir = config.get("save_dir")
        if save_dir:
            self.save_dir_edit.setText(save_dir)

        preset_camera_id = config.get("camera_id")
        if self.current_source_type == "camera" and preset_camera_id is not None:
            cam_index = self.camera_combo.findData(preset_camera_id)
            if cam_index >= 0:
                self.camera_combo.setCurrentIndex(cam_index)

        self.update_button_states()
        self.log_message(f"任务预设已应用: {preset_name}")

    def delete_selected_preset(self):
        """删除当前选中的任务预设"""
        preset_name = self.preset_combo.currentText()
        if not preset_name or preset_name == "请选择预设":
            QMessageBox.information(self, "提示", "请先选择一个预设")
            return

        # 二次确认：首次点击进入确认态，二次点击执行删除
        if self._delete_confirm_target != preset_name:
            self._delete_confirm_target = preset_name
            self.delete_preset_btn.setText("确认删除")
            self.delete_preset_btn.setToolTip(f"再次点击删除预设「{preset_name}」")
            self.delete_preset_btn.setStyleSheet(
                "background:#dc2626;color:#ffffff;border:1px solid #b91c1c;border-radius:12px;"
            )
            return

        self.preset_data.pop(preset_name, None)
        if self._write_task_presets_to_file():
            self._delete_confirm_target = None
            self.delete_preset_btn.setText("删除预设")
            self.delete_preset_btn.setToolTip("删除当前选中的任务预设")
            self.delete_preset_btn.setStyleSheet("")
            self._refresh_preset_combo()
            self.log_message(f"任务预设已删除: {preset_name}")

    def eventFilter(self, obj, event):
        """当处于“确认删除”状态时，点击其它区域会自动取消高亮。"""
        try:
            if (
                self._delete_confirm_target
                and event.type() == QEvent.MouseButtonPress
                and hasattr(self, "delete_preset_btn")
            ):
                global_pos = event.globalPos()
                btn_pos = self.delete_preset_btn.mapFromGlobal(global_pos)
                if not self.delete_preset_btn.rect().contains(btn_pos):
                    # 点击在删除按钮之外：还原为普通状态
                    self._delete_confirm_target = None
                    self.delete_preset_btn.setText("删除预设")
                    self.delete_preset_btn.setToolTip("删除当前选中的任务预设")
                    self.delete_preset_btn.setStyleSheet("")
        except Exception:
            pass

        return super().eventFilter(obj, event)

    def update_button_states(self):
        """更新按钮状态"""
        has_model = self.model is not None

        if self.current_source_type == "camera":
            has_source = self.camera_combo.currentData() != -1
            self.select_file_btn.setEnabled(False)
        else:
            has_source = self.current_source_path is not None
            self.select_file_btn.setEnabled(True)

        self.start_btn.setEnabled(has_model and has_source)

    def select_file(self):
        """选择文件或文件夹"""
        if self.current_source_type == "image":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "",
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.webp);;所有文件 (*)"
            )
        elif self.current_source_type == "video":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "",
                "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;所有文件 (*)"
            )
        elif self.current_source_type == "batch":
            file_path = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        else:
            return

        if file_path:
            self.current_source_path = file_path
            self._current_file_full_path = file_path
            self._update_current_file_display()
            self.log_message(f"已选择: {file_path}")
            self.update_button_states()
            self._status_source.setText(Path(file_path).name)
            self._sync_wireframe_overview_from_status()

            if self.current_source_type in ["image", "video"]:
                self.preview_file(file_path)

    def preview_file(self, file_path):
        """预览文件"""
        try:
            if self.current_source_type == "image":
                img = cv2.imread(file_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self._last_preview_original = np.ascontiguousarray(img_rgb)
                    self._last_preview_result = None
                    self.display_image(img_rgb, self.original_label)
                    self.result_label.clear()
                    self.result_label.setText("等待检测结果...")
        except Exception as e:
            self.log_message(f"预览文件失败: {str(e)}")

    def start_detection(self):
        """开始检测"""
        self._set_progress_state("running")
        if hasattr(self, "progress_bar"):
            self.progress_bar.setValue(0)
        if self.tab_widget.currentIndex() == 2:
            if not self.model:
                self.log_message("错误: 模型未加载")
                return
            camera_ids = self._get_monitor_camera_ids()
            if not camera_ids:
                self.log_message("错误: 请在右侧输入源选择可用摄像头")
                return
            self.monitor_tab.set_shared_model(self.model)
            self.monitor_tab.start_monitoring(
                shared_model=self.model, camera_ids=camera_ids)
            if self.monitor_tab.is_monitoring_active():
                self.update_detection_ui_state(True)
                self.log_message("开始设备监控…")
            return

        if not self.model:
            self.log_message("错误: 模型未加载")
            return

        if self.current_source_type == "batch":
            self.start_batch_detection()
        else:
            self.start_single_detection()

    def start_single_detection(self):
        """开始单个检测"""
        camera_id = 0
        if self.current_source_type == "camera":
            camera_id = self.camera_combo.currentData()
            if camera_id == -1:
                self.log_message("错误: 没有可用的摄像头")
                return

        self.detection_thread = DetectionThread(
            self.model, self.current_source_type, self.current_source_path, camera_id, self.confidence_threshold
        )
        self.detection_thread.result_ready.connect(self.on_detection_result)
        self.detection_thread.progress_updated.connect(
            self._on_main_progress)
        self.detection_thread.status_changed.connect(
            self.statusBar().showMessage)
        self.detection_thread.error_occurred.connect(self.log_message)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.fps_updated.connect(self._on_fps_updated)

        self.update_detection_ui_state(True)
        self.tab_widget.setCurrentIndex(0)  # 切换到实时检测

        self.detection_thread.start()
        self.log_message(f"开始{self.current_source_type}检测…")

    def start_batch_detection(self):
        """开始批量检测"""
        self.batch_results.clear()

        self.batch_detection_thread = BatchDetectionThread(
            self.model, self.current_source_path, self.confidence_threshold
        )
        self.batch_detection_thread.result_ready.connect(self.on_batch_result)
        self.batch_detection_thread.progress_updated.connect(
            self._on_main_progress)
        self.batch_detection_thread.current_file_changed.connect(
            self.statusBar().showMessage)
        self.batch_detection_thread.finished.connect(self.on_batch_finished)

        self.update_detection_ui_state(True)
        self.tab_widget.setCurrentIndex(1)  # 切换到批量结果

        self.batch_detection_thread.start()
        self.log_message("开始批量检测…")

    def update_detection_ui_state(self, detecting):
        """更新检测状态的UI"""
        is_monitor_tab = self.tab_widget.currentIndex() == 2
        self.start_btn.setEnabled(not detecting)
        self.pause_btn.setEnabled(
            detecting and (is_monitor_tab or self.current_source_type != "batch"))
        self.stop_btn.setEnabled(detecting)
        self.source_combo.setEnabled(not detecting and not is_monitor_tab)
        self.select_file_btn.setEnabled(
            not detecting and not is_monitor_tab and self.current_source_type != "camera")
        self.model_combo.setEnabled(not detecting)
        self._update_header_pills()

    def pause_detection(self):
        """暂停/恢复检测"""
        if self.tab_widget.currentIndex() == 2:
            if self.monitor_tab.is_monitoring_active():
                self.monitor_tab.stop_monitoring()
                if self.monitor_tab.is_monitoring_paused():
                    self.pause_btn.setText("继续")
                    self._set_btn_icon(self.pause_btn, "play", "#6366f1")
                    self.log_message("监控已暂停")
                else:
                    self.pause_btn.setText("暂停")
                    self._set_btn_icon(self.pause_btn, "pause", "#6366f1")
                    self.log_message("监控已恢复")
            return

        if self.detection_thread and self.detection_thread.is_running:
            if self.detection_thread.is_paused:
                self.detection_thread.resume()
                self.pause_btn.setText("暂停")
                self._set_btn_icon(self.pause_btn, "pause", "#6366f1")
                self.log_message("检测已恢复")
            else:
                self.detection_thread.pause()
                self.pause_btn.setText("继续")
                self._set_btn_icon(self.pause_btn, "play", "#6366f1")
                self.log_message("检测已暂停")

    def stop_detection(self):
        """停止检测"""
        if self.tab_widget.currentIndex() == 2:
            self.monitor_tab.clear_monitoring()
            self.on_detection_finished()
            self.log_message("设备监控已停止")
            return

        if self.detection_thread and self.detection_thread.is_running:
            self.detection_thread.stop()
            self.detection_thread.wait()

        if self.batch_detection_thread and self.batch_detection_thread.is_running:
            self.batch_detection_thread.stop()
            self.batch_detection_thread.wait()

        self.on_detection_finished()

    def on_detection_result(self, original_img, result_img, inference_time, results, class_names):
        """检测结果回调"""
        if original_img is not None:
            self._last_preview_original = np.ascontiguousarray(original_img)
        if result_img is not None:
            self._last_preview_result = np.ascontiguousarray(result_img)
        # 显示图像
        self.display_image(original_img, self.original_label)
        self.display_image(result_img, self.result_label)

        # 更新结果详情
        self.result_detail_widget.update_results(
            results, class_names, inference_time)
        self._refresh_wireframe_sidebar(results, class_names, inference_time)

        # 记录日志（简化版，避免过多输出）
        if results and results[0].boxes and len(results[0].boxes) > 0:
            object_count = len(results[0].boxes)
            self._status_objects.setText(str(object_count))
            self._status_latency.setText(f"{inference_time*1000:.0f} 毫秒")

            # 统计类别
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_counts = {}
            for cls in classes:
                class_name = class_names[cls] if cls < len(
                    class_names) else f"类别{cls}"
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            class_summary = ", ".join(
                [f"{name}:{count}" for name, count in class_counts.items()])
            self.log_message(
                f"检测到 {object_count} 个目标: {class_summary}（耗时 {inference_time:.3f} 秒）")
        else:
            self.log_message(f"未检测到目标（耗时 {inference_time:.3f} 秒）")
            self._status_objects.setText("0")
            self._status_latency.setText(f"{inference_time*1000:.0f} 毫秒")

    def _on_fps_updated(self, fps):
        try:
            self._status_fps.setText(f"{fps:.1f}")
        except Exception:
            self._status_fps.setText("-")
        self._sync_wireframe_overview_from_status()

    def on_batch_result(self, file_path, original_img, result_img, inference_time, results, class_names):
        """批量检测结果回调"""
        # 计算目标数量
        object_count = len(
            results[0].boxes) if results and results[0].boxes else 0

        # 保存结果
        result_data = {
            'file_path': file_path,
            'original_img': original_img,
            'result_img': result_img,
            'inference_time': inference_time,
            'results': results,
            'class_names': class_names,
            'object_count': object_count
        }

        self.batch_results.append(result_data)

        # 显示第一个结果
        if len(self.batch_results) == 1:
            self.current_batch_index = 0
            self.show_batch_result(0)

        self.update_batch_navigation()

        # 记录日志
        filename = Path(file_path).name
        if object_count > 0:
            self.log_message(
                f"{filename}: {object_count} 个目标（{inference_time:.3f} 秒）")
        else:
            self.log_message(f"{filename}: 无目标（{inference_time:.3f} 秒）")

    def on_batch_finished(self):
        """批量检测完成"""
        total_count = len(self.batch_results)
        total_objects = sum(result['object_count']
                            for result in self.batch_results)

        self.log_message(
            f"批量检测完成：共 {total_count} 张图片，检出 {total_objects} 个目标")
        self.statusBar().showMessage(
            f"批量检测完成 - {total_count} 张图片，{total_objects} 个目标")

        self.save_results_btn.setEnabled(True)
        self.clear_results_btn.setEnabled(True)
        self.result_index_label.setText(f"1/{len(self.batch_results)}")
        self.on_detection_finished(completed=True)

    def on_detection_finished(self, completed=False):
        """检测完成回调"""
        if completed or self.progress_bar.value() >= 100:
            self.progress_bar.setValue(100)
            self._set_progress_state("done")
        else:
            self.progress_bar.setValue(0)
            self._set_progress_state("idle")
        self.update_detection_ui_state(False)
        self.pause_btn.setText("暂停")
        self._set_btn_icon(self.pause_btn, "pause", "#6366f1")

    def show_batch_result(self, index):
        """显示批量结果"""
        if 0 <= index < len(self.batch_results):
            result = self.batch_results[index]

            self.display_image(result['original_img'],
                               self.batch_original_label)
            self.display_image(result['result_img'], self.batch_result_label)

            filename = Path(result['file_path']).name
            object_count = result['object_count']
            inference_time = result['inference_time']

            info_lines = []
            info_lines.append(f"<b>文件</b>：{filename}")
            info_lines.append(f"<b>检测目标</b>：{object_count} 个")
            info_lines.append(f"<b>推理耗时</b>：{inference_time:.3f} 秒")

            if result['results'] and result['results'][0].boxes and len(result['results'][0].boxes) > 0:
                # 显示类别统计
                classes = result['results'][0].boxes.cls.cpu(
                ).numpy().astype(int)
                confidences = result['results'][0].boxes.conf.cpu().numpy()

                class_counts = {}
                for cls in classes:
                    class_name = result['class_names'][cls] if cls < len(
                        result['class_names']) else f"类别{cls}"
                    class_counts[class_name] = class_counts.get(
                        class_name, 0) + 1

                class_summary = ", ".join(
                    [f"{name}:{count}" for name, count in class_counts.items()])
                info_lines.append(f"<b>类别统计</b>：{class_summary}")
                info_lines.append(f"<b>平均置信度</b>：{np.mean(confidences):.3f}")

            self.batch_info_label.setText("<br/>".join(info_lines))
            self.result_index_label.setText(
                f"{index + 1}/{len(self.batch_results)}")

    def show_prev_result(self):
        """显示上一个结果"""
        if self.current_batch_index > 0:
            self.current_batch_index -= 1
            self.show_batch_result(self.current_batch_index)
            self.update_batch_navigation()

    def show_next_result(self):
        """显示下一个结果"""
        if self.current_batch_index < len(self.batch_results) - 1:
            self.current_batch_index += 1
            self.show_batch_result(self.current_batch_index)
            self.update_batch_navigation()

    def update_batch_navigation(self):
        """更新批量结果导航"""
        has_results = len(self.batch_results) > 0
        self.prev_result_btn.setEnabled(
            has_results and self.current_batch_index > 0)
        self.next_result_btn.setEnabled(
            has_results and self.current_batch_index < len(self.batch_results) - 1)

    def clear_batch_results(self):
        self.batch_results.clear()
        self.batch_result_label.clear()
        self.batch_original_label.clear()
        self.batch_result_label.setText("结果预览\n检测完成后显示")
        self.batch_original_label.setText("原图预览\n选择文件夹后运行检测")
        self.batch_info_label.setText("选择文件夹开始批量检测。")
        self.result_index_label.setText("0/0")
        self.save_results_btn.setEnabled(False)
        self.next_result_btn.setEnabled(False)
        self.prev_result_btn.setEnabled(False)
        self.clear_results_btn.setEnabled(False)

    def save_batch_results(self):
        """保存批量检测结果"""
        if not self.batch_results:
            QMessageBox.information(self, "提示", "没有可保存的结果")
            return

        start_dir = self.default_save_dir if Path(
            self.default_save_dir).exists() else str(base_dir)
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", start_dir)
        if not save_dir:
            return

        try:
            save_path = Path(save_dir)
            self.default_save_dir = str(save_path)
            self.save_dir_edit.setText(self.default_save_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = save_path / f"detection_results_{timestamp}"
            result_dir.mkdir(exist_ok=True)

            # 保存检测结果图片
            for i, result in enumerate(self.batch_results):
                file_name = Path(result['file_path']).stem
                result_img = cv2.cvtColor(
                    result['result_img'], cv2.COLOR_RGB2BGR)
                result_save_path = result_dir / f"{file_name}_result.jpg"
                cv2.imwrite(str(result_save_path), result_img)

            # 保存检测报告
            self.save_detection_report(result_dir)

            QMessageBox.information(self, "成功", f"结果已保存到:\n{result_dir}")
            self.log_message(f"结果已保存到: {result_dir}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            self.log_message(f"保存失败: {str(e)}")

    def save_detection_report(self, result_dir):
        """保存检测报告"""
        report_path = result_dir / "detection_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("🎯 Enhanced Object Detection System - 批量检测报告\n")
            f.write("=" * 60 + "\n")
            f.write(
                f"📅 处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🎚️ 置信度阈值: {self.confidence_threshold}\n")
            f.write(f"📂 处理图片数量: {len(self.batch_results)}\n")
            f.write(
                f"🎯 总检测目标数: {sum(r['object_count'] for r in self.batch_results)}\n")
            f.write("\n📊 详细结果:\n")
            f.write("-" * 60 + "\n")

            for i, result in enumerate(self.batch_results, 1):
                f.write(f"{i}. 📁 {Path(result['file_path']).name}\n")
                f.write(f"   🎯 检测目标: {result['object_count']} 个\n")
                f.write(f"   ⏱️ 推理耗时: {result['inference_time']:.3f} 秒\n")

                if result['results'] and result['results'][0].boxes and len(result['results'][0].boxes) > 0:
                    confidences = result['results'][0].boxes.conf.cpu().numpy()
                    classes = result['results'][0].boxes.cls.cpu(
                    ).numpy().astype(int)

                    f.write(
                        f"   📈 置信度范围: {np.min(confidences):.3f} - {np.max(confidences):.3f}\n")

                    # 类别统计
                    class_counts = {}
                    for cls in classes:
                        class_name = result['class_names'][cls] if cls < len(
                            result['class_names']) else f"类别{cls}"
                        class_counts[class_name] = class_counts.get(
                            class_name, 0) + 1

                    f.write("   📊 类别分布: " + ", ".join(
                        [f"{name}:{count}" for name, count in class_counts.items()]) + "\n")

                f.write("\n")

    def clear_display_windows(self):
        """清空显示窗口"""
        self._last_preview_original = None
        self._last_preview_result = None
        self.original_label.clear()
        self.original_label.clear()
        self.result_label.clear()
        self.original_label.setText("暂无输入源\n请选择图片、视频或摄像头以开始预览")
        self.result_label.setText("暂无结果\n开始检测后将显示框与类别标签")

    def display_image(self, img_array, label):
        """显示图像"""
        if img_array is None:
            return

        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        lw = max(label.width(), 1)
        lh = max(label.height(), 1)
        scaled_pixmap = pixmap.scaled(
            lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """窗口缩放时按等比例重绘最后一帧预览"""
        super().resizeEvent(event)
        if hasattr(self, "_font_resize_timer"):
            self._font_resize_timer.start(90)
        if self._last_preview_original is not None:
            self.display_image(self._last_preview_original,
                               self.original_label)
        if self._last_preview_result is not None:
            self.display_image(self._last_preview_result, self.result_label)

    def log_message(self, message):
        """添加日志消息（[时:分:秒] 前缀，与标注稿一致）"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{timestamp}] {message}")

        # 限制日志行数
        max_lines = 1000
        lines = self.log_text.toPlainText().split("\n")
        if len(lines) > max_lines:
            keep_lines = lines[-500:]
            self.log_text.setPlainText("\n".join(keep_lines))

        # 自动滚动到底部
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        self.log_message("日志已清除")

    def showEvent(self, event):
        """窗口显示后加载任务预设"""
        super().showEvent(event)
        self._apply_ui_font_scale()
        if not hasattr(self, "_presets_initialized"):
            self._load_task_presets_from_file()
            self._refresh_preset_combo()
            self._presets_initialized = True

    def create_enhanced_icon(self, size=64):
        """创建增强的应用图标"""
        icon = QIcon()
        icon.addFile("./assets/icons/dimension_logo.png")

        return icon


def main():
    app = QApplication(sys.argv)
    app.setFont(StyleManager.application_ui_font())

    # 设置应用程序信息
    app.setApplicationName("Dimension 目标检测系统")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Dimension Team")

    # 设置高DPI缩放
    # app.setAttribute(Qt.AA_EnableHighDpiScaling)
    # app.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 创建主窗口（首次启动默认最大化，占满工作区）
    window = EnhancedDetectionUI()
    window.showMaximized()

    # 启动消息
    window.log_message("Dimension 目标检测系统已启动")
    window.log_message("支持：实时检测、批量结果、多路监控、任务预设与运行日志")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
