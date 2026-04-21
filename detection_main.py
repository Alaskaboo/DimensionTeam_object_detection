import ast
import csv
import functools
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from ultralytics import YOLO

# 处理打包环境：资源目录与可写数据目录分离
if getattr(sys, 'frozen', False):
    # 打包环境：资源在 _MEIPASS，可写数据放在 exe 同级目录
    base_dir = Path(sys._MEIPASS)
    data_dir = Path(sys.executable).resolve().parent
else:
    # 开发环境：资源与数据都在项目根目录
    base_dir = Path(__file__).parent
    data_dir = base_dir

# 权重目录：统一在外部 models/ 下管理 .pt；子目录仅区分来源，列表扫描会递归收录。
MODELS_ROOT = data_dir / "models"
MODELS_DIR_CUSTOM = MODELS_ROOT / "custom"
MODELS_DIR_OFFICIAL = MODELS_ROOT / "official"
RESULTS_DIR = data_dir / "results"
LOGS_DIR = data_dir / "logs"

# 确保外部可写目录存在（直接运行 detection_main.py 时也能正常工作）
for _p in (MODELS_ROOT, MODELS_DIR_CUSTOM, MODELS_DIR_OFFICIAL, RESULTS_DIR, LOGS_DIR):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

from theme_icons import ThemeIcons
from task_history_store import TaskHistoryStore

_overlay_cjk_font_cache: dict[int, ImageFont.ImageFont] = {}


def _load_overlay_cjk_font(size_px: int) -> ImageFont.FreeTypeFont:
    """叠加层用字体：须支持中文路径；按字号缓存。"""
    size_px = max(10, min(40, int(size_px)))
    if size_px in _overlay_cjk_font_cache:
        return _overlay_cjk_font_cache[size_px]
    paths: list[str] = []
    bundled = base_dir / "assets" / "fonts" / "NotoSansSC-Regular.ttf"
    if bundled.is_file():
        paths.append(str(bundled))
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        paths.extend([
            os.path.join(windir, "Fonts", "msyh.ttc"),
            os.path.join(windir, "Fonts", "msyhbd.ttc"),
            os.path.join(windir, "Fonts", "simhei.ttf"),
            os.path.join(windir, "Fonts", "simsun.ttc"),
        ])
    elif sys.platform == "darwin":
        paths.extend([
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ])
    else:
        paths.extend([
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ])
    font = None
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        try:
            font = ImageFont.truetype(p, size_px, index=0)
            break
        except OSError:
            try:
                font = ImageFont.truetype(p, size_px)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _overlay_cjk_font_cache[size_px] = font
    return font


def _overlay_text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(0, bbox[2] - bbox[0])


def _fit_overlay_line_text(
        text: str, font, draw: ImageDraw.ImageDraw, max_width: int) -> str:
    if not text:
        return text
    if _overlay_text_width(draw, text, font) <= max_width:
        return text
    ell = "…"
    if _overlay_text_width(draw, ell, font) > max_width:
        return ell
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cand = text[:mid] + ell
        if _overlay_text_width(draw, cand, font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell if lo > 0 else ell


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


class ModelFileDownloadThread(QThread):
    """在后台线程下载 .pt，避免长时间占用 GUI 主线程导致界面卡顿。"""

    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, url: str, save_path: Path, chunk_size: int = 512 * 1024):
        super().__init__()
        self._url = url
        self._save_path = Path(save_path)
        self._chunk_size = max(8192, int(chunk_size))

    def run(self):
        part_path = self._save_path.with_suffix(self._save_path.suffix + ".part")
        try:
            with requests.get(
                self._url,
                stream=True,
                timeout=(30, 600),
            ) as response:
                response.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self._chunk_size):
                        if self.isInterruptionRequested():
                            raise RuntimeError("下载已取消")
                        if chunk:
                            f.write(chunk)
            if self.isInterruptionRequested():
                raise RuntimeError("下载已取消")
            if self._save_path.exists():
                self._save_path.unlink()
            part_path.replace(self._save_path)
            self.finished_ok.emit(str(self._save_path.resolve()))
        except Exception as e:
            try:
                if part_path.exists():
                    part_path.unlink()
            except OSError:
                pass
            self.failed.emit(str(e))


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
    OP_COL_W = 168
    OP_ROW_H = 38
    OP_BTN_H = 26

    COLUMN_HEADERS_LOCAL = ["模型名称", "大小", "修改时间", "路径"]
    COLUMN_HEADERS_NETWORK = ["模型名称", "大小(MB)", "修改时间", "类别数量", "状态", "操作"]
    COLUMN_HEADERS_OFFICIAL_NETWORK = [
        "模型名称", "大小(MB)", "修改时间", "类别数量", "状态", "操作"]
    PRIVATE_RELEASE_REPO = "Alaskaboo/DimensionTeam_object_detection"
    PRIVATE_RELEASE_TAG = "train_weights"

    def __init__(self, model_manager, parent=None):
        super().__init__(parent)
        self.model_manager = model_manager
        self.selected_model = None
        self.network_models = []
        self.official_network_models = []
        self._active_downloads = {}
        self.init_ui()
        self.load_network_models()
        self.load_official_network_models()

    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("高级模型选择")
        self.setModal(True)
        self.resize(900, 700)
        self.setObjectName("modelSelectionDialog")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # 创建标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setObjectName("modelSelectTabs")
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

        self._download_toast = QLabel("")
        self._download_toast.setObjectName("modelDownloadToast")
        self._download_toast.setWordWrap(True)
        self._download_toast.setVisible(False)
        layout.addWidget(self._download_toast)
        self._download_toast_timer = QTimer(self)
        self._download_toast_timer.setSingleShot(True)
        self._download_toast_timer.timeout.connect(
            lambda: self._download_toast.setVisible(False))

        # 按钮区域
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setObjectName("modelSelectButtonBox")
        ok_btn = button_box.button(QDialogButtonBox.Ok)
        cancel_btn = button_box.button(QDialogButtonBox.Cancel)
        if ok_btn is not None:
            ok_btn.setText("应用模型")
            ok_btn.setProperty("variant", "skyPrimary")
        if cancel_btn is not None:
            cancel_btn.setText("取消")
            cancel_btn.setProperty("variant", "secondary")
        layout.addWidget(button_box)

        self.setStyleSheet(StyleManager.get_main_stylesheet(1.0))
        self._apply_dialog_theme()

    def setup_local_tab(self):
        """设置本地模型标签页"""
        layout = QVBoxLayout(self.local_tab)

        # 路径选择组
        path_group = QGroupBox("自定义模型路径")
        path_group.setObjectName("modelSelectGroup")
        path_layout = QHBoxLayout(path_group)

        self.path_edit = QLineEdit()
        self.path_edit.setObjectName("modelSelectPathEdit")
        self.path_edit.setPlaceholderText("输入自定义模型目录路径...")
        path_layout.addWidget(self.path_edit)

        browse_btn = QPushButton("浏览")
        browse_btn.setProperty("variant", "skyPrimary")
        browse_btn.setIcon(ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_btn.setIconSize(QSize(18, 18))
        browse_btn.clicked.connect(self.browse_path)
        path_layout.addWidget(browse_btn)

        refresh_btn = QPushButton("刷新")
        refresh_btn.setProperty("variant", "skyPrimary")
        refresh_btn.setIcon(ThemeIcons.icon("refresh", 18, "#ffffff"))
        refresh_btn.setIconSize(QSize(18, 18))
        refresh_btn.clicked.connect(self.refresh_models)
        path_layout.addWidget(refresh_btn)

        layout.addWidget(path_group)

        # 模型列表组
        models_group = QGroupBox("可用模型")
        models_group.setObjectName("modelSelectGroup")
        models_layout = QVBoxLayout(models_group)

        self.model_table = self._create_table(
            self.COLUMN_HEADERS_LOCAL, 4, "modelSelectTable")
        self.model_table.doubleClicked.connect(self.accept)
        self.model_table.setMinimumHeight(450)
        header = self.model_table.horizontalHeader()
        header.setSectionResizeMode(
            self.MODEL_NAME_COL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(
            self.SIZE_COL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(
            self.MODIFIED_COL, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.PATH_COL, QHeaderView.Stretch)
        self.model_table.setTextElideMode(Qt.ElideMiddle)
        models_layout.addWidget(self.model_table)

        layout.addWidget(models_group)
        self.refresh_models()

    def setup_network_tab(self):
        """设置网络模型标签页"""
        layout = QVBoxLayout(self.network_tab)

        # 下载路径组
        path_group = QGroupBox("下载路径")
        path_group.setObjectName("modelSelectGroup")
        path_layout = QHBoxLayout(path_group)

        self.download_path_edit = QLineEdit()
        self.download_path_edit.setObjectName("modelSelectPathEdit")
        self.download_path_edit.setText(
            str(MODELS_DIR_CUSTOM.absolute()))
        self.download_path_edit.setPlaceholderText("模型下载目录路径...")
        path_layout.addWidget(self.download_path_edit)

        browse_download_btn = QPushButton("浏览")
        browse_download_btn.setProperty("variant", "skyPrimary")
        browse_download_btn.setIcon(
            ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_download_btn.setIconSize(QSize(18, 18))
        browse_download_btn.clicked.connect(self.browse_download_path)
        path_layout.addWidget(browse_download_btn)

        layout.addWidget(path_group)

        # 网络模型组
        models_group = QGroupBox("网络模型资源")
        models_group.setObjectName("modelSelectGroup")
        models_layout = QVBoxLayout(models_group)

        self.network_table = self._create_table(
            self.COLUMN_HEADERS_NETWORK, 6, "modelSelectTable")
        self.network_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.network_table.customContextMenuRequested.connect(
            self.show_network_context_menu)
        self.network_table.doubleClicked.connect(self.show_network_model_info)
        self.network_table.setMinimumHeight(450)
        n_header = self.network_table.horizontalHeader()
        n_header.setSectionResizeMode(self.MODEL_NAME_COL, QHeaderView.Stretch)
        n_header.setSectionResizeMode(
            self.SIZE_COL, QHeaderView.ResizeToContents)
        n_header.setSectionResizeMode(
            self.MODIFIED_COL, QHeaderView.ResizeToContents)
        n_header.setSectionResizeMode(
            self.STATUS_COL - 1, QHeaderView.ResizeToContents)
        n_header.setSectionResizeMode(
            self.STATUS_COL, QHeaderView.ResizeToContents)
        n_header.setSectionResizeMode(self.ACTION_COL, QHeaderView.Fixed)
        self.network_table.setColumnWidth(self.ACTION_COL, self.OP_COL_W)
        models_layout.addWidget(self.network_table)

        layout.addWidget(models_group)

    def setup_official_network_tab(self):
        """设置官方网络模型标签页"""
        layout = QVBoxLayout(self.official_network_tab)

        # 下载路径组
        path_group = QGroupBox("路径设置")
        path_group.setObjectName("modelSelectGroup")
        path_layout = QHBoxLayout(path_group)

        self.official_download_path_edit = QLineEdit()
        self.official_download_path_edit.setObjectName("modelSelectPathEdit")
        self.official_download_path_edit.setText(
            str(MODELS_DIR_OFFICIAL.absolute()))
        self.official_download_path_edit.setPlaceholderText("官方模型下载目录路径...")
        path_layout.addWidget(self.official_download_path_edit)

        browse_official_btn = QPushButton("浏览")
        browse_official_btn.setProperty("variant", "skyPrimary")
        browse_official_btn.setIcon(
            ThemeIcons.icon("folder_open", 18, "#ffffff"))
        browse_official_btn.setIconSize(QSize(18, 18))
        browse_official_btn.clicked.connect(self.browse_official_download_path)
        path_layout.addWidget(browse_official_btn)

        layout.addWidget(path_group)

        # 官方网络模型组
        models_group = QGroupBox("官方YOLO模型资源")
        models_group.setObjectName("modelSelectGroup")
        models_layout = QVBoxLayout(models_group)

        self.official_network_table = self._create_table(
            self.COLUMN_HEADERS_OFFICIAL_NETWORK, 6, "modelSelectTable")
        self.official_network_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.official_network_table.customContextMenuRequested.connect(
            self.show_official_network_context_menu)
        self.official_network_table.doubleClicked.connect(
            self.show_official_network_model_info)
        self.official_network_table.setMinimumHeight(450)
        o_header = self.official_network_table.horizontalHeader()
        o_header.setSectionResizeMode(self.MODEL_NAME_COL, QHeaderView.Stretch)
        o_header.setSectionResizeMode(
            self.SIZE_COL, QHeaderView.ResizeToContents)
        o_header.setSectionResizeMode(
            self.MODIFIED_COL, QHeaderView.ResizeToContents)
        o_header.setSectionResizeMode(
            self.STATUS_COL - 1, QHeaderView.ResizeToContents)
        o_header.setSectionResizeMode(
            self.STATUS_COL, QHeaderView.ResizeToContents)
        o_header.setSectionResizeMode(self.ACTION_COL, QHeaderView.Fixed)
        self.official_network_table.setColumnWidth(
            self.ACTION_COL, self.OP_COL_W)
        models_layout.addWidget(self.official_network_table)

        layout.addWidget(models_group)

    def setup_help_tab(self):
        """设置帮助标签页"""
        help_host = self.help_tab
        help_host.setObjectName("helpRoot")
        layout = QVBoxLayout(help_host)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        section_wrap = QWidget()
        section_wrap.setObjectName("helpNavStrip")
        section_wrap.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        section_wrap.setMinimumHeight(56)
        section_wrap.setMaximumHeight(56)
        section_row = QHBoxLayout(section_wrap)
        section_row.setContentsMargins(12, 8, 12, 8)
        section_row.setSpacing(10)
        layout.addWidget(section_wrap)

        help_shell = QWidget()
        help_shell.setObjectName("helpShell")
        shell_lay = QHBoxLayout(help_shell)
        shell_lay.setContentsMargins(10, 8, 10, 10)
        shell_lay.setSpacing(10)

        article_wrap = QWidget()
        article_wrap.setObjectName("helpArticleWrap")
        article_wrap.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        article_lay = QHBoxLayout(article_wrap)
        article_lay.setContentsMargins(0, 0, 0, 0)
        article_lay.setSpacing(0)

        _help_ff = StyleManager.help_document_font_family_css()
        _help_mono = StyleManager.MONO_FONT_FAMILY

        viewer = QTextEdit()
        viewer.setReadOnly(True)
        viewer.setLineWrapMode(QTextEdit.WidgetWidth)
        viewer.setMinimumHeight(520)
        viewer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        viewer.setFont(StyleManager.help_document_qfont())
        viewer.setObjectName("helpMarkdownViewer")
        viewer.document().setDocumentMargin(6)
        _help_md_css = """
            html, body {
                font-family: __HELP_FF__;
                font-size: 16px;
                line-height: 2.0;
                color: #1e293b;
                text-align: left;
                margin: 0;
                padding: 0;
            }
            h1, h2, h3, h4 {
                font-family: __HELP_FF__;
                text-align: left;
            }
            h1 {
                font-size: 30px;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: #0f172a;
                margin: 0 0 22px 0;
                padding-bottom: 14px;
                border-bottom: 1px solid #e8eef5;
            }
            h2 {
                font-size: 24px;
                font-weight: 600;
                color: #1e293b;
                margin: 30px 0 12px 0;
                line-height: 1.45;
            }
            h3 {
                font-size: 19px;
                font-weight: 600;
                color: #334155;
                margin: 22px 0 10px 0;
                line-height: 1.5;
            }
            p {
                font-size: 16px;
                color: #334155;
                text-align: left;
                margin: 0 0 18px 0;
                line-height: 2.0;
            }
            li {
                font-size: 16px;
                color: #334155;
                text-align: left;
                margin: 10px 0;
                line-height: 2.0;
            }
            ul, ol {
                margin: 12px 0 22px 24px;
                padding-left: 8px;
                text-align: left;
            }
            hr {
                border: none;
                border-top: 1px solid #e8eef5;
                margin: 26px 0;
            }
            code {
                font-family: __HELP_MONO__;
                font-size: 14px;
                background: #f1f5f9;
                color: #475569;
                padding: 2px 6px;
                border-radius: 4px;
            }
            pre, pre code {
                font-family: __HELP_MONO__;
                font-size: 13px;
                line-height: 1.7;
            }
            table {
                border-collapse: collapse;
                margin: 18px 0 24px 0;
                text-align: left;
                font-family: __HELP_FF__;
            }
            th, td {
                border: 1px solid #e8eef5;
                padding: 10px 14px;
                line-height: 1.8;
                font-family: __HELP_FF__;
            }
            th {
                background: #f8fafc;
                font-weight: 600;
                color: #334155;
            }
            blockquote {
                border-left: 3px solid #c7d2fe;
                margin: 18px 0;
                padding: 14px 18px;
                color: #475569;
                background: #f8fafc;
                border-radius: 0 8px 8px 0;
                text-align: left;
                line-height: 1.95;
                font-family: __HELP_FF__;
            }
        """
        viewer.document().setDefaultStyleSheet(
            _help_md_css.replace("__HELP_FF__", _help_ff).replace(
                "__HELP_MONO__", _help_mono))
        # 占满 article 区域宽度，不再两侧留空、也不再限制 maxWidth。
        viewer.setMinimumWidth(200)
        viewer.setAlignment(Qt.AlignmentFlag.AlignLeft)
        article_lay.addWidget(viewer, 1)
        shell_lay.addWidget(article_wrap, 1)

        toc_panel = QFrame()
        toc_panel.setObjectName("helpTocPanel")
        toc_panel.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        toc_panel.setFixedWidth(214)
        toc_v = QVBoxLayout(toc_panel)
        toc_v.setContentsMargins(8, 8, 8, 8)
        toc_v.setSpacing(6)
        toc_header = QWidget()
        toc_header_lay = QHBoxLayout(toc_header)
        toc_header_lay.setContentsMargins(0, 0, 0, 0)
        toc_header_lay.setSpacing(8)
        toc_icon = QLabel()
        toc_icon.setPixmap(ThemeIcons.pixmap("list", 16, "#64748b"))
        toc_title = QLabel("目录")
        toc_title.setObjectName("helpTocTitle")
        toc_header_lay.addWidget(toc_icon, 0, Qt.AlignVCenter)
        toc_header_lay.addWidget(toc_title, 0, Qt.AlignVCenter)
        toc_header_lay.addStretch(1)
        toc_tree = QTreeWidget()
        toc_tree.setObjectName("helpTocTree")
        toc_tree.setHeaderHidden(True)
        toc_tree.setRootIsDecorated(True)
        toc_tree.setAnimated(True)
        toc_tree.setIndentation(12)
        toc_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        toc_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        toc_tree.setUniformRowHeights(False)
        toc_v.addWidget(toc_header)
        toc_v.addWidget(toc_tree, 1)
        shell_lay.addWidget(toc_panel, 0)

        layout.addWidget(help_shell, 1)

        help_host._help_md_viewer = viewer
        help_host._help_toc_tree = toc_tree
        help_host._help_section_row = section_row
        help_host._help_last_md = ""
        _HEADING_IDX_ROLE = Qt.ItemDataRole.UserRole + 32

        _help_heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

        def _parse_md_headings(md_text: str) -> list[tuple[int, str]]:
            out: list[tuple[int, str]] = []
            for line in (md_text or "").splitlines():
                m = _help_heading_re.match(line.strip())
                if m:
                    out.append((len(m.group(1)), m.group(2).strip()))
            return out

        def _jump_help_to_heading(title: str) -> None:
            t = (title or "").strip()
            if not t:
                return
            help_host._help_spy_suppress = True
            doc = viewer.document()
            for i in range(doc.blockCount()):
                block = doc.findBlockByNumber(i)
                line = block.text().strip()
                if line == t:
                    cur = QTextCursor(block)
                    viewer.setTextCursor(cur)
                    viewer.ensureCursorVisible()
                    QTimer.singleShot(
                        120, lambda: setattr(help_host, "_help_spy_suppress", False))
                    return
            for i in range(doc.blockCount()):
                block = doc.findBlockByNumber(i)
                line = block.text().strip()
                if line.startswith(t):
                    cur = QTextCursor(block)
                    viewer.setTextCursor(cur)
                    viewer.ensureCursorVisible()
                    QTimer.singleShot(
                        120, lambda: setattr(help_host, "_help_spy_suppress", False))
                    return
            QTimer.singleShot(
                0, lambda: setattr(help_host, "_help_spy_suppress", False))

        def _normalize_md_to_text(md_text: str) -> tuple[str, dict[int, int]]:
            """将 Markdown 规范化为易读纯文本，并返回标题行级别映射。"""
            if not md_text:
                return "", {}

            def _clean_inline(s: str) -> str:
                s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)  # link
                s = re.sub(r"`([^`]+)`", r"\1", s)  # inline code
                s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)  # bold **
                s = re.sub(r"__([^_]+)__", r"\1", s)  # bold __
                s = re.sub(r"\*([^*]+)\*", r"\1", s)  # italic *
                s = re.sub(r"_([^_]+)_", r"\1", s)  # italic _
                s = re.sub(r"~~([^~]+)~~", r"\1", s)  # strikethrough
                return s.strip()

            out: list[str] = []
            heading_levels_raw: dict[int, int] = {}
            in_code = False
            for raw in md_text.splitlines():
                line = raw.rstrip()
                stripped = line.strip()

                if stripped.startswith("```"):
                    in_code = not in_code
                    if out and out[-1] != "":
                        out.append("")
                    continue

                if in_code:
                    out.append(line)
                    continue

                hm = _help_heading_re.match(stripped)
                if hm:
                    level = len(hm.group(1))
                    title = _clean_inline(hm.group(2))
                    if out and out[-1] != "":
                        out.append("")
                    heading_line = len(out)
                    out.append(title)
                    heading_levels_raw[heading_line] = level
                    out.append("")
                    continue

                if re.match(r"^\s*[-*+]\s+", line):
                    txt = _clean_inline(re.sub(r"^\s*[-*+]\s+", "", line))
                    out.append(f"• {txt}" if txt else "")
                    continue

                om = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
                if om:
                    num = om.group(1)
                    txt = _clean_inline(om.group(2))
                    out.append(f"{num}. {txt}" if txt else f"{num}.")
                    continue

                if stripped.startswith(">"):
                    out.append(_clean_inline(stripped.lstrip("> ").strip()))
                    continue

                if stripped.startswith("|") and stripped.endswith("|"):
                    # 跳过 markdown 表头分隔行
                    if re.match(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$", stripped):
                        continue
                    cols = [c.strip() for c in stripped.strip("|").split("|")]
                    cols = [_clean_inline(c) for c in cols if c.strip()]
                    out.append("  |  ".join(cols))
                    continue

                if stripped == "":
                    if out and out[-1] != "":
                        out.append("")
                else:
                    out.append(_clean_inline(stripped))

            # 去掉首尾空行和重复空行
            cleaned: list[str] = []
            old_to_new: dict[int, int] = {}
            for i, s in enumerate(out):
                if s == "" and cleaned and cleaned[-1] == "":
                    continue
                old_to_new[i] = len(cleaned)
                cleaned.append(s)
            lead_blanks = 0
            while lead_blanks < len(cleaned) and cleaned[lead_blanks] == "":
                lead_blanks += 1
            tail = len(cleaned)
            while tail > lead_blanks and cleaned[tail - 1] == "":
                tail -= 1
            final_lines = cleaned[lead_blanks:tail]
            heading_levels: dict[int, int] = {}
            for old_idx, level in heading_levels_raw.items():
                if old_idx not in old_to_new:
                    continue
                new_idx = old_to_new[old_idx] - lead_blanks
                if 0 <= new_idx < len(final_lines):
                    heading_levels[new_idx] = level
            return "\n".join(final_lines), heading_levels

        def _apply_help_plain_text_styles(heading_levels: dict[int, int]) -> None:
            """根据标题级别给纯文本应用层级样式（字号/字重/段距）。"""
            doc = viewer.document()
            default_bf = QTextBlockFormat()
            default_bf.setAlignment(Qt.AlignmentFlag.AlignLeft)
            default_bf.setTopMargin(0)
            default_bf.setBottomMargin(8)

            default_cf = QTextCharFormat()
            default_cf.setFontFamilies([
                "Times New Roman",
                "Microsoft YaHei UI",
                "PingFang SC",
            ])
            default_cf.setFontPointSize(16.0)
            default_cf.setForeground(QColor("#1e293b"))
            default_cf.setFontWeight(QFont.Weight.Normal)

            heading_sizes = {1: 30.0, 2: 24.0,
                             3: 20.0, 4: 18.0, 5: 17.0, 6: 16.0}
            heading_weights = {
                1: QFont.Weight.Bold,
                2: QFont.Weight.DemiBold,
                3: QFont.Weight.DemiBold,
                4: QFont.Weight.Medium,
                5: QFont.Weight.Medium,
                6: QFont.Weight.Medium,
            }

            cur = QTextCursor(doc)
            cur.beginEditBlock()
            blk = doc.begin()
            while blk.isValid():
                line_no = blk.blockNumber()
                level = int(heading_levels.get(line_no, 0))

                bf = QTextBlockFormat(default_bf)
                cf = QTextCharFormat(default_cf)
                if level > 0:
                    lv = max(1, min(6, level))
                    cf.setFontPointSize(heading_sizes.get(lv, 18.0))
                    cf.setFontWeight(heading_weights.get(
                        lv, QFont.Weight.Medium))
                    cf.setForeground(QColor("#0f172a"))
                    if lv == 1:
                        bf.setTopMargin(10)
                        bf.setBottomMargin(12)
                    elif lv == 2:
                        bf.setTopMargin(8)
                        bf.setBottomMargin(9)
                    else:
                        bf.setTopMargin(6)
                        bf.setBottomMargin(8)

                cur.setPosition(blk.position())
                cur.setBlockFormat(bf)
                cur.select(QTextCursor.SelectionType.BlockUnderCursor)
                cur.mergeCharFormat(cf)
                blk = blk.next()
            cur.endEditBlock()

        def _rebuild_help_toc(md_text: str) -> None:
            toc_tree.clear()
            headings = _parse_md_headings(md_text)
            if not headings:
                empty = QTreeWidgetItem(["（本节无标题）"])
                empty.setFlags(Qt.NoItemFlags)
                toc_tree.addTopLevelItem(empty)
                help_host._help_heading_blocks = []
                return
            stack: list[tuple[int, QTreeWidgetItem]] = []
            h_idx = 0
            for level, title in headings:
                node = QTreeWidgetItem([title])
                node.setData(0, Qt.UserRole, title)
                node.setData(0, _HEADING_IDX_ROLE, h_idx)
                node.setToolTip(0, title)
                h_idx += 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                if not stack:
                    toc_tree.addTopLevelItem(node)
                else:
                    stack[-1][1].addChild(node)
                stack.append((level, node))
            toc_tree.collapseAll()

        def _index_help_heading_blocks() -> None:
            md_text = getattr(help_host, "_help_last_md", "") or ""
            headings = _parse_md_headings(md_text)
            doc = viewer.document()
            blocks: list = []
            for _level, title in headings:
                blk = None
                for i in range(doc.blockCount()):
                    b = doc.findBlockByNumber(i)
                    if b.text().strip() == title:
                        blk = b
                        break
                blocks.append(blk)
            help_host._help_heading_blocks = blocks

        def _sync_toc_from_scroll(_value=None) -> None:
            if getattr(help_host, "_help_spy_suppress", False):
                return
            blocks = getattr(help_host, "_help_heading_blocks", None)
            if not blocks:
                return
            lay = viewer.document().documentLayout()
            if lay is None:
                return
            scroll_y = float(viewer.verticalScrollBar().value())
            active_idx = -1
            for i, block in enumerate(blocks):
                if block is None:
                    continue
                try:
                    top = lay.blockBoundingRect(block).top()
                except Exception:
                    continue
                if top <= scroll_y + 12.0:
                    active_idx = i
            if active_idx < 0:
                return

            it = QTreeWidgetItemIterator(toc_tree)
            while it.value():
                cur = it.value()
                idx = cur.data(0, _HEADING_IDX_ROLE)
                if idx is not None and int(idx) == active_idx:
                    toc_tree.blockSignals(True)
                    toc_tree.setCurrentItem(cur)
                    toc_tree.scrollToItem(
                        cur, QAbstractItemView.ScrollHint.EnsureVisible)
                    toc_tree.blockSignals(False)
                    return
                it += 1

        def _on_toc_tree_clicked(item: QTreeWidgetItem, _column: int) -> None:
            if item is None:
                return
            title = item.data(0, Qt.UserRole)
            if title:
                _jump_help_to_heading(str(title))

        toc_tree.itemClicked.connect(_on_toc_tree_clicked)
        viewer.verticalScrollBar().valueChanged.connect(_sync_toc_from_scroll)

        fixed_nav = ["使用说明", "模型说明", "常见问题", "软件介绍"]
        nav_icon_map = {
            "使用说明": "book",
            "模型说明": "cpu",
            "常见问题": "help",
            "软件介绍": "sparkles",
        }
        section_file_map = {
            "软件介绍": base_dir / "docs" / "help_sections" / "software_intro.md",
            "使用说明": base_dir / "docs" / "help_sections" / "software_manual.md",
            "模型说明": base_dir / "docs" / "help_sections" / "model_guide.md",
            "常见问题": base_dir / "docs" / "help_sections" / "faq.md",
        }

        def _clear_section_row():
            while section_row.count():
                item = section_row.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        def _set_view_content(sec: str):
            md_text = help_host._help_section_map.get(sec, "")
            help_host._help_last_md = md_text
            plain_text, heading_levels = _normalize_md_to_text(md_text)
            viewer.setPlainText(plain_text)
            _rebuild_help_toc(md_text)
            sb = viewer.verticalScrollBar()

            def _after_layout():
                _apply_help_plain_text_styles(heading_levels)
                sb.setValue(0)
                _index_help_heading_blocks()
                _sync_toc_from_scroll()

            QTimer.singleShot(0, _after_layout)

        def _render_help(section_map: dict):
            _clear_section_row()

            current = getattr(help_host, "_help_current_section", "使用说明")
            if current not in section_map:
                current = "使用说明"
            help_host._help_section_map = section_map

            btn_group = QButtonGroup(help_host)
            btn_group.setExclusive(True)
            help_host._help_section_btn_group = btn_group

            for title in fixed_nav:
                btn = QPushButton(title)
                btn.setObjectName("helpNavBtn")
                btn.setCheckable(True)
                btn.setProperty("variant", "secondary")
                btn.setMinimumHeight(30)
                icon_name = nav_icon_map.get(title, "file_text")
                btn.setIcon(ThemeIcons.icon(icon_name, 13, "#6366f1"))
                btn.setIconSize(QSize(13, 13))
                btn_group.addButton(btn)

                def _on_click(_checked=False, sec=title):
                    help_host._help_current_section = sec
                    _set_view_content(sec)

                btn.clicked.connect(_on_click)
                section_row.addWidget(btn, 0, Qt.AlignLeft)
                if title == current:
                    btn.setChecked(True)

            section_row.addStretch(1)
            _set_view_content(current)
            help_host._help_current_section = current

        def _reload_help_from_md():
            section_map = {}
            for title in fixed_nav:
                p = section_file_map[title]
                try:
                    if p.exists():
                        section_map[title] = p.read_text(encoding="utf-8")
                    else:
                        section_map[title] = f"# {title}\n\n未找到文档：`{p.relative_to(base_dir)}`。"
                except Exception as e:
                    section_map[title] = f"# {title}\n\n读取文档失败：\n\n`{e}`"
            _render_help(section_map)

        help_host._reload_help_from_md = _reload_help_from_md
        _reload_help_from_md()

    def _create_table(self, headers, column_count, object_name="modelSelectTable"):
        """创建标准表格控件"""
        table = QTableWidget()
        table.setObjectName(object_name)
        table.setColumnCount(column_count)
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        vheader = table.verticalHeader()
        vheader.setSectionResizeMode(QHeaderView.Fixed)
        vheader.setDefaultSectionSize(self.OP_ROW_H)
        vheader.setMinimumSectionSize(30)
        table.setWordWrap(False)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setAlternatingRowColors(True)
        return table

    def _apply_dialog_theme(self):
        self.setStyleSheet(self.styleSheet() + """
            QDialog#modelSelectionDialog {
                background: #f8fafc;
            }
            QTabWidget#modelSelectTabs::pane {
                border: 1px solid #dbe3ff;
                border-radius: 12px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar#dialogTabBar::tab {
                min-height: 34px;
                min-width: 128px;
                padding: 7px 14px;
                margin-right: 4px;
                border: 1px solid #dbe3ff;
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                background: #eef2ff;
                color: #475569;
                font-weight: 600;
            }
            QTabBar#dialogTabBar::tab:selected {
                color: #312e81;
                background: #ffffff;
                border-color: #c7d2fe;
            }
            QTabBar#dialogTabBar::tab:hover:!selected {
                background: #e2e8ff;
                color: #3730a3;
            }
            QGroupBox#modelSelectGroup {
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
                color: #1e293b;
                background: #ffffff;
            }
            QGroupBox#modelSelectGroup::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #3730a3;
            }
            QLineEdit#modelSelectPathEdit {
                min-height: 34px;
                border: 1px solid #cbd5e1;
                border-radius: 10px;
                padding: 0 10px;
                background: #ffffff;
                color: #0f172a;
            }
            QLineEdit#modelSelectPathEdit:focus {
                border-color: #818cf8;
                background: #f8faff;
            }
            QTableWidget#modelSelectTable {
                border: 1px solid #dbe3ff;
                border-radius: 10px;
                gridline-color: #e2e8f0;
                background: #ffffff;
                alternate-background-color: #f8faff;
                selection-background-color: #e0e7ff;
                selection-color: #1e1b4b;
            }
            QTableWidget#modelSelectTable::item {
                padding: 2px 6px;
                border: none;
            }
            QTableWidget#modelSelectTable::item:selected {
                background: #dbeafe;
                color: #1e1b4b;
            }
            QTableWidget#modelSelectTable QHeaderView::section {
                background: #eef2ff;
                color: #312e81;
                border: none;
                border-bottom: 1px solid #dbe3ff;
                padding: 8px 6px;
                font-weight: 700;
            }
            QDialogButtonBox#modelSelectButtonBox QPushButton {
                min-height: 34px;
                min-width: 92px;
            }
            QPushButton#modelOpActionBtn {
                min-height: 24px;
                max-height: 24px;
                padding: 0px 6px;
                border-radius: 7px;
                font-size: 11px;
                font-weight: 600;
                text-align: center;
                background: #ffffff;
                color: #334155;
                border: 1px solid #dbe3ff;
            }
            QPushButton#modelOpActionBtn:hover {
                background: #f8fafc;
                border-color: #c7d2fe;
                color: #312e81;
            }
            QPushButton#modelOpActionBtn:pressed {
                background: #eef2ff;
            }
        """)

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
                path_item = QTableWidgetItem(model['path'])
                path_item.setToolTip(model['path'])
                self.model_table.setItem(row, self.PATH_COL, path_item)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"刷新模型列表失败: {str(e)}")

    def _load_models_csv(
        self,
        csv_name: str,
        attr_name: str,
        refresh_fn,
        label: str,
    ) -> None:
        """从 csv_reports 读取模型列表并刷新对应表格。"""
        try:
            csv_path = base_dir / "csv_reports" / csv_name
            if not csv_path.exists():
                QMessageBox.warning(
                    self, "警告", f"未找到{label}模型数据文件 {csv_path}")
                return
            models_data = self._read_csv_with_encodings(csv_path)
            if not models_data:
                QMessageBox.warning(
                    self, "警告", f"无法正确读取{label}模型数据文件")
                return
            setattr(self, attr_name, models_data)
            refresh_fn()
        except Exception as e:
            QMessageBox.critical(
                self, "错误", f"加载{label}模型数据失败: {str(e)}")

    def load_network_models(self):
        """加载网络模型数据（优先实时读取 GitHub Release assets，失败回退 CSV）。"""
        try:
            models = self._fetch_private_release_assets()
            if models:
                self.network_models = models
                self.refresh_network_models()
                return
        except Exception:
            # 实时读取失败时回退到 CSV，避免界面不可用。
            pass
        self._load_models_csv(
            "pt_files_report.csv",
            "network_models",
            self.refresh_network_models,
            "网络",
        )

    def load_official_network_models(self):
        """加载官方网络模型数据"""
        self._load_models_csv(
            "YOLO_pt_files_report.csv",
            "official_network_models",
            self.refresh_official_network_models,
            "官方网络",
        )

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

    def _fetch_private_release_assets(self) -> list[dict]:
        """从 GitHub Release 读取私有模型资产列表。"""
        api_url = (
            f"https://api.github.com/repos/{self.PRIVATE_RELEASE_REPO}/"
            f"releases/tags/{self.PRIVATE_RELEASE_TAG}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "DimensionTeam-object-detection",
        }
        resp = requests.get(api_url, headers=headers, timeout=20)
        resp.raise_for_status()
        payload = resp.json() if isinstance(resp.json(), dict) else {}
        assets = payload.get("assets", []) if isinstance(payload, dict) else []
        if not isinstance(assets, list):
            return []

        # CSV 中可能包含类别统计信息，优先合并进实时资产列表。
        meta_map: dict[str, dict] = {}
        csv_path = base_dir / "csv_reports" / "pt_files_report.csv"
        if csv_path.exists():
            for row in self._read_csv_with_encodings(csv_path):
                name = str(row.get("文件名", "") or "").strip()
                if name:
                    meta_map[name] = row

        def _fmt_mb(size_b: int) -> str:
            mb = max(0.0, float(size_b) / (1024.0 * 1024.0))
            s = f"{mb:.2f}".rstrip("0").rstrip(".")
            return s if s else "0"

        models: list[dict] = []
        for a in assets:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name", "") or "")
            if not name.lower().endswith(".pt"):
                continue
            size_mb = _fmt_mb(int(a.get("size", 0) or 0))
            updated_raw = str(a.get("updated_at", "") or "")
            try:
                dt = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
                modified = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                modified = updated_raw

            meta = meta_map.get(name, {})
            models.append({
                "文件名": name,
                "大小(MB)": size_mb,
                "修改日期": modified,
                "类别数量": str(meta.get("类别数量", "-")),
                "类别信息": str(meta.get("类别信息", "")),
                "下载链接": str(a.get("browser_download_url", "") or ""),
            })
        return models

    def _populate_network_model_rows(
        self,
        models: list,
        table: QTableWidget,
        path_edit: QLineEdit,
        add_row_buttons,
    ) -> None:
        """填充网络 / 官方网络模型表的共用逻辑。"""
        table.setRowCount(len(models))
        for row, model in enumerate(models):
            full_name = str(model.get('文件名', '') or '')
            short_name = self._short_model_display_name(full_name)
            name_item = QTableWidgetItem(short_name)
            name_item.setToolTip(full_name)
            table.setItem(row, self.MODEL_NAME_COL, name_item)
            table.setItem(
                row, self.SIZE_COL, QTableWidgetItem(f"{model['大小(MB)']} MB"))
            table.setItem(
                row, self.MODIFIED_COL, QTableWidgetItem(model['修改日期']))
            table.setItem(
                row, self.STATUS_COL - 1, QTableWidgetItem(model['类别数量']))

            download_path = Path(path_edit.text())
            local_path = download_path / model['文件名']
            is_downloaded = local_path.exists()
            status_text = "已下载" if is_downloaded else "未下载"
            status_color = (
                QColor("#27ae60") if is_downloaded else QColor("#e74c3c"))
            status_item = QTableWidgetItem(status_text)
            status_item.setForeground(status_color)
            table.setItem(row, self.STATUS_COL, status_item)
            add_row_buttons(row, model)

    @staticmethod
    def _short_model_display_name(name: str, keep: int = 18) -> str:
        """模型名过长时中间省略，保证表格可读性。"""
        n = (name or "").strip()
        if len(n) <= keep:
            return n
        head = max(6, keep // 2)
        tail = max(6, keep - head - 1)
        return f"{n[:head]}…{n[-tail:]}"

    def refresh_network_models(self):
        """刷新网络模型列表"""
        self._populate_network_model_rows(
            self.network_models,
            self.network_table,
            self.download_path_edit,
            self._create_operation_buttons,
        )

    def refresh_official_network_models(self):
        """刷新官方网络模型列表"""
        self._populate_network_model_rows(
            self.official_network_models,
            self.official_network_table,
            self.official_download_path_edit,
            self._create_official_operation_buttons,
        )

    def _build_model_download_url(self, model_name, official=False):
        if official:
            return f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}"
        return f"https://github.com/Alaskaboo/DimensionTeam_object_detection/releases/download/train_weights/{model_name}"

    def _create_model_operation_buttons(self, row, model, table, download_path_edit, download_handler, copy_handler):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(6)

        download_btn = QPushButton("下载")
        download_btn.setObjectName("modelOpActionBtn")
        download_btn.setIcon(ThemeIcons.icon("download", 14, "#4f46e5"))
        download_btn.setIconSize(QSize(12, 12))
        self._fit_model_op_button(download_btn, "下载")
        download_btn.clicked.connect(functools.partial(download_handler, row))

        copy_btn = QPushButton("复制")
        copy_btn.setObjectName("modelOpActionBtn")
        copy_btn.setIcon(ThemeIcons.icon("link", 14, "#4f46e5"))
        copy_btn.setIconSize(QSize(12, 12))
        self._fit_model_op_button(copy_btn, "复制")
        copy_btn.clicked.connect(functools.partial(copy_handler, model))

        local_path = Path(download_path_edit.text()) / model['文件名']
        if local_path.exists():
            download_btn.setText("已下载")
            download_btn.setIcon(QIcon())
            download_btn.setEnabled(False)

        layout.addWidget(download_btn)
        layout.addWidget(copy_btn)
        layout.setAlignment(Qt.AlignCenter)
        widget.setMinimumHeight(self.OP_ROW_H - 2)
        table.setCellWidget(row, self.ACTION_COL, widget)
        if table.rowHeight(row) < self.OP_ROW_H:
            table.setRowHeight(row, self.OP_ROW_H)

    def _fit_model_op_button(self, btn: QPushButton, text: str):
        """按按钮文本计算按钮宽度，兼顾可读与不挤压。"""
        fm = btn.fontMetrics()
        text_w = fm.horizontalAdvance(text)
        # 图标 + 间距 + 左右内边距 + 安全余量
        w = max(64, int(text_w + 38))
        btn.setFixedSize(w, self.OP_BTN_H)

    def _create_operation_buttons(self, row, model):
        """创建操作按钮"""
        self._create_model_operation_buttons(
            row, model, self.network_table, self.download_path_edit,
            self.download_network_model, self.copy_download_link
        )

    def _create_official_operation_buttons(self, row, model):
        """创建官方模型操作按钮"""
        self._create_model_operation_buttons(
            row, model, self.official_network_table, self.official_download_path_edit,
            self.download_official_network_model, self.copy_official_download_link
        )

    def _show_model_info_dialog(self, model, title):
        class_info_raw = str(model.get('类别信息', '') or '')
        try:
            class_info = ast.literal_eval(
                class_info_raw) if class_info_raw else {}
            class_text = "\n".join(
                [f"{k}: {v}" for k, v in class_info.items()])
            if not class_text:
                class_text = "（暂无）"
        except Exception:
            class_text = class_info_raw or "（暂无）"

        info = (
            f"模型名称: {model.get('文件名', '-')}\n"
            f"大小: {model.get('大小(MB)', '-')} MB\n"
            f"修改时间: {model.get('修改日期', '-')}\n"
            f"类别数量: {model.get('类别数量', '-')}\n\n"
            f"类别信息:\n{class_text}"
        )
        # 使用自定义对话框替代 QMessageBox，以支持最大高度与滚动显示
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
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

    def _show_table_model_info(
        self, table: QTableWidget, models: list, title: str
    ) -> None:
        row = table.currentRow()
        if row < 0 or row >= len(models):
            return
        self._show_model_info_dialog(models[row], title)

    def show_network_model_info(self):
        """显示网络模型详细信息"""
        self._show_table_model_info(
            self.network_table, self.network_models, "模型详细信息")

    def show_official_network_model_info(self):
        """显示官方网络模型详细信息"""
        self._show_table_model_info(
            self.official_network_table,
            self.official_network_models,
            "官方模型详细信息",
        )

    def _exec_network_download_menu(self, table: QTableWidget, download_fn, pos):
        row = table.currentRow()
        if row < 0:
            return
        menu = QMenu(self)
        download_action = menu.addAction("📥 下载模型")
        download_action.triggered.connect(lambda: download_fn(row))
        menu.exec(table.viewport().mapToGlobal(pos))

    def show_network_context_menu(self, pos):
        """显示网络模型右键菜单"""
        self._exec_network_download_menu(
            self.network_table, self.download_network_model, pos)

    def show_official_network_context_menu(self, pos):
        """显示官方网络模型右键菜单"""
        self._exec_network_download_menu(
            self.official_network_table, self.download_official_network_model, pos)

    def _flash_download_toast(self, html: str, kind: str = "ok") -> None:
        """底部非阻塞提示：ok成功 / err 失败 / info 说明。"""
        self._download_toast.setVisible(True)
        if kind == "err":
            border, bg = "#f87171", "#fef2f2"
        elif kind == "info":
            border, bg = "#93c5fd", "#eff6ff"
        else:
            border, bg = "#34d399", "#ecfdf5"
        self._download_toast.setStyleSheet(f"""
            QLabel#modelDownloadToast {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 10px 14px;
                font-size: 13px;
                color: #111827;
            }}
        """)
        self._download_toast.setText(html)
        self._download_toast_timer.start(9000 if kind == "err" else 6500)

    def _update_download_window_title_badge(self) -> None:
        n = sum(1 for t in self._active_downloads.values() if t.isRunning())
        if n:
            self.setWindowTitle(f"高级模型选择 — 下载中 ({n})")
        else:
            self.setWindowTitle("高级模型选择")

    def _on_download_thread_cleanup(
        self, task_key, thread: ModelFileDownloadThread
    ) -> None:
        if self._active_downloads.get(task_key) is thread:
            del self._active_downloads[task_key]
        self._update_download_window_title_badge()

    def _on_model_download_success(
        self,
        row: int,
        table: QTableWidget,
        model_name: str,
        download_done_text: str,
        path_str: str,
    ) -> None:
        status_item = table.item(row, self.STATUS_COL)
        if status_item:
            status_item.setText("已下载")
            status_item.setForeground(QColor("#27ae60"))
        widget = table.cellWidget(row, self.ACTION_COL)
        if widget:
            for btn in widget.findChildren(QPushButton):
                if "下载" in btn.text():
                    btn.setText("已下载")
                    btn.setEnabled(False)
        n = sum(1 for t in self._active_downloads.values() if t.isRunning())
        extra = ""
        if n > 1:
            extra = (
                f"<br/><span style='color:#6b7280;font-size:12px;'>"
                f"另有约 {n - 1} 个任务进行中（见标题栏计数）</span>"
            )
        self._flash_download_toast(
            f"<b>已完成</b> · {download_done_text} {model_name}{extra}"
            f"<br/><span style='color:#4b5563;font-size:12px;'>{path_str}</span>",
            "ok",
        )

    def _on_model_download_failed(
        self, row: int, table: QTableWidget, model_name: str, msg: str
    ) -> None:
        status_item = table.item(row, self.STATUS_COL)
        if status_item:
            if "取消" in msg:
                status_item.setText("未下载")
            else:
                status_item.setText("下载失败")
            status_item.setForeground(QColor("#e74c3c"))
        if "取消" in msg:
            self._flash_download_toast(
                f"<b>已中断</b><br/>{model_name}", "info")
        else:
            self._flash_download_toast(
                f"<b>下载失败</b> · {model_name}"
                f"<br/><span style='font-size:12px;'>{msg}</span>",
                "err",
            )

    def _download_model_by_row(self, row, models, table, download_path_edit, download_done_text, official=False):
        if row >= len(models):
            return

        model = models[row]
        model_name = model['文件名']
        task_key = (id(table), model_name)
        existing = self._active_downloads.get(task_key)
        if existing is not None and existing.isRunning():
            self._flash_download_toast(
                f"<b>请勿重复</b><br/>「{model_name}」已在下载中。", "info")
            return

        download_dir = Path(download_path_edit.text())

        try:
            download_dir.mkdir(parents=True, exist_ok=True)
            local_path = download_dir / model_name

            if local_path.exists():
                reply = QMessageBox.question(
                    self, "确认覆盖",
                    f"模型文件 {model_name} 已存在，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            status_item = table.item(row, self.STATUS_COL)
            if status_item:
                status_item.setText("下载中...")
                status_item.setForeground(QColor("#f39c12"))

            url = str(model.get("下载链接", "") or "")
            if not url:
                url = self._build_model_download_url(
                    model_name, official=official)

            thread = ModelFileDownloadThread(url, local_path)
            self._active_downloads[task_key] = thread
            thread.finished_ok.connect(
                functools.partial(
                    self._on_model_download_success,
                    row,
                    table,
                    model_name,
                    download_done_text,
                )
            )
            thread.failed.connect(
                functools.partial(
                    self._on_model_download_failed,
                    row,
                    table,
                    model_name,
                )
            )
            thread.finished.connect(
                functools.partial(
                    self._on_download_thread_cleanup, task_key, thread))
            self._update_download_window_title_badge()
            thread.start()

        except Exception as e:
            status_item = table.item(row, self.STATUS_COL)
            if status_item:
                status_item.setText("下载失败")
                status_item.setForeground(QColor("#e74c3c"))
            self._flash_download_toast(
                f"<b>无法开始下载</b><br/>{str(e)}", "err")

    def download_network_model(self, row):
        """下载网络模型"""
        self._download_model_by_row(
            row, self.network_models, self.network_table, self.download_path_edit, "模型", official=False
        )

    def download_official_network_model(self, row):
        """下载官方网络模型"""
        self._download_model_by_row(
            row, self.official_network_models, self.official_network_table,
            self.official_download_path_edit, "官方模型", official=True
        )

    def _copy_model_download_link(self, model, success_text, official=False):
        if not model or '文件名' not in model:
            raise ValueError("模型数据无效")
        url = str(model.get("下载链接", "") or "")
        if not url:
            url = self._build_model_download_url(
                model['文件名'], official=official)
        QApplication.clipboard().setText(url)
        QMessageBox.information(self, "复制成功", success_text)

    def copy_download_link(self, model):
        """复制下载链接到剪贴板"""
        try:
            self._copy_model_download_link(
                model, "下载链接已复制到剪贴板", official=False)
        except Exception as e:
            QMessageBox.critical(self, "复制失败", f"错误: {str(e)}")

    def copy_official_download_link(self, model):
        """复制官方模型下载链接到剪贴板"""
        try:
            self._copy_model_download_link(
                model, "官方模型下载链接已复制到剪贴板", official=True)
        except Exception as e:
            QMessageBox.critical(self, "复制失败", f"错误: {str(e)}")

    def closeEvent(self, event):
        active = [t for t in self._active_downloads.values() if t.isRunning()]
        if active:
            reply = QMessageBox.question(
                self,
                "正在下载",
                f"当前有 {len(active)} 个模型正在下载，确定关闭？\n"
                "关闭后将尝试中断所有任务并删除临时文件。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            for t in active:
                t.requestInterruption()
            for t in active:
                t.wait(8000)
            self._active_downloads.clear()
            self._update_download_window_title_badge()
        event.accept()

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

    def _accept_if_local_model_ready(
        self,
        table: QTableWidget,
        models: list,
        path_edit: QLineEdit,
        not_downloaded_msg: str,
    ) -> None:
        row = table.currentRow()
        if row < 0:
            return
        model = models[row]
        local_path = Path(path_edit.text()) / model['文件名']
        if not local_path.exists():
            QMessageBox.warning(self, "警告", not_downloaded_msg)
            return
        self.selected_model = str(local_path)
        super().accept()

    def _handle_network_selection(self):
        """处理网络模型选择"""
        self._accept_if_local_model_ready(
            self.network_table,
            self.network_models,
            self.download_path_edit,
            "请先下载选中的网络模型！",
        )

    def _handle_official_network_selection(self):
        """处理官方网络模型选择"""
        self._accept_if_local_model_ready(
            self.official_network_table,
            self.official_network_models,
            self.official_download_path_edit,
            "请先下载选中的官方网络模型！",
        )


def _format_export_local_time() -> str:
    """本地时间字符串：导出、历史快照与报告共用格式。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


_DETECTION_EXPORT_META_ORDER = (
    "应用",
    "主界面页签",
    "结果记录时间",
    "推算推理开始时间",
    "推算推理结束时间",
    "本帧推理耗时_s",
    "理论推理帧率_FPS",
    "模型文件路径",
    "模型名称",
    "置信度阈值",
    "输入类型",
    "输入来源",
    "检测目标总数",
    "类别种数",
    "平均置信度",
    "置信度最小值",
    "置信度最大值",
    "类别分布",
    "备注",
)

_DETECTION_EXPORT_DETAIL_FIELDS = [
    "帧序号", "帧时间", "序号", "类别", "置信度", "x1", "y1", "x2", "y2",
    "宽", "高", "面积",
]


def _confirm_dialog(parent, title: str, text: str, ok_text: str = "确认删除", cancel_text: str = "取消") -> bool:
    """统一确认弹窗样式与文案，避免系统默认 Yes/No 风格突兀。"""
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(text)
    box.setStandardButtons(
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    yes_btn = box.button(QMessageBox.StandardButton.Yes)
    no_btn = box.button(QMessageBox.StandardButton.No)
    if yes_btn is not None:
        yes_btn.setText(ok_text)
    if no_btn is not None:
        no_btn.setText(cancel_text)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    box.setEscapeButton(QMessageBox.StandardButton.No)
    box.setStyleSheet(
        """
        QMessageBox {
            background: #f8fafc;
        }
        QMessageBox QLabel {
            color: #0f172a;
            font-size: 13px;
            min-width: 0px;
            padding: 0px;
            margin: 0px;
        }
        QMessageBox QPushButton {
            min-width: 72px;
            min-height: 30px;
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid #cbd5e1;
            background: #ffffff;
            color: #334155;
            font-weight: 600;
        }
        QMessageBox QPushButton:hover {
            background: #f1f5f9;
            border-color: #94a3b8;
        }
        QMessageBox QPushButton:pressed {
            background: #e2e8f0;
        }
        """
    )
    return box.exec() == QMessageBox.StandardButton.Yes


class PathCellLineEdit(QLineEdit):
    """历史来源路径单元格：可横向查看，失焦后恢复初始展示。"""

    def _restore_default_view(self):
        self.deselect()
        self.setCursorPosition(0)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._restore_default_view()


class TaskHistoryPrefsDialog(QDialog):
    """历史任务首选项：条数上限、清空全部、存储说明。"""

    def __init__(
        self,
        parent,
        base_dir: Path,
        initial_max: int,
        on_purged=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("历史任务 — 首选项")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._base_dir = Path(base_dir)
        self._db_path = self._base_dir / "task_history.db"
        self._on_purged = on_purged

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        intro = QLabel("配置历史记录保留策略。清空历史仅影响本地数据库文件。")
        intro.setWordWrap(True)
        intro.setObjectName("wfMutedHint")
        root.addWidget(intro)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self._max_spin = QSpinBox()
        self._max_spin.setObjectName("prefsMaxSpinField")
        self._max_spin.setRange(50, 20_000)
        self._max_spin.setValue(initial_max)
        self._max_spin.setSuffix(" 条")
        self._max_spin.setMinimumHeight(30)
        self._max_spin.setToolTip("超过该数量时，系统会自动删除最旧记录。")
        form.addRow("最多保留记录：", self._max_spin)
        root.addLayout(form)

        db_row = QVBoxLayout()
        db_row.setContentsMargins(0, 0, 0, 0)
        db_row.setSpacing(6)
        db_title = QLabel("本地数据库文件")
        db_title.setObjectName("toolbarFieldLabel")
        db_row.addWidget(db_title)
        lab_db = QLineEdit(str(self._db_path.resolve()))
        lab_db.setObjectName("pathReadonlyField")
        lab_db.setReadOnly(True)
        lab_db.setMinimumHeight(32)
        lab_db.setCursorPosition(0)
        db_row.addWidget(lab_db)
        root.addLayout(db_row)

        hint = QLabel(
            "单机默认使用 SQLite，无需额外数据库服务；备份时复制上方文件即可。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("wfMutedHint")
        root.addWidget(hint)

        root.addSpacing(2)
        purge_btn = QPushButton("清空全部历史记录…")
        purge_btn.setProperty("variant", "secondary")
        purge_btn.setIcon(ThemeIcons.icon("trash", 16, "#dc2626"))
        purge_btn.setIconSize(QSize(16, 16))
        purge_btn.setToolTip("该操作不可恢复，请谨慎执行。")
        purge_btn.clicked.connect(self._on_purge_all)
        root.addWidget(purge_btn)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_btn = bb.button(QDialogButtonBox.StandardButton.Ok)
        cancel_btn = bb.button(QDialogButtonBox.StandardButton.Cancel)
        if ok_btn is not None:
            ok_btn.setText("保存设置")
            ok_btn.setProperty("variant", "skyPrimary")
        if cancel_btn is not None:
            cancel_btn.setText("取消")
            cancel_btn.setProperty("variant", "secondary")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        root.addWidget(bb)

    def _on_purge_all(self):
        ok = _confirm_dialog(
            self,
            "确认清空",
            "将删除所有历史记录且不可恢复，是否继续？",
            ok_text="确认清空",
            cancel_text="取消",
        )
        if not ok:
            return
        TaskHistoryStore(self._db_path).delete_all()
        if callable(self._on_purged):
            self._on_purged()
        QMessageBox.information(self, "完成", "已清空全部历史记录。")

    def max_records(self) -> int:
        return self._max_spin.value()


class HistoryTableScrollHost(QScrollArea):
    """历史表格外层横向滚动：内容区固定为「列宽之和」，避免 QTableWidget 在 Tab 内被压扁列宽。

    竖向仍使用表格自带的垂直滚动条；横向只在本区域底部滚动条（或 Shift+滚轮）移动。
    """

    def __init__(self, table: QTableWidget):
        super().__init__()
        self._table = table
        self._inner = QWidget()
        self.setWidgetResizable(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        il = QVBoxLayout(self._inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(0)
        il.addWidget(table)
        self.setWidget(self._inner)
        table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)
        table.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

    def set_content_width(self, w: int) -> None:
        w = max(400, int(w))
        self._inner.setFixedWidth(w)
        self._table.setFixedWidth(w)
        self._sync_inner_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_inner_height()

    def _sync_inner_height(self) -> None:
        h = max(120, self.viewport().height())
        self._inner.setFixedHeight(h)
        self._table.setFixedHeight(h)


class TaskHistoryTable(QTableWidget):
    """支持 Shift+滚轮横向滚动（委托给外层 HistoryTableScrollHost）。"""

    def wheelEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            w = self.parentWidget()
            while w is not None:
                if isinstance(w, QScrollArea):
                    hbar = w.horizontalScrollBar()
                    if hbar is not None:
                        d = event.angleDelta().y()
                        if d:
                            hbar.setValue(hbar.value() - d)
                            event.accept()
                            return
                    break
                w = w.parentWidget()
        super().wheelEvent(event)


class TaskHistoryWidget(QWidget):
    """底部「历史任务」：SQLite 持久化，支持多选删除与首选项。"""

    _HEADERS = [
        "",
        "状态",
        "任务类型",
        "模型",
        "目标数",
        "总耗时(s)",
        "平均FPS",
        "帧进度",
        "累计推理(s)",
        "开始时间",
        "结束时间",
        "备注",
        "来源",
    ]
    _SOURCE_COL = 12  # 来源列（0=勾选，1…12 为数据列）
    # 数据列默认像素宽（与表头顺序一致）；总宽大于视口时应出现横向滚动条
    _HISTORY_DATA_COL_WIDTHS = (
        # 状态, 任务类型, 模型, 目标数, 总耗时, FPS, 帧进度, 累计推理, 开始时间, 结束时间, 备注, 来源
        180, 130, 140, 88, 98, 98, 118, 122, 178, 178, 320, 420,
    )

    def __init__(self, base_dir: Path):
        super().__init__()
        self._base_dir = Path(base_dir)
        self._prefs_path = self._base_dir / "task_history_prefs.json"
        self._store = TaskHistoryStore(self._base_dir / "task_history.db")
        self._prefs = self._load_prefs()
        self._mode_filter = "all"
        self._current_page = 1
        self._total_rows = 0
        self._total_pages = 1
        self._build_ui()
        self._reload_from_store()

    def _load_prefs(self) -> dict:
        d = {"page_size": 25}
        if self._prefs_path.exists():
            try:
                d.update(json.loads(
                    self._prefs_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        # 兼容旧首选项键
        if "page_size" not in d:
            d["page_size"] = int(d.get("max_records", 25) or 25)
        return d

    def _save_prefs(self) -> None:
        self._prefs_path.write_text(
            json.dumps(self._prefs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _page_size(self) -> int:
        return max(1, int(self._prefs.get("page_size", 20)))

    def _apply_history_column_geometry(self) -> None:
        """钉死列宽并同步外层横向滚动区域的内容宽度（列宽之和 + 勾选列）。"""
        tw = getattr(self, "table", None)
        if tw is None:
            return
        ncol = tw.columnCount()
        if ncol < 2:
            return
        hh = tw.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setCascadingSectionResizes(False)
        hh.setMinimumSectionSize(48)
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        tw.setColumnWidth(0, 48)
        widths = self._HISTORY_DATA_COL_WIDTHS
        for col in range(1, ncol):
            hh.setSectionResizeMode(col, QHeaderView.ResizeMode.Fixed)
            wi = widths[col - 1] if col - 1 < len(widths) else 96
            hh.resizeSection(col, int(wi))
        total_w = sum(tw.columnWidth(i) for i in range(ncol))
        host = getattr(self, "_history_table_scroll", None)
        if host is not None:
            host.set_content_width(total_w)
        self._place_header_select_checkbox()

    def _place_header_select_checkbox(self) -> None:
        """将表头全选框定位在首列表头中心，保持与行内复选框同风格。"""
        table = getattr(self, "table", None)
        cb = getattr(self, "_header_select_all_cb", None)
        if table is None or cb is None:
            return
        header = table.horizontalHeader()
        if header is None:
            return
        x = header.sectionPosition(0) - int(header.offset())
        w = int(table.columnWidth(0))
        h = int(header.height())
        ind = max(16, cb.sizeHint().height())
        cb.resize(ind, ind)
        cb.move(x + max(0, (w - ind) // 2), max(0, (h - ind) // 2))
        cb.raise_()

    def showEvent(self, event):
        super().showEvent(event)
        # 切换 Tab 后布局完成再钉列宽（否则仍可能被父布局压一轮）
        QTimer.singleShot(0, self._apply_history_column_geometry)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        bar = QHBoxLayout()
        bar.setContentsMargins(8, 4, 8, 0)
        bar.setSpacing(8)

        self.prefs_btn = QPushButton("首选项")
        self.prefs_btn.setObjectName("toolBtn")
        self.prefs_btn.setIcon(ThemeIcons.icon("settings", 16, "#6366f1"))
        self.prefs_btn.setIconSize(QSize(16, 16))
        self.prefs_btn.setMinimumHeight(32)
        self.prefs_btn.clicked.connect(self._on_prefs)
        bar.addWidget(self.prefs_btn, 0)

        self.del_sel_btn = QPushButton("批量删除")
        self.del_sel_btn.setObjectName("toolBtn")
        self.del_sel_btn.setIcon(ThemeIcons.icon("trash", 16, "#6366f1"))
        self.del_sel_btn.setIconSize(QSize(16, 16))
        self.del_sel_btn.setMinimumHeight(32)
        self.del_sel_btn.clicked.connect(self._delete_selected)
        bar.addWidget(self.del_sel_btn, 0)

        bar.addStretch(1)
        layout.addLayout(bar)

        ncol = len(self._HEADERS)
        self.table = TaskHistoryTable(0, ncol)
        self.table.setObjectName("wfHistoryTable")
        self.table.setHorizontalHeaderLabels(self._HEADERS)
        _tips = {
            1: "完成 / 手动停止 / 异常 / 未完成",
            5: "整次任务总耗时（秒）：含读帧、推理、节流等",
            6: "平均处理帧率：视频为 已处理帧/墙钟；单图为 1/墙钟；批量为 张数/累计推理秒",
            7: "视频：已处理/总帧；批量：张数/张数；其它为 —",
            8: "各帧推理时间之和（秒）",
            9: "点击「开始检测」的时间（若可获知）",
            10: "任务结束并写入历史的时间",
        }
        for hi, tip in _tips.items():
            hitem = self.table.horizontalHeaderItem(hi)
            if hitem is not None and tip:
                hitem.setToolTip(tip)
        for hi in range(ncol):
            item = self.table.horizontalHeaderItem(hi)
            if item:
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        sel_header = self.table.horizontalHeaderItem(0)
        if sel_header is not None:
            sel_header.setToolTip("全选/取消全选")
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.table.setMinimumHeight(160)
        self.table.setWordWrap(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = self.table.horizontalHeader()
        self._header_select_all_cb = QCheckBox(header)
        self._header_select_all_cb.setObjectName("historyRowCheck")
        self._header_select_all_cb.setTristate(False)
        self._header_select_all_cb.setToolTip("全选/取消全选")
        self._header_select_all_cb.stateChanged.connect(
            self._on_header_checkbox_state_changed)
        header.sectionResized.connect(lambda *_: self._place_header_select_checkbox())
        header.sectionMoved.connect(lambda *_: self._place_header_select_checkbox())
        self._place_header_select_checkbox()
        self._history_table_scroll = HistoryTableScrollHost(self.table)
        self._history_table_scroll.setObjectName("wfHistoryTableScroll")
        self._apply_history_column_geometry()
        _vh = self.table.verticalHeader()
        _vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        _vh.setDefaultSectionSize(42)
        _vh.setMinimumSectionSize(38)
        self.table.setItemDelegate(NoFocusTableItemDelegate(self.table))
        self.table.setToolTip(
            "提示：底部横向滚动条在外层区域；"
            "或在表内按住 Shift 再滚动鼠标滚轮左右查看")
        layout.addWidget(self._history_table_scroll, 1)

        pager = QHBoxLayout()
        pager.setContentsMargins(8, 0, 8, 4)
        pager.setSpacing(10)
        self.page_total_label = QLabel("共 0 条")
        self.page_total_label.setObjectName("wfMutedHint")
        self.page_total_label.setMinimumHeight(34)
        self.page_size_label = QLabel("每页")
        self.page_size_label.setObjectName("wfMutedHint")
        self.page_size_label.setMinimumHeight(34)
        self.page_size_combo = QComboBox()
        self.page_size_combo.setObjectName("historyPageSizeCombo")
        self.page_size_combo.addItems(["10", "20", "25", "50", "100", "200"])
        cur_ps = str(self._page_size())
        if self.page_size_combo.findText(cur_ps) < 0:
            self.page_size_combo.addItem(cur_ps)
        self.page_size_combo.setCurrentText(cur_ps)
        self.page_size_combo.setMinimumHeight(34)
        self.page_size_combo.setFixedWidth(84)
        self.page_size_combo.setToolTip("每页条数")
        self.page_size_combo.currentTextChanged.connect(
            self._on_page_size_combo_changed)
        self.page_count_hint = QLabel("条")
        self.page_count_hint.setObjectName("wfMutedHint")
        self.page_count_hint.setMinimumHeight(34)

        self.page_prev_btn = QPushButton("上一页")
        self.page_next_btn = QPushButton("下一页")
        for b in (self.page_prev_btn, self.page_next_btn):
            b.setObjectName("toolBtn")
            b.setMinimumHeight(34)
            b.setMinimumWidth(74)
        self.page_prev_btn.clicked.connect(
            lambda: self._goto_page(self._current_page - 1))
        self.page_next_btn.clicked.connect(
            lambda: self._goto_page(self._current_page + 1))

        self.page_num_row = QWidget()
        self.page_num_row.setObjectName("historyPageNumRow")
        self.page_num_layout = QHBoxLayout(self.page_num_row)
        self.page_num_layout.setContentsMargins(0, 0, 0, 0)
        self.page_num_layout.setSpacing(4)
        self._page_token_widgets = []

        self.page_jump_label = QLabel("跳至")
        self.page_jump_label.setObjectName("wfMutedHint")
        self.page_jump_label.setMinimumHeight(34)
        self.page_jump_edit = QLineEdit()
        self.page_jump_edit.setObjectName("historyPageJumpEdit")
        self.page_jump_edit.setFixedWidth(56)
        self.page_jump_edit.setMinimumHeight(34)
        self.page_jump_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.page_jump_edit.setPlaceholderText("")
        self.page_jump_edit.setValidator(
            QIntValidator(1, 999999, self.page_jump_edit))
        self.page_jump_edit.setToolTip("输入页码后按回车")
        self.page_jump_edit.returnPressed.connect(self._on_jump_page)
        self.page_jump_tail = QLabel("页")
        self.page_jump_tail.setObjectName("wfMutedHint")
        self.page_jump_tail.setMinimumHeight(34)

        self.page_info_label = QLabel("1 / 1")
        self.page_info_label.setObjectName("wfMutedHint")
        self.page_info_label.setMinimumHeight(34)
        pager.addWidget(self.page_total_label, 0)
        pager.addSpacing(8)
        pager.addWidget(self.page_size_label, 0)
        pager.addWidget(self.page_size_combo, 0)
        pager.addWidget(self.page_count_hint, 0)
        pager.addStretch(1)
        pager.addWidget(self.page_prev_btn, 0)
        pager.addWidget(self.page_num_row, 0)
        pager.addWidget(self.page_next_btn, 0)
        pager.addSpacing(12)
        pager.addWidget(self.page_jump_label, 0)
        pager.addWidget(self.page_jump_edit, 0)
        pager.addWidget(self.page_jump_tail, 0)
        pager.addSpacing(10)
        pager.addWidget(self.page_info_label, 0)
        layout.addLayout(pager)

    def _on_prefs(self):
        dlg = TaskHistoryPrefsDialog(
            self.window(),
            self._base_dir,
            self._page_size(),
            on_purged=self._reload_from_store,
        )
        if dlg.exec() != QDialog.Accepted:
            return
        self._reload_from_store()
        win = self.window()
        if hasattr(win, "log_message"):
            win.log_message("历史任务首选项已保存。")

    def set_mode_filter(self, mode_key: str) -> None:
        mk = (mode_key or "all").strip().lower()
        if mk not in ("all", "image", "video", "batch", "camera", "monitor"):
            mk = "all"
        if mk != self._mode_filter:
            self._mode_filter = mk
            self._current_page = 1
            self._reload_from_store()

    def _refresh_pagination_ui(self) -> None:
        self.page_total_label.setText(f"共 {self._total_rows} 条")
        self.page_info_label.setText(
            f"{self._current_page} / {self._total_pages}")
        has_prev = self._current_page > 1
        has_next = self._current_page < self._total_pages
        self.page_prev_btn.setEnabled(has_prev)
        self.page_next_btn.setEnabled(has_next)
        self._rebuild_page_tokens()

    def _goto_page(self, page: int) -> None:
        p = max(1, min(int(page), max(1, int(self._total_pages))))
        if p != self._current_page:
            self._current_page = p
            self._reload_from_store()

    def _on_page_size_changed(self, value: int) -> None:
        self._prefs["page_size"] = max(1, int(value))
        self._save_prefs()
        self._current_page = 1
        self._reload_from_store()

    def _on_page_size_combo_changed(self, text: str) -> None:
        try:
            value = int(text)
        except Exception:
            return
        self._on_page_size_changed(value)

    def _on_jump_page(self) -> None:
        txt = (self.page_jump_edit.text() or "").strip()
        if not txt:
            return
        try:
            pg = int(txt)
        except Exception:
            return
        self._goto_page(pg)
        self.page_jump_edit.selectAll()

    def _clear_page_tokens(self) -> None:
        while self.page_num_layout.count():
            item = self.page_num_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._page_token_widgets = []

    def _rebuild_page_tokens(self) -> None:
        self._clear_page_tokens()
        total = int(self._total_pages)
        cur = int(self._current_page)

        def add_page_btn(n: int) -> None:
            b = QPushButton(str(n))
            b.setObjectName("pageNumBtn")
            b.setProperty("activePage", n == cur)
            b.setMinimumHeight(30)
            b.setMinimumWidth(30)
            b.clicked.connect(lambda _=False, p=n: self._goto_page(p))
            self.page_num_layout.addWidget(b, 0)
            self._page_token_widgets.append(b)

        def add_ellipsis() -> None:
            lb = QLabel("...")
            lb.setObjectName("wfMutedHint")
            lb.setStyleSheet(
                "font-size: 13px; color:#94a3b8; font-weight:700;")
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setFixedWidth(24)
            self.page_num_layout.addWidget(lb, 0)

        if total <= 7:
            pages = list(range(1, total + 1))
        else:
            pages = [1]
            left = max(2, cur - 1)
            right = min(total - 1, cur + 1)
            if left > 2:
                pages.append("...")
            pages.extend(range(left, right + 1))
            if right < total - 1:
                pages.append("...")
            pages.append(total)

        for p in pages:
            if p == "...":
                add_ellipsis()
            else:
                add_page_btn(int(p))

    def _toggle_select_all(self):
        """根据表头状态切换全选。"""
        boxes = self._collect_row_checkboxes()
        if not boxes:
            return
        should_check_all = not all(cb.isChecked() for cb in boxes)
        for cb in boxes:
            cb.setChecked(should_check_all)
        self._sync_header_checkbox_state()

    def _collect_row_checkboxes(self) -> list[QCheckBox]:
        boxes: list[QCheckBox] = []
        for r in range(self.table.rowCount()):
            wrap = self.table.cellWidget(r, 0)
            cb = wrap.findChild(QCheckBox) if wrap is not None else None
            if cb is not None:
                boxes.append(cb)
        return boxes

    def _sync_header_checkbox_state(self) -> None:
        """仅当全部选中时表头显示勾选；否则不勾选。"""
        header_cb = getattr(self, "_header_select_all_cb", None)
        if header_cb is None:
            return
        boxes = self._collect_row_checkboxes()
        all_checked = bool(boxes) and all(cb.isChecked() for cb in boxes)
        header_cb.blockSignals(True)
        header_cb.setChecked(all_checked)
        header_cb.blockSignals(False)

    def _on_header_checkbox_state_changed(self, state) -> None:
        boxes = self._collect_row_checkboxes()
        if not boxes:
            self._sync_header_checkbox_state()
            return
        # 兼容 PySide6 不同信号重载：state 可能是 Qt.CheckState 或 int
        if isinstance(state, Qt.CheckState):
            should_check = state == Qt.CheckState.Checked
        else:
            should_check = state == 2
        for cb in boxes:
            cb.blockSignals(True)
            cb.setChecked(should_check)
            cb.blockSignals(False)
        self._sync_header_checkbox_state()

    def _delete_selected(self):
        ids = []
        for r in range(self.table.rowCount()):
            wrap = self.table.cellWidget(r, 0)
            cb = wrap.findChild(QCheckBox) if wrap is not None else None
            if cb is not None and cb.isChecked():
                rid = cb.property("row_id")
                if rid is not None:
                    ids.append(int(rid))
        if not ids:
            QMessageBox.information(self.window(), "提示", "请先勾选要删除的记录。")
            return
        ok = _confirm_dialog(
            self.window(),
            "确认删除",
            f"确定删除选中的 {len(ids)} 条历史记录？",
            ok_text="确认删除",
            cancel_text="取消",
        )
        if ok:
            self._delete_ids(ids)

    def _delete_ids(self, ids: list):
        if not ids:
            return
        self._store.delete_ids(ids)
        total_after = int(self._store.count())
        page_size = self._page_size()
        last_page = max(1, (total_after + page_size - 1) // page_size)
        if self._current_page > last_page:
            self._current_page = last_page
        self._reload_from_store()

    def _reload_from_store(self):
        if not hasattr(self, "table"):
            return
        self._total_rows = int(self._store.count_filtered(self._mode_filter))
        page_size = self._page_size()
        self._total_pages = max(
            1, (self._total_rows + page_size - 1) // page_size)
        self._current_page = max(1, min(self._current_page, self._total_pages))
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(0)
        for row in self._store.list_page_filtered(self._current_page, page_size, self._mode_filter):
            r = self.table.rowCount()
            self.table.insertRow(r)
            self._fill_row(r, row)
        self.table.blockSignals(False)
        self._sync_header_checkbox_state()
        self._apply_history_column_geometry()
        self._refresh_pagination_ui()

    def _fill_row(self, table_row: int, row: tuple):
        (
            nid,
            ended_at,
            started_at,
            task_type,
            source,
            model,
            objects,
            wall_s,
            note,
            det_status,
            avg_proc_fps,
            frames_done,
            frames_total,
            sum_infer_s,
        ) = row
        # 旧数据仅有 note、无 det_status 列时，备注里常见「用户手动停止｜结束 …」
        det_display = (det_status or "").strip()
        if not det_display:
            note_s = (note or "").strip()
            if "｜" in note_s:
                det_display = note_s.split("｜", 1)[0].strip()
            elif " | " in note_s:
                det_display = note_s.split(" | ", 1)[0].strip()
        cb = QCheckBox()
        cb.setObjectName("historyRowCheck")
        cb.setTristate(False)
        cb.setChecked(False)
        cb.setProperty("row_id", int(nid))
        cb.stateChanged.connect(lambda _=0: self._sync_header_checkbox_state())
        cb_wrap = QWidget()
        cb_lay = QHBoxLayout(cb_wrap)
        cb_lay.setContentsMargins(0, 0, 0, 0)
        cb_lay.setSpacing(0)
        cb_lay.addWidget(cb, 0, Qt.AlignmentFlag.AlignCenter)
        self.table.setCellWidget(table_row, 0, cb_wrap)
        fd, ft = int(frames_done), int(frames_total)
        if ft > 0:
            frame_txt = f"{fd}/{ft}"
        elif fd > 0:
            frame_txt = f"{fd}/—"
        else:
            frame_txt = "—"
        vals = [
            det_display,
            task_type or "",
            model or "",
            str(objects),
            f"{float(wall_s):.4f}",
            f"{float(avg_proc_fps):.2f}" if float(
                avg_proc_fps) > 1e-9 else "—",
            frame_txt,
            f"{float(sum_infer_s):.4f}" if float(sum_infer_s) > 1e-9 else "—",
            started_at or "",
            ended_at or "",
            note or "",
            source or "",
        ]
        align_cv = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        for c, text in enumerate(vals, start=1):
            if c == self._SOURCE_COL:
                # 来源列用只读输入框承载长路径：可点选/拖拽选中并左右移动查看全路径
                src_edit = PathCellLineEdit(text)
                src_edit.setObjectName("historyPathCellEdit")
                src_edit.setReadOnly(True)
                src_edit.setFrame(False)
                src_edit.setCursorPosition(0)
                src_edit.setFont(self.table.font())
                src_edit.setAlignment(
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                src_edit.setToolTip(text)
                self.table.setCellWidget(table_row, c, src_edit)
                continue
            it = QTableWidgetItem(text)
            it.setTextAlignment(align_cv)
            if c == 11:
                it.setToolTip(text)
            self.table.setItem(table_row, c, it)

    def add_record(
        self,
        time_str: str,
        task_type: str,
        source: str,
        model: str,
        objects: int,
        inference_s: float,
        note: str = "",
        *,
        started_at: str = "",
        det_status: str = "",
        avg_proc_fps: float = 0.0,
        frames_done: int = 0,
        frames_total: int = 0,
        sum_infer_s: float = 0.0,
    ):
        self._store.add(
            time_str,
            task_type,
            source,
            model,
            objects,
            inference_s,
            note,
            started_at=started_at,
            det_status=det_status,
            avg_proc_fps=avg_proc_fps,
            frames_done=frames_done,
            frames_total=frames_total,
            sum_infer_s=sum_infer_s,
        )
        self._current_page = 1
        self._reload_from_store()


class NoFocusTableItemDelegate(QStyledItemDelegate):
    """绘制单元格时不画焦点虚线框，点击后仍保留选中底色。"""

    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QStyle.StateFlag.State_HasFocus
        opt.state &= ~QStyle.StateFlag.State_KeyboardFocusChange
        super().paint(painter, opt, index)


class DetectionResultWidget(QWidget):
    """检测结果显示组件"""

    def __init__(self):
        super().__init__()
        self._detail_csv_rows = []
        self._video_detail_rows = []
        self._video_last_frame_no = 0
        self._export_meta = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sheet = QFrame()
        sheet.setObjectName("wfResultSheet")
        sl = QHBoxLayout(sheet)
        sl.setContentsMargins(8, 8, 8, 8)
        sl.setSpacing(0)

        left_panel = QWidget()
        left_panel.setObjectName("wfResultLeftPanel")
        lv = QVBoxLayout(left_panel)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)

        title_texts = [
            "序号", "类别", "置信度",
            "左上 (x1, y1)", "右下 (x2, y2)", "尺寸 (w×h)", "面积",
        ]
        _tips = (
            None,
            None,
            None,
            None,
            None,
            "宽 × 高（像素）",
            "round(宽 × 高)",
        )
        hdr_row = QHBoxLayout()
        hdr_row.setContentsMargins(0, 0, 0, 0)
        hdr_row.setSpacing(0)
        self._detail_col_title_labels = []
        for idx, (text, tip) in enumerate(zip(title_texts, _tips)):
            lab = QLabel(text)
            lab.setObjectName("wfResultColTitle")
            lab.setAlignment(Qt.AlignCenter)
            lab.setMinimumHeight(40)
            if idx == len(title_texts) - 1:
                lab.setProperty("edgeLast", True)
            if tip:
                lab.setToolTip(tip)
            hdr_row.addWidget(lab, 1)
            self._detail_col_title_labels.append(lab)
        hdr_wrap = QWidget()
        hdr_wrap.setObjectName("wfResultColHeaderRow")
        hdr_wrap.setLayout(hdr_row)
        lv.addWidget(hdr_wrap, 0)

        self.result_table = QTableWidget()
        self.result_table.setObjectName("wfResultTable")
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(title_texts)
        _hdr = self.result_table.horizontalHeader()
        _hdr.setVisible(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.result_table.setMinimumHeight(148)
        self.result_table.setMaximumHeight(300)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        self.result_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result_table.setShowGrid(False)
        self.result_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_table.verticalScrollBar().rangeChanged.connect(
            lambda *_: QTimer.singleShot(0,
                                         self._sync_result_table_column_widths)
        )
        self.result_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.result_table.setItemDelegate(
            NoFocusTableItemDelegate(self.result_table))
        lv.addWidget(self.result_table, 1)

        sl.addWidget(left_panel, 1)
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
        """七列表体与自定义表头等分同宽（严格执行均匀列宽）。"""
        t = self.result_table
        vw = t.viewport().width()
        if vw < 80:
            return
        n = 7
        base = max(1, vw // n)
        for i in range(n):
            t.setColumnWidth(i, base)
        rem = vw - sum(t.columnWidth(i) for i in range(n))
        if rem > 0:
            t.setColumnWidth(n - 1, t.columnWidth(n - 1) + rem)

    def export_detection_detail(self, fmt: str):
        """导出检测明细。fmt: csv / json / txt / xlsx"""
        if not self._detail_csv_rows:
            QMessageBox.information(
                self.window(), "提示", "暂无检测明细可导出，请先运行检测。")
            return
        stem = f"detection_detail_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dialogs = {
            "csv": (
                "导出检测明细 — CSV",
                f"{stem}.csv",
                "CSV 表格 (*.csv)",
            ),
            "json": (
                "导出检测明细 — JSON",
                f"{stem}.json",
                "JSON 文件 (*.json)",
            ),
            "txt": (
                "导出检测明细 — 文本",
                f"{stem}.txt",
                "文本文件 (*.txt)",
            ),
            "xlsx": (
                "导出检测明细 — Excel",
                f"{stem}.xlsx",
                "Excel 工作簿 (*.xlsx)",
            ),
        }
        if fmt not in dialogs:
            return
        title, default_name, filt = dialogs[fmt]
        save_path, _ = QFileDialog.getSaveFileName(
            self.window(),
            title,
            str((data_dir / default_name).resolve()),
            f"{filt};;所有文件 (*)",
        )
        if not save_path:
            return
        try:
            if fmt == "csv":
                self._write_export_csv(save_path)
            elif fmt == "json":
                self._write_export_json(save_path)
            elif fmt == "txt":
                self._write_export_txt(save_path)
            else:
                if not self._write_export_xlsx(save_path):
                    return
            win = self.window()
            if hasattr(win, "log_message"):
                win.log_message(f"检测明细已导出 ({fmt.upper()}): {save_path}")
        except Exception as e:
            QMessageBox.warning(self.window(), "导出失败", str(e))

    def _write_export_csv(self, save_path: str):
        export_time = _format_export_local_time()
        with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["【指标项】", "【值】"])
            w.writerow(["导出时间", export_time])
            for key in _DETECTION_EXPORT_META_ORDER:
                if key in self._export_meta:
                    val = self._export_meta[key]
                    w.writerow([key, val if val is not None else ""])
            w.writerow([])
            w.writerow(["【目标明细】", ""])
            dw = csv.DictWriter(
                f, fieldnames=_DETECTION_EXPORT_DETAIL_FIELDS,
                extrasaction="ignore")
            dw.writeheader()
            dw.writerows(self._detail_csv_rows)

    def _write_export_json(self, save_path: str):
        export_time = _format_export_local_time()
        metrics = {
            k: self._export_meta[k]
            for k in _DETECTION_EXPORT_META_ORDER
            if k in self._export_meta
        }
        payload = {
            "format": "Dimension_detection_export",
            "version": 1,
            "export_time_local": export_time,
            "metrics": metrics,
            "detections": self._detail_csv_rows,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _write_export_txt(self, save_path: str):
        export_time = _format_export_local_time()
        lines = [
            "Dimension 目标检测 — 检测明细导出",
            "=" * 48,
            f"导出时间: {export_time}",
            "",
            "【汇总指标】",
            "-" * 48,
        ]
        for key in _DETECTION_EXPORT_META_ORDER:
            if key in self._export_meta:
                v = self._export_meta[key]
                lines.append(f"{key}: {v if v is not None else ''}")
        lines.extend([
            "",
            "【目标明细】",
            "-" * 48,
            "\t".join(_DETECTION_EXPORT_DETAIL_FIELDS),
        ])
        for row in self._detail_csv_rows:
            lines.append(
                "\t".join(
                    str(row.get(k, "")) for k in _DETECTION_EXPORT_DETAIL_FIELDS))
        Path(save_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_export_xlsx(self, save_path: str) -> bool:
        try:
            from openpyxl import Workbook
        except ImportError:
            QMessageBox.warning(
                self.window(),
                "缺少依赖",
                "导出 Excel 需要安装 openpyxl：\npip install openpyxl",
            )
            return False
        export_time = _format_export_local_time()
        wb = Workbook()
        ws1 = wb.active
        ws1.title = "汇总指标"
        ws1.append(["指标项", "值"])
        ws1.append(["导出时间", export_time])
        for key in _DETECTION_EXPORT_META_ORDER:
            if key in self._export_meta:
                v = self._export_meta[key]
                ws1.append([key, v if v is not None else ""])
        ws1.column_dimensions["A"].width = 28
        ws1.column_dimensions["B"].width = 86
        ws2 = wb.create_sheet("目标明细")
        ws2.append(_DETECTION_EXPORT_DETAIL_FIELDS)
        for row in self._detail_csv_rows:
            ws2.append([row.get(k) for k in _DETECTION_EXPORT_DETAIL_FIELDS])
        ws2.column_dimensions["A"].width = 10
        ws2.column_dimensions["B"].width = 8
        ws2.column_dimensions["C"].width = 16
        ws2.column_dimensions["D"].width = 12
        wb.save(save_path)
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_result_table_column_widths()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self._sync_result_table_column_widths)

    def update_results(self, results, class_names, inference_time):
        """更新检测结果"""
        win = self.window()
        src_type = getattr(win, "current_source_type", "") or ""
        is_stream_source = src_type in ("video", "camera")
        if not results or not results[0].boxes or len(results[0].boxes) == 0:
            is_stream_running = bool(
                hasattr(win, "current_source_type")
                and getattr(win, "current_source_type", None) in ("video", "camera")
                and hasattr(win, "detection_thread")
                and getattr(win.detection_thread, "is_running", False)
            )
            # 流检测（视频/摄像头）过程中遇到“当前帧无目标”时，也记录空帧，
            # 这样导出可保持逐帧时间轴完整。
            if is_stream_running and is_stream_source:
                frame_no = int(
                    getattr(win, "_video_export_frame_index", 0) or 0)
                if frame_no <= 0:
                    frame_no = self._video_last_frame_no + 1
                self._video_last_frame_no = max(
                    self._video_last_frame_no, frame_no)
                frame_time_str = datetime.now().strftime("%H:%M:%S")
                self._video_detail_rows.append({
                    "帧序号": frame_no,
                    "帧时间": frame_time_str,
                    "序号": 0,
                    "类别": "无目标",
                    "置信度": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "宽": "",
                    "高": "",
                    "面积": "",
                })
                self._detail_csv_rows = list(self._video_detail_rows)
            if not is_stream_running:
                self.result_table.setRowCount(0)
                self._detail_csv_rows = []
                self._video_detail_rows = []
                self._video_last_frame_no = 0
                self._export_meta = {}
            if hasattr(win, "_sync_export_detail_button_state"):
                win._sync_export_detail_button_state(
                    bool(self._detail_csv_rows))
            elif hasattr(win, "export_detail_btn"):
                win.export_detail_btn.setEnabled(bool(self._detail_csv_rows))
            if is_stream_running and self._detail_csv_rows:
                self.stats_label.setText("当前帧未检测到目标（保留上一帧明细）")
            else:
                self.stats_label.setText("未检测到目标")
            return

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        frame_time_str = datetime.now().strftime("%H:%M:%S")

        # 更新表格
        self.result_table.setRowCount(len(confidences))
        frame_rows = []

        class_counts = {}
        for i, (conf, cls, box) in enumerate(zip(confidences, classes, xyxy)):
            class_name = class_names[cls] if cls < len(
                class_names) else f"类别{cls}"

            # 统计类别数量
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            idx_item = QTableWidgetItem(str(i + 1))
            idx_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 0, idx_item)
            cls_item = QTableWidgetItem(class_name)
            cls_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 1, cls_item)

            # 置信度带颜色
            conf_item = QTableWidgetItem(f"{conf:.3f}")
            if conf > 0.8:
                conf_item.setBackground(QColor(16, 185, 129, 55))
            elif conf > 0.5:
                conf_item.setBackground(QColor(245, 158, 11, 55))
            else:
                conf_item.setBackground(QColor(248, 113, 113, 55))
            conf_item.setTextAlignment(Qt.AlignCenter)
            self.result_table.setItem(i, 2, conf_item)

            bw = box[2] - box[0]
            bh = box[3] - box[1]
            area_px = max(0, int(round(bw * bh)))
            tl_txt = f"{box[0]:.1f}, {box[1]:.1f}"
            br_txt = f"{box[2]:.1f}, {box[3]:.1f}"
            size_txt = f"{bw:.2f} × {bh:.2f}"
            cell_tl = QTableWidgetItem(tl_txt)
            cell_tl.setTextAlignment(Qt.AlignCenter)
            cell_tl.setToolTip(f"({box[0]:.4f}, {box[1]:.4f})")
            self.result_table.setItem(i, 3, cell_tl)
            cell_br = QTableWidgetItem(br_txt)
            cell_br.setTextAlignment(Qt.AlignCenter)
            cell_br.setToolTip(f"({box[2]:.4f}, {box[3]:.4f})")
            self.result_table.setItem(i, 4, cell_br)
            cell_sz = QTableWidgetItem(size_txt)
            cell_sz.setTextAlignment(Qt.AlignCenter)
            cell_sz.setToolTip(f"宽 {bw:.4f} px · 高 {bh:.4f} px")
            self.result_table.setItem(i, 5, cell_sz)
            cell_area = QTableWidgetItem(str(area_px))
            cell_area.setTextAlignment(Qt.AlignCenter)
            cell_area.setToolTip(f"{size_txt} → {area_px}")
            self.result_table.setItem(i, 6, cell_area)

            frame_rows.append({
                "帧序号": "",
                "帧时间": frame_time_str,
                "序号": i + 1,
                "类别": class_name,
                "置信度": round(float(conf), 6),
                "x1": round(float(box[0]), 2),
                "y1": round(float(box[1]), 2),
                "x2": round(float(box[2]), 2),
                "y2": round(float(box[3]), 2),
                "宽": round(float(bw), 2),
                "高": round(float(bh), 2),
                "面积": area_px,
            })

        end_local = datetime.now()
        try:
            inf_s = float(inference_time)
            start_local = end_local - timedelta(seconds=inf_s)
        except Exception:
            inf_s = 0.0
            start_local = end_local

        model_path = ""
        if hasattr(win, "_loaded_model_path") and win._loaded_model_path:
            model_path = str(win._loaded_model_path)
        elif hasattr(win, "model") and win.model is not None:
            ckpt = getattr(win.model, "ckpt_path", None)
            if ckpt:
                try:
                    model_path = str(Path(ckpt).resolve())
                except Exception:
                    model_path = str(ckpt)

        model_name = ""
        if hasattr(win, "model_combo"):
            model_name = win.model_combo.currentText()

        main_tab = ""
        if hasattr(win, "tab_widget"):
            _ti = win.tab_widget.currentIndex()
            _tabs = ["文件检测", "批量分析", "设备监控", "历史任务"]
            if 0 <= _ti < len(_tabs):
                main_tab = _tabs[_ti]

        src_type_cn = {
            "image": "单张图片",
            "video": "视频文件",
            "camera": "摄像头",
            "batch": "文件夹批量",
        }.get(src_type, src_type)

        src_detail = ""
        if hasattr(win, "current_file_label"):
            src_detail = (
                win.current_file_label.toolTip().strip()
                or win.current_file_label.text().strip()
            )

        total_objects = len(confidences)
        avg_confidence = float(np.mean(confidences))
        cmin = float(np.min(confidences))
        cmax = float(np.max(confidences))
        class_summary_str = " | ".join(
            [f"{n}:{c}" for n, c in class_counts.items()])

        tfps = ""
        if inf_s > 1e-12:
            tfps = round(1.0 / inf_s, 4)

        if is_stream_source:
            frame_no = int(getattr(win, "_video_export_frame_index", 0) or 0)
            if frame_no <= 0:
                frame_no = self._video_last_frame_no + 1
            self._video_last_frame_no = max(
                self._video_last_frame_no, frame_no)
            for row in frame_rows:
                new_row = dict(row)
                new_row["帧序号"] = frame_no
                self._video_detail_rows.append(new_row)
            self._detail_csv_rows = list(self._video_detail_rows)
        else:
            self._video_detail_rows = []
            self._video_last_frame_no = 0
            self._detail_csv_rows = frame_rows

        start_meta = start_local.strftime("%Y-%m-%d %H:%M:%S")
        end_meta = end_local.strftime("%Y-%m-%d %H:%M:%S")
        if is_stream_source:
            start_meta = (
                getattr(win, "_video_export_started_at", "")
                or getattr(win, "_history_started_at_str", "")
                or start_meta
            )
            end_meta = getattr(win, "_video_export_ended_at", "") or end_meta

        self._export_meta = {
            "应用": "Dimension 目标检测系统",
            "主界面页签": main_tab or "—",
            "结果记录时间": end_meta,
            "推算推理开始时间": start_meta,
            "推算推理结束时间": end_meta,
            "本帧推理耗时_s": round(inf_s, 6),
            "理论推理帧率_FPS": tfps,
            "模型文件路径": model_path or "—",
            "模型名称": model_name or "—",
            "置信度阈值": getattr(win, "confidence_threshold", 0.25),
            "输入类型": src_type_cn or "—",
            "输入来源": src_detail or "—",
            "检测目标总数": total_objects,
            "类别种数": len(class_counts),
            "平均置信度": round(avg_confidence, 6),
            "置信度最小值": round(cmin, 6),
            "置信度最大值": round(cmax, 6),
            "类别分布": class_summary_str,
            "备注": (
                "流任务导出为逐帧累计明细（含帧序号，含无目标帧）；开始/结束时间为本次任务时间。"
                if is_stream_source else
                "推算推理起止时间：由本机收到该帧结果的时刻与引擎返回的本帧推理耗时反推，"
                "用于时间轴对照；非 GPU 硬件级精确计时。"
            ),
        }

        if hasattr(win, "_sync_export_detail_button_state"):
            win._sync_export_detail_button_state(True)
        elif hasattr(win, "export_detail_btn"):
            win.export_detail_btn.setEnabled(True)
        self._sync_result_table_column_widths()

        # 更新统计信息
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

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

        win = self.window()
        if hasattr(win, "_update_history_snapshot"):
            win._update_history_snapshot(results, class_names, inference_time)

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
    def help_document_font_family_css() -> str:
        """帮助 Markdown正文：西文/数字 Times New Roman，中文回退雅黑等（与 QFont 顺序一致）。"""
        return (
            '"Times New Roman", "Microsoft YaHei UI", "PingFang SC", '
            '"Hiragino Sans GB", "Segoe UI", serif'
        )

    @staticmethod
    def help_document_qfont() -> QFont:
        """与 help_document_font_family_css 一致的 QTextDocument 默认字体（避免与样式表冲突）。"""
        f = QFont()
        f.setFamilies([
            "Times New Roman",
            "Microsoft YaHei UI",
            "PingFang SC",
        ])
        f.setPointSizeF(12.0)
        f.setStyleHint(QFont.StyleHint.Serif)
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
                height: 10px;
                margin-right: 6px;
                image: url("./assets/icons/chevron_down_dark.svg");
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

            /* 任务预设行：默认白底；悬停时靛色渐变（图标由代码切换为白色） */
            QToolButton#presetNewIconBtn {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QToolButton#presetNewIconBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
                border: none;
            }
            QToolButton#presetNewIconBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
                border: none;
            }
            QToolButton#presetNewIconBtn:disabled {
                background: #e2e8f0;
                border: 1px solid #e2e8f0;
            }

            QToolButton#presetSaveIconBtn {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QToolButton#presetSaveIconBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
                border: none;
            }
            QToolButton#presetSaveIconBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
                border: none;
            }
            QToolButton#presetSaveIconBtn:disabled {
                background: #e2e8f0;
                border-color: #e2e8f0;
            }

            QToolButton#presetDeleteBtn {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 0px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QToolButton#presetDeleteBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
                border: none;
            }
            QToolButton#presetDeleteBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
                border: none;
            }
            QToolButton#presetDeleteBtn:disabled {
                background: #e2e8f0;
                border-color: #e2e8f0;
            }

            /* 任务控制标题栏齿轮 + 模型行刷新按钮 */
            QToolButton#taskCardHeaderSettingsBtn {
                background: transparent;
                border: none;
                border-radius: 8px;
                padding: 4px;
                min-width: 30px;
                max-width: 30px;
                min-height: 30px;
                max-height: 30px;
            }
            QToolButton#taskCardHeaderSettingsBtn:hover {
                background: #e0e7ff;
            }
            QToolButton#taskCardHeaderSettingsBtn:pressed {
                background: #c7d2fe;
            }
            QToolButton#taskCardHeaderSettingsBtn:disabled {
                background: transparent;
            }
            QToolButton#modelRefreshBtn {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 4px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QToolButton#modelRefreshBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #a5b4fc, stop:1 #818cf8);
                border: none;
            }
            QToolButton#modelRefreshBtn:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6366f1, stop:1 #4f46e5);
                border: none;
            }
            QToolButton#modelRefreshBtn:disabled {
                background: #e2e8f0;
                border-color: #e2e8f0;
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
                background: #fafbfc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }

            QWidget#wfResultColHeaderRow {
                background: transparent;
                border: 1px solid #e2e8f0;
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
            }
            QLabel#wfResultColTitle {
                background: #f1f5f9;
                color: #475569;
                font-size: 12px;
                font-weight: 600;
                padding: 10px 6px;
                border-right: 1px solid #e2e8f0;
                border-bottom: 1px solid #e2e8f0;
                min-height: 40px;
            }
            QLabel#wfResultColTitle[edgeLast="true"] {
                border-right: none;
            }
            QWidget#wfResultLeftPanel {
                background: transparent;
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
                border: 1px solid #e2e8f0;
                border-top: none;
                border-radius: 0 0 10px 10px;
                background: #ffffff;
                font-size: 13px;
                color: #0f172a;
                alternate-background-color: #f8fafc;
            }
            QTableWidget#wfResultTable::item {
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid #eef2f7;
                outline: none;
            }
            QTableWidget#wfResultTable::item:focus {
                outline: none;
                border: none;
            }
            QTableWidget#wfResultTable:focus {
                outline: none;
            }
            QTableWidget#wfResultTable::item:selected {
                background: rgba(99, 102, 241, 0.12);
                color: #0f172a;
            }
            QTableWidget#wfResultTable QHeaderView::section {
                background: #f1f5f9;
                color: #475569;
                padding: 10px 10px;
                font-size: 12px;
                font-weight: 600;
                border: none;
                border-bottom: 1px solid #e2e8f0;
            }

            QTableWidget#wfHistoryTable {
                border: 1px solid #dbe3ef;
                border-radius: 12px;
                background: #ffffff;
                gridline-color: #e9eef5;
                font-size: 12px;
                color: #0f172a;
                alternate-background-color: #fbfdff;
            }
            QTableWidget#wfHistoryTable::item {
                padding: 7px 10px;
                border: none;
                border-bottom: 1px solid #eef2f7;
                outline: none;
            }
            QTableWidget#wfHistoryTable::item:hover {
                background: #f5f8ff;
            }
            QTableWidget#wfHistoryTable::item:focus {
                outline: none;
                border: none;
            }
            QTableWidget#wfHistoryTable:focus {
                outline: none;
            }
            QTableWidget#wfHistoryTable::item:selected {
                background: rgba(99, 102, 241, 0.14);
                color: #0f172a;
            }
            QTableWidget#wfHistoryTable QHeaderView::section {
                background: #f6f8fc;
                color: #334155;
                padding: 9px 8px;
                font-size: 12px;
                font-weight: 700;
                border: none;
                border-right: 1px solid #e6edf6;
                border-bottom: 1px solid #dbe3ef;
            }
            QScrollArea#wfHistoryTableScroll {
                border: 1px solid #dbe3ef;
                border-radius: 12px;
                background: #ffffff;
            }
            QScrollArea#wfHistoryTableScroll > QWidget > QWidget {
                background: #ffffff;
            }
            QScrollArea#wfHistoryTableScroll QScrollBar:horizontal {
                height: 10px;
                background: #f1f5f9;
                border-radius: 5px;
                margin: 4px 8px 6px 8px;
            }
            QScrollArea#wfHistoryTableScroll QScrollBar::handle:horizontal {
                background: #c7d2fe;
                min-width: 56px;
                border-radius: 5px;
            }
            QScrollArea#wfHistoryTableScroll QScrollBar::handle:horizontal:hover {
                background: #a5b4fc;
            }
            QScrollArea#wfHistoryTableScroll QScrollBar::add-line:horizontal,
            QScrollArea#wfHistoryTableScroll QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollArea#wfHistoryTableScroll QScrollBar::add-page:horizontal,
            QScrollArea#wfHistoryTableScroll QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            QPushButton#pageNumBtn {
                min-width: 30px;
                max-width: 34px;
                min-height: 30px;
                border: none;
                border-radius: 4px;
                background: transparent;
                color: #475569;
                font-size: 12px;
                font-weight: 600;
                padding: 0px 6px;
            }
            QPushButton#pageNumBtn:hover {
                background: #eef2ff;
                color: #334155;
            }
            QPushButton#pageNumBtn[activePage="true"] {
                background: #2563eb;
                color: #ffffff;
                font-weight: 700;
            }
            QSpinBox {
                font-size: 13px;
                padding: 2px 8px;
            }
            QLineEdit#historyPathCellEdit {
                min-height: 0px;
                padding: 0px 6px;
                border: none;
                background: transparent;
                color: #0f172a;
                font-size: 13px;
            }
            QLineEdit#historyPathCellEdit:focus {
                border: none;
                background: transparent;
            }
            QCheckBox#historyRowCheck {
                spacing: 0px;
                padding: 0px;
                margin: 0px;
                background: transparent;
            }
            QCheckBox#historyRowCheck::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #94a3b8;
                background: #ffffff;
            }
            QCheckBox#historyRowCheck::indicator:hover {
                border: 1px solid #6366f1;
                background: #eef2ff;
            }
            QCheckBox#historyRowCheck::indicator:checked {
                border: 1px solid #6366f1;
                background: #ffffff;
                image: url("./assets/icons/check_mark_blue.svg");
            }
            QCheckBox#historyRowCheck::indicator:disabled {
                border: 1px solid #cbd5e1;
                background: #f1f5f9;
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
            QLabel#previewPlaceholder:focus {
                border: 2px dashed #cbd5e1;
                outline: none;
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

            QToolButton[variant="secondary"] {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                color: #475569;
                padding: 8px 14px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 10px;
                min-height: 32px;
            }
            QToolButton[variant="secondary"]:hover {
                background: #f8fafc;
                border-color: #c7d2fe;
                color: #312e81;
            }
            QToolButton[variant="secondary"]:pressed {
                background: #eef2ff;
            }
            QToolButton[variant="secondary"]:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }

            /* 任务概览：导出检测明细 — 使用 QPushButton 保证图标+文字居中 */
            QPushButton#exportDetailMenuBtn {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                color: #475569;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 600;
                border-radius: 10px;
                min-height: 32px;
                text-align: center;
            }
            QPushButton#exportDetailMenuBtn:hover {
                background: #f8fafc;
                border-color: #c7d2fe;
                color: #312e81;
            }
            QPushButton#exportDetailMenuBtn:pressed {
                background: #eef2ff;
            }
            QPushButton#exportDetailMenuBtn:disabled {
                background: #e2e8f0;
                color: #94a3b8;
            }
            QPushButton#exportDetailMenuBtn::menu-indicator {
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 10px;
                height: 10px;
                margin-right: 6px;
                image: url("./assets/icons/chevron_down_dark.svg");
            }

            QMenu#exportDetailMenu {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                padding: 6px 0;
            }
            QMenu#exportDetailMenu::item {
                padding: 9px 28px 9px 16px;
                color: #334155;
                font-size: 12px;
                font-weight: 500;
            }
            QMenu#exportDetailMenu::item:selected {
                background: #eef2ff;
                color: #312e81;
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
            QPushButton#dangerPresetBtn {
                background: #dc2626;
                border: none;
                color: #f8fafc;
                font-weight: 600;
            }
            QPushButton#dangerPresetBtn:hover {
                background: #b91c1c;
            }
            QPushButton#dangerPresetBtn:pressed {
                background: #991b1b;
            }
            QPushButton#dangerPresetBtn:disabled {
                background: #fecaca;
                color: #ffffff;
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
                padding: 6px 30px 6px 10px;
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
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border: none;
                border-left: 1px solid #e2e8f0;
                background: #f8fafc;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QComboBox::drop-down:hover {
                background: #eef2ff;
            }
            QComboBox::drop-down:pressed {
                background: #e0e7ff;
            }
            QComboBox::down-arrow {
                image: url("./assets/icons/chevron_down_dark.svg");
                width: 12px;
                height: 12px;
            }

            /* 模型下拉：与置信度 spin右侧条同宽、同底色；弹出列表限高+滚动条 */
            QComboBox#modelComboField {
                padding: 6px 24px 6px 10px;
            }
            QComboBox#modelComboField::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                border-left: 1px solid #e2e8f0;
                background: #f8fafc;
                border-top-right-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            QComboBox#modelComboField::drop-down:hover {
                background: #eef2ff;
            }
            QComboBox#modelComboField::drop-down:pressed {
                background: #e0e7ff;
            }
            QComboBox#modelComboField::down-arrow {
                width: 10px;
                height: 10px;
            }
            QComboBox#modelComboField QAbstractItemView {
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                padding: 4px;
                outline: none;
                selection-background-color: #e0e7ff;
                selection-color: #0f172a;
            }
            QComboBox#modelComboField QAbstractItemView::item {
                min-height: 28px;
                padding: 4px 8px;
                border-radius: 8px;
            }
            QComboBox#modelComboField QAbstractItemView::item:hover {
                background: #f1f5f9;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar:vertical {
                width: 20px;
                background: #f8fafc;
                border: none;
                border-left: 1px solid #e2e8f0;
                margin: 0;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::handle:vertical {
                background: #cbd5e1;
                border-radius: 5px;
                min-height: 28px;
                margin: 3px 4px 3px 3px;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::handle:vertical:hover {
                background: #a5b4fc;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::add-line:vertical,
            QComboBox#modelComboField QAbstractItemView QScrollBar::sub-line:vertical {
                height: 22px;
                subcontrol-origin: margin;
                border: none;
                background: #f8fafc;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::add-line:vertical:hover,
            QComboBox#modelComboField QAbstractItemView QScrollBar::sub-line:vertical:hover {
                background: #eef2ff;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::add-line:vertical:pressed,
            QComboBox#modelComboField QAbstractItemView QScrollBar::sub-line:vertical:pressed {
                background: #e0e7ff;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::up-arrow {
                image: url("./assets/icons/chevron_up_dark.svg");
                width: 10px;
                height: 10px;
            }
            QComboBox#modelComboField QAbstractItemView QScrollBar::down-arrow {
                image: url("./assets/icons/chevron_down_dark.svg");
                width: 10px;
                height: 10px;
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
                padding: 6px 24px 6px 10px;
                border: 1px solid #e2e8f0;
                border-radius: 10px;
                background: #ffffff;
                min-width: 80px;
                font-size: 12px;
                color: #0f172a;
            }
            QDoubleSpinBox#confSpinField {
                padding: 6px 24px 6px 10px;
            }
            QSpinBox#prefsMaxSpinField {
                padding: 6px 24px 6px 10px;
            }
            QDoubleSpinBox#confSpinField::up-button,
            QDoubleSpinBox#confSpinField::down-button {
                width: 20px;
                border: none;
                border-left: 1px solid #e2e8f0;
                background: #f8fafc;
            }
            QSpinBox#prefsMaxSpinField::up-button,
            QSpinBox#prefsMaxSpinField::down-button {
                width: 20px;
                border: none;
                border-left: 1px solid #e2e8f0;
                background: #f8fafc;
            }
            QDoubleSpinBox#confSpinField::up-button {
                border-top-right-radius: 10px;
            }
            QSpinBox#prefsMaxSpinField::up-button {
                border-top-right-radius: 10px;
            }
            QDoubleSpinBox#confSpinField::down-button {
                border-top: 1px solid #e2e8f0;
                border-bottom-right-radius: 10px;
            }
            QSpinBox#prefsMaxSpinField::down-button {
                border-top: 1px solid #e2e8f0;
                border-bottom-right-radius: 10px;
            }
            QDoubleSpinBox#confSpinField::up-button:hover,
            QDoubleSpinBox#confSpinField::down-button:hover {
                background: #eef2ff;
            }
            QSpinBox#prefsMaxSpinField::up-button:hover,
            QSpinBox#prefsMaxSpinField::down-button:hover {
                background: #eef2ff;
            }
            QDoubleSpinBox#confSpinField::up-button:pressed,
            QDoubleSpinBox#confSpinField::down-button:pressed {
                background: #e0e7ff;
            }
            QSpinBox#prefsMaxSpinField::up-button:pressed,
            QSpinBox#prefsMaxSpinField::down-button:pressed {
                background: #e0e7ff;
            }
            QDoubleSpinBox#confSpinField::up-arrow {
                image: url("./assets/icons/chevron_up_dark.svg");
                width: 10px;
                height: 10px;
            }
            QSpinBox#prefsMaxSpinField::up-arrow {
                image: url("./assets/icons/chevron_up_dark.svg");
                width: 10px;
                height: 10px;
            }
            QDoubleSpinBox#confSpinField::down-arrow {
                image: url("./assets/icons/chevron_down_dark.svg");
                width: 10px;
                height: 10px;
            }
            QSpinBox#prefsMaxSpinField::down-arrow {
                image: url("./assets/icons/chevron_down_dark.svg");
                width: 10px;
                height: 10px;
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
                padding: 10px 14px;
                margin-right: 8px;
                margin-bottom: 0px;
                margin-top: 0px;
                font-weight: 600;
                font-size: 12px;
                color: #64748b;
                min-height: 38px;
                min-width: 96px;
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
        self.models_roots: list[Path] = [MODELS_ROOT]
        self.current_model = None
        self.class_names = []

    def scan_models(self, custom_path=None):
        """扫描模型文件（递归子目录，多根目录去重）。"""
        models: list[dict] = []
        seen_resolved: set[Path] = set()
        roots: list[Path] = []
        if custom_path and str(custom_path).strip():
            cp = Path(custom_path).expanduser()
            if cp.is_dir():
                roots.append(cp.resolve())
        for r in self.models_roots:
            if r.is_dir():
                roots.append(r.resolve())
        for root in roots:
            try:
                for pt_file in sorted(root.rglob("*.pt")):
                    key = pt_file.resolve()
                    if key in seen_resolved:
                        continue
                    seen_resolved.add(key)
                    models.append({
                        'name': pt_file.name,
                        'path': str(pt_file),
                        'size': self._get_file_size(pt_file),
                        'modified': self._get_modification_time(pt_file)
                    })
            except Exception as e:
                print(f"扫描目录 {root} 时出错: {e}")

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
    # 视频：已处理帧数、总帧数、剩余秒数(按视频帧率估)、平均处理FPS(墙钟)
    video_time_hint = Signal(int, int, float, float)
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
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = 0
        class_names = list(self.model.names.values())

        self.status_changed.emit(f"开始处理视频 (共{total_frames}帧)...")

        video_t0 = time.time()
        while cap.isOpened() and self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            infer_t0 = time.time()
            results = self.model(
                frame, conf=self.confidence_threshold, verbose=False)
            infer_t1 = time.time()
            infer_s = infer_t1 - infer_t0

            original_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            frame_count += 1
            wall_elapsed = time.time() - video_t0
            proc_fps = frame_count / max(wall_elapsed, 1e-6)

            self.result_ready.emit(
                original_img, result_img, infer_s, results, class_names)

            if total_frames > 0:
                progress = int((frame_count / total_frames) * 100)
                self.progress_updated.emit(progress)
                left = max(0, total_frames - frame_count)
                if vid_fps > 1e-3:
                    rem_s = left / vid_fps
                else:
                    rem_s = left * max(infer_s, 1e-6)
                self.video_time_hint.emit(
                    frame_count, total_frames, rem_s, proc_fps)
            else:
                self.video_time_hint.emit(frame_count, 0, 0.0, proc_fps)

            # 视频文件：不调用 _update_fps（避免用单帧推理速度冒充整段处理速度）

            # 状态更新（每30帧更新一次）
            if frame_count % 30 == 0:
                if total_frames > 0:
                    self.status_changed.emit(
                        f"处理中... {frame_count}/{total_frames} 帧 (平均 {proc_fps:.1f} FPS)")
                else:
                    self.status_changed.emit(
                        f"处理中... 已处理 {frame_count} 帧 (平均 {proc_fps:.1f} FPS)")

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
        self._loaded_model_path = ""
        self.detection_thread = None
        self.batch_detection_thread = None
        self.current_source_type = 'image'
        self.current_source_path = None
        self.default_save_dir = str((data_dir / "results").absolute())
        self.confidence_threshold = 0.25
        self.batch_results = []
        self.current_batch_index = 0
        self._history_batch_mode = False
        self._history_last_snapshot = None
        self._history_run_start_mono = None
        self._history_pending_stop_reason = None
        self._history_last_error = ""
        self._history_run_sum_infer_s = 0.0
        self._history_video_frames_done = 0
        self._history_video_frames_total = 0
        self._history_started_at_str = ""
        self._video_export_started_at = ""
        self._video_export_ended_at = ""
        self._video_export_frame_index = 0
        self._tab_sync_source_change = False
        self.preset_data = {}
        self._delete_confirm_target = None
        self.task_preset_file = data_dir / "task_presets.json"

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
        self._last_realtime_overlay_ctx = None
        self._video_time_overlay = None  # (cur,tot,rem_sec,avg_proc_fps)|None
        self._current_file_full_path = None
        self.init_ui()
        self.setWindowIcon(self.create_enhanced_icon())

        self._last_applied_font_scale = None
        self._font_resize_timer = QTimer(self)
        self._font_resize_timer.setSingleShot(True)
        self._font_resize_timer.timeout.connect(self._apply_ui_font_scale)
        self._apply_ui_font_scale()

        # 应用级点击事件：保证子控件/空白区点击也能被捕获。
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

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
        # 顶部角标运行区：随字号缩放同步高度，防止按钮被 Tab 顶栏裁切
        compact_h = max(34, int(round(34 * min(s, 1.15))))
        corner_h = max(40, int(round(40 * min(s, 1.15))))
        for btn in (getattr(self, "start_btn", None), getattr(self, "pause_btn", None), getattr(self, "stop_btn", None)):
            if btn is not None:
                btn.setMinimumHeight(compact_h)
        if hasattr(self, "tab_widget"):
            tab_bar = self.tab_widget.tabBar()
            if tab_bar is not None:
                # 标签含图标+中文，需高于右侧角标区，避免第三项等被纵向裁切
                tab_bar_h = max(46, int(round(46 * min(s, 1.15))))
                tab_bar.setMinimumHeight(tab_bar_h)
            corner = self.tab_widget.cornerWidget(Qt.Corner.TopRightCorner)
            if corner is not None:
                corner.setMinimumHeight(corner_h)
        # 顶部运行按钮宽度参考预设区控件，保证上下视觉统一
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
        self.header_help_btn = QPushButton("使用帮助")
        self.header_help_btn.setObjectName("headerToolBtn")
        self._set_btn_icon(self.header_help_btn, "help", "#e5e7eb", 16)
        self.header_help_btn.clicked.connect(self.open_usage_help_dialog)
        header_layout.addWidget(self.header_help_btn, 0, Qt.AlignVCenter)
        wrap_layout.addWidget(app_header)
        self.result_detail_widget = DetectionResultWidget()
        self.result_detail_widget.stats_label.hide()

        center = self._build_main_workspace()
        wrap_layout.addWidget(center, 1)

        self.realtime_detail_panel = QFrame()
        self.realtime_detail_panel.setObjectName("bottomDrawer")
        panel_layout = QVBoxLayout(self.realtime_detail_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(0)
        self.log_text.setFont(StyleManager.log_mono_font(10))
        panel_layout.addWidget(self.result_detail_widget)

        self._main_left_column_layout.addWidget(self.realtime_detail_panel, 0)
        self.tab_widget.currentChanged.connect(self._on_main_tab_changed)
        self._on_main_tab_changed(self.tab_widget.currentIndex())

        main_layout.addWidget(content_wrap, 1)

        # 状态栏
        self.statusBar().showMessage("就绪 — 请选择模型与检测源")
        # 首次启动时同步：主 Tab 与输入源下拉保持单一状态来源
        if hasattr(self, "tab_widget"):
            self._sync_source_options_for_tab(self.tab_widget.currentIndex())

        self._setup_model_dir_watcher()

        # 尝试加载默认模型
        self.try_load_default_model()
        self._update_header_pills()
        self._update_current_file_display()

    def _on_main_tab_changed(self, index: int):
        """文件检测/批量分析共用底部检测明细；其它主 Tab 占满主画布。"""
        if hasattr(self, "realtime_detail_panel"):
            self.realtime_detail_panel.setVisible(index in (0, 1))
        self._sync_source_options_for_tab(index)
        # 避免“文件检测”实时结果串到“批量分析”页签：切页时按当前页语义重绘底部明细。
        if index == 0 and hasattr(self, "result_detail_widget"):
            # 从批量回到文件时，底部明细仍停留在批量表：按最后一次文件检测回调状态恢复。
            ctx = getattr(self, "_last_realtime_overlay_ctx", None)
            if ctx:
                results = ctx.get("results")
                class_names = ctx.get("class_names") or []
                infer_s = float(ctx.get("inference_time") or 0.0)
                self.result_detail_widget.update_results(
                    results, class_names, infer_s)
                try:
                    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                        self._status_objects.setText(str(len(results[0].boxes)))
                    else:
                        self._status_objects.setText("0")
                    self._status_latency.setText(f"{infer_s * 1000:.0f} 毫秒")
                    vt = getattr(self, "_video_time_overlay", None)
                    if (
                        getattr(self, "current_source_type", None) == "video"
                        and vt is not None
                        and len(vt) >= 4
                        and float(vt[3]) > 1e-6
                    ):
                        self._status_fps.setText(f"{float(vt[3]):.1f}")
                    elif infer_s > 1e-9:
                        self._status_fps.setText(f"{1.0 / infer_s:.2f}")
                    else:
                        self._status_fps.setText("-")
                except Exception:
                    self._status_objects.setText("-")
                    self._status_fps.setText("-")
                    self._status_latency.setText("-")
                self._sync_wireframe_overview_from_status()
            else:
                self.result_detail_widget.update_results(None, [], 0.0)
                self._status_objects.setText("-")
                self._status_fps.setText("-")
                self._status_latency.setText("-")
                self._sync_wireframe_overview_from_status()
        if index == 1 and hasattr(self, "result_detail_widget"):
            if getattr(self, "batch_results", None):
                idx = int(getattr(self, "current_batch_index", 0) or 0)
                idx = max(0, min(idx, len(self.batch_results) - 1))
                self.current_batch_index = idx
                self.show_batch_result(idx)
            else:
                self.result_detail_widget.update_results(None, [], 0.0)

    def _sync_source_options_for_tab(self, tab_index: int):
        """让右侧输入源选项与主 Tab 语义一致，避免双导航冲突。"""
        if not hasattr(self, "source_combo"):
            return
        tab_source_map = {
            0: ["单张图片", "视频文件"],       # 文件检测（摄像头仅在设备监控 Tab）
            1: ["文件夹批量"],                  # 批量分析
            2: ["摄像头"],                      # 设备监控
            3: ["单张图片", "视频文件"],       # 历史任务（与文件检测侧栏一致）
        }
        allowed = tab_source_map.get(tab_index, ["单张图片", "视频文件"])
        previous = self.source_combo.currentText()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(allowed)
        self.source_combo.blockSignals(False)
        target = previous if previous in allowed else allowed[0]
        self.source_combo.setCurrentText(target)
        # 强制同步内部 source_type，避免下拉文本与实际选择逻辑不一致。
        # 主 Tab 切换引发的同步不应清空文件检测画面/路径状态。
        self._tab_sync_source_change = True
        try:
            self.on_source_changed(target)
        finally:
            self._tab_sync_source_change = False
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

    def open_usage_help_dialog(self):
        """主页右上角全局使用帮助。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("使用帮助")
        dlg.resize(1020, 740)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # 复用原“高级模型选择”中的帮助内容，保持原封不动。
        help_widget = QWidget()
        prev_help_tab = getattr(self, "help_tab", None)
        self.help_tab = help_widget
        try:
            ModelSelectionDialog.setup_help_tab(self)
        finally:
            if prev_help_tab is not None:
                self.help_tab = prev_help_tab
            else:
                try:
                    delattr(self, "help_tab")
                except Exception:
                    pass
        lay.addWidget(help_widget, 1)

        # 热更新：帮助窗口打开期间，保存 docs/help_sections/*.md 会自动回显。
        help_dir = (base_dir / "docs" / "help_sections").resolve()
        reload_help = getattr(help_widget, "_reload_help_from_md", None)
        if callable(reload_help):
            watcher = QFileSystemWatcher(dlg)

            def _reload_help_md():
                reload_help()

            if help_dir.exists():
                watcher.addPath(str(help_dir))
                for p in help_dir.glob("*.md"):
                    watcher.addPath(str(p))

            def _on_file_changed(_path: str):
                # 某些平台会在保存后丢失监听，重载后重新挂载。
                _reload_help_md()
                for p in help_dir.glob("*.md"):
                    sp = str(p)
                    if sp not in watcher.files():
                        watcher.addPath(sp)

            def _on_dir_changed(_path: str):
                if help_dir.exists() and str(help_dir) not in watcher.directories():
                    watcher.addPath(str(help_dir))
                for p in help_dir.glob("*.md"):
                    sp = str(p)
                    if sp not in watcher.files():
                        watcher.addPath(sp)
                _reload_help_md()

            watcher.fileChanged.connect(_on_file_changed)
            watcher.directoryChanged.connect(_on_dir_changed)
            dlg._help_md_watcher = watcher

        _dlg_help_ff = StyleManager.help_document_font_family_css()
        dlg.setStyleSheet(StyleManager.get_main_stylesheet(1.0) + """
            QWidget#helpRoot {
                background: #eef2f6;
            }
            QWidget#helpShell {
                background: #eef2f6;
            }
            QWidget#helpArticleWrap {
                background: transparent;
            }
            QWidget#helpNavStrip {
                background: #ffffff;
                border: none;
                border-bottom: 1px solid #e2e8f0;
            }
            QTextEdit#helpMarkdownViewer {
                font-family: __DLG_HELP_FF__;
                border: none;
                border-radius: 8px;
                background: #ffffff;
                padding: 18px 20px 22px 20px;
                selection-background-color: #dbeafe;
                selection-color: #0f172a;
                color: #334155;
            }
            QTextEdit#helpMarkdownViewer QScrollBar:vertical {
                width: 6px;
                background: transparent;
                margin: 4px 2px 4px 0;
                border: none;
            }
            QTextEdit#helpMarkdownViewer QScrollBar::handle:vertical {
                min-height: 36px;
                background: #cbd5e1;
                border-radius: 4px;
            }
            QTextEdit#helpMarkdownViewer QScrollBar::handle:vertical:hover {
                background: #94a3b8;
            }
            QTextEdit#helpMarkdownViewer QScrollBar::add-line:vertical,
            QTextEdit#helpMarkdownViewer QScrollBar::sub-line:vertical {
                height: 0;
                width: 0;
            }
            QFrame#helpTocPanel {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #e2e8f0;
                border-radius: 10px;
            }
            QLabel#helpTocTitle {
                font-family: __DLG_HELP_FF__;
                color: #475569;
                font-weight: 600;
                font-size: 13px;
                letter-spacing: 0.02em;
                padding: 0;
            }
            QTreeWidget#helpTocTree {
                font-family: __DLG_HELP_FF__;
                border: none;
                background: transparent;
                padding: 0;
                font-size: 12px;
                color: #475569;
                outline: none;
            }
            QTreeWidget#helpTocTree::item {
                padding: 4px 5px;
                border-radius: 6px;
                min-height: 20px;
            }
            QTreeWidget#helpTocTree::item:hover {
                background: #f1f5f9;
                color: #0f172a;
            }
            QTreeWidget#helpTocTree::item:selected {
                background: #eff6ff;
                color: #1d4ed8;
                border: none;
                border-left: 3px solid #3b82f6;
                padding-left: 3px;
            }
            QTreeWidget#helpTocTree QScrollBar:vertical {
                width: 5px;
                background: transparent;
                margin: 2px 0;
            }
            QTreeWidget#helpTocTree QScrollBar::handle:vertical {
                min-height: 24px;
                background: #e2e8f0;
                border-radius: 3px;
            }
            QTreeWidget#helpTocTree QScrollBar::handle:vertical:hover {
                background: #cbd5e1;
            }
            QTreeWidget#helpTocTree QScrollBar::add-line:vertical,
            QTreeWidget#helpTocTree QScrollBar::sub-line:vertical {
                height: 0;
            }
            QPushButton#helpNavBtn {
                font-family: __DLG_HELP_FF__;
                min-height: 36px;
                padding: 0 14px;
                border-radius: 10px;
                border: 1px solid transparent;
                background: transparent;
                color: #475569;
                font-weight: 600;
                font-size: 14px;
                text-align: left;
            }
            QPushButton#helpNavBtn:hover {
                background: #f8fafc;
                border-color: #e2e8f0;
                color: #1e293b;
            }
            QPushButton#helpNavBtn:checked {
                background: #eef2ff;
                border: 1px solid #c7d2fe;
                color: #4338ca;
                font-weight: 600;
            }
        """.replace("__DLG_HELP_FF__", _dlg_help_ff))
        dlg.exec()

    @staticmethod
    def _set_btn_icon(btn, name: str, color: str = "#ffffff", size: int = 18):
        btn.setIcon(ThemeIcons.icon(name, size, color))
        btn.setIconSize(QSize(size, size))

    @staticmethod
    def _set_btn_icon_keep_color(btn, name: str, color: str = "#6366f1", size: int = 16):
        """与工具按钮统一：细线图标 + 禁用态保持同色。"""
        btn.setIcon(ThemeIcons.icon_same_when_disabled(name, size, color))
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
        """顶部任务预设入口：定位到右侧任务概览中的预设区。"""
        if hasattr(self, "preset_combo"):
            self.preset_combo.setFocus()
            self.log_message("已定位到任务概览中的任务预设，可进行新建/修改/删除。")

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
        self._set_btn_icon_keep_color(self.start_btn, "play", "#6366f1", 16)
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
        self._set_btn_icon_keep_color(self.pause_btn, "pause", "#6366f1", 16)
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
        self._set_btn_icon_keep_color(self.stop_btn, "square", "#6366f1", 16)
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
        """右侧栏：任务控制 + 任务概览（含任务预设）+ 运行日志；输入源在顶部浮层。"""
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
        tv.addWidget(self._wireframe_card_header(
            "任务控制", "settings", header_settings_btn=True))
        task_body = QWidget()
        tb = QVBoxLayout(task_body)
        tb.setContentsMargins(10, 12, 10, 12)
        tb.setSpacing(10)
        tb.addWidget(self._create_top_controls_corner(), 0)
        r_model = QHBoxLayout()
        align_row(r_model)
        r_model.addWidget(self._toolbar_label("模型"))
        self.model_combo = QComboBox()
        self.model_combo.setObjectName("modelComboField")
        self.model_combo.setMinimumWidth(96)
        self.model_combo.setMinimumHeight(32)
        self.model_combo.setMaxVisibleItems(10)
        self.model_combo.view().setUniformItemSizes(True)
        self.model_combo.view().setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        self.init_model_combo()
        r_model.addWidget(self.model_combo, 1)
        self.model_refresh_btn = QToolButton()
        self.model_refresh_btn.setObjectName("modelRefreshBtn")
        self.model_refresh_btn.setIcon(
            ThemeIcons.icon("refresh", 18, "#6366f1"))
        self.model_refresh_btn.setIconSize(QSize(18, 18))
        self.model_refresh_btn.setAutoRaise(True)
        self.model_refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.model_refresh_btn.setToolTip("重新扫描模型目录（也可在保存目录变更后自动刷新）")
        self.model_refresh_btn.clicked.connect(self.refresh_model_combo)
        r_model.addWidget(self.model_refresh_btn, 0)
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
        self.conf_spinbox.setObjectName("confSpinField")
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
        self.source_combo.addItems(["单张图片", "视频文件"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        r2_mode.addWidget(self.source_combo, 1)
        self.select_file_btn = QPushButton("浏览…")
        self._set_btn_icon(self.select_file_btn, "folder_open", "#6366f1")
        self.select_file_btn.clicked.connect(self.select_file)
        self.select_file_btn.setObjectName("toolBtn")
        self.select_file_btn.setMinimumHeight(34)
        self.select_file_btn.setMinimumWidth(80)
        self.select_file_btn.setToolTip(
            "根据模式选择图片、视频或批量目录；摄像头仅在「设备监控」页选择")
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

        r_pre_top = QHBoxLayout()
        align_row(r_pre_top)
        r_pre_top.addWidget(self._toolbar_label("任务预设", 64))
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("presetComboField")
        self.preset_combo.setMinimumHeight(32)
        self.preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.currentTextChanged.connect(
            self.on_preset_selection_changed)
        r_pre_top.addWidget(self.preset_combo, 1)
        preset_icon_row = QHBoxLayout()
        preset_icon_row.setSpacing(4)
        preset_icon_row.setContentsMargins(0, 0, 0, 0)
        self.new_preset_btn = QToolButton()
        self.new_preset_btn.setObjectName("presetNewIconBtn")
        self.new_preset_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.new_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_btn_icon_keep_color(
            self.new_preset_btn, "folder_plus", "#6366f1", 16)
        self.new_preset_btn.clicked.connect(self.create_new_preset)
        self.new_preset_btn.setFixedSize(32, 32)
        self.new_preset_btn.setToolTip("新建预设：基于当前配置创建一个新任务预设")
        preset_icon_row.addWidget(self.new_preset_btn)
        self.save_preset_btn = QToolButton()
        self.save_preset_btn.setObjectName("presetSaveIconBtn")
        self.save_preset_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.save_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_btn_icon_keep_color(
            self.save_preset_btn, "save", "#6366f1", 16)
        self.save_preset_btn.clicked.connect(self.save_current_preset)
        self.save_preset_btn.setEnabled(False)
        self.save_preset_btn.setFixedSize(32, 32)
        self.save_preset_btn.setToolTip("修改预设：将当前配置保存到已选预设")
        preset_icon_row.addWidget(self.save_preset_btn)
        self.delete_preset_btn = QToolButton()
        self.delete_preset_btn.setObjectName("presetDeleteBtn")
        self.delete_preset_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.delete_preset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_btn_icon_keep_color(
            self.delete_preset_btn, "trash", "#6366f1", 16)
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        self.delete_preset_btn.setEnabled(False)
        self.delete_preset_btn.setFixedSize(32, 32)
        self.delete_preset_btn.setToolTip("删除预设：删除当前选中的任务预设")
        preset_icon_row.addWidget(self.delete_preset_btn)
        r_pre_top.addLayout(preset_icon_row)
        ob.addLayout(r_pre_top)
        for _pb in (
                self.new_preset_btn,
                self.save_preset_btn,
                self.delete_preset_btn,
        ):
            _pb.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            _pb.installEventFilter(self)

        mode_row = QHBoxLayout()
        align_row(mode_row)
        mode_row.addWidget(self._toolbar_label("当前模式", 64))
        self.overview_mode_value = QComboBox()
        self.overview_mode_value.setObjectName("modeComboField")
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
        # 全局叠加开关：文件检测与批量分析共用
        overlay_row = QHBoxLayout()
        overlay_row.setContentsMargins(0, 2, 0, 0)
        overlay_row.setSpacing(8)
        overlay_row.addWidget(self._toolbar_label("图上指标", 64))
        self.overlay_metrics_check = QCheckBox("显示")
        self.overlay_metrics_check.setChecked(True)
        self.overlay_metrics_check.setToolTip(
            "在结果图上显示指标（不显示路径）；视频时含按视频帧率估算的剩余时间")
        self.overlay_metrics_check.toggled.connect(
            self._on_overlay_toggle_changed)
        overlay_row.addWidget(self.overlay_metrics_check, 0)
        overlay_row.addStretch(1)
        ob.addLayout(overlay_row)

        # 右侧指标卡取消后，将内部状态映射到隐藏状态标签，维持内部逻辑一致。
        self._wf_stat_labels = {
            "objects": self._status_objects,
            "fps": self._status_fps,
            "latency": self._status_latency,
            "model": self._status_model,
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
        self.export_detail_btn = QPushButton()
        self.export_detail_btn.setObjectName("exportDetailMenuBtn")
        self.export_detail_btn.setText("导出检测明细")
        self.export_detail_btn.setIcon(
            ThemeIcons.icon_same_when_disabled("download", 16, "#6366f1"))
        self.export_detail_btn.setIconSize(QSize(16, 16))
        self.export_detail_btn.setProperty("variant", "secondary")
        self.export_detail_btn.setMinimumHeight(34)
        self.export_detail_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.export_detail_btn.setToolTip(
            "选择导出格式：CSV / JSON / 文本 / Excel（含汇总指标与目标明细）")
        _export_menu = QMenu(self.export_detail_btn)
        _export_menu.setObjectName("exportDetailMenu")
        for _fid, _label in (
            ("csv", "CSV 表格（.csv）"),
            ("json", "JSON 数据（.json）"),
            ("txt", "纯文本报告（.txt）"),
            ("xlsx", "Excel 工作簿（.xlsx）"),
        ):
            _a = _export_menu.addAction(_label)
            _a.setData(_fid)
        self.export_detail_btn.setMenu(_export_menu)
        _export_menu.triggered.connect(self._on_export_detail_format)
        self.export_detail_btn.setEnabled(False)
        ov_btns.addWidget(self.export_detail_btn, 1)
        _eq_w = max(
            self.open_result_dir_btn.sizeHint().width(),
            self.export_detail_btn.sizeHint().width(),
        )
        self.open_result_dir_btn.setMinimumWidth(_eq_w)
        self.export_detail_btn.setMinimumWidth(_eq_w)
        self.open_result_dir_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.export_detail_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

    def _wireframe_card_header(
        self, title: str, icon_name: str, header_settings_btn: bool = False
    ):
        """线框稿卡片顶栏：标题占位条 + 文案 + 右侧图标（可选可点击齿轮）。"""
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
        if header_settings_btn:
            btn = QToolButton()
            btn.setObjectName("taskCardHeaderSettingsBtn")
            btn.setIcon(ThemeIcons.icon(icon_name, 18, "#64748b"))
            btn.setIconSize(QSize(18, 18))
            btn.setAutoRaise(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip("高级模型选择（浏览、下载与管理权重）")
            btn.clicked.connect(self.show_model_selection_dialog)
            self.task_control_header_settings_btn = btn
            hl.addWidget(btn, 0, Qt.AlignVCenter)
        else:
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
            if hasattr(self, "overview_model_metric_value"):
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

    def _on_export_detail_format(self, action):
        """侧栏导出菜单：按所选格式写出检测明细。"""
        fmt = action.data()
        if not isinstance(fmt, str) or not hasattr(self, "result_detail_widget"):
            return
        self.result_detail_widget.export_detection_detail(fmt)

    def _history_task_type_label(self) -> str:
        if hasattr(self, "tab_widget") and self.tab_widget.currentIndex() == 2:
            return "设备监控"
        st = getattr(self, "current_source_type", "") or ""
        return {
            "image": "文件 · 图片",
            "video": "文件 · 视频",
            "camera": "设备监控 · 摄像头",
        }.get(st, st or "文件检测")

    def _history_source_summary(self) -> str:
        # 优先使用当前任务真实来源路径（含从预设读取的 source_path），避免受界面显示文本影响
        st = getattr(self, "current_source_type", "") or ""
        if st == "camera":
            cam_text = self.camera_combo.currentText() if hasattr(
                self, "camera_combo") else ""
            return cam_text.strip() or "摄像头"
        t = (getattr(self, "current_source_path", None) or "").strip()
        if not t:
            t = (getattr(self, "_current_file_full_path", None) or "").strip()
        if not t and hasattr(self, "current_file_label"):
            t = (self.current_file_label.toolTip() or "").strip() or (
                self.current_file_label.text() or "").strip()
        if len(t) > 96:
            return t[:44] + "…" + t[-44:]
        return t or "-"

    def _update_history_snapshot(self, results, class_names, inference_time):
        """更新写入历史表用的快照（视频每帧覆盖；停止时在 _flush 中补全状态与墙钟耗时）。"""
        object_count = 0
        class_note = ""
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            object_count = len(results[0].boxes)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            class_counts = {}
            for cls in classes:
                cname = class_names[cls] if cls < len(
                    class_names) else f"类别{cls}"
                class_counts[cname] = class_counts.get(cname, 0) + 1
            class_note = ", ".join(
                [f"{n}:{c}" for n, c in class_counts.items()])
        model = self.model_combo.currentText() if hasattr(
            self, "model_combo") else "-"
        self._history_last_snapshot = {
            "time_str": _format_export_local_time(),
            "task_type": self._history_task_type_label(),
            "source": self._history_source_summary(),
            "model": model,
            "objects": object_count,
            "inference_s": float(inference_time),
            "note": class_note[:160],
        }

    def _flush_history_task_snapshot(self):
        """将文件检测快照写入历史表：补充结束状态、结束时间、墙钟耗时与视频进度。"""
        if getattr(self, "_history_batch_mode", False):
            return
        snap = getattr(self, "_history_last_snapshot", None)
        if not snap or not hasattr(self, "task_history_widget"):
            return
        wall_s = 0.0
        if self._history_run_start_mono is not None:
            wall_s = max(0.0, time.monotonic() - self._history_run_start_mono)
        else:
            wall_s = float(snap.get("inference_s") or 0.0)

        reason = self._history_pending_stop_reason
        st = getattr(self, "current_source_type", "") or ""
        if reason is None:
            if st == "image":
                reason = "completed"
            elif st == "video":
                tot = int(self._history_video_frames_total)
                cur = int(self._history_video_frames_done)
                if tot > 0:
                    reason = "completed" if cur >= tot else "incomplete"
                else:
                    reason = "completed"
            else:
                reason = "completed"

        reason_cn = {
            "completed": "检测已完成",
            "user_stop": "用户手动停止",
            "error": "检测异常结束",
            "incomplete": "检测未完成",
        }.get(reason, str(reason))

        end_ts = _format_export_local_time()
        parts = []
        sum_inf = float(self._history_run_sum_infer_s or 0.0)
        if reason in ("error", "incomplete"):
            if st == "video" and self._history_video_frames_total > 0:
                parts.append(
                    f"帧进度 {self._history_video_frames_done}/"
                    f"{self._history_video_frames_total}")
            if sum_inf > 0:
                parts.append(f"累计推理 {sum_inf:.2f}s")
        if reason == "error" and self._history_last_error:
            parts.append(self._history_last_error[:200])
        cls_note = (snap.get("note") or "").strip()
        if cls_note:
            parts.append(cls_note)

        snap["note"] = " ； ".join(parts)[:650] if parts else "—"
        snap["inference_s"] = wall_s
        snap["time_str"] = end_ts

        if st == "image":
            fd, ft = 1, 1
        elif st == "video":
            fd = int(self._history_video_frames_done)
            ft = int(self._history_video_frames_total)
        else:
            fd = int(self._history_video_frames_done)
            ft = int(self._history_video_frames_total)
        if wall_s > 1e-9:
            if st == "video" and fd > 0:
                avg_fps = fd / wall_s
            elif st == "image":
                avg_fps = 1.0 / wall_s
            elif fd > 0:
                avg_fps = fd / wall_s
            else:
                avg_fps = 0.0
        else:
            avg_fps = 0.0

        self.task_history_widget.add_record(
            time_str=end_ts,
            task_type=snap["task_type"],
            source=snap["source"],
            model=snap["model"],
            objects=int(snap["objects"]),
            inference_s=wall_s,
            note=snap["note"],
            started_at=getattr(self, "_history_started_at_str", "") or "",
            det_status=reason_cn,
            avg_proc_fps=float(avg_fps),
            frames_done=fd,
            frames_total=ft,
            sum_infer_s=sum_inf,
        )
        self._history_last_snapshot = None

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
        mtb.setElideMode(Qt.TextElideMode.ElideNone)
        mtb.show()

        realtime_tab = self.create_realtime_tab()
        batch_tab = self.create_batch_tab()
        self.monitor_tab = MonitoringWidget(
            self.model_manager, self.camera_manager)
        self.tab_widget.addTab(
            realtime_tab, ThemeIcons.icon("radio", 17, "#0ea5e9"), "文件检测")
        self.tab_widget.addTab(
            batch_tab, ThemeIcons.icon("folders", 17, "#0ea5e9"), "批量分析")
        self.tab_widget.addTab(
            self.monitor_tab, ThemeIcons.icon("monitor", 17, "#0ea5e9"), "设备监控")
        self.task_history_widget = TaskHistoryWidget(data_dir)
        self.tab_widget.addTab(
            self.task_history_widget,
            ThemeIcons.icon("list", 17, "#0ea5e9"),
            "历史任务",
        )

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
        rh.setSpacing(0)
        rh.addWidget(left_col, 16)
        rh.addWidget(right, 7)
        return row

    def create_realtime_tab(self):
        """创建文件检测标签页（单张图片 / 视频文件）。"""
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

        self.original_label = QLabel(
            "暂无输入源\n请选择图片或视频；摄像头请使用「设备监控」页")
        self.original_label.setObjectName("previewPlaceholder")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(240, 180)
        self.original_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.original_label.setScaledContents(False)
        self.original_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.original_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction)
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
        self.result_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction)
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
        # 与批量明细表、右侧指标和图上叠加信息重复，默认隐藏。
        self.batch_info_label.setVisible(False)
        layout.addWidget(self.batch_info_label)

        return widget

    def init_model_combo(self):
        """首次填充模型下拉（不触发切换回调，由 try_load_default_model 加载）。"""
        self.refresh_model_combo(reload_if_changed=False)

    def _model_watch_directory_candidates(self):
        """与扫描根目录一致，供 QFileSystemWatcher 监视（非递归，子目录变更多数仍会触发）。"""
        seen = set()
        out = []
        for p in (
            MODELS_ROOT,
            MODELS_DIR_CUSTOM,
            MODELS_DIR_OFFICIAL,
        ):
            try:
                pr = Path(p).resolve()
                key = str(pr)
                if pr.is_dir() and key not in seen:
                    seen.add(key)
                    out.append(key)
            except OSError:
                continue
        return out

    def _setup_model_dir_watcher(self):
        self._model_dir_watch = QFileSystemWatcher(self)
        self._model_refresh_debounce = QTimer(self)
        self._model_refresh_debounce.setSingleShot(True)
        self._model_refresh_debounce.setInterval(420)
        self._model_refresh_debounce.timeout.connect(
            self._debounced_refresh_model_combo)
        for d in self._model_watch_directory_candidates():
            self._model_dir_watch.addPath(d)
        self._model_dir_watch.directoryChanged.connect(
            self._on_model_watch_dirs_changed)

    def _on_model_watch_dirs_changed(self, _path: str):
        self._model_refresh_debounce.start()

    def _debounced_refresh_model_combo(self):
        self.refresh_model_combo()

    def refresh_model_combo(self, reload_if_changed: bool = True):
        """重新扫描 .pt 并刷新下拉框；尽量保持当前选中名；必要时重新加载模型。"""
        if not hasattr(self, "model_combo"):
            return
        prev_text = self.model_combo.currentText()
        was_enabled = self.model_combo.isEnabled()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = self.model_manager.scan_models()
        if not models:
            self.model_combo.addItem("无可用模型")
            self.model_combo.setEnabled(False)
        else:
            self.model_combo.addItems([m["name"] for m in models])
            self.model_combo.setEnabled(was_enabled)
            pick = None
            if prev_text and prev_text != "无可用模型":
                if self.model_combo.findText(prev_text) >= 0:
                    pick = prev_text
            if pick is None:
                prev_path = getattr(self, "_loaded_model_path", "") or ""
                if prev_path:
                    try:
                        bn = Path(prev_path).name
                        if self.model_combo.findText(bn) >= 0:
                            pick = bn
                    except OSError:
                        pass
            if pick:
                self.model_combo.setCurrentText(pick)
            elif self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)
        if not reload_if_changed:
            return
        new_text = self.model_combo.currentText()
        if new_text == "无可用模型":
            return
        if new_text != prev_text:
            self.on_model_changed(new_text)
            return
        prev_path = getattr(self, "_loaded_model_path", "") or ""
        if prev_path and new_text == prev_text:
            try:
                if not Path(prev_path).is_file():
                    self.on_model_changed(new_text)
            except OSError:
                self.on_model_changed(new_text)

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
            self._loaded_model_path = str(Path(model_path).resolve())
            self.log_message(f"模型加载成功: {Path(model_path).name}")
            # self.update_button_states()
            self._update_header_pills()
            return True
        except Exception as e:
            self.log_message(f"模型加载失败: {str(e)}")
            self.model = None
            self._loaded_model_path = ""
            self._update_header_pills()
            return False

    def show_model_selection_dialog(self):
        """显示模型选择对话框"""
        dialog = ModelSelectionDialog(self.model_manager, self)
        if dialog.exec() == QDialog.Accepted and dialog.selected_model:
            path = dialog.selected_model
            model_name = Path(path).name
            self.refresh_model_combo(reload_if_changed=False)
            self.model_combo.blockSignals(True)
            ix = self.model_combo.findText(model_name)
            if ix >= 0:
                self.model_combo.setCurrentIndex(ix)
            else:
                self.model_combo.addItem(model_name)
                self.model_combo.setCurrentText(model_name)
            self.model_combo.blockSignals(False)
            self.load_model(path)

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

        is_tab_sync = bool(getattr(self, "_tab_sync_source_change", False))
        if not is_tab_sync:
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
            self.default_save_dir).exists() else str(data_dir)
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
        save_txt = ""
        if hasattr(self, "save_dir_edit"):
            save_txt = self.save_dir_edit.text().strip()
        save_dir = save_txt or self.default_save_dir
        return {
            "model_name": self.model_combo.currentText(),
            "confidence_threshold": float(self.confidence_threshold),
            "source_type": self.current_source_type,
            "source_path": self.current_source_path,
            "save_dir": save_dir,
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
            self.delete_preset_btn.setToolTip("删除预设：删除当前选中的任务预设")
            self._reset_delete_preset_btn_surface_after_confirm()

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
            elif source_type == "camera":
                self.tab_widget.setCurrentIndex(2)
            elif source_type in ("image", "video"):
                self.tab_widget.setCurrentIndex(0)
        source_label = self._source_type_to_label(source_type)
        source_index = self.source_combo.findText(source_label)
        if source_index >= 0:
            self.source_combo.setCurrentIndex(source_index)

        source_path = config.get("source_path")
        if self.current_source_type == "camera":
            pass
        elif source_path:
            path_obj = Path(source_path)
            if path_obj.exists():
                resolved = str(path_obj.resolve())
                self.current_source_path = resolved
                self._current_file_full_path = resolved
                self._update_current_file_display()
                if self.current_source_type in ["image", "video"]:
                    self.preview_file(resolved)
            else:
                self.current_source_path = None
                self._current_file_full_path = None
                self._update_current_file_display()
                self.log_message(f"预设中的源路径已不存在，已清空: {source_path}")
        else:
            if self.current_source_type in ("image", "video", "batch"):
                self.current_source_path = None
                self._current_file_full_path = None
                self._update_current_file_display()
                self.log_message(
                    "提示：该预设未保存输入文件/目录路径；请先在右侧选好路径后点击「修改预设」保存到预设。"
                )

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
            self.delete_preset_btn.setToolTip(
                f"确认删除：再次点击将删除预设「{preset_name}」")
            self._set_btn_icon_keep_color(
                self.delete_preset_btn, "trash", "#ffffff", 16)
            self.delete_preset_btn.setStyleSheet(
                "QToolButton#presetDeleteBtn { background:#dc2626; color:#ffffff; "
                "border:1px solid #b91c1c; border-radius:8px; padding:0px; }"
                "QToolButton#presetDeleteBtn:hover { background:#b91c1c; "
                "border-color:#991b1b; }"
            )
            return

        self.preset_data.pop(preset_name, None)
        if self._write_task_presets_to_file():
            self._delete_confirm_target = None
            self.delete_preset_btn.setToolTip("删除预设：删除当前选中的任务预设")
            self._reset_delete_preset_btn_surface_after_confirm()
            self._refresh_preset_combo()
            self.log_message(f"任务预设已删除: {preset_name}")

    def _apply_preset_tool_icon_for_toolbar_hover(
            self, btn, icon_name: str, hovered: bool):
        """预设行图标按钮：白底时为靛色图标，悬停渐变底时为白色图标。"""
        if btn is getattr(self, "delete_preset_btn", None) and getattr(
                self, "_delete_confirm_target", None):
            self._set_btn_icon_keep_color(btn, "trash", "#ffffff", 16)
            return
        color = "#ffffff" if hovered else "#6366f1"
        self._set_btn_icon_keep_color(btn, icon_name, color, 16)

    def _reset_delete_preset_btn_surface_after_confirm(self):
        """清除确认删除的红底样式，并按是否仍悬停恢复图标颜色。"""
        btn = getattr(self, "delete_preset_btn", None)
        if btn is None:
            return
        btn.setStyleSheet("")
        if btn.underMouse():
            self._apply_preset_tool_icon_for_toolbar_hover(btn, "trash", True)
        else:
            self._set_btn_icon_keep_color(btn, "trash", "#6366f1", 16)

    def _reset_history_path_cells_on_outside_click(self, global_pos):
        """点击路径框外部时，恢复历史来源路径的默认显示。"""
        thw = getattr(self, "task_history_widget", None)
        tb = getattr(thw, "table", None)
        if tb is None:
            return
        source_col = getattr(thw, "_SOURCE_COL", 12)
        for r in range(tb.rowCount()):
            src_edit = tb.cellWidget(r, source_col)
            if not isinstance(src_edit, PathCellLineEdit):
                continue
            local_pos = src_edit.mapFromGlobal(global_pos)
            if not src_edit.rect().contains(local_pos):
                src_edit._restore_default_view()

    def eventFilter(self, obj, event):
        """当处于“确认删除”状态时，点击其它区域会自动取消高亮。"""
        try:
            if event.type() in (QEvent.HoverEnter, QEvent.HoverLeave):
                hovered = event.type() == QEvent.HoverEnter
                np = getattr(self, "new_preset_btn", None)
                sp = getattr(self, "save_preset_btn", None)
                dp = getattr(self, "delete_preset_btn", None)
                if obj is np and np is not None:
                    self._apply_preset_tool_icon_for_toolbar_hover(
                        np, "folder_plus", hovered)
                elif obj is sp and sp is not None:
                    self._apply_preset_tool_icon_for_toolbar_hover(
                        sp, "save", hovered)
                elif obj is dp and dp is not None:
                    self._apply_preset_tool_icon_for_toolbar_hover(
                        dp, "trash", hovered)
            if event.type() == QEvent.MouseButtonPress:
                global_pos = event.globalPos()
                # 历史来源路径：应用内点击路径框外部时恢复默认展示。
                self._reset_history_path_cells_on_outside_click(global_pos)

                # 历史任务表：点击表格外任意区域时，取消当前行选中高亮
                thw = getattr(self, "task_history_widget", None)
                tb = getattr(thw, "table", None)
                if tb is not None and tb.selectionModel() and tb.selectionModel().hasSelection():
                    vp_pos = tb.viewport().mapFromGlobal(global_pos)
                    if not tb.viewport().rect().contains(vp_pos):
                        tb.clearSelection()
                        tb.setCurrentCell(-1, -1)

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
                    self.delete_preset_btn.setToolTip(
                        "删除预设：删除当前选中的任务预设")
                    self._reset_delete_preset_btn_surface_after_confirm()
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
            self._history_last_snapshot = None
            self._history_batch_mode = False
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
        self._history_last_snapshot = None
        self._history_batch_mode = False
        self._history_pending_stop_reason = None
        self._history_last_error = ""
        self._history_run_sum_infer_s = 0.0
        self._history_video_frames_done = 0
        self._history_video_frames_total = 0
        self._history_run_start_mono = time.monotonic()
        self._history_started_at_str = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        self._video_export_frame_index = 0
        if self.current_source_type in ("video", "camera"):
            self._video_export_started_at = self._history_started_at_str
        else:
            self._video_export_started_at = ""
        self._video_export_ended_at = ""
        self._video_time_overlay = None
        camera_id = 0
        if self.current_source_type == "camera":
            camera_id = self.camera_combo.currentData()
            if camera_id == -1:
                self.log_message("错误: 没有可用的摄像头")
                self._history_run_start_mono = None
                return

        self.detection_thread = DetectionThread(
            self.model, self.current_source_type, self.current_source_path, camera_id, self.confidence_threshold
        )
        self.detection_thread.result_ready.connect(self.on_detection_result)
        self.detection_thread.progress_updated.connect(
            self._on_main_progress)
        self.detection_thread.status_changed.connect(
            self.statusBar().showMessage)
        self.detection_thread.error_occurred.connect(
            self._on_detection_thread_error)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.fps_updated.connect(self._on_fps_updated)
        self.detection_thread.video_time_hint.connect(
            self._on_video_time_hint)

        self.update_detection_ui_state(True)
        self.tab_widget.setCurrentIndex(0)  # 切换到文件检测

        self.detection_thread.start()
        self.log_message(f"开始{self.current_source_type}检测…")

    def start_batch_detection(self):
        """开始批量检测"""
        self._history_last_snapshot = None
        self._history_batch_mode = True
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
        tcs = getattr(self, "task_control_header_settings_btn", None)
        if tcs is not None:
            tcs.setEnabled(not detecting)
        mrb = getattr(self, "model_refresh_btn", None)
        if mrb is not None:
            mrb.setEnabled(not detecting)
        self._sync_export_detail_button_state()
        self._update_header_pills()

    def _sync_export_detail_button_state(self, has_detail: bool | None = None):
        """导出明细按钮状态同步：视频检测进行中禁用，结束后按是否有明细决定。"""
        btn = getattr(self, "export_detail_btn", None)
        if btn is None:
            return
        dt = getattr(self, "detection_thread", None)
        video_running = bool(
            getattr(self, "current_source_type", None) == "video"
            and dt is not None
            and getattr(dt, "is_running", False)
        )
        if video_running:
            btn.setEnabled(False)
            btn.setToolTip("视频检测进行中，请结束或停止后导出检测明细")
            return
        if has_detail is None:
            rdw = getattr(self, "result_detail_widget", None)
            rows = getattr(rdw, "_detail_csv_rows", None)
            has_detail = bool(rows)
        btn.setEnabled(bool(has_detail))
        btn.setToolTip("选择导出格式：CSV / JSON / 文本 / Excel（含汇总指标与目标明细）")

    def pause_detection(self):
        """暂停/恢复检测"""
        if self.tab_widget.currentIndex() == 2:
            if self.monitor_tab.is_monitoring_active():
                self.monitor_tab.stop_monitoring()
                if self.monitor_tab.is_monitoring_paused():
                    self.pause_btn.setText("继续")
                    self._set_btn_icon_keep_color(
                        self.pause_btn, "play", "#6366f1", 16)
                    self.log_message("监控已暂停")
                else:
                    self.pause_btn.setText("暂停")
                    self._set_btn_icon_keep_color(
                        self.pause_btn, "pause", "#6366f1", 16)
                    self.log_message("监控已恢复")
            return

        if self.detection_thread and self.detection_thread.is_running:
            if self.detection_thread.is_paused:
                self.detection_thread.resume()
                self.pause_btn.setText("暂停")
                self._set_btn_icon_keep_color(
                    self.pause_btn, "pause", "#6366f1", 16)
                self.log_message("检测已恢复")
            else:
                self.detection_thread.pause()
                self.pause_btn.setText("继续")
                self._set_btn_icon_keep_color(
                    self.pause_btn, "play", "#6366f1", 16)
                self.log_message("检测已暂停")

    def stop_detection(self):
        """停止检测"""
        if self.tab_widget.currentIndex() == 2:
            self.monitor_tab.clear_monitoring()
            self.on_detection_finished()
            self.log_message("设备监控已停止")
            return

        if self.detection_thread and self.detection_thread.is_running:
            self._history_pending_stop_reason = "user_stop"
            self.detection_thread.stop()
            self.detection_thread.wait()

        if self.batch_detection_thread and self.batch_detection_thread.is_running:
            self.batch_detection_thread.stop()
            self.batch_detection_thread.wait()

        self.on_detection_finished()

    def _history_ensure_snapshot_for_error(self):
        """出错时尚未产生任何结果帧时补一条空快照，便于历史表落盘。"""
        if getattr(self, "_history_last_snapshot", None) is not None:
            return
        if getattr(self, "_history_run_start_mono", None) is None:
            return
        model = self.model_combo.currentText() if hasattr(
            self, "model_combo") else "-"
        self._history_last_snapshot = {
            "time_str": _format_export_local_time(),
            "task_type": self._history_task_type_label(),
            "source": self._history_source_summary(),
            "model": model,
            "objects": 0,
            "inference_s": 0.0,
            "note": "",
        }

    def _on_detection_thread_error(self, msg: str):
        self._history_pending_stop_reason = "error"
        self._history_last_error = (str(msg) or "").strip()[:400]
        self._history_ensure_snapshot_for_error()
        self.log_message(str(msg))

    def on_detection_result(self, original_img, result_img, inference_time, results, class_names):
        """检测结果回调"""
        is_file_tab = bool(
            getattr(self, "tab_widget", None) is not None
            and self.tab_widget.currentIndex() == 0
        )
        if getattr(self, "current_source_type", None) in ("video", "camera"):
            self._video_export_frame_index += 1
            self._video_export_ended_at = _format_export_local_time()
        try:
            self._history_run_sum_infer_s += float(inference_time)
        except Exception:
            pass
        if original_img is not None:
            self._last_preview_original = np.ascontiguousarray(original_img)
        if result_img is not None:
            self._last_preview_result = np.ascontiguousarray(result_img)
        self._last_realtime_overlay_ctx = {
            "results": results,
            "class_names": class_names,
            "inference_time": inference_time,
        }
        # 显示图像
        self.display_image(original_img, self.original_label)
        self.display_image(
            self._render_realtime_overlay_image(result_img), self.result_label)

        # 仅在“文件检测”主 Tab 下刷新共享明细，避免串到批量模式
        if is_file_tab:
            self.result_detail_widget.update_results(
                results, class_names, inference_time)
        self._refresh_wireframe_sidebar(results, class_names, inference_time)

        # 记录日志（简化版，避免过多输出）
        if results and results[0].boxes and len(results[0].boxes) > 0:
            object_count = len(results[0].boxes)
            if is_file_tab:
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
            if is_file_tab:
                self._status_objects.setText("0")
                self._status_latency.setText(f"{inference_time*1000:.0f} 毫秒")

        self._update_history_snapshot(
            results, class_names, inference_time)

    def _on_fps_updated(self, fps):
        try:
            self._status_fps.setText(f"{fps:.1f}")
        except Exception:
            self._status_fps.setText("-")
        self._sync_wireframe_overview_from_status()

    @staticmethod
    def _format_video_remaining_cn(sec: float) -> str:
        if sec <= 0:
            return "即将完成"
        if sec >= 3600 * 24:
            return "剩余 —"
        total = int(round(sec))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"剩余约 {h}:{m:02d}:{s:02d}"
        return f"剩余约 {m}:{s:02d}"

    def _on_video_time_hint(self, cur: int, total: int, rem_s: float, proc_fps: float):
        self._video_time_overlay = (
            int(cur), int(total), float(rem_s), float(proc_fps))
        self._history_video_frames_done = int(cur)
        self._history_video_frames_total = int(total)
        try:
            if getattr(self, "current_source_type", None) == "video":
                self._status_fps.setText(f"{float(proc_fps):.1f}")
                self._sync_wireframe_overview_from_status()
        except Exception:
            pass
        if getattr(self, "tab_widget", None) is None:
            return
        if self.tab_widget.currentIndex() != 0:
            return
        if getattr(self, "current_source_type", None) != "video":
            return
        if self._last_preview_result is None or not self._is_overlay_enabled():
            return
        self.display_image(
            self._render_realtime_overlay_image(self._last_preview_result),
            self.result_label,
        )

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

        # 仅在“批量分析”主 Tab 下显示首个批量结果，避免覆盖文件检测明细
        if len(self.batch_results) == 1 and getattr(self, "tab_widget", None) is not None and self.tab_widget.currentIndex() == 1:
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

        if hasattr(self, "task_history_widget") and total_count > 0:
            sum_inf = sum(float(r["inference_time"])
                          for r in self.batch_results)
            avg_inf = sum_inf / total_count
            src = (self.current_source_path or "").strip() or "-"
            ended = _format_export_local_time()
            avg_fps = (
                float(total_count) / max(float(sum_inf), 1e-9)
                if total_count > 0 else 0.0)
            self.task_history_widget.add_record(
                time_str=ended,
                task_type="批量分析",
                source=src,
                model=self.model_combo.currentText() if hasattr(
                    self, "model_combo") else "-",
                objects=total_objects,
                inference_s=round(sum_inf, 4),
                note=f"共 {total_count} 张 · 均 {avg_inf:.4f}s/张",
                started_at="",
                det_status="检测已完成",
                avg_proc_fps=avg_fps,
                frames_done=total_count,
                frames_total=total_count,
                sum_infer_s=float(sum_inf),
            )

        self.log_message(
            f"批量检测完成：共 {total_count} 张图片，检出 {total_objects} 个目标")
        self.statusBar().showMessage(
            f"批量检测完成 - {total_count} 张图片，{total_objects} 个目标")

        self.save_results_btn.setEnabled(True)
        self.clear_results_btn.setEnabled(True)
        self.result_index_label.setText(f"1/{len(self.batch_results)}")
        self.on_detection_finished(completed=True)
        self._history_batch_mode = False

    def _is_overlay_enabled(self) -> bool:
        return not hasattr(self, "overlay_metrics_check") or self.overlay_metrics_check.isChecked()

    def _render_overlay_image(self, base_img, line1: str, line2: str | None = None):
        if base_img is None or not self._is_overlay_enabled():
            return base_img
        rgb = np.ascontiguousarray(base_img.copy())
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return base_img
        h, w = rgb.shape[:2]
        margin_x, margin_y = 14, 18
        pad_x, pad_y = 14, 12
        line_gap = 8
        top_lead = 3
        font_size = int(max(13, min(21, w / 50.0)))
        font = _load_overlay_cjk_font(font_size)

        lines: list[str] = [line1.strip()] if line1 else []
        if line2 and str(line2).strip():
            lines.append(str(line2).strip())
        if not lines:
            return base_img

        pil_base = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        layer = Image.new("RGBA", pil_base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        dummy = ImageDraw.Draw(Image.new("RGB", (max(w, 4), max(h, 4))))

        max_text_w = max(80, w - margin_x * 2 - pad_x * 2)
        fitted: list[str] = []
        for raw in lines:
            fitted.append(_fit_overlay_line_text(raw, font, dummy, max_text_w))

        bbs = [dummy.textbbox((0, 0), t, font=font) for t in fitted]
        text_heights = [b[3] - b[1] for b in bbs]
        text_block_h = sum(text_heights) + line_gap * max(0, len(fitted) - 1)
        box_inner_w = max(b[2] - b[0] for b in bbs)
        box_w = min(w - margin_x * 2, box_inner_w + pad_x * 2)
        box_h = pad_y * 2 + top_lead + text_block_h
        x1 = max(margin_x, (w - box_w) // 2)
        y1 = margin_y
        x2 = min(w - margin_x, x1 + box_w)
        y2 = min(h - margin_y, y1 + box_h)
        rad = int(max(8, min(16, font_size)))

        fill = (30, 41, 59, 118)
        outline = (129, 140, 248, 140)
        draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=rad, fill=fill, outline=outline, width=1)

        y_cursor = y1 + pad_y + top_lead
        for i, t in enumerate(fitted):
            bb0 = dummy.textbbox((0, 0), t, font=font)
            line_w = bb0[2] - bb0[0]
            tx = x1 + (box_w - line_w) // 2
            bb = draw.textbbox((tx, y_cursor), t, font=font)
            if bb[1] < y1 + 2:
                y_cursor += y1 + 2 - bb[1]
                bb = draw.textbbox((tx, y_cursor), t, font=font)
            col = (248, 250, 252, 255) if i == 0 else (199, 210, 254, 255)
            draw.text((tx, y_cursor), t, font=font, fill=col)
            y_cursor = bb[3] + line_gap

        out = Image.alpha_composite(pil_base, layer)
        return np.ascontiguousarray(np.asarray(out.convert("RGB")))

    def _render_batch_overlay_image(self, result: dict):
        """按需在批量结果图上叠加简要指标信息（不显示路径/文件名）。"""
        object_count = int(result.get("object_count", 0))
        infer_s = float(result.get("inference_time", 0.0))
        fps = (1.0 / infer_s) if infer_s > 1e-9 else 0.0
        line1 = (
            f"目标 {object_count}  ·  耗时 {infer_s * 1000:.1f} ms  ·  {fps:.1f} FPS")
        return self._render_overlay_image(result.get("result_img"), line1, None)

    def _render_realtime_overlay_image(self, result_img):
        ctx = self._last_realtime_overlay_ctx or {}
        infer_s = float(ctx.get("inference_time", 0.0) or 0.0)
        inst_fps = (1.0 / infer_s) if infer_s > 1e-9 else 0.0
        results = ctx.get("results")
        object_count = 0
        if results and results[0].boxes is not None:
            object_count = len(results[0].boxes)
        if getattr(self, "current_source_type", None) == "video":
            vt = getattr(self, "_video_time_overlay", None)
            proc_fps = float(vt[3]) if vt is not None and len(vt) >= 4 else 0.0
            if proc_fps > 1e-6:
                line1 = (
                    f"目标 {object_count}  ·  本帧 {infer_s * 1000:.1f} ms"
                    f"  ·  平均 {proc_fps:.1f} FPS")
            else:
                line1 = (
                    f"目标 {object_count}  ·  本帧 {infer_s * 1000:.1f} ms"
                    f"  ·  本帧 {inst_fps:.1f} FPS")
        else:
            line1 = (
                f"目标 {object_count}  ·  耗时 {infer_s * 1000:.1f} ms"
                f"  ·  {inst_fps:.1f} FPS")
        line2 = None
        if getattr(self, "current_source_type", None) == "video":
            vt = getattr(self, "_video_time_overlay", None)
            if vt is not None:
                cur, tot, rem = vt[0], vt[1], vt[2]
                if tot > 0:
                    line2 = (
                        f"进度 {cur}/{tot}  ·  {self._format_video_remaining_cn(rem)}")
                elif cur > 0:
                    line2 = f"已处理 {cur} 帧"
        return self._render_overlay_image(result_img, line1, line2)

    def _on_overlay_toggle_changed(self, _checked: bool):
        """切换图上指标显示时，刷新当前预览画面。"""
        if getattr(self, "tab_widget", None) is None:
            return
        idx = self.tab_widget.currentIndex()
        if idx == 0 and self._last_preview_result is not None:
            self.display_image(
                self._render_realtime_overlay_image(self._last_preview_result),
                self.result_label,
            )
        elif idx == 1 and self.batch_results:
            self.show_batch_result(self.current_batch_index)

    def on_detection_finished(self, completed=False):
        """检测完成回调"""
        self._video_time_overlay = None
        if getattr(self, "current_source_type", None) in ("video", "camera"):
            if not self._video_export_ended_at:
                self._video_export_ended_at = _format_export_local_time()
        try:
            self._flush_history_task_snapshot()
        except Exception:
            pass
        finally:
            self._history_run_start_mono = None
            self._history_pending_stop_reason = None
            self._history_last_error = ""
            self._history_run_sum_infer_s = 0.0
            self._history_video_frames_done = 0
            self._history_video_frames_total = 0
            self._history_started_at_str = ""
        if completed or self.progress_bar.value() >= 100:
            self.progress_bar.setValue(100)
            self._set_progress_state("done")
        else:
            self.progress_bar.setValue(0)
            self._set_progress_state("idle")
        self.update_detection_ui_state(False)
        self.pause_btn.setText("暂停")
        self._set_btn_icon_keep_color(self.pause_btn, "pause", "#6366f1", 16)
        bt = getattr(self, "batch_detection_thread", None)
        if bt is None or not getattr(bt, "is_running", False):
            self._history_batch_mode = False

    def show_batch_result(self, index):
        """显示批量结果"""
        if 0 <= index < len(self.batch_results):
            result = self.batch_results[index]

            self.display_image(result['original_img'],
                               self.batch_original_label)
            self.display_image(
                self._render_batch_overlay_image(result), self.batch_result_label)

            # 批量与文件检测共用同一个检测明细组件：仅在批量主 Tab 下更新，防止串页
            if hasattr(self, "result_detail_widget") and getattr(self, "tab_widget", None) is not None and self.tab_widget.currentIndex() == 1:
                self.result_detail_widget.update_results(
                    result['results'], result['class_names'], result['inference_time'])
            # 右侧指标卡同步当前批量结果
            self._status_objects.setText(str(result['object_count']))
            self._status_latency.setText(
                f"{result['inference_time']*1000:.0f} 毫秒")
            if result['inference_time'] > 1e-9:
                self._status_fps.setText(f"{1.0/result['inference_time']:.2f}")
            else:
                self._status_fps.setText("-")
            self._sync_wireframe_overview_from_status()

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
        self.result_index_label.setText("0/0")
        if hasattr(self, "result_detail_widget"):
            self.result_detail_widget.update_results(None, [], 0.0)
        self._status_objects.setText("-")
        self._status_fps.setText("-")
        self._status_latency.setText("-")
        self._sync_wireframe_overview_from_status()
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
            self.default_save_dir).exists() else str(data_dir)
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
        """批量保存时导出“检测明细”同款数据（CSV/JSON）。"""
        export_time = _format_export_local_time()
        total_images = len(self.batch_results)
        total_objects = sum(int(r.get("object_count", 0))
                            for r in self.batch_results)
        total_infer_s = sum(float(r.get("inference_time", 0.0))
                            for r in self.batch_results)
        avg_infer_s = (total_infer_s / total_images) if total_images else 0.0
        model_name = self.model_combo.currentText() if hasattr(
            self, "model_combo") else "-"
        src_folder = self.current_source_path or "-"

        metrics = {
            "应用": "Dimension 目标检测系统",
            "主界面页签": "批量分析",
            "结果记录时间": export_time,
            "推算推理开始时间": export_time,
            "推算推理结束时间": export_time,
            "本帧推理耗时_s": round(avg_infer_s, 6),
            "理论推理帧率_FPS": round(1.0 / avg_infer_s, 4) if avg_infer_s > 1e-12 else "",
            "模型文件路径": getattr(self, "_loaded_model_path", "") or "—",
            "模型名称": model_name or "—",
            "置信度阈值": getattr(self, "confidence_threshold", 0.25),
            "输入类型": "文件夹批量",
            "输入来源": src_folder,
            "检测目标总数": total_objects,
            "类别种数": "",
            "类别统计": "",
            "平均置信度": "",
            "置信度最小值": "",
            "置信度最大值": "",
            "状态": "批量完成",
            "设备信息": "",
            "备注": f"共 {total_images} 张，平均耗时 {avg_infer_s:.4f}s/张",
        }

        detail_rows = []
        row_no = 1
        for item in self.batch_results:
            file_name = Path(item.get("file_path", "")).name
            results = item.get("results")
            class_names = item.get("class_names", []) or []
            if not results or not results[0].boxes or len(results[0].boxes) == 0:
                continue
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            for conf, cls, box in zip(confidences, classes, xyxy):
                class_name = class_names[cls] if cls < len(
                    class_names) else f"类别{cls}"
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                area_px = max(0, int(round(bw * bh)))
                detail_rows.append({
                    "来源文件": file_name,
                    "序号": row_no,
                    "类别": class_name,
                    "置信度": round(float(conf), 6),
                    "x1": round(float(box[0]), 2),
                    "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2),
                    "y2": round(float(box[3]), 2),
                    "宽": round(float(bw), 2),
                    "高": round(float(bh), 2),
                    "面积": area_px,
                })
                row_no += 1

        csv_path = result_dir / "detection_detail.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["【指标项】", "【值】"])
            w.writerow(["导出时间", export_time])
            for key in _DETECTION_EXPORT_META_ORDER:
                if key in metrics:
                    val = metrics[key]
                    w.writerow([key, val if val is not None else ""])
            w.writerow([])
            w.writerow(["【目标明细】", ""])
            detail_fields = ["来源文件"] + list(_DETECTION_EXPORT_DETAIL_FIELDS)
            dw = csv.DictWriter(
                f, fieldnames=detail_fields, extrasaction="ignore")
            dw.writeheader()
            dw.writerows(detail_rows)

        json_path = result_dir / "detection_detail.json"
        payload = {
            "format": "Dimension_batch_detection_export",
            "version": 1,
            "export_time_local": export_time,
            "metrics": {
                k: metrics[k]
                for k in _DETECTION_EXPORT_META_ORDER
                if k in metrics
            },
            "detections": detail_rows,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def clear_display_windows(self):
        """清空显示窗口"""
        self._last_preview_original = None
        self._last_preview_result = None
        self.original_label.clear()
        self.result_label.clear()
        self.original_label.setText(
            "暂无输入源\n请选择图片或视频；摄像头请使用「设备监控」页")
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
            self.display_image(
                self._render_realtime_overlay_image(self._last_preview_result),
                self.result_label,
            )

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
        icon_file = base_dir / "assets" / "icons" / "dimension_logo.png"
        icon.addFile(str(icon_file))

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
    window.log_message("Dimension目标检测系统已启动")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
