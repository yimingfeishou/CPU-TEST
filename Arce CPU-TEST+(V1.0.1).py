import sys
import time
import math
import os
import numpy as np
import platform
import ctypes
from ctypes import wintypes
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                            QTextEdit, QGroupBox, QMessageBox, QLineEdit, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon
import threading
from Crypto.Cipher import AES # pyright: ignore[reportMissingImports]
from Crypto.Util.Padding import pad, unpad # pyright: ignore[reportMissingImports]
import random
import datetime

# 软件版本信息
VERSION = "1.0.1"

def resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和打包后环境"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 测试配置 - 支持单独设置每个测试的时长
TEST_CONFIG = {
    "duration": {
        "single_core": 12,
        "multi_core": 12,
        "memory": 12,
        "crypto": 12
    },
    "matrix_size": 1024,
    "prime_limit": 2000000,
    "memory_size": 500000,
    "crypto_rounds": 8,
    "weights": {
        "single_core": 0.3,
        "multi_core": 0.4,
        "memory": 0.2,
        "crypto": 0.1
    },
    "developer_mode": False  # 开发者模式标志
}

# 默认测试时长，用于恢复默认设置
DEFAULT_DURATIONS = {
    "single_core": 12,
    "multi_core": 12,
    "memory": 12,
    "crypto": 12
}

# 测试项名称映射
TEST_NAME_MAP = {
    "single_core": "单核性能测试",
    "multi_core": "多核性能测试",
    "memory": "内存性能测试",
    "crypto": "加密性能测试"
}

class TestThread(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    log_updated = pyqtSignal(str)
    
    def __init__(self, test_type, parent=None):
        super().__init__(parent)
        self.test_type = test_type
        self.running = True
        # 获取当前测试项的时长
        self.duration = TEST_CONFIG["duration"][test_type]
        
    def stop(self):
        self.running = False
        self.log_updated.emit(f"[{self.test_type}] 收到停止信号")

class SingleCoreTestThread(TestThread):
    def run(self):
        try:
            mode_tag = "（开发者模式）" if TEST_CONFIG["developer_mode"] else ""
            self.log_updated.emit(f"单核测试开始，持续 {self.duration} 秒{mode_tag}")
            start_time = time.time()
            end_time = start_time + self.duration
            operations = 0
            
            while time.time() < end_time and self.running:
                # 基础计算任务
                calculate_primes(TEST_CONFIG["prime_limit"] // 10)
                matrix_multiplication(TEST_CONFIG["matrix_size"] // 2)
                precision_calc = np.sum(np.exp(np.random.rand(100000)))
                
                # 开发者模式增强算法
                if TEST_CONFIG["developer_mode"]:
                    self.log_updated.emit("单核测试（开发者模式）：圆周率估算")
                    # 蒙特卡洛法计算圆周率（高复杂度浮点运算）
                    pi_samples = 1000000  # 100万次采样
                    x = np.random.rand(pi_samples)
                    y = np.random.rand(pi_samples)
                    pi_estimate = 4 * np.sum(x**2 + y**2 < 1) / pi_samples
                
                operations += 1
                
                elapsed = time.time() - start_time
                progress = int((elapsed / self.duration) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / self.duration
                self.log_updated.emit(f"单核测试完成，得分: {score:.4f}")
                self.result_ready.emit(score)
            else:
                self.log_updated.emit("单核测试被中止")
                
        except Exception as e:
            err_msg = f"单核测试错误: {str(e)}"
            self.log_updated.emit(err_msg)
            self.error_occurred.emit(err_msg)

class MultiCoreTestThread(TestThread):
    def run(self):
        try:
            mode_tag = "（开发者模式）" if TEST_CONFIG["developer_mode"] else ""
            self.log_updated.emit(f"多核测试开始，持续 {self.duration} 秒{mode_tag}")
            cpu_cores = os.cpu_count() or 4
            self.log_updated.emit(f"检测到{cpu_cores}个CPU核心")
            start_time = time.time()
            end_time = start_time + self.duration
            results = [0] * cpu_cores
            
            def thread_worker(index):
                ops = 0
                while time.time() < end_time and self.running:
                    matrix = np.random.rand(200, 200) + np.eye(200) * 0.5
                    np.linalg.inv(matrix)
                    
                    # 开发者模式增强算法
                    if TEST_CONFIG["developer_mode"]:
                        # 不同核心分配不同高复杂度任务
                        if index % 2 == 0:
                            np.linalg.svd(matrix)  # 奇异值分解
                        else:
                            np.linalg.eig(matrix)  # 特征值分解
                        # 增加大矩阵乘法
                        large_matrix = np.random.rand(500, 500)
                        np.dot(large_matrix, large_matrix.T)
                        
                    ops += 1
                results[index] = ops
            
            threads = []
            for i in range(cpu_cores):
                t = threading.Thread(target=thread_worker, args=(i,), daemon=True)
                threads.append(t)
                t.start()
                self.log_updated.emit(f"启动多核工作线程 {i+1}/{cpu_cores}")
            
            while time.time() < end_time and self.running:
                elapsed = time.time() - start_time
                progress = int((elapsed / self.duration) * 100)
                self.progress_updated.emit(progress)
                time.sleep(0.01)
            
            for t in threads:
                t.join(timeout=2)
            self.log_updated.emit("多核工作线程已结束")
            
            if self.running:
                total_ops = sum(results)
                score = total_ops / self.duration
                self.log_updated.emit(f"多核测试完成，总操作数: {total_ops}，得分: {score:.4f}")
                self.result_ready.emit(score)
            else:
                self.log_updated.emit("多核测试被中止")
                
        except Exception as e:
            err_msg = f"多核测试错误: {str(e)}"
            self.log_updated.emit(err_msg)
            self.error_occurred.emit(err_msg)

class MemoryTestThread(TestThread):
    def run(self):
        try:
            mode_tag = "（开发者模式）" if TEST_CONFIG["developer_mode"] else ""
            self.log_updated.emit(f"内存测试开始，持续 {self.duration} 秒{mode_tag}")
            start_time = time.time()
            end_time = start_time + self.duration
            operations = 0
            
            while time.time() < end_time and self.running:
                # 基础内存操作
                array = np.random.rand(TEST_CONFIG["memory_size"]).astype(np.float64)
                sum1 = array.sum()
                sum2 = np.sum(array **2)
                std_dev = np.sqrt((sum2 / len(array)) - (sum1 / len(array))** 2)
                array = np.roll(array, shift=1000)
                array = array * 0.999 + 0.001
                
                # 开发者模式增强算法（修复维度匹配问题）
                if TEST_CONFIG["developer_mode"]:
                    self.log_updated.emit("内存测试（开发者模式）：数组操作")
                    # 1. 生成大型数组并重塑为 (m, k)，确保转置后列数与原数组相同
                    large_array = np.random.rand(TEST_CONFIG["memory_size"] * 5).astype(np.float64)
                    # 固定列数k=1000，确保能被数组长度整除
                    k = 1000  # 固定列数
                    m = len(large_array) // k  # 计算行数
                    large_array = large_array[:m*k]  # 截断为整数长度
                    multi_array = large_array.reshape(m, k)  # 形状：(m, 1000)
                    
                    # 2. 转置后形状为 (1000, m)，再重塑为 (m, 1000)，确保列数与原数组相同
                    transposed = multi_array.T.reshape(m, k)  # 形状：(m, 1000)
                    
                    # 3. 沿行方向（axis=0）拼接，此时两者列数均为1000，维度匹配
                    combined = np.concatenate([multi_array, transposed], axis=0)  # 拼接后形状：(2m, 1000)
                    
                    # 强制内存回收
                    del large_array, multi_array, transposed, combined
                    
                operations += 1
                
                elapsed = time.time() - start_time
                progress = int((elapsed / self.duration) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / self.duration
                self.log_updated.emit(f"内存测试完成，得分: {score:.4f}")
                self.result_ready.emit(score)
            else:
                self.log_updated.emit("内存测试被中止")
                
        except Exception as e:
            err_msg = f"内存测试错误: {str(e)}"
            self.log_updated.emit(err_msg)
            self.error_occurred.emit(err_msg)

class CryptoTestThread(TestThread):
    def run(self):
        try:
            mode_tag = "（开发者模式）" if TEST_CONFIG["developer_mode"] else ""
            self.log_updated.emit(f"加密测试开始，持续 {self.duration} 秒{mode_tag}")
            start_time = time.time()
            end_time = start_time + self.duration
            operations = 0
            
            # 开发者模式使用256位密钥，普通模式128位
            key = os.urandom(32) if TEST_CONFIG["developer_mode"] else os.urandom(16)
            key_size = 256 if TEST_CONFIG["developer_mode"] else 128
            self.log_updated.emit(f"使用AES-{key_size}位密钥进行测试")
            
            while time.time() < end_time and self.running:
                # 开发者模式增加数据量
                data_size = 8192 if TEST_CONFIG["developer_mode"] else 1024
                data = os.urandom(data_size)
                
                # 开发者模式使用GCM模式（带认证），普通模式CBC
                if TEST_CONFIG["developer_mode"]:
                    cipher = AES.new(key, AES.MODE_GCM)
                    ct, tag = cipher.encrypt_and_digest(data)
                    nonce = cipher.nonce
                    # 解密+认证验证
                    cipher_dec = AES.new(key, AES.MODE_GCM, nonce=nonce)
                    pt = cipher_dec.decrypt_and_verify(ct, tag)
                else:
                    cipher = AES.new(key, AES.MODE_CBC)
                    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
                    iv = cipher.iv
                    cipher_dec = AES.new(key, AES.MODE_CBC, iv=iv)
                    pt = unpad(cipher_dec.decrypt(ct_bytes), AES.block_size)
                
                # 验证解密正确性
                if pt != data:
                    self.log_updated.emit("加密解密验证失败")
                
                operations += 1
                
                elapsed = time.time() - start_time
                progress = int((elapsed / self.duration) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / self.duration
                self.log_updated.emit(f"加密测试完成，得分: {score:.4f}")
                self.result_ready.emit(score)
            else:
                self.log_updated.emit("加密测试被中止")
                
        except Exception as e:
            err_msg = f"加密测试错误: {str(e)}"
            self.log_updated.emit(err_msg)
            self.error_occurred.emit(err_msg)

# 辅助函数
def calculate_primes(limit):
    """优化的素数筛法，增加计算量"""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    primes = np.nonzero(sieve)[0]
    np.sum(primes)
    return primes

def matrix_multiplication(size):
    """矩阵运算，开发者模式下增加复杂度"""
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    c = np.dot(a, b)
    c = np.dot(c, a.T)
    c = c + np.eye(size)
    
    # 开发者模式增强算法
    if TEST_CONFIG["developer_mode"]:
        # 增加多重矩阵分解运算
        eig_vals = np.linalg.eigvals(c)
        u, s, vh = np.linalg.svd(c)
        inv_c = np.linalg.inv(c)
        pow_c = np.linalg.matrix_power(c, 5)
        
    return c

# Windows API 调用获取CPU信息（修复版）
class CPUInfo:
    @staticmethod
    def get_cpu_info():
        """获取CPU信息，仅支持Windows系统"""
        if platform.system() != "Windows":
            return "CPU信息获取仅支持Windows系统"
            
        try:
            # 手动定义Windows注册表常量
            HKEY_LOCAL_MACHINE = 0x80000002
            KEY_READ = 0x20019
            KEY_WOW64_64KEY = 0x0100
            
            # 获取CPU核心数等基础信息
            class SYSTEM_INFO(ctypes.Structure):
                _fields_ = [
                    ("wProcessorArchitecture", wintypes.WORD),
                    ("wReserved", wintypes.WORD),
                    ("dwPageSize", wintypes.DWORD),
                    ("lpMinimumApplicationAddress", wintypes.LPVOID),
                    ("lpMaximumApplicationAddress", wintypes.LPVOID),
                    ("dwActiveProcessorMask", ctypes.c_size_t),
                    ("dwNumberOfProcessors", wintypes.DWORD),
                    ("dwProcessorType", wintypes.DWORD),
                    ("dwAllocationGranularity", wintypes.DWORD),
                    ("wProcessorLevel", wintypes.WORD),
                    ("wProcessorRevision", wintypes.WORD),
                ]

            sysinfo = SYSTEM_INFO()
            ctypes.windll.kernel32.GetSystemInfo(ctypes.byref(sysinfo))
            
            # 获取CPU型号
            cpu_name = ""
            hkey = wintypes.HKEY()
            if ctypes.windll.advapi32.RegOpenKeyExW(
                HKEY_LOCAL_MACHINE,
                "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                0,
                KEY_READ | KEY_WOW64_64KEY,
                ctypes.byref(hkey)
            ) == 0:
                buf = ctypes.create_unicode_buffer(256)
                buf_size = wintypes.DWORD(ctypes.sizeof(buf))
                if ctypes.windll.advapi32.RegQueryValueExW(
                    hkey,
                    "ProcessorNameString",
                    None,
                    None,
                    ctypes.byref(buf),
                    ctypes.byref(buf_size)
                ) == 0:
                    cpu_name = buf.value
                ctypes.windll.advapi32.RegCloseKey(hkey)
            
            # 获取当前CPU频率
            freq = 0
            if ctypes.windll.advapi32.RegOpenKeyExW(
                HKEY_LOCAL_MACHINE,
                "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                0,
                KEY_READ | KEY_WOW64_64KEY,
                ctypes.byref(hkey)
            ) == 0:
                buf = wintypes.DWORD()
                buf_size = wintypes.DWORD(ctypes.sizeof(buf))
                if ctypes.windll.advapi32.RegQueryValueExW(
                    hkey,
                    "~MHz",
                    None,
                    None,
                    ctypes.byref(buf),
                    ctypes.byref(buf_size)
                ) == 0:
                    freq = buf.value
                ctypes.windll.advapi32.RegCloseKey(hkey)
            
            # 收集并返回信息
            info = []
            if cpu_name:
                info.append(f"CPU型号: {cpu_name}")
            info.append(f"核心数: {sysinfo.dwNumberOfProcessors}")
            if freq > 0:
                info.append(f"当前频率: {freq} MHz")
            
            return "\n".join(info)
        except Exception as e:
            return f"获取CPU信息失败: {str(e)}"

class CPUTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arce CPU-TEST+")
        self.setGeometry(100, 100, 1000, 800)
        
        # 尝试设置图标
        try:
            icon_path = resource_path("1.ico")
            self.setWindowIcon(QIcon(icon_path))
            app = QApplication.instance()
            if app:
                app.setWindowIcon(QIcon(icon_path))
        except:
            pass  # 图标加载失败不影响主程序
        
        self.is_testing = False
        self.test_threads = []
        self.test_results = {}
        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("Arce CPU-TEST+")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 版本信息
        version_label = QLabel(f"版本: {VERSION}")
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(version_label)
        
        # CLI
        cli_group = QGroupBox("开发命令")
        cli_layout = QHBoxLayout()
        cli_group.setLayout(cli_layout)
        main_layout.addWidget(cli_group)
        
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("输入命令 (输入 'help' 查看帮助)")
        self.command_input.returnPressed.connect(self.execute_command)
        cli_layout.addWidget(self.command_input)
        
        self.command_button = QPushButton("执行")
        self.command_button.clicked.connect(self.execute_command)
        cli_layout.addWidget(self.command_button)
        
        # 测试控制区
        test_group = QGroupBox("测试控制")
        test_layout = QVBoxLayout()
        test_group.setLayout(test_layout)
        main_layout.addWidget(test_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        test_layout.addWidget(self.progress_bar)
        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始CPU测试")
        self.start_button.clicked.connect(self.start_tests)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("停止测试")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white;")
        self.stop_button.clicked.connect(self.stop_tests)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        test_layout.addLayout(button_layout)
        
        self.status_label = QLabel("就绪")
        test_layout.addWidget(self.status_label)
        
        # 日志区域（包含导出功能）
        log_group = QGroupBox("测试日志")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 日志控制栏（添加导出按钮）
        log_control_layout = QHBoxLayout()
        self.export_log_btn = QPushButton("导出日志")
        self.export_log_btn.clicked.connect(self.export_log)
        log_control_layout.addWidget(self.export_log_btn)
        log_layout.addLayout(log_control_layout)
        
        # 日志显示区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        log_layout.addWidget(self.log_text)
        
        # 结果区域
        result_group = QGroupBox("测试结果")
        result_layout = QVBoxLayout()
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setHtml("测试结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        
        # 初始日志
        self.log(f"Arce CPU-TEST+ 版本 {VERSION} 启动")
        self.log("输入 'help' 查看可用命令")
    
    def log(self, message):
        """带时间戳的日志输出"""
        timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
        self.log_text.append(timestamp + message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def export_log(self):
        """导出日志到文本文件"""
        # 获取当前日志内容
        log_content = self.log_text.toPlainText()
        if not log_content:
            QMessageBox.information(self, "提示", "日志为空，无需导出")
            return
        
        # 生成默认文件名（带时间戳）
        default_filename = f"cpu_test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # 弹出文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出日志",
            default_filename,
            "文本文件 (*.txt);;所有文件 (*)"
        )
        
        if not file_path:  # 用户取消保存
            return
        
        try:
            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(log_content)
            self.log(f"日志已成功导出到: {file_path}")
            QMessageBox.information(self, "成功", f"日志已导出至:\n{file_path}")
        except Exception as e:
            error_msg = f"日志导出失败: {str(e)}"
            self.log(error_msg)
            QMessageBox.critical(self, "失败", error_msg)
    
    def execute_command(self):
        """执行CLI命令"""
        command = self.command_input.text().strip()
        self.command_input.clear()
    
        if not command:
            return
        
        self.log(f"执行命令: {command}")
    
        if command == "help":
            self.show_help()
        elif command == "list-test":
            self.list_tests()
        elif command == "test-time-re":
            self.restore_default_time()
        elif command.startswith("test-time-"):
            parts = command.split("-")
            if len(parts) == 4 and parts[3].isdigit():
                test_item = parts[2]
                try:
                    duration = int(parts[3])
                    self.set_test_time(test_item, duration)
                except ValueError:
                    self.log("error: 时间必须是整数")
            elif len(parts) == 4 and parts[3] == "seetime":
                test_item = parts[2]
                self.show_test_time(test_item)
            else:
                self.log("error: 命令格式错误，正确格式: test-time-测试项-时间 或 test-time-测试项-seetime")
        elif command == "dev-test":
            self.toggle_developer_mode()
        elif command == "ver":
            self.show_version()
        else:
            self.log(f"error: 未知命令 '{command}'，输入 'help' 查看帮助")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
可用命令:
- help: 显示帮助信息
- list-test: 列出所有测试项
- test-time-测试项-时间: 设置指定测试项的时间（1s-65536s）
- test-time-测试项-seetime: 查看测试项的时长
- test-time-re: 恢复默认时长
- dev-test: 开启/关闭开发者模式
- ver: 显示软件版本
        """
        self.log(help_text.strip())
    
    def list_tests(self):
        """列出所有测试项"""
        self.log("所有测试项:")
        for test_key, test_name in TEST_NAME_MAP.items():
            self.log(f"- {test_key}: {test_name} (当前时长: {TEST_CONFIG['duration'][test_key]}秒)")
    
    def set_test_time(self, test_item, duration):
        """设置测试项时长"""
        if test_item not in TEST_CONFIG["duration"]:
            self.log(f"error: 测试项 '{test_item}' 不存在")
            return
            
        if duration < 1 or duration > 65536:
            self.log("error: 时间必须在1-65536秒之间")
            return
            
        TEST_CONFIG["duration"][test_item] = duration
        self.log(f"done: {TEST_NAME_MAP.get(test_item, test_item)} 时长已设置为 {duration} 秒")
    
    def show_test_time(self, test_item):
        """显示测试项时长"""
        if test_item not in TEST_CONFIG["duration"]:
            self.log(f"error: 测试项 '{test_item}' 不存在")
            return
            
        duration = TEST_CONFIG["duration"][test_item]
        self.log(f"{TEST_NAME_MAP.get(test_item, test_item)} 当前时长: {duration} 秒")
    
    def restore_default_time(self):
        """恢复默认时长"""
        TEST_CONFIG["duration"] = DEFAULT_DURATIONS.copy()
        self.log("done: 所有测试项时长已恢复默认值")
    
    def toggle_developer_mode(self):
        """切换开发者模式"""
        TEST_CONFIG["developer_mode"] = not TEST_CONFIG["developer_mode"]
        if TEST_CONFIG["developer_mode"]:
            self.log("开发者模式已开启")
            # 显示CPU信息
            cpu_info = CPUInfo.get_cpu_info()
            self.log("CPU信息:")
            for line in cpu_info.split("\n"):
                self.log(f"  {line}")
            self.log("开发者模式将使用更复杂的测试算法")
        else:
            self.log("开发者模式已关闭，将使用标准测试算法")
    
    def show_version(self):
        """显示软件版本"""
        self.log(f"Arce CPU-TEST+ 版本 {VERSION}")
    
    def start_tests(self):
        if self.is_testing:
            return
        
        self.log("开始测试序列")
        self.is_testing = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.result_text.setHtml("测试进行中...")
        self.test_results = {}
        self.test_threads = []
        
        self.test_sequence = [
            ("single_core", "单核性能测试", SingleCoreTestThread),
            ("multi_core", "多核性能测试", MultiCoreTestThread),
            ("memory", "内存性能测试", MemoryTestThread),
            ("crypto", "加密性能测试", CryptoTestThread)
        ]
        self.current_test_index = 0
        self.run_next_test()
    
    def run_next_test(self):
        if self.current_test_index >= len(self.test_sequence):
            self.calculate_and_show_results()
            return
        
        test_key, test_name, test_class = self.test_sequence[self.current_test_index]
        self.status_label.setText(f"正在进行: {test_name}")
        self.progress_bar.setValue(0)
        self.log(f"开始{test_name}（{self.current_test_index+1}/4）")
        
        thread = test_class(test_key)
        thread.progress_updated.connect(self.update_test_progress)
        thread.result_ready.connect(lambda result, key=test_key: self.on_test_complete(result, key))
        thread.error_occurred.connect(self.on_test_error)
        thread.log_updated.connect(self.log)
        self.test_threads.append(thread)
        thread.start()
    
    def update_test_progress(self, value):
        total_tests = len(self.test_sequence)
        overall_progress = (self.current_test_index / total_tests) * 100 + (value / total_tests)
        self.progress_bar.setValue(int(overall_progress))
    
    def on_test_complete(self, result, test_key):
        self.test_results[test_key] = result
        self.log(f"{test_key}测试完成，结果: {result:.4f}")
        self.current_test_index += 1
        self.run_next_test()
    
    def on_test_error(self, error_msg):
        self.stop_tests()
        self.log(f"测试错误: {error_msg}")
        QMessageBox.critical(self, "测试错误", f"错误详情:\n{error_msg}")
        self.result_text.setHtml(f"测试失败: {error_msg}")
    
    def calculate_and_show_results(self):
        weights = TEST_CONFIG["weights"]
        total_score = (
            self.test_results["single_core"] * weights["single_core"] +
            self.test_results["multi_core"] * weights["multi_core"] +
            self.test_results["memory"] * weights["memory"] +
            self.test_results["crypto"] * weights["crypto"]
        )
        
        cpu_cores = os.cpu_count() or 4
        self.log(f"所有测试完成，综合得分: {total_score:.4f}")
        
        result_html = f"""
        <h3>CPU测试结果</h3>
        <p><strong>单核性能得分:</strong> {self.format_number(self.test_results['single_core'])}</p>
        <p><strong>多核性能得分:</strong> {self.format_number(self.test_results['multi_core'])}</p>
        <p><strong>内存性能得分:</strong> {self.format_number(self.test_results['memory'])}</p>
        <p><strong>加密性能得分:</strong> {self.format_number(self.test_results['crypto'])}</p>
        <hr>
        <p><strong>综合性能得分:</strong> {self.format_number(total_score)}</p>
        <p>测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>CPU核心数: {cpu_cores}</p>
        """
        
        if TEST_CONFIG["developer_mode"]:
            result_html += f"<p><em>在开发者模式下执行测试</em></p>"
            cpu_info = CPUInfo.get_cpu_info()
            result_html += f"<hr><p><strong>CPU信息:</strong><br>{cpu_info.replace('\n', '<br>')}</p>"
        
        self.result_text.setHtml(result_html)
        self.status_label.setText("测试完成")
        self.progress_bar.setValue(100)
        self.is_testing = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def stop_tests(self):
        if not self.is_testing:
            return
        
        self.log("开始停止所有测试...")
        for thread in self.test_threads:
            if thread.isRunning():
                thread.stop()
                thread.wait(1000)
                if thread.isRunning():
                    self.log(f"警告: {thread.test_type}线程无法正常停止")
        
        self.is_testing = False
        self.progress_bar.setValue(0)
        self.result_text.setHtml("测试已中止")
        self.status_label.setText("已停止")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("所有测试已停止")
    
    def format_number(self, num):
        if num > 1e9:
            return f"{num / 1e9:.2f} G"
        elif num > 1e6:
            return f"{num / 1e6:.2f} M"
        elif num > 1e3:
            return f"{num / 1e3:.2f} K"
        return f"{round(num, 4)}"
    
    def closeEvent(self, event):
        self.stop_tests()
        event.accept()

if __name__ == "__main__":
    # 禁用numpy多线程，避免与程序线程池冲突
    np.set_printoptions(threshold=np.inf)
    try:
        np.config.set_option('num_threads', 1)
    except:
        pass
    
    app = QApplication(sys.argv)
    window = CPUTestApp()
    window.show()
    sys.exit(app.exec())
