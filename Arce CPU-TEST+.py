import sys
import time
import math
import os
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QProgressBar, 
                            QTextEdit, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtGui import QIcon
import threading

def resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和打包后环境"""
    try:
        # PyInstaller创建的临时文件夹
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 提升测试压力，确保CPU高负载
TEST_CONFIG = {
    "duration": 12,  # 适度延长测试时间，让CPU稳定高负载
    "matrix_size": 1024,  # 增大矩阵尺寸（计算量与尺寸三次方成正比）
    "prime_limit": 2000000,  # 扩大素数计算范围
    "memory_size": 500000,  # 内存数据量适中，避免OOM但保证操作密集
    "crypto_rounds": 8,  # 增加加密轮数
    "weights": {
        "single_core": 0.3,
        "multi_core": 0.4,
        "memory": 0.2,
        "crypto": 0.1
    }
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
        
    def stop(self):
        self.running = False
        self.log_updated.emit(f"[{self.test_type}] 收到停止信号")

class SingleCoreTestThread(TestThread):
    def run(self):
        try:
            self.log_updated.emit("单核测试开始")
            start_time = time.time()
            end_time = start_time + TEST_CONFIG["duration"]
            operations = 0
            
            # 无休眠循环，持续计算
            while time.time() < end_time and self.running:
                # 组合多个计算密集型任务，确保CPU持续工作
                calculate_primes(TEST_CONFIG["prime_limit"] // 10)  # 素数计算
                matrix_multiplication(TEST_CONFIG["matrix_size"] // 2)  # 矩阵乘法
                # 新增浮点数精度计算，增加CPU负载
                precision_calc = np.sum(np.exp(np.random.rand(100000)))  # 指数+求和
                operations += 1
                
                # 仅在更新进度时短暂检查，不阻塞计算
                elapsed = time.time() - start_time
                progress = int((elapsed / TEST_CONFIG["duration"]) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / TEST_CONFIG["duration"]
                self.log_updated.emit(f"单核测试完成，得分: {score}")
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
            self.log_updated.emit("多核测试开始")
            cpu_cores = os.cpu_count() or 4  # 获取实际CPU核心数（而非线程数）
            self.log_updated.emit(f"检测到{cpu_cores}个CPU核心")
            start_time = time.time()
            end_time = start_time + TEST_CONFIG["duration"]
            results = [0] * cpu_cores
            
            # 工作线程：无休眠，持续计算
            def thread_worker(index):
                ops = 0
                while time.time() < end_time and self.running:
                    # 高计算量任务：矩阵求逆（计算密集）+ 随机数生成（CPU依赖）
                    matrix = np.random.rand(200, 200) + np.eye(200) * 0.5  # 确保可逆
                    np.linalg.inv(matrix)  # 矩阵求逆（高CPU消耗）
                    ops += 1
                results[index] = ops
            
            # 启动与核心数相同的线程（满负载）
            threads = []
            for i in range(cpu_cores):
                t = threading.Thread(target=thread_worker, args=(i,), daemon=True)
                threads.append(t)
                t.start()
                self.log_updated.emit(f"启动多核工作线程 {i+1}/{cpu_cores}")
            
            # 进度更新（不阻塞工作线程）
            while time.time() < end_time and self.running:
                elapsed = time.time() - start_time
                progress = int((elapsed / TEST_CONFIG["duration"]) * 100)
                self.progress_updated.emit(progress)
                time.sleep(0.01)  # 极短休眠，仅为更新UI
            
            # 等待所有工作线程结束
            for t in threads:
                t.join(timeout=2)
            self.log_updated.emit("多核工作线程已结束")
            
            if self.running:
                total_ops = sum(results)
                score = total_ops / TEST_CONFIG["duration"]
                self.log_updated.emit(f"多核测试完成，总操作数: {total_ops}，得分: {score}")
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
            self.log_updated.emit("内存测试开始")
            start_time = time.time()
            end_time = start_time + TEST_CONFIG["duration"]
            operations = 0
            
            # 内存操作+计算结合，避免纯内存访问CPU占用低
            while time.time() < end_time and self.running:
                # 1. 生成大数组（内存IO）
                array = np.random.rand(TEST_CONFIG["memory_size"]).astype(np.float64)
                # 2. 多次计算（CPU参与）：求和+平方和+标准差
                sum1 = array.sum()
                sum2 = np.sum(array **2)
                std_dev = np.sqrt((sum2 / len(array)) - (sum1 / len(array))** 2)
                # 3. 数组变换（内存+CPU结合）
                array = np.roll(array, shift=1000)  # 滚动数组（内存重排）
                array = array * 0.999 + 0.001  # 元素级计算
                operations += 1
                
                # 更新进度
                elapsed = time.time() - start_time
                progress = int((elapsed / TEST_CONFIG["duration"]) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / TEST_CONFIG["duration"]
                self.log_updated.emit(f"内存测试完成，得分: {score}")
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
            self.log_updated.emit("加密测试开始")
            start_time = time.time()
            end_time = start_time + TEST_CONFIG["duration"]
            operations = 0
            
            # 加密+解密组合，增加CPU负载
            while time.time() < end_time and self.running:
                # 生成更多数据
                data = bytearray(np.random.randint(0, 256, 1024, dtype=np.uint8))
                # 多轮加密+解密（增加计算量）
                encrypted = aes_like_encrypt(data, TEST_CONFIG["crypto_rounds"])
                decrypted = aes_like_decrypt(encrypted, TEST_CONFIG["crypto_rounds"])
                operations += 1
                
                # 更新进度
                elapsed = time.time() - start_time
                progress = int((elapsed / TEST_CONFIG["duration"]) * 100)
                self.progress_updated.emit(progress)
            
            if self.running:
                score = operations / TEST_CONFIG["duration"]
                self.log_updated.emit(f"加密测试完成，得分: {score}")
                self.result_ready.emit(score)
            else:
                self.log_updated.emit("加密测试被中止")
                
        except Exception as e:
            err_msg = f"加密测试错误: {str(e)}"
            self.log_updated.emit(err_msg)
            self.error_occurred.emit(err_msg)

# 辅助函数（强化计算强度）
def calculate_primes(limit):
    """优化的素数筛法，增加计算量"""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False  # 向量化操作，高效但CPU密集
    # 新增素数求和，增加计算步骤
    primes = np.nonzero(sieve)[0]
    np.sum(primes)  # 额外计算，提升CPU负载
    return primes

def matrix_multiplication(size):
    """更大矩阵+多次运算，提升CPU占用"""
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    # 多次矩阵运算（乘法+转置+加法）
    c = np.dot(a, b)
    c = np.dot(c, a.T)  # 转置后再乘
    c = c + np.eye(size)  # 加单位矩阵
    return c

# 只展示需要修改的加密相关函数，其他代码保持不变

def aes_like_encrypt(data, rounds):
    """增强版类AES加密，修复uint8范围问题"""
    block_size = 16
    pad_length = block_size - (len(data) % block_size)
    data += bytes([pad_length]) * pad_length
    
    # 生成初始密钥（严格限制在0-255）
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    round_keys = [key]
    
    # 密钥扩展（修复可能产生256的问题）
    for _ in range(rounds):
        # 确保计算结果严格在0-255范围内
        new_key = np.uint8((round_keys[-1] * 0x1f + 0x2d) % 256)
        round_keys.append(new_key)
    
    encrypted = bytearray()
    for i in range(0, len(data), block_size):
        block = np.frombuffer(data[i:i+block_size], dtype=np.uint8)
        block ^= round_keys[0]  # 初始轮
        
        # 多轮加密（每轮确保值在uint8范围内）
        for round in range(1, rounds):
            # 字节替换：强制转换为uint8，避免超出范围
            block = np.uint8((block * 0x13 + 0x17) % 256)
            # 行移位
            for row in range(4):
                block[row::4] = np.roll(block[row::4], -row)
            # 轮密钥加
            block ^= round_keys[round]
        
        encrypted.extend(block.tobytes())
    return encrypted

def aes_like_encrypt(data, rounds):
    """彻底修复uint8溢出问题的加密函数"""
    block_size = 16
    pad_length = block_size - (len(data) % block_size)
    data += bytes([pad_length]) * pad_length  # 填充数据
    
    # 生成初始密钥（严格限制0-255）
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    round_keys = [key.copy()]  # 用copy避免引用问题
    
    # 密钥扩展（彻底避免256）
    for _ in range(rounds):
        prev_key = round_keys[-1].astype(int)  # 转为int计算
        # 简化密钥扩展算法，确保结果在0-255
        new_key = (prev_key * 17 + 23) % 256  # 17和23是质数，降低碰撞概率
        round_keys.append(np.uint8(new_key))  # 强制转换为uint8
    
    encrypted = bytearray()
    for i in range(0, len(data), block_size):
        # 提取块并转为uint8数组
        block = np.frombuffer(data[i:i+block_size], dtype=np.uint8).astype(int)
        
        # 初始轮密钥加
        block ^= round_keys[0].astype(int)
        
        # 多轮加密
        for round in range(1, rounds):
            # 1. 字节替换（简化算法，确保无溢出）
            block = (block * 13 + 11) % 256  # 13和11为质数，结果必然在0-255
            
            # 2. 行移位（仅移位，不改变值）
            for row in range(4):
                block[row::4] = np.roll(block[row::4], -row)
            
            # 3. 轮密钥加
            block ^= round_keys[round].astype(int)
        
        # 转换为uint8并添加到结果
        encrypted.extend(np.uint8(block).tobytes())
    
    return encrypted

def aes_like_decrypt(data, rounds):
    """配套解密函数（与加密对应，修复溢出）"""
    block_size = 16
    
    # 生成与加密相同的密钥（必须与加密一致才能解密）
    key = np.random.randint(0, 256, 16, dtype=np.uint8)
    round_keys = [key.copy()]
    for _ in range(rounds):
        prev_key = round_keys[-1].astype(int)
        new_key = (prev_key * 17 + 23) % 256
        round_keys.append(np.uint8(new_key))
    
    decrypted = bytearray()
    for i in range(0, len(data), block_size):
        # 提取块并转为int数组计算
        block = np.frombuffer(data[i:i+block_size], dtype=np.uint8).astype(int)
        
        # 最后一轮密钥加
        block ^= round_keys[-1].astype(int)
        
        # 多轮解密
        for round in range(rounds-1, 0, -1):
            # 1. 逆行移位
            for row in range(4):
                block[row::4] = np.roll(block[row::4], row)
            
            # 2. 逆字节替换（确保结果在0-255）
            inv_mult = 241  # 13的乘法逆元（13*241=3133 → 3133%256=1）
            inv_add = 185   # 逆加法常量（(x - 11) * 241 %256 = (x*241 - 11*241) %256 → 11*241=2651%256=185）
            block = ((block - inv_add) * inv_mult) % 256  # 先减后乘，确保无溢出
            
            # 3. 轮密钥加
            block ^= round_keys[round].astype(int)
        
        # 转换为uint8并添加到结果
        decrypted_block = np.uint8(block)
        decrypted.extend(decrypted_block.tobytes())
    
    # 去除填充
    pad_length = decrypted[-1]
    return decrypted[:-pad_length]

class CPUTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arce CPU-TEST+")
        self.setGeometry(100, 100, 900, 700)
        
        # 修复图标加载
        icon_path = resource_path("1.ico")
        self.setWindowIcon(QIcon(icon_path))
        
        # 设置应用程序图标
        app = QApplication.instance()
        if app:
            app.setWindowIcon(QIcon(icon_path))
        
        self.is_testing = False
        self.test_threads = []
        self.test_results = {}
        self.init_ui()
        
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
        
        # 日志区域
        log_group = QGroupBox("测试日志")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
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
    
    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S] ")
        self.log_text.append(timestamp + message)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
    
    def start_tests(self):
        if self.is_testing:
            return
        
        self.log("开始测试")
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
        overall_progress = self.current_test_index * 25 + (value // 4)
        self.progress_bar.setValue(overall_progress)
    
    def on_test_complete(self, result, test_key):
        self.test_results[test_key] = result
        self.log(f"{test_key}测试完成，结果: {result}")
        self.current_test_index += 1
        # 测试切换无延迟，保持高负载连续性
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
        self.log(f"所有测试完成，综合得分: {total_score}")
        
        result_html = f"""
        <h3>CPU测试结果</h3>
        <p><strong>单核性能得分:</strong> {self.format_number(self.test_results['single_core'])}</p>
        <p><strong>多核性能得分:</strong> {self.format_number(self.test_results['multi_core'])}</p>
        <p><strong>内存性能得分:</strong> {self.format_number(self.test_results['memory'])}</p>
        <p><strong>加密性能得分:</strong> {self.format_number(self.test_results['crypto'])}</p>
        <hr>
        <p><strong>综合性能得分:</strong> {self.format_number(total_score)}</p>
        <p>测试完成时间: {time.strftime('%H:%M:%S')}</p>
        <p>CPU核心数: {cpu_cores}</p>
        """
        
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
        return f"{round(num, 2)}"
    
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
    
    # 在主程序中也设置应用程序图标
    icon_path = resource_path("1.ico")
    app.setWindowIcon(QIcon(icon_path))
    
    window = CPUTestApp()
    window.show()
    sys.exit(app.exec())