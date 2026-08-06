# logger_setup.py
import logging
import sys
from datetime import datetime

def setup_logger(name="workflow", log_file=None, level=logging.INFO):
    """
    配置全局 Logger
    :param name: Logger 名称
    :param log_file: 日志文件路径 (如果为None则不输出到文件)
    :param level: 日志级别 (logging.INFO / DEBUG)
    :return: logger 对象
    """
    # 1. 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 Handler (防止重复打印)
    if logger.handlers:
        return logger

    # 2. 定义格式：时间 - 级别 - 模块 - 消息
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 3. 控制台输出 (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 4. 文件输出 (FileHandler) - 可选
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger