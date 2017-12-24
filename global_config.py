# -*- coding: utf-8 -*-

import logging
import inspect, os

NUM_PROCESS = os.cpu_count()


def config_log():
    ################################
    ### 日志配置
    _run_script = inspect.stack()[-1][1]
    # 运行脚本名
    RUN_SCRIPT = os.path.basename(_run_script)
    _run_script_no_ext = os.path.splitext(RUN_SCRIPT)[0]
    # 日志文件名
    LOG_FILENAME = '%s.log' % _run_script_no_ext

    # 配置日志输出文件及级别
    FORMAT = '%(asctime)s::%(levelname)s::%(message)s'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT, filemode='a')

    # 同时输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(FORMAT))
    logging.getLogger('').addHandler(console)
