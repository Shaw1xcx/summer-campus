from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, sleep
import csv
import sys
import sysconfig

# 读取 prepare_demo.py 生成的 8 项任务，不在计时阶段临时改变输入。
with Path("data/thread-tasks.csv").open(encoding="utf-8") as file:
    tasks = list(csv.DictReader(file))
cpu_tasks = [task for task in tasks if task["kind"] == "cpu"]
io_tasks = [task for task in tasks if task["kind"] == "io"]

def cpu_work(task):
    """纯 Python 整数循环：大部分时间都在执行 Python 字节码。"""
    value = int(task["seed"])
    checksum = 0
    for _ in range(int(task["rounds"])):
        value = (value * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        checksum ^= value
    return checksum

def io_wait(task):
    """sleep 模拟磁盘或网络等待；等待期间线程不占用 CPU。"""
    sleep(float(task["delay_seconds"]))
    return task["task"]

def measure(work, selected_tasks, use_threads):
    start = perf_counter()
    if use_threads:
        with ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(work, selected_tasks))
    else:
        result = [work(task) for task in selected_tasks]
    return result, perf_counter() - start

cpu_serial, cpu_serial_s = measure(cpu_work, cpu_tasks, False)
cpu_threads, cpu_threads_s = measure(cpu_work, cpu_tasks, True)
io_serial, io_serial_s = measure(io_wait, io_tasks, False)
io_threads, io_threads_s = measure(io_wait, io_tasks, True)

free_threaded_build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
gil_enabled = sys._is_gil_enabled() if hasattr(sys, "_is_gil_enabled") else True
print("Free-threaded build:", free_threaded_build)
print("GIL currently enabled:", gil_enabled)
print("CPU tasks:", len(cpu_tasks), "×", cpu_tasks[0]["rounds"], "rounds")
print("CPU serial / threads:", round(cpu_serial_s, 3), "/",
      round(cpu_threads_s, 3), "s")
print("CPU thread speedup:", round(cpu_serial_s / cpu_threads_s, 2), "x")
print("I/O tasks:", len(io_tasks), "×", io_tasks[0]["delay_seconds"], "s wait")
print("I/O serial / threads:", round(io_serial_s, 3), "/",
      round(io_threads_s, 3), "s")
print("I/O thread speedup:", round(io_serial_s / io_threads_s, 2), "x")
print("CPU results equal:", cpu_serial == cpu_threads)
print("I/O results equal :", io_serial == io_threads)
