from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
import csv

with Path("data/orders.csv").open(encoding="utf-8") as file:
    orders = list(csv.DictReader(file))
initial_stock = int(orders[0]["initial_stock"])
quantities = [int(order["quantity"]) for order in orders]

# 错误版：两线程先后执行 Python 字节码，但“检查”和“扣减”不是同一步。
# Barrier 故意让两张订单都在库存仍为 10 时完成检查，再继续扣减。
stock = initial_stock
checked = Barrier(2)
def unsafe_buy(quantity):
    global stock
    if stock >= quantity:
        checked.wait()
        stock -= quantity
        return True
    return False

with ThreadPoolExecutor(max_workers=2) as pool:
    unsafe_results = list(pool.map(unsafe_buy, quantities))
print("unsafe:", unsafe_results, "final stock:", stock)

# 正确版：Lock 把“检查库存 + 扣减库存”包成不可交错的临界区。
stock = initial_stock
stock_lock = Lock()
def safe_buy(quantity):
    global stock
    with stock_lock:
        if stock >= quantity:
            stock -= quantity
            return True
        return False

with ThreadPoolExecutor(max_workers=2) as pool:
    safe_results = list(pool.map(safe_buy, quantities))
print("safe  :", safe_results, "final stock:", stock)
