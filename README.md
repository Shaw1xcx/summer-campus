# 数据处理技术补充

本文件夹包含讲义、可独立运行的课堂代码，以及代码使用的全部虚构数据。

## 文件结构

- `数据处理技术补充.html`：学生阅读的完整讲义。
- `code/demos/`：从讲义提取出的 15 段 Python 演示代码，文件名与主题对应。
- `code/results/`：15 段代码的实际运行输出和汇总记录。
- `code/run_all_demos.py`：重新提取并依次执行讲义中全部 Python 代码。
- `data/`：课堂使用的虚构输入数据，以及演示生成的 Parquet、分区目录和 Zarr 数据。

## 运行方式

先在本文件夹打开终端，并安装讲义开头列出的依赖。随后执行：

```bash
python code/run_all_demos.py
```

也可以单独运行某个示例，例如：

```bash
python code/demos/12_zarr_chunks.py
```

所有代码均使用相对路径 `data/...`，因此应当把当前工作目录保持在本文件夹根目录。
