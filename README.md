# Mandelbrot Visualizer: Multiprocessing vs. Multithreading

This application visualizes the Mandelbrot set while showcasing the performance differences between multiprocessing and multithreading in Python.

**Why Multiprocessing vs. Multithreading?**  
Python's Global Interpreter Lock (GIL) generally restricts efficient CPU utilization when using threads for CPU-bound tasks. Multiprocessing overcomes this by creating separate processes, which leads to better performance. This application allows you to switch between both modes and observe their performance.


## Installation

- `python -m venv .venv`
- `source .venv/bin/activate` on Linux or `.venv\Scripts\activate` on Windows
- `pip install requirements.txt`
- `python app.py`
