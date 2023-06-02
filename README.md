# Mandelbrot Set Visualizer in Python

This visualizer illustrates the performance differences between Multiprocessing and Multithreading in Python using the Mandelbrot set as an example.

Python's Global Interpreter Lock (GIL) typically results in Multithreading performing less efficiently than Multiprocessing, particularly for tasks that require heavy use of the CPU.

## Libraries

`pygame` for rendering graphics  
`cython` for speeding up numerical computations  
`numpy` for array manipulation and access

## Usage

`pip install requirements.txt`
`python main.py`

Note: you may need to compile `calculate_mandelbrot.pyx`, run `python setup.py build_ext --inplace`

## Controls

`up`: Zoom in  
`down`: Zoom out  
`(w, a, s, d)`: Move the scene  
`(right or left)`: Increase/decrease the number of iterations  
`t`: switch to Multithreading  
`p`: switch to Multiprocessing  

![screenshot](screenshot.png)
