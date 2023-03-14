# mandelbrot-python-visualizer

Visualize the mandelbrot in Python using Multiprocessing or Multithreading to demonstrae performance differences.  

Due to the Python Global Interpreter Lock (GIL) multithreading has worst performance than multiprocessing.  

## Used libraries

`pygame` for displaying graphics  
`numba` for accelerating numerical computations  
`numpy` for easier access of arrays  


## Controls

`up`: Zoom in  
`down`: Zoom out  
`(w, a, s, d)`: Move the scene  
`(right or left)`: Increase/decrease the number of iterations  
`t`: switch to Multithreading  
`p`: switch to Multiprocessing  

![screenshot](screenshot.png)
