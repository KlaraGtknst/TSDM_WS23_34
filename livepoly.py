import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .polynomial.orthpoly import OrthogonalPolyDB

def slide_and_evaluate(poly, Y, step=1):
    window = poly.size_
    #last regression needs at least window y-data points 
    return np.array([poly.transform_evaluate(Y[idx:min(idx+window, len(Y))]) for idx in np.arange(0, len(Y)-window, step)])
class LivePolyRegr2SS:
    def __init__(self, x, y, window=10, degree=3, step = 1, gt_nov = None, dims_ss= (0, 1), animate=True, max_abs = None):
        self.x = x
        self.y = y
        if gt_nov is None:
            gt_nov = np.zeros(len(y))
        if len(x) != len(y) or (gt_nov is not None and len(gt_nov) != len(y)):
            raise "Shapes do not match"
        self.gt_nov = np.array([np.any(gt_nov[idx:idx+step]) for idx in np.arange(0, len(gt_nov), step)]) # gt convert to steps
        gt_colors = {False: "b", True: "r"}
        self.gt_colored = np.array([gt_colors[t] for t in self.gt_nov])
        self.window = window
        self.degree = degree
        self.step = step
        self.frame_cnt = int(np.ceil((len(x)-self.window)/self.step))
        self.shape_space_dots = []
        self.dims_ss = dims_ss
        self.max_abs = max_abs
        self.calc_regression()
        # init figure
        self.fig = plt.figure(figsize=(9,4))
        self.subplot_poly = self.fig.add_subplot(211)
        self.subplot_shapespace = self.fig.add_subplot(212)
        self.poly_regr, = self.subplot_poly.plot([], [], 'r', zorder=2)
        self.plot_static()
        # execute the animation
        if animate:
            self.exec_animation()
        else:
            self.plot_shape_space()
    
    def plot_static(self):
        """ Plots the static stuff """
        self.subplot_poly.plot(self.x, self.y, zorder=1)
    
    def calc_regression(self):
        poly = OrthogonalPolyDB(degree=self.degree, size=self.window)
        self.poly_result = slide_and_evaluate(poly, self.y, self.step)
        self.ss_calculated = False
    
    def exec_animation(self):
        print(self.frame_cnt)
        for i in range(self.frame_cnt):
            self.animate(i)
            self.subplot_poly.relim()
            self.subplot_poly.autoscale_view()
            self.fig.canvas.draw()
        
    def animate(self, frame):
        X_win = self.x[np.arange(frame*self.step, min(frame*self.step+self.window, len(self.x)))]
        Y_win = self.poly_result[frame][1]
        self.poly_regr.set_data(X_win, Y_win)
        
        color = plt.cm.jet((frame%100)/100)
        if frame > 0 and frame%80==0:
            # redraw shapespace
            self.subplot_shapespace.cla()
            plt_idx = frame+1
            self.subplot_shapespace.scatter(self.shape_space[:plt_idx, self.dims_ss[0]], 
                                            self.shape_space[:plt_idx, self.dims_ss[1]], 
                                            c="black", s=5)
        else:
            self.subplot_shapespace.scatter(self.shape_space[frame, self.dims_ss[0]], 
                                            self.shape_space[frame, self.dims_ss[1]],
                                            c=np.array([color]).reshape(1,-1), s=10)
    
    @property
    def shape_space(self):
        if self.ss_calculated == False:
            self.ss_calculated = True
            self._shape_space = np.array([a for a in self.poly_result[:, 0]])
            max_abs = self.max_abs if self.max_abs is not None else np.max(np.abs(self._shape_space), axis = 0)
            self._shape_space = self._shape_space/max_abs
            self.max_abs = max_abs
        return self._shape_space
    
    def plot_shape_space(self, target=plt):
        # redraw shapespace
        target.scatter(self.shape_space[:, self.dims_ss[0]], 
                       self.shape_space[:, self.dims_ss[1]], 
                       c="black", s=1)
    
    def get_gt(self):
        return self.gt_nov
