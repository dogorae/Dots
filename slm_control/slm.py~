import numpy as np
import matplotlib.pyplot as plt
import csv
import numpy as np
from . import _slm_win as slm

def changeMode(SLMNumber, mode):
    slm.SLM_Ctrl_Open(SLMNumber)
    slm.SLM_Ctrl_WriteVI(SLMNumber,mode)     # 0:Memory 1:DVI
    slm.SLM_Ctrl_Close(SLMNumber)

def display(filepath):
    slm.SLM_Ctrl_Open (1)
    slm.SLM_Ctrl_WriteGS(1,1023)
    slm.SLM_Ctrl_WriteMI_CSV(1,1, 0, filepath)
    slm.SLM_Ctrl_WriteDS(1,1)
    slm.SLM_Ctrl_Close (1)

class Canvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.height, self.width), dtype=np.uint32)
        
    def add_superpixel(self, *superpixels):
        def add_thing(self, thing, x, y):
            ylen, xlen = np.shape(thing)
            padded_thing = np.pad(thing, [(y,self.height-y-ylen),(x,self.width-x-xlen)], constant_values=0)
            self.canvas += padded_thing
            
        for superpixel in superpixels:
            add_thing(self, superpixel.canvas, superpixel.x, superpixel.y)
        
    def show(self):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(self.canvas, cmap='gray')
    
    def save(self, filepath):
        def add_padding():
            ylen, xlen = np.shape(self.canvas)
            padded_canvas = np.append([np.arange(xlen)], self.canvas, axis=0)
            padded_canvas = np.append([[e] for e in (np.arange(ylen+1) -1)], padded_canvas, axis=1)
            padded_canvas = padded_canvas.astype(str)
            padded_canvas[0][0] = "Y/X"
            return padded_canvas
    
        np.savetxt(filepath, add_padding(), fmt="%s", delimiter=",")


class Superpixel():
    def __init__(self, pos=(0,0), offset=0, step=200, width=15, height=15, period=5, vertical=True):
        self.x, self.y = pos #tuple
        if vertical:
            self.width = width
            self.height = height
        else:
            self.width = height
            self.height = width
        self.canvas = np.zeros((height, width), dtype=np.uint32)
        self.generate_grating(offset, step, period, vertical)
    
    def generate_grating(self, offset, step, period, vertical):
        def generate_sawtooth():
            mod_base = step * period
            return ((np.arange(self.width, dtype=np.uint32) * step) % mod_base)

        def duplicate_rows(row):
            return np.array([row] * self.height)
        
        if vertical:
            self.canvas += duplicate_rows((generate_sawtooth() + offset) % 1024)
        else:
            self.canvas += duplicate_rows((generate_sawtooth() + offset) % 1024).T

if __name__ == "__main__":
    changeMode(1,0)
    display(r"C:\santec\SLM-200\Files\santec_logo.csv")
