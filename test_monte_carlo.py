#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:21:23 2021

@author: md703
"""

from IPython import get_ipython
get_ipython().magic('clear')
get_ipython().magic('reset -f')
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
plt.rcParams.update({'mathtext.default': 'regular'})
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 300


# %% parameters
# Example: R=5, N=500
def testCircle(R, N):
    circleX, circleY = getCircleCoord(R)
    
    r_inverse = R * np.sqrt(np.random.uniform(size=N))
    
    phi_inverse = 2 * np.pi * np.random.uniform(size=N)
    
    x_inverse = r_inverse * np.cos(phi_inverse)
    
    y_inverse = r_inverse * np.sin(phi_inverse)
    
    plt.plot(circleX, circleY)
    plt.axes().set_aspect("equal")
    plt.plot(x_inverse, y_inverse, ".", label="Simulated points, N = {}".format(N))
    plt.xlim(-1.3*R, 1.3*R)
    plt.ylim(-1.3*R, 1.3*R)
    plt.legend(fontsize="small")
    plt.grid()
    plt.show()


def testPI(squareSideLength, N, showFigure=False):
    # sampling points
    squareX = -squareSideLength/2 + squareSideLength * np.random.uniform(size=N)
    squareY = -squareSideLength/2 + squareSideLength * np.random.uniform(size=N)
    
    # estimate pi
    dist2cen = np.sqrt(squareX**2 + squareY**2)    
    inCircleNum = sum(dist2cen<=squareSideLength/2)    
    estimatedPI = 4 * inCircleNum/N
    
    if showFigure:
        # plot square
        # plt.plot([-squareSideLength/2, -squareSideLength/2], [-squareSideLength/2, squareSideLength/2], color="k")
        # plt.plot([-squareSideLength/2, squareSideLength/2], [squareSideLength/2, squareSideLength/2], color="k")
        # plt.plot([squareSideLength/2, squareSideLength/2], [squareSideLength/2, -squareSideLength/2], color="k")
        # plt.plot([squareSideLength/2, -squareSideLength/2], [-squareSideLength/2, -squareSideLength/2], color="k", label="Square")        
        # plot sampling points by monte carlo method
        plt.axes().set_aspect("equal")
        plt.plot(squareX, squareY, ".", color="green", label="Generated points, N = {}".format(N))
        # get circle coordinates
        circleX, circleY = getCircleCoord(squareSideLength/2)
        # plot circle
        plt.plot(circleX, circleY, color="k", label="Circle")        
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.yticks([-1, -0.5, 0, 0.5, 1])
        plt.xlim(-squareSideLength/2, squareSideLength/2)
        plt.ylim(-squareSideLength/2, squareSideLength/2)
        plt.legend(fontsize="small", bbox_to_anchor=(1.01, 1.01))
        plt.title("Estimated pi = {}".format(estimatedPI))
        plt.show()
    
    return estimatedPI 


def getCircleCoord(R):
    phiArr = np.linspace(0, 2*np.pi, num=100)
    circleX = R * np.cos(phiArr)
    circleY = R * np.sin(phiArr)
    return circleX, circleY


if __name__ == "__main__":
    cvSet = []
    for _ in range(1000):
        est = []
        for _ in range(10):
            est.append(testPI(2, 1000))
        est = np.array(est)
        cv = est.std(ddof=1) / est.mean()
        cvSet.append(cv)
    cvSet = np.array(cvSet)
    cvSet.mean()