# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 20:09:27 2019

@author: mico
"""

##This is Linear Regression with univariate function. 
import matplotlib.pyplot as plt
import random

def draw_bestline(): #This creates a randomly generated line we're trying to optimize. 
    
    w_val = random.randint(1, 10) #This will generate a singular random w_val. This w_val is what we'll be optimizing.
    b_val = random.randint(1, 10) #This will generate a singular random b_val. This b_val is what we'll also be optimizing.
  
    x_values_for_line = [1, 2, 3]
    y_values_for_line = []
    for el in range(len(x_values_for_line)):
        line = (w_val * x_values_for_line[el]) + b_val #This is the function we'll get out of those values.
        y_values_for_line.append(line)
    
    plot_points = generate_datapoints()
    actual_y_points = plot_points[1]
    
    print(actual_y_points)
    
    sumDerW = 0 #Initialize the sums for both the derivatives W and B.
    sumDerB = 0
    learning_rate = .001 #This is the learning rate that I'll be multiplying the w_val and b_val with.
    
    GD_y_values = [] #This will hold the y_values_ taken from the GD algorithm.
    
    #From line 36 to 49, this is the gradient descent algorithm that let's my program find the optimal weights for
    #the univariate function I'm optimizing. 
    
    for i in range(500):
        for x, y in zip(x_values_for_line, actual_y_points):
            sumDerW = sumDerW + (w_val*x + b_val - y) * x 
            sumDerB = sumDerB + (w_val*x + b_val - y) * 1
    
        sumDerW = (2/len(x_values_for_line)) * sumDerW
        sumDerB = (2/len(x_values_for_line)) * sumDerB
    
        w_val = w_val - (learning_rate)*(sumDerW)
        b_val = b_val - (learning_rate)*(sumDerB)
        
        
        print("This is the first weight: ", w_val)
        print("This is the bias weight: " , b_val)
    
    for el in range(len(x_values_for_line)):
         line = (w_val * x_values_for_line[el]) + b_val #This is the function we'll get out of those values.
         GD_y_values.append(line)

    return y_values_for_line, GD_y_values

def generate_datapoints(): #Allows me to generate random datapoints to use on my error function. 
    x_plot = [1, 2, 3]
    y_plot = [3, 4, 5]
    
    return x_plot, y_plot

def draw_graph(): #This function generates our points that we're trying to make a best fit line for. 
    plot_points = generate_datapoints()
    x_plot = plot_points[0]
    y_plot = plot_points[1]
    
    lines = draw_bestline()
    
    plt.scatter(x_plot, y_plot)
    plt.plot(x_plot, lines[1])

        
def main():
    
    draw_bestline()
    draw_graph()

main()