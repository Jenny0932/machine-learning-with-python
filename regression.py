from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import random

#xs = np.array([1,2,3,4,5,6,7], dtype=np.float64)
#ys = np.array([4,5,8,5,7,6,9], dtype=np.float64)



def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation =='pos':
            val+=step
        elif correlation and correlation =='neg':
            val-=step
    xs = [i for i in range(len(ys))]        
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
    
    

def best_fit_slope_and_intercept(xs,ys):
    m = ( ( mean(xs)*mean(ys) - mean(xs*ys) )  /
        ( mean(xs)*mean(xs) -  mean(xs*xs) ) )    
    b = mean(ys) - m*mean(xs)
    return m,b

def squared_error(ys_orig, ys_line):    
    return sum((ys_orig-ys_line)*(ys_orig-ys_line))
    
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squred_error_regr = squared_error(ys_orig, ys_line)
    squred_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 -  ( squred_error_regr / squred_error_y_mean )



xs, ys           = create_dataset(40, 10, 1, correlation='pos')    
m,b              = best_fit_slope_and_intercept(xs,ys)
regiression_line = [m*x+b for x in xs]

predict_x = 8
predict_y = m*predict_x + b
r_squared =  coefficient_of_determination(ys, regiression_line)

print(r_squared)
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y,s=100,color='r')
plt.plot(xs,regiression_line)
plt.show()