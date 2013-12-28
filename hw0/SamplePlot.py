import matplotlib.pyplot as plt
import math
import numpy as np

x = np.linspace(-2, 3, 51)
y1 = [math.exp(i) for i in x]
y2 = [math.exp(2*i) for i in x]

plt.plot(x, y1, '-b')
plt.plot(x, y2, '--g')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sample plot')
plt.show()
