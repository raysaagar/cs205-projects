import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import math

# Main
if __name__ == '__main__':
	x = np.linspace(1,100,51)
	y1 = [math.log(i,2) for i in x]
	y2 = [(i-1) for i in x]
	plt.plot(x,y1,'-b')
	plt.plot(x,y2,'--g')
	plt.xlabel('Number of bags')
	plt.ylabel('Time to count')
	plt.title('Counting Time for Bags, parallel vs. serial')
	plt.show()
