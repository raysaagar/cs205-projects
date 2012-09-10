import numpy as np     # Use numpy package with alias np

a = np.arange(0,1,0.3) # [0,0.3,0.6,0.9], step is float
b = np.array([[0,1], [2,3]])  # 2x2 array
B = np.matrix([[0,1], [2,3]]) # 2x2 matrix
C = np.matrix('4 5; 6 7')     # 2x2 matrix
z = np.zeros([2,2])    # 2x2 ARRAY of zeros
Z = np.asmatrix(z)     # 2x2 MATRIX of zeros

blockMat = np.bmat('B C;C Z') # 4x4 matrix from blocks

b2 = b*b               # Element-wise multiplication
B2 = B*B               # Matrix multiplication
