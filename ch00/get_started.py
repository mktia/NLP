import numpy as np

x = np.array([1, 2, 3])
print(x.__class__)
# output: <class 'numpy.ndarray'>
print(x.shape)
# output: (3,)
print(x.ndim)
# output: 1

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
# output: (2, 3)
print(W.ndim)
# output: 2

W = np.array([[1, 2], [3, 4]])
X = np.array([[5, 6], [7, 8]])
print(W + X)
# output: [[ 6  8]
#          [10 12]]
print(W * X)
# output: [[ 5 12]
#          [21 32]]

# Broadcast (Repeat node)
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
B = np.array([[5, 6], [5, 6]])
print(A * 10)
# output: [[10 20]
#          [30 40]]
print(A * b)
# output: [[ 5 12]
#          [15 24]]
print(A * B)
# output: [[ 5 12]
#          [15 24]]

a = np.array([1, 2])
b = np.array([3, 4])
print(a * b)
# output: [3 8]
print(np.dot(a, b))
# output: 11
