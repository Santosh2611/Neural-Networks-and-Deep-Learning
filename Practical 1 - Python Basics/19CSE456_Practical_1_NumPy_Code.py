print("\n1. Import the numpy package under the name np and print the numpy version and the configuration:")
import numpy as np
print("NumPy version: ", np.__version__)
print("NumPy configuration: ", np.show_config())

print("\n2. Convert a list of numeric value into a one-dimensional NumPy array:")
num_list = [1, 2, 3, 4, 5]
np_array = np.array(num_list)
print(np_array)

print("\n3. Create a null vector of size 5:")
null_vector = np.zeros(5)
print(null_vector)

print("\n4. Create a null vector of size 12 but the fourth value which is 100:")
null_vector = np.zeros(12)
null_vector[3] = 100
print(null_vector)

print("\n5. Create a vector with values ranging from 29 to 60:")
range_vector = np.arange(29, 61)
print(range_vector)

print("\n6. Create a 3x3 matrix with values ranging from 50 to 58:")
matrix_3x3 = np.arange(50, 59).reshape(3, 3)
print(matrix_3x3)

print("\n7. Find indices of non-zero elements from [5,8,0,9,0,0]:")
arr = np.array([5, 8, 0, 9, 0, 0])
non_zero_indices = np.nonzero(arr)
print(non_zero_indices)

print("\n8. Create a 5x5 identity matrix:")
identity_matrix = np.identity(5)
print(identity_matrix)

print("\n9. Create a 3x3x3 array with random values:")
random_array = np.random.rand(3, 3, 3)
print(random_array)

print("\n10. Create a 9x9 array with random values and find the minimum and maximum values:")
random_array = np.random.rand(9, 9)
min_val = np.min(random_array)
max_val = np.max(random_array)
print("Minimum value: ", min_val)
print("Maximum value: ", max_val)

print("\n11. Create a random vector of size 20 and find the mean value:")
random_array = np.random.rand(20)
mean_val = np.mean(random_array)
print("Mean value: ", mean_val)

print("\n12. Create a 6x6 matrix with values 1,2,3,4 just below the diagonal:")
matrix_6x6 = np.diag(1 + np.arange(4), k=-1)
print(matrix_6x6)

print("\n13. Consider a (8,9,10) shape array, what is the index (x,y,z) of the 100th element?")
shape_array = np.zeros((8, 9, 10))
x, y, z = np.unravel_index(99, shape_array.shape)
print("Index of 100th element: ", x, y, z)

print("\n14. Consider two random array A and B, check if they are equal:")
array_A = np.random.rand(3, 3)
array_B = np.random.rand(3, 3)
equal_arrays = np.array_equal(array_A, array_B)
print("Are the arrays equal? ", equal_arrays)

print("\n15. How to sort an array of shape (5, 4) by the nth column?")
array_5x4 = np.random.rand(5, 4)
print("Array before sorting: \n", array_5x4)
nth_column = 1  # sorting by the second column
sorted_array = array_5x4[array_5x4[:, nth_column].argsort()]
print("Array after sorting by the nth column: \n", sorted_array)

print("\n16. Consider an array a = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array b = [[1,2,3,4], [2,3,4,5], 3,4,5,6], ..., [11,12,13,14]]?")
a = np.arange(1, 15)
b = np.array([a[i:i+4] for i in range(0, len(a)-3)])
print(b)

print("\n17. How to find the most frequent value in an array?")
array = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5])
values, counts = np.unique(array, return_counts=True)
index = np.argmax(counts)
most_frequent_value = values[index]
print("Most frequent value: ", most_frequent_value)
