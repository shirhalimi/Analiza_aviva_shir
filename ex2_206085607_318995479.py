#finf the pivot in the matrix
def find_pivot(matrix):
    m = len(matrix)
    IDMatrix = [[float(i ==j) for i in range(m)] for j in range(m)]
    for j in range(m):
        row = max(range(j, m), key=lambda i: abs(matrix[i][j]))
        if j != row:
            # Swap the rows
            IDMatrix[j], IDMatrix[row] = IDMatrix[row], IDMatrix[j]
    return IDMatrix

#A function that copies the matrix it receives to a new matrix and returns the new matrix
def copy_matrix(mat,size):
    len_rows = len_cols=size; #Length of the line and the column

    new_mat = zero_matrix(len_rows, len_cols)

    for i in range(len_rows):
        for j in range(len_rows):
            new_mat[i][j] = mat[i][j]

    return new_mat

def mul_mat_veq(matrix_A, b):
    x = []
    for i in range(len(matrix_A[0])):  # this loops through columns of the matrix
         total = 0
         for j in range(len(matrix_A[0])):  # this loops through vector coordinates & rows of matrix
            total += b[j] * matrix_A[i][j]
         x.append(total)

    return x

#A function that receives rows and columns and returns a new matrix the size
# of the rows and columns it received and all its members are equal to 0

def zero_matrix(rows, cols):
    zeroM= [[0.0] * rows for i in range(cols)]
    return zeroM

def unitmatrix(matrix,size):
    IDMatrix = [[float(i == j) for i in range(size)] for j in range(size)]
    return IDMatrix


#A function that multiplies between two matrices and returns
# a new matrix which is the value of the two doubled matrices

def matrix_multiply(A,B):
    len_rowsA = len(A)  #Length of the line In matrix A
    len_colsA = len(A[0])  #Length of the column In matrix A

    len_rowsB = len(B) #Length of the line In matrix B
    len_colsB = len(B[0])  #Length of the column In matrix B

    if len_colsA != len_rowsB: #Test that matrix A and matrix B have the same size
        print('The matrix has be a square matrix')
        exit()

    new_mat = zero_matrix(len_rowsA, len_colsB)

    for i in range(len_rowsA):
        for j in range(len_colsB):
            sum = 0
            for k in range(len_colsA):
                sum += A[i][k] * B[k][j]
            new_mat[i][j] = sum

    return new_mat


#Solving a system of equations using an elementary matrix
def Solve_elementary_matrices_Gauss(MatrixA, B, size):

    matrix_A = copy_matrix(MatrixA,size)
    matrix_I = unitmatrix(MatrixA,size)


    len_mat_A = size# The length of a matrix A



    index = 0 # index stands for focus diagonal OR the current diagonal
    #Check the first value in the matrix is different from 0 and if so we will switch between the rows
    for i in range(len_mat_A):
        max =  abs(matrix_A[i][i])
        for j in range(i,len_mat_A):
            if abs(matrix_A[i][j])>max:
                temp = matrix_A[j]
                matrix_A[j]= matrix_A[i]
                matrix_A[i]= temp
                max = abs(matrix_A[i][i])

    diagonal = 1. / matrix_A[index][index]# Current diagonal or diagonal




    for j in range(len_mat_A): # using j to indicate cycling thru columns
        matrix_A[index][j] = diagonal* matrix_A[index][j]
        matrix_I[index][j] = diagonal * matrix_I[index][j]

    list_of_len = list(range(len_mat_A))

    for i in list_of_len[0:index] + list_of_len[index + 1:]:  # * skip row with fd in it.
        cr = matrix_A[i][index]  # cr stands for "current row".
        for j in range(len_mat_A):  # cr - cr * fdRow, but one element at a time.
            matrix_A[i][j] = matrix_A[i][j] - cr * matrix_A[index][j]
            matrix_I[i][j] = matrix_I[i][j] - cr * matrix_I[index][j]


    for index in range(1, len_mat_A):  # stands for focus diagonal
        diagonal= 1.0 / matrix_A[index][index]
        # FIRST: scale fd row with fd inverse.
        for j in range(len_mat_A):  # Use j to indicate column looping.
            matrix_A[index][j] *= diagonal
            matrix_I[index][j] *= diagonal


        # SECOND: operate on all rows except cd row.
        for i in list_of_len[:index] + list_of_len[index + 1:]:  # * skip row with fd in it.
            cr = matrix_A[i][index]  # cr stands for "current row".
            for j in range(len_mat_A):  # cr - cr * fdRow, but one element at a time.
                matrix_A[i][j] = matrix_A[i][j] - cr * matrix_A[index][j]
                matrix_I[i][j] = matrix_I[i][j] - cr * matrix_I[index][j]

    x = mul_mat_veq(matrix_I, B)

    return  x


#Decomposition into L or  method
def find_L_U(matrix,size):
    #Replace rows
    #the row with the highest member in the first column will be in first row
    max = matrix[0]
    d = 0
    for i in range(size):
         if abs(matrix[i][d])>max[d]:
             max = matrix[i]
             k = i

    temp = matrix[0]
    matrix[0]=max
    matrix[k]= temp

    len_matrix = size
    L = [[0.0] * len_matrix for i in range(len_matrix)]#Building and resetting L matrix
    U = [[0.0] * len_matrix for i in range(len_matrix)]#Building and resetting U matrix
    piv = find_pivot(matrix) #pivot
    mul_piv_mat = matrix_multiply(piv, matrix)
    for j in range(len_matrix):
        L[j][j] = 1.0
        for i in range(j+1):
            sum1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = mul_piv_mat[i][j] - sum1
        for i in range(j, len_matrix):
            sum2 = sum(U[k][j] * L[i][k] for k in range(j))
            if U[j][j] != 0:
                L[i][j] = (mul_piv_mat[i][j] - sum2) / U[j][j]
    return (L, U)


matrix_A = [[0,1,1,1], [1,1,2,1], [2,2,4,0],[1,2,1,1]]# matrix A
B = [1,1,1]
size_matrix_A=4;


if size_matrix_A<4: #if the size of the matrix is less then 4 solve in Gauss method
        x = Solve_elementary_matrices_Gauss(matrix_A, B, size_matrix_A)  # Reverse matrix A^-1
        print("The solve is: ",x)
else:#if the size of the matrix is more then 4 solve in Decomposition into L or  method
        l, u = find_L_U(matrix_A,size_matrix_A)
        print("The solve is:\nL= ",l)
        print("U= ",u)