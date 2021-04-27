# Check whether the matrix has a dominant diagonal
def Enough_conditions(Matrix):
    '''

    :param Matrix: Matrix
    :return:Returns true if the sum of the absolute values of the I-row members except the main diagonal
             members is less than the absolute value of the main diagonal member, otherwise false.
    '''
    flage = False
    if abs(Matrix[0][0]) > abs(Matrix[0][1]) + abs(Matrix[0][2]) and abs(Matrix[1][1]) > abs(Matrix[1][0]) + abs(
            Matrix[1][2]) and abs(Matrix[2][2]) > abs(Matrix[2][0]) + abs(Matrix[2][1]):
        flage = True

    return flage


# Creating a zero matrix
def zero_matrix(rows, cols):
    matA = []
    for i in range(rows):
        matA.append([])
        for j in range(cols):
            matA[-1].append(0.0)

    return matA


# Arrange the matrix by pivoting
def pivoting(Matrix):
    '''

    :param Matrix: Matrix
    :return:Matrix arranged by pivoting
    '''
    temp = Matrix[0]
    for i in range(len(Matrix)):
        for j in range(1, len(Matrix)):
            if Matrix[i][i] < Matrix[j][i]:
                temp = Matrix[i]
                Matrix[i] = Matrix[j]
                Matrix[j] = temp

    return Matrix


# Creating the coefficient matrix
def Creating_Matrix_A(Matrix):
    new_Mtarix = zero_matrix(3, 3)
    for i in range(len(Matrix)):
        for j in range(3):
            new_Mtarix[i][j] = Matrix[i][j]

    return new_Mtarix


# Create the result vector
def Creating_a_vector_B(Matrix):
    '''

    :param Matrix: Matrix
    :return: result vector B
    '''
    n = len(Matrix)
    vector_B = []
    for i in range(len(Matrix)):
        vector_B.append(Matrix[i][n])
    return vector_B


def yacobi(A, B):
    '''
    Solving a system of equations according to the yacobi method

    :param A: Matrix A
    :param B:Result vector B
    '''
    if Enough_conditions(A):
        print("There is a dominant diagonal in the matrix")
    else:
        print("There is no dominant diagonal in the matrix")

    f1 = lambda x, y, z: (B[0] - A[0][1] * y - A[0][2] * z) / A[0][0]
    f2 = lambda x, y, z: (B[1] - A[1][0] * x - A[1][2] * z) / A[1][1]
    f3 = lambda x, y, z: (B[2] - A[2][0] * x - A[2][1] * y) / A[2][2]
    # Initial setup
    x0 = 0
    y0 = 0
    z0 = 0
    count = 1

    # Reading tolerable error
    e = 0.00001

    # Implementation of Jacobi Iteration
    print('\nCount\tx\ty\tz\n')

    condition = True

    while condition:
        x1 = f1(x0, y0, z0)
        y1 = f2(x0, y0, z0)
        z1 = f3(x0, y0, z0)
        print('%d\t%0.4f\t%0.4f\t%0.4f\n' % (count, x1, y1, z1))
        e1 = abs(x0 - x1);
        e2 = abs(y0 - y1);
        e3 = abs(z0 - z1);

        count += 1
        x0 = x1
        y0 = y1
        z0 = z1

        condition = e1 > e and e2 > e and e3 > e

    print('\nSolution: x=%0.3f, y=%0.3f and z = %0.3f\n' % (x1, y1, z1))
    print("The number of iterations is : %d\n" % (count - 1))


def Gauss(A, B):
    '''
    Solving a system of equations according to the Gaussian method

    :param A: Matrix A
    :param B:Result vector B

    '''
    f1 = lambda x, y, z: (B[0] - A[0][1] * y - A[0][2] * z) / A[0][0]
    f2 = lambda x, y, z: (B[1] - A[1][0] * x - A[1][2] * z) / A[1][1]
    f3 = lambda x, y, z: (B[2] - A[2][0] * x - A[2][1] * y) / A[2][2]
    # Initial setup
    x0 = 0
    y0 = 0
    z0 = 0
    count = 1

    # Reading tolerable error
    e = 0.00001

    # Implementation of Jacobi Iteration
    print('\nCount\tx\ty\tz\n')

    condition = True

    while condition:
        x1 = f1(x0, y0, z0)
        y1 = f2(x1, y0, z0)
        z1 = f3(x1, y1, z0)
        print('%d\t%0.4f\t%0.4f\t%0.4f\n' % (count, x1, y1, z1))
        e1 = abs(x0 - x1);
        e2 = abs(y0 - y1);
        e3 = abs(z0 - z1);

        count += 1
        x0 = x1
        y0 = y1
        z0 = z1

        condition = e1 > e and e2 > e and e3 > e

    print('\nSolution: x=%0.3f, y=%0.3f and z = %0.3f\n' % (x1, y1, z1))
    print("The number of iterations is : %d\n" % (count - 1))


M = [[2, 10, 4, 6], [0, 4, 5, 5], [4, 2, 0, 2]]  # Matrix M
MATRIX = pivoting(M)
A = Creating_Matrix_A(MATRIX)  # The coefficient matrix
B = Creating_a_vector_B(MATRIX)  # Result vector
choice = int(input(
    "Please enter which method you would like to use:\nEnter 1 for the Yakobi method\nEnter 2 for the Gauss Seidel method\n"))
if (choice == 1):
    yacobi(A, B)
else:
    Gauss(A, B)
