import numpy as np

def problem_1(x_1,x_2):
    if (x_1 == 4 and x_2==2):
        print("Hurrah you answered correctly")
    else:
        print("Sorry wrong, try again")


def problem_2(x_1,x_2,x_3):

    if (x_1 == 6 and x_2==6 and x_3==-3):
        print("Hurrah you answered correctly")
    else:
        print("Sorry wrong, try again")

def problem_3(K):

    K_true = np.array([[3],[3],[3],[3],[3]])
    if (K==K_true):
        print("Hurrah you answered correctly")
    else:
        print("Sorry wrong, try again")


def problem_5(K_3):

    K_true = np.array([[9],[3],[1]])
    if (K_3==K_true):
        print("Hurrah you answered correctly")
    else:
        print("Sorry wrong, try again")

def temperature(z):

    Noise = np.random.rand(len(z),1)*4
    T = 20-6.5*z + Noise

    return T
