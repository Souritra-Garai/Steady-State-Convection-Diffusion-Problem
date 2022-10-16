# Written by Souritra Garai
# Date - 30th September 2020

import numpy as np
# from numba import jit

# Definition of class to solve standard pentadiagonal matrix equations
class solver :

    def __init__(self) :

        self.__D = None
        self.__E = None
        self.__F = None
        self.__G = None
        self.__H = None

        self.__L = None
        self.__U = None

        self.__m = None

        self.__ready_to_decom = False
        self.__ready_to_solve = False

    def set_m(self, m) :

        if type(m) != int and m <= 0:

            raise TypeError('Input has to be +ve integer!!')

        if type(self.__m) != type(None) :

            if self.__m == m :

                raise RuntimeWarning('Previously set value of m=', self.__m, ' matches current input!!')

            else :

                raise RuntimeError('Previously set value of m=', self.__m, ' does not match current input!!')
        
        else :
            
            self.__m = m

        pass

    def set_D(self, D) :

        if type(self.__m) == type(None) :

            self.__D = np.array(D, dtype=np.float).flatten()
            self.__m = self.__D.shape[0]

        else :

            self.__D = np.array(D, dtype=np.float).flatten()

            if self.__D.shape[0] != self.__m :

                raise RuntimeError('Length of D ~= m !!')

        pass

    def set_E(self, E) :

        if type(self.__m) == type(None) :

            self.__E = np.array(E, dtype=np.float).flatten()
            self.__m = self.__E.shape[0]

        else :

            self.__E = np.array(E, dtype=np.float).flatten()

            if self.__E.shape[0] != self.__m :

                raise RuntimeError('Length of E ~= m !!')

        pass

    def set_F(self, F) :

        if type(self.__m) == type(None) :

            self.__F = np.array(F, dtype=np.float).flatten()
            self.__m = self.__F.shape[0]

        else :

            self.__F = np.array(F, dtype=np.float).flatten()

            if self.__F.shape[0] != self.__m :

                raise RuntimeError('Length of F ~= m !!')

        pass

    def set_G(self, G) :

        if type(self.__m) == type(None) :

            self.__G = np.array(G, dtype=np.float).flatten()
            self.__m = self.__G.shape[0]

        else :

            self.__G = np.array(G, dtype=np.float).flatten()

            if self.__G.shape[0] != self.__m :

                raise RuntimeError('Length of G ~= m !!')

        pass

    def set_H(self, H) :

        if type(self.__m) == type(None) :

            self.__H = np.array(H, dtype=np.float).flatten()
            self.__m = self.__H.shape[0]

        else :

            self.__H = np.array(H, dtype=np.float).flatten()

            if self.__H.shape[0] != self.__m :

                raise RuntimeError('Length of H ~= m !!')

        pass

    def check_if_initialised(self) :

        undefined_variables = ''

        if type(self.__m) == type(None) :

            undefined_variables += 'm, '

        if type(self.__D) == type(None) :

            undefined_variables += 'D, '
            
        if type(self.__E) == type(None) :

            undefined_variables += 'E, '

        if type(self.__F) == type(None) :

            undefined_variables += 'F, '

        if type(self.__G) == type(None) :

            undefined_variables += 'G, '
        
        if type(self.__H) == type(None) :

            undefined_variables += 'H, '

        if len(undefined_variables) != 0 :

            raise RuntimeError('The parameters ' + undefined_variables[:-2] + ' are still undefined!!')

        self.__ready_to_decom = True
        pass

    # @jit(nopython=True)
    def LU_Decomposition(self) :

        if not self.__ready_to_decom :

            raise RuntimeError('Not ready for LU decomposition!! Call check_if_initialised first!!')

        self.__U = np.zeros((3, self.__m), dtype=np.float)
        self.__L = np.zeros((3, self.__m), dtype=np.float)

        self.__U[0, :] = self.__F
        self.__U[1, :] = self.__G
        self.__U[2, :] = self.__H
        self.__L[2, :] = 1

        for i in range(0, self.__m-2) :

            self.__L[1, i+1] = self.__E[i+1] / self.__U[0, i]
            self.__U[0, i+1] -= self.__L[1, i+1] * self.__U[1, i]
            self.__U[1, i+1] -= self.__L[1, i+1] * self.__U[2, i]

            self.__L[0, i+2] = self.__D[i+2] / self.__U[0, i]
            self.__U[0, i+2] -= self.__L[0, i+2] * self.__U[2, i]
            self.__E[i+2] -= self.__L[0, i+2] * self.__U[1, i]

        i = self.__m - 2
        self.__L[1, i+1] = self.__E[i+1] / self.__U[0, i]
        self.__U[0, i+1] -= self.__L[1, i+1] * self.__U[1, i]

        self.__ready_to_solve = True
        pass

    # @jit(nopython=True)
    def solve_Ldb(self, b) :

        d = np.zeros((self.__m), dtype=np.float)

        d[0] = b[0]

        d[1] = b[1] - self.__L[1, 1]*d[0]

        for i in range(2, self.__m) :

            d[i] = b[i] - d[i-1]*self.__L[1, i] - d[i-2]*self.__L[0, i] 

        return d

    # @jit(nopython=True)
    def solve_Uxd(self, d) :

        x = np.zeros((self.__m), dtype=np.float)

        x[self.__m-1] = d[self.__m-1] / self.__U[0, self.__m-1]
        x[self.__m-2] = ( d[self.__m-2] - self.__U[1, self.__m-2]*x[self.__m-1] ) / self.__U[0, self.__m-2]

        for i in range(self.__m-3, -1, -1) :

            x[i] = ( d[i] - self.__U[1, i]*x[i+1] - self.__U[2, i]*x[i+2]) / self.__U[0, i]
    
        return x

    def solve(self, b_input) :

        b = np.array(b_input, dtype=np.float)

        if b.shape[0] != self.__m :

            raise RuntimeError('Size mismatch, b is not equal to m!!')

        if not self.__ready_to_solve :
            
            self.check_if_initialised()
            self.LU_Decomposition()

        return self.solve_Uxd(self.solve_Ldb(b))

    @classmethod
    def generate_solver(cls, D, E, F, G, H) :

        my_obj = cls()
        my_obj.set_D(D)
        my_obj.set_E(E)
        my_obj.set_F(F)
        my_obj.set_G(G)
        my_obj.set_H(H)

        my_obj.check_if_initialised()
        my_obj.LU_Decomposition()

        return my_obj

if __name__ == "__main__":

    A = np.array([  [   5,  6,  17   ],
                    [   7,  11,  9   ],
                    [   19,  3,  2   ]   ])

    b = [1, 2, 3]

    D = [0, 0, 19]
    E = [0, 7, 3]
    F = [5, 11, 2]
    G = [6, 9, 0]
    H = [17, 0, 0]
    # my_solver = solver()
    # my_solver.set_m(3)
    # my_solver.set_E([0, 7, 3])
    # my_solver.set_F([5, 11, 2])
    # my_solver.set_G([6, 9, 0])

    my_solver = solver.generate_solver(D, E, F, G, H)

    print(A)
    print(b)
    print( my_solver.solve(b) )
    print( np.linalg.solve(A, b) )



