import numpy as np

def create_coefficient_array(ip2=0, ip1=0, i=0, im1=0, im2=0) :

    return np.array([ip2, ip1, i, im1, im2], dtype=float)

def multiply(phi, coeff_arr) :

    if phi.ndim == 1 :

        if phi.shape[0] == 3 :

            return np.sum( np.multiply( phi, coeff_arr[1:-1] ) )
        
        else :

            return np.sum( np.multiply( phi, coeff_arr ) )

    else :

        if phi.shape[1] == 3 :

            return np.matmul( phi, coeff_arr[1:-1] )

        else :

            return np.matmul( phi, coeff_arr.transpose() )

First_Derivative_1st_Order_Forward_Difference_Coeff  = create_coefficient_array(    ip1 = 1,
                                                                                    i   =-1 )

First_Derivative_1st_Order_Backward_Difference_Coeff = create_coefficient_array(    i   = 1,
                                                                                    im1 =-1 )

First_Derivative_2nd_Order_Central_Difference_Coeff = create_coefficient_array( ip1 = 0.5,
                                                                                im1 =-0.5 )

First_Derivative_3rd_Order_Forward_Difference_Coeff  = create_coefficient_array(    ip2 =-1,
                                                                                    ip1 = 6,
                                                                                    i   =-3,
                                                                                    im1 =-2 ) / 6

First_Derivative_3rd_Order_Backward_Difference_Coeff = create_coefficient_array(    ip1 = 2,
                                                                                    i   = 3,
                                                                                    im1 =-6,
                                                                                    im2 = 1 ) / 6

Second_Derivative_2nd_Order_Central_Difference_Coeff = create_coefficient_array(    ip1 = 1,
                                                                                    i   =-2,
                                                                                    im1 = 1 )

Second_Derivative_4th_Order_Central_Difference_Coeff = create_coefficient_array(    ip2 = -1,
                                                                                    ip1 = 16,
                                                                                    i   =-30,
                                                                                    im1 = 16,
                                                                                    im2 = -1    ) / 12

# Explicit Scheme
def Coefficients_3rd_Order_Explicit_Scheme(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_3rd_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_3rd_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_4th_Order_Central_Difference_Coeff)

    return diffusion - convection

def Coefficients_2nd_Order_Explicit_Scheme(d, c) :

    convection = c * np.copy(First_Derivative_2nd_Order_Central_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return diffusion - convection

def Coefficients_1st_Order_Explicit_Scheme(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_1st_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_1st_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return diffusion - convection

# Implicit Scheme
def Coefficients_3rd_Order_Implicit_Scheme(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_3rd_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_3rd_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_4th_Order_Central_Difference_Coeff)

    return convection - diffusion + create_coefficient_array(i=1)

def Coefficients_2nd_Order_Implicit_Scheme(d, c) :

    convection = c * np.copy(First_Derivative_2nd_Order_Central_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return convection - diffusion + create_coefficient_array(i=1)

def Coefficients_1st_Order_Implicit_Scheme(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_1st_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_1st_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return convection - diffusion + create_coefficient_array(i=1)

# Crank Nicolson Method
# Implicit Scheme
def Coefficients_3rd_Order_Crank_Nicolson_Method(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_3rd_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_3rd_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_4th_Order_Central_Difference_Coeff)

    return 0.5 * (convection - diffusion) + create_coefficient_array(i=1), 0.5 * (diffusion - convection)

def Coefficients_2nd_Order_Crank_Nicolson_Method(d, c) :
    
    convection = c * np.copy(First_Derivative_2nd_Order_Central_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return 0.5 * (convection - diffusion) + create_coefficient_array(i=1), 0.5 * (diffusion - convection)

def Coefficients_1st_Order_Crank_Nicolson_Method(d, c) :

    if c < 0 :

        convection = c * np.copy(First_Derivative_1st_Order_Forward_Difference_Coeff)

    else :

        convection = c * np.copy(First_Derivative_1st_Order_Backward_Difference_Coeff)

    diffusion = d * np.copy(Second_Derivative_2nd_Order_Central_Difference_Coeff)

    return 0.5 * (convection - diffusion) + create_coefficient_array(i=1), 0.5 * (diffusion - convection)
