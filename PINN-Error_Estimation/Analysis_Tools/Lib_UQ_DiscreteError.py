#######################################################
#Original File at :
# /home/adhika/Desktop/Mesh_Refinement/Codes/Extrapolate
#######################################################

import numpy as np
#from scipy.optimize import curve_fit

###########################################
# MAIN
###########################################

def EE_GCI(x_List, f_List, p_Value = None):
    # Cons
    Safety_Factoy = 1.25
    #print(*x_List, sep='\t')
    #print(*f_List, sep='\t')
    if p_Value == None:
        p_Value = Calc_p(x_List, f_List, "Convergence")
    #print(p_Value)

    # Vars
    f12  = f_List[1] - f_List[0]
    r12p = np.power((x_List[1] / x_List[0]), p_Value)
    f10  = f12 / (r12p - 1.)

    # Main
    GCI_f0         = f_List[0] - f10
    GCI_Number     = Safety_Factoy * np.abs(f10)
    GCI_Percentage = GCI_Number / f_List[0]

    return GCI_f0, GCI_Number

def EE_Gradient(x_List, f_List):
    # Basic Param
    g12 = (f_List[1] - f_List[0]) / (x_List[1] - x_List[0])
    g23 = (f_List[2] - f_List[1]) / (x_List[2] - x_List[1])
    #r12 = x_List[1] / x_List[0]
    r23 = x_List[2] / x_List[1]
    
    # Specific Param
    h23_min = (5 * r23 + 7) / (r23 + 11) / (r23 + 1) * (x_List[2] - x_List[1]) + x_List[1]
    h12_max = (x_List[0] + x_List[1]) / 2
    g0 = 1 / (1 / g12 - (1 / g23 - 1 / g12) / (h23_min / h12_max - 1))
    #print(g12, g23, r23, h23_min, h12_max, g0)

    Condition = "A"
    if (abs(g12) > abs(g23)):
        Condition = "B"
    if g0 * g12 < 0:
        Condition = "C"


    if Condition == "A":
        Val_A = f_List[0] - g12 * x_List[0]
        Val_B = f_List[0]
    elif Condition == "B":
        Val_A = f_List[0] - (g0 + g12) / 2 * h12_max
        Val_B = f_List[0] #- g12 * x_List[0]
    elif Condition == "C":
        Val_A = 101. * f_List[0]
        Val_B = -99. * f_List[0]

    #print(Condition_A, Val_A, Val_B)
    GCI_f0         = (Val_A + Val_B) / 2
    GCI_Number     = np.abs(Val_B - Val_A) / 2
    GCI_Percentage = GCI_Number / GCI_f0

    return GCI_f0, GCI_Number

"""
def EE_GCI_LSQ(x_List, f_List, p_Formal = 2):
    CV_Param, _ = curve_fit(Exp_Func_Tempplate, x_List, f_List)
    p_Value     = CV_Param[2]
    
    f12  = f_List[1] - f_List[0]
    r12  = x_List[1] / x_List[0]
    EE1  = f12 / (np.power(r12, p_Value ) - 1.)
    EE2  = f12 / (np.power(r12, p_Formal) - 1.)
    
    Largef  = np.tile(f_list, (len(f_List), 1))
    dm      = np.max(np.abs(Largef - np.transpose(Largef)))
    f_guess = Exp_Func_Tempplate(x_List, CV_Param[0], CV_Param[1], CV_Param[2])
""" 
#######################################################
# Support
#######################################################
def Calc_p(h_List, f_List, Mode, Conv_Crit = 1e-6, Max_Iter = 100):
    # Mode = "Convergence or Set_Iter"

    # Constants
    P_Current = 0.5
    P_URF     = 0.2
    P_Conv    = 1

    # Params
    r12 = h_List[1] / h_List[0]
    r13 = h_List[2] / h_List[0]
    if np.abs(f_List[0]) >=1e-8:
        f21 = f_List[1] / f_List[0]
        f31 = f_List[2] / f_List[0]
        f_Param = (f31 - f21) / (f21 - 1.)
    else:
        f_Param = (f_List[2]  - f_List[1]) / (f_List[1]  - f_List[0])
    #if np.abs(f21 - 1.) <= 1e-6:
    #    return 1.

    r_Param = np.log(r13)

    if Mode == "Set_Iter":
        for i in range(Max_Iter):
            r12p  = np.power(r12, P_Current)
            LHS   = r12p + (r12p - 1.) * f_Param
            P_New = np.log(LHS) / r_Param
            P_Current = P_New * P_URF + (1. - P_URF) * P_Current
    elif Mode == "Convergence":
        while P_Conv > Conv_Crit:
            r12p  = np.power(r12, P_Current)
            LHS   = r12p + (r12p - 1.) * f_Param
            P_New = np.log(LHS) / r_Param
            P_Conv2 = np.abs(P_New / P_Current - 1.) * P_URF
            while P_Conv2 >= P_Conv:
                P_URF = P_URF / 2.
                P_Conv2 = np.abs(P_New / P_Current - 1.) * P_URF
                if P_URF <= 1e-6:
                    P_Conv2 = 0
                    P_Conv  = 0
            P_Current = P_New * P_URF + (1. - P_URF) * P_Current
            P_Conv    = P_Conv2
            #print(P_Current, P_Conv, P_Conv2)

    return P_Current
    
def Exp_Func_Tempplate(x, a, b, c):
    return a + b * np.exp(c * x)
