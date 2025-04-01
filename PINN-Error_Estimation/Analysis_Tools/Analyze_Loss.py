
import numpy as np
from Lib_UQ_DiscreteError import EE_Gradient, Calc_p


# Param
MF1             = "HP_ID8_3/"
Good_Data_Limit = [0, 19]

# Param2
FN       = MF1 + "Beta_Loss2.txt"
Out_Name = MF1 + "ExtGrad_AllCombin_P.txt"


# Read Data (NO FILTERING)
print("Reading")
Raw_Data = np.loadtxt(FN, unpack=True)
h_Raw    = Raw_Data[0][Good_Data_Limit[0]:Good_Data_Limit[1]]
f_Raw    = Raw_Data[5][Good_Data_Limit[0]:Good_Data_Limit[1]]
n        = len(h_Raw)
Lower_Lim= f_Raw[n-1]
print("Total_Data", n)

# Extrapolate
print("Extrapolating")
h = 1. / h_Raw
f = 1. / f_Raw
Ext_Results = []
Ext_Details = []

for i in range(n - 2):
    for j in range(i + 1, n - 1):
        for k in range(j + 1, n):
            h_List = [h[k], h[j], h[i]]
            f_List = [f[k], f[j], f[i]]
            f0, U = EE_Gradient(h_List, f_List)
            Min_Val = f0 - U
            Max_Val = f0 + U
            p_Value = Calc_p(h_List, f_List, 'Set_Iter', Max_Iter=500)

            if Min_Val > 0:
                Real_Val     = 1. / Min_Val
                Temp_Details = [i + 1, j + 1, k + 1, Real_Val, p_Value]
                if Real_Val >= Lower_Lim:
                    Ext_Results.append(Real_Val)
                    Ext_Details.append(Temp_Details)

                        

# Save Extrapolation
print("Saving")
np.savetxt(Out_Name, (Ext_Details), fmt='%d\t%d\t%d\t%.8e\t%.8e')
print("Total Good Ext:", len(Ext_Results))

# Result
Upper_Lim  = np.min(Ext_Results)
print("Minimum Estimate:", Lower_Lim)
print("Maximum Estimate:", Upper_Lim)

