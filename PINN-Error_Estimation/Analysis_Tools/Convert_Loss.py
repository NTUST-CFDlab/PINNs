####################################################################
# Task:	Read Data (filter out important info)
#	Calc LD
#	Plot Raw data (Mode 0, 1, 2)
####################################################################

import numpy as np
import matplotlib.pyplot as plt

MF1          = "HP_ID8_3/"
Total_Data   = 640
LD_Weight    = 1. 
LM_Index     = -3

def Calc_LD(Beta, LM, Weights, Alpha = 4.2):
    All_LD = np.zeros(len(Beta))
    for i in range(len(Beta)):
        # Init
        Converge_Criteria = 1e-6
        Current_Error     = 1.
        Min_Val, Max_Val  = 0., 80.
        Cur_Val           = 0.
        Cur_Beta2, Ref_LM = np.square(Beta[i]), LM[i]/Weights[1]
        #print(Cur_Beta2, Ref_LM)
        #while Current_Error > Converge_Criteria:
        for j in range(200):
            Cur_Val = (Min_Val + Max_Val) / 2.
            Cur_LM  = 2. * Cur_Val / (1. + np.exp(-Alpha * np.log10(Cur_Val/Cur_Beta2) + Alpha/2.))
            Dif     = Cur_LM - Ref_LM
            #print(i, Min_Val, Max_Val, Cur_Val, Cur_LM, Dif)
            if Dif < 0:
                Min_Val = Cur_Val
            else:
                Max_Val = Cur_Val
            #Current_Error = np.abs(Dif)

        All_LD[i] = Cur_Val
    return All_LD /Weights[0]

# Read Data
print("Reading Data")
FN       = MF1 + "Beta_Loss.txt"
Raw_Data = np.loadtxt(FN, unpack = True)
Beta     = Raw_Data[0]
Iter     = Raw_Data[1]
LF       = Raw_Data[-1]
LM       = Raw_Data[LM_Index]
D_Iter   = np.diff(np.insert(Iter, 0, 0))

# LD
print("Converting")
LD_M0= Calc_LD(Beta, LM, [LD_Weight, 1], Alpha = 4.2)
L2   = np.sqrt(LD_M0/Total_Data)

# Print Result
print("Printing")
All_Export_Data = np.array([Beta, LM, LF, LD_M0, D_Iter, L2])
if Beta[0] > Beta[1]:
    All_Export_Data  = np.flip(All_Export_Data, axis = 1)
np.savetxt(MF1 + "Beta_Loss2.txt", np.transpose(All_Export_Data), fmt='%.4e\t%.8e\t%.8e\t%.8e\t%d\t%.8e')
	


