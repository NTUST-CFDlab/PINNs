# ------------------------------------------------------
# Version 1.3.3
# ------------------------------------------------------

# Import
from Case_Details import *
from NN_Create import *
from NN_Training import *
from Point_Functions import *
from General_Functions import *
from Report_Functions import *
from Case_Info import *

# Load Classes
print("Loading Classes")
Case_Info = Case_Info_Class()
CD = Case_Details_Class()
PINN_Solver = PINN_Solver_Class()
RR = Report_Result()

# Applying Basic Vlaues
print("Applying Basic Values")
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)
tf.random.set_seed(0)

# Preperations
print("Preparing PINN")
CD.Initialize_Values(Case_Info)
Init_Case(CD.MF)
Set_NN_Model(Case_Info, CD)
Generate_Points(Case_Info, CD)
RR.Initialize_Values(Case_Info, CD)
PINN_Solver.Initialize_Training_Info(Case_Info, CD, RR, DTYPE)

# Training
print("Begin Training")
PINN_Solver.Begin_Training()

# Final Report
Plot_Final_LossCurve3(Case_Info)
Print_Last_Loss()
Rename_Reports_Folder(CD.Case_Name)
#Export_Tecplot_File(CD, 0., NN_Solver.DTYPE)
