
# Import
#import tensorflow as tf
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#from matplotlib.colors import Normalize

from time import time
from Report_Loss import *
from Report_Images import *

class Report_Result():
    def Initialize_Values(self, Case_Info, CD):
        self.CD = CD
        #self.Print_Status, self.Print_Types = Case_Info.Class_RP.Load_Report_Print_Setting()
        self.Print_Status = [True, False, True]	#[Print_Final, Print_Checkpoint, Print_Backup]
        self.Print_Types  = [True, True, False] #[Print_Output_Results, Print_Residual, Print_Velocity_Profile]
        
        self.Mark_Loc    = Calc_Marker_Location(Case_Info)
        self.R_Img_Class = []
        
        All_Settings      = Case_Info.Class_RP.Load_Image_Setting("Report")
        self.Total_Domain = All_Settings[0]
        for i in range(All_Settings[0]):
            self.R_Img_Class.append(Report_xyu_PColorMesh(CD, i, All_Settings))

        # Load LF Setting
        #self.R_LF = Report_LF()

        # Load Image Setting
        #[self.n_Image, self.Print_ID, Plot_MinMax, Domain_Info, Sampling, Size] = Case_Info.Class_RP.Load_Image_Setting("Report")
        
            #Local_Domain_Info = [Domain_Info[0][i], Domain_Info[1][i], Domain_Info[2][i], Domain_Info[3][i]]
            #self.R_Data[i].Initialize_Values(CD, Local_Domain_Info, Plot_MinMax, Sampling[i], Size[i])
        #self.R_Vel_Profile = Vel_Profile()
        #self.R_Vel_Profile.Initialize_Values(CD, Case_Info)


    def Report_Current_Result(self, NN_Solver, Mode, Save_Name):
        # Final =       SaveNN --> LF_Image --> LF_Data --> (all) Calc --> Print_Data --> Print_Image
        # Checkpoint =  SaveNN --> LF_Data --> (all) Calc --> Print_Data
        # Backup =      SaveNN --> LF_Image --> LF_Data --> (all) Calc --> Print_Data --> Print_Image

        if Mode == "F":
            NN_Save_Name = self.CD.MF + 'NN/'+ Save_Name
            Image_Save_Name = "FINAL" + "-"
            Print_Status = self.Print_Status[0]
        elif Mode == "C":
            NN_Save_Name = self.CD.MF + 'NN/Backup-LS-' + Save_Name
            Image_Save_Name = Save_Name + "-"
            Print_Status = self.Print_Status[1]
        elif Mode == "B":
            NN_Save_Name = self.CD.MF + 'NN/Backup-iter-' + Save_Name
            Image_Save_Name = Save_Name + "-"
            Print_Status = self.Print_Status[2]
        elif Mode == "E":
            NN_Save_Name = self.CD.MF + 'NN/Backup-Beta-' + Save_Name
            Image_Save_Name = "Beta_" + Save_Name + "-"
            Print_Status = self.Print_Status[2]	# Dont want to change case info

        NN_Solver.model.save(NN_Save_Name)
        self.Write_Simulation_Progress(Mode, NN_Solver.iter, NN_Solver.current_loss, NN_Solver.Initial_Time)
        Write_LF_Data(NN_Solver, self.Mark_Loc, MF = self.CD.MF)
        Plot_LF_Image(NN_Solver.hist, "Total", "Total", Save_Name, MF = self.CD.MF)

        if Mode == "E":
            Print_Beta_Loss(Save_Name + "\t" + str(NN_Solver.iter))
        for i in range(self.Total_Domain):
            #Image_Save_Name = Image_Save_Name + str(PR_ID)
            if Print_Status:
                self.R_Img_Class[i].Update_All(Mode, str(NN_Solver.iter))
                #PrintRes = False
                #if (self.Print_Types[0] or self.Print_Types[1]):
                #    PrintRes = True
                #    self.R_Data[PR_ID].Calc_Sim_Param()
                    #if (self.CD.Analytical_Exist):
                    #    self.R_Data[PR_ID].Calc_Analytical_Error()
                #self.R_Data[PR_ID].Print_Result_Data(NN_Solver.iter, NN_Solver.current_loss, NN_Solver.Initial_Time, Mode, PrintRes)
                #self.R_Data[PR_ID].Print_Result_Image(Image_Save_Name, self.Print_Types, self.Print_ID)

                # Velocity profile
                #if self.Print_Types[2]:
                #    self.R_Vel_Profile.Calc_Sim_Param()
                #    self.R_Vel_Profile.Print_Result_Data(Save_Name)
                #    self.R_Vel_Profile.Print_Result_Image(Save_Name)
    
    def Write_Simulation_Progress(self, Mode, Iter, Total_Loss, Initial_Time):
        File_Holder = open(self.CD.MF + "Progress.txt", "a+")
        
        str_total   = Mode + "\t"
        str_total  += str(Iter) + "\t"
        str_total  += str(Total_Loss) + "\t"
        str_total  += str(time() - Initial_Time)
        
        File_Holder.writelines(str_total + "\n")
        File_Holder.close()






########################################
# Support
########################################

def Calc_Marker_Location(Case_Info):
    #[LF_Types, LF_Weigths, LF_PG_Conv] = Case_Info.Load_LF_Info()
    LF_Types, _, LF_PG_Conv, _, LF_Setting = Case_Info.Load_Weights_List()
    Mark_Loc = [0]
    #Total_Length = 1
    
    # For every loss type (e.g. Data, GE, BC_D)
    for i in range(len(LF_Types)-1):
        #Value = Mark_Loc[Total_Length-1]
        Value = Mark_Loc[i]
        
        # For every point group
        for j in range(len(LF_PG_Conv[i])):
            ID = LF_PG_Conv[i][j]
            Value += len(LF_Setting[ID])-1
            
        Mark_Loc.append(Value)
        #Total_Length += 1
        
    Mark_Loc.append(0)  # Just to make sure that it doesnot create any index error
    return Mark_Loc
