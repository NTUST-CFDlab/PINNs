##########################################
# Version 1.1	Added option for Unifrom sampling 
#		Change Report to file
##########################################


# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Points_Exclusion import *
from Points_Generation import *
from Points_Special_Processing import *

def Generate_Points(Case_Info, CD):
    #Point_Gen_Types, Point_Grouping, CD.LF_Setting = Case_Info.Load_Point_Gen_Info()
    Scr_BP, Scr_DP, Scr_CP, Scr_UP = Case_Info.Load_Point_Gen_Info()
    _, _, _, Point_Grouping, CD.LF_Setting = Case_Info.Load_Weights_List()
    #Total_Coor_Var = CD.n_Input
    #Total_BC_Vars = CD.n_Output
    WF = Write_To_File(CD.MF + "Points_Report.txt")

    # Report
    WF.PrintLN("-------------------------------------------------------------------------------")
    WF.PrintLN("Points Report")
    WF.PrintLN("-------------------------------------------------------------------------------\n\n")

    # Set
    Set_Names  = ["Boundary", "Data", "Collocation"]
    All_Set    = [Scr_BP, Scr_DP, Scr_CP]
    All_Loc    = [[], [], []]
    #All_Val    = [[], []]
    All_Val    = [[], [], []]
    All_Points = [[], [], []]
    CD.Obj_Set = []

    # Generate
    WF.PrintLN("-------------------------------------------------------------------------------")
    WF.PrintLN("1. Generate Points")
    WF.PrintLN("-------------------------------------------------------------------------------")
    
    for Set_Type in range(len(All_Set)):
        for i in range(len(All_Set[Set_Type])):
            WF.PrintLN(i, All_Set[Set_Type][i][0], All_Set[Set_Type][i][1])
            # Generate Points
            if All_Set[Set_Type][i][0] == "Gen":
                # Generate
                Domain_Param = All_Set[Set_Type][i][2]
                Points_Param = All_Set[Set_Type][i][3]
                if All_Set[Set_Type][i][1] == "Unif_Box":
                    Tmp_Coor, Tmp_TP = Gen_Points_Unif_Box(Domain_Param, Points_Param)
                    WF.PrintMinMax(Tmp_Coor)
                    #WF.PrintLN(np.min(Tmp_Coor, axis = 1))
                    #WF.PrintLN(np.max(Tmp_Coor, axis = 1))
                if All_Set[Set_Type][i][1] == "Obj_Cylinder":
                    Tmp_Coor, Tmp_TP = Gen_Points_Obj_Cylinder(Domain_Param, Points_Param)
                    Obj_Param = ["Cylinder", Domain_Param]
                    CD.Obj_Set.append(Obj_Param)

                # Recap
                WF.PrintLN("Generate " + All_Set[Set_Type][i][1], Tmp_TP)
                WF.PrintMinMax(Tmp_Coor)
                #WF.PrintLN("Min", np.min(Tmp_Coor, axis = 1))
                #WF.PrintLN("Max", np.max(Tmp_Coor, axis = 1))
                All_Loc[Set_Type].append(Tmp_Coor)
                All_Points[Set_Type].append(Tmp_TP)

            # Generate Points (FILE)
            elif All_Set[Set_Type][i][0] == "Gen_File":
                Tmp_Coor, Tmp_Vals, Tmp_TP = Gen_Points_File(All_Set[Set_Type][i][1], All_Set[Set_Type][i][2], All_Set[Set_Type][i][3])
                WF.PrintLN("Generate From File" + All_Set[Set_Type][i][1], Tmp_TP)
                All_Loc[Set_Type].append(Tmp_Coor)
                All_Val[Set_Type].append(Tmp_Vals)
                All_Points[Set_Type].append(Tmp_TP)
            elif All_Set[Set_Type][i][0] == "Gen_File_Sep":
                Tmp_Coor, _, Tmp_TP = Gen_Points_File(All_Set[Set_Type][i][1][0], All_Set[Set_Type][i][2], [-1])
                _, Tmp_Vals, Tmp_TP = Gen_Points_File(All_Set[Set_Type][i][1][1], [-1], All_Set[Set_Type][i][3])
                WF.PrintLN("Generate From File" + All_Set[Set_Type][i][1][0], Tmp_TP)
                WF.PrintLN("Generate From File" + All_Set[Set_Type][i][1][1], Tmp_TP)
                WF.PrintMinMax(Tmp_Coor)
                #WF.PrintLN("Min", np.min(Tmp_Coor, axis = 1))
                #WF.PrintLN("Max", np.max(Tmp_Coor, axis = 1))
                All_Loc[Set_Type].append(Tmp_Coor)
                All_Val[Set_Type].append(Tmp_Vals)
                All_Points[Set_Type].append(Tmp_TP)


            # Exclude Points
            elif All_Set[Set_Type][i][0] == "Exclude" or All_Set[Set_Type][i][0] == "Include":
                # Params
                P_ID      = All_Set[Set_Type][i][2]
                Dom_Param = All_Set[Set_Type][i][3]
                Dir_Param = All_Set[Set_Type][i][0]

                # Data
                Data_Available = True
                if Set_Type == 2:
                    Data_Available = False
                elif len(All_Val[Set_Type]) <= P_ID:
                    Data_Available = False
                if not(Data_Available):
                    Tmp_Ref_Data = np.copy(All_Loc[Set_Type][P_ID])
                else:
                    Tmp_Ref_Data = np.copy(All_Val[Set_Type][P_ID])

                # Exclusion
                if All_Set[Set_Type][i][1] == "Poly_2D":
                    Tmp_Coor, Tmp_Data, Tmp_TP = Ex_Points_Poly_2D(All_Loc[Set_Type][P_ID], Tmp_Ref_Data, Dom_Param, Dir_Param)
                elif All_Set[Set_Type][i][1] == "Cylinder_2D":
                    Tmp_Coor, Tmp_Data, Tmp_TP = Ex_Points_Cylinder_2D(All_Loc[Set_Type][P_ID], Tmp_Ref_Data, Dom_Param)
                elif All_Set[Set_Type][i][1] == "Box_MD":
                    Tmp_Coor, Tmp_Data, Tmp_TP = Ex_Points_Box_MD(All_Loc[Set_Type][P_ID], Tmp_Ref_Data, Dom_Param, Dir_Param)
                elif All_Set[Set_Type][i][1] == "Unif_Sampling":
                    Tmp_Coor, Tmp_Data, Tmp_TP = Ex_Points_Uniform_Sampling(All_Loc[Set_Type][P_ID], Tmp_Ref_Data, Dom_Param, Dir_Param)
                elif All_Set[Set_Type][i][1] == "Slice_Sturctured":
                    Tmp_Coor, Tmp_Data, Tmp_TP = Ex_Points_Slice_Sturctured(All_Loc[Set_Type][P_ID], Tmp_Ref_Data, Dom_Param)

                # Recap
                WF.PrintLN(All_Set[Set_Type][i][0] + " " + All_Set[Set_Type][i][1], Tmp_TP)
                WF.PrintMinMax(Tmp_Coor)
                #WF.PrintLN("Min", np.min(Tmp_Coor, axis = 1))
                #WF.PrintLN("Max", np.max(Tmp_Coor, axis = 1))
                All_Loc[Set_Type][P_ID]    = Tmp_Coor
                All_Points[Set_Type][P_ID] = Tmp_TP
                if Data_Available:
                    All_Val[Set_Type][P_ID] = Tmp_Data

            # Other Special Points Settings
            elif All_Set[Set_Type][i][0] == "SP":
                if All_Set[Set_Type][i][1] == "Translate":
                    P_ID = All_Set[Set_Type][i][2]
                    for j in range(len(All_Loc[Set_Type][P_ID])):
                        All_Loc[Set_Type][P_ID][j] += All_Set[Set_Type][i][3][j]
                elif All_Set[Set_Type][i][1] == "Data_Deviation":
                    P_ID = All_Set[Set_Type][i][2]
                    WF.PrintLN("Deviation Stat")
                    for j in range(len(All_Val[Set_Type][P_ID])):
                        All_Val[Set_Type][P_ID][j] += All_Set[Set_Type][i][3][j]
                        WF.PrintLN(np.min(All_Val[Set_Type][P_ID][j]), np.max(All_Val[Set_Type][P_ID][j]), np.average(All_Val[Set_Type][P_ID][j]))
                elif All_Set[Set_Type][i][1] == "Pressure_Ref":
                    P_ID      = All_Set[Set_Type][i][2]
                    P_Ref_Loc = All_Set[Set_Type][i][3]
                    Tmp_Coor, Tmp_Data, Tmp_TP = Obtain_Pressure_Reference(All_Loc[Set_Type][P_ID], All_Val[Set_Type][P_ID], P_Ref_Loc)
                    All_Loc[Set_Type].append(Tmp_Coor)
                    All_Val[Set_Type].append(Tmp_Vals)
                    All_Points[Set_Type].append(Tmp_TP)
                elif All_Set[Set_Type][i][1] == "Change_Values":
                    P_ID     = All_Set[Set_Type][i][2]
                    Var_Type = All_Set[Set_Type][i][3][0]
                    Var_ID   = All_Set[Set_Type][i][3][1]
                    New_Val  = All_Set[Set_Type][i][3][2]
                    if Var_Type == "x" or Var_Type == "X" or Var_Type == "Coor":
                        All_Loc[Set_Type][P_ID][Var_ID] = New_Val
                    elif Var_Type == "u" or Var_Type == "U" or Var_Type == "Data":
                        All_Val[Set_Type][P_ID][Var_ID] = New_Val      
                elif All_Set[Set_Type][i][1] == "Filter_Discrete_EE":
                    # Filter by Discrete Error Estimate Range
                    F_ID     = All_Set[Set_Type][i][2][0]
                    E_ID     = All_Set[Set_Type][i][2][1]
                    Acceptable_Range = All_Set[Set_Type][i][3]
                    
                    Tmp_Coor, Tmp_Data, Tmp_TP = SP_Filter_Discrete_Error_Estimate(All_Loc[Set_Type][F_ID], All_Val[Set_Type][F_ID], All_Val[Set_Type][E_ID], Acceptable_Range, WF)
                    for j in range(len(Tmp_Data)):
                        All_Loc[Set_Type].append(Tmp_Coor[j])
                        All_Val[Set_Type].append(Tmp_Data[j])
                        All_Points[Set_Type].append(Tmp_TP[j])
                    
            WF.PrintLN("")

    # Add Unsteady axis
    for i in range(len(Scr_UP)):
        if Scr_UP[i][0] == "ALL":
            WF.PrintLN("Adding Unsteady Axis to all coor", Scr_UP[i][1])
            for Set_Type in range(len(All_Loc)):
                for j in range(len(All_Loc[Set_Type])):
                    Ori_Shape = All_Loc[Set_Type][j].shape
                    #print(All_Loc[Set_Type][j].shape)
                    Unsteady_Points = np.ones((Ori_Shape[1])) * Scr_UP[i][1]
                    Temp_1D = np.reshape(All_Loc[Set_Type][j], -1)
                    Temp_1D = np.insert(Temp_1D, 0, Unsteady_Points)
                    All_Loc[Set_Type][j] = np.reshape(Temp_1D, (Ori_Shape[0] + 1, Ori_Shape[1]))
                    #print(All_Loc[Set_Type][j].shape)

    # BC_Gen_Vals
    for i in range(len(All_Loc[0])):
        WF.PrintLN("Calc BC Values", i, All_Set[0][i][1])
        BC_Vals = Case_Info.BC_EQ.Calc_BC(All_Loc[0][i], All_Set[0][i][4])
        #print(np.min(BC_Vals), np.max(BC_Vals))
        All_Val[0].append(np.array(BC_Vals))


    # Recap
    WF.PrintLN("-------------------------------------------------------------------------------")
    for Set_Type in range(len(All_Set)):
        Total_Points = 0
        for i in range(len(All_Points[Set_Type])):
            Total_Points += All_Points[Set_Type][i]
        WF.PrintLN(Set_Names[Set_Type], "Points Generated, Total: \t" + str(Total_Points) + " Points")
    WF.PrintLN("-------------------------------------------------------------------------------\n")

    # Compile & Plot
    Compile_Points(CD, Point_Grouping, All_Loc, All_Val, WF)
    M_Plot_Points(CD, Case_Info, WF)

    # SW
    if CD.Apply_Spatial_Weight:
        WF.PrintLN("Calculating Spatial Weights")
        CD.SW = Case_Info.SW_EQ.Calc_SW(CD.X_C[CD.CP_ID])

    # Report End
    WF.PrintLN("-------------------------------------------------------------------------------")
    WF.StopWrite()


# Compile
def Compile_Points(CD, Point_Grouping, All_Coor, All_Data, WF):
    Total_Group    = len(Point_Grouping)
    CD.Total_Group = Total_Group
    CD.X_C         = [0 for x in range(Total_Group)]
    CD.U_C         = [0 for x in range(Total_Group)]
    Names_Conv     = [["BC", 0], ["DA", 1], ["C", 2]]

    WF.PrintLN("-------------------------------------------------------------------------------")
    WF.PrintLN("2. Compiling Points")
    WF.PrintLN("-------------------------------------------------------------------------------")
    
    for i in range(Total_Group):
        Coor_List   = []
        Result_List = []
        Set_Type    = Get_Set_ID(Point_Grouping[i][0], Names_Conv)
        #print(i, Set_Type, Total_Group)

        if Set_Type < 2:
            for j in range(1, len(Point_Grouping[i])):
                k = Point_Grouping[i][j]
                New_Shape1 = [len(All_Coor[Set_Type][k]), -1, 1]
                New_Shape2 = [len(All_Data[Set_Type][k]), -1, 1]
                #print(k, New_Shape1, New_Shape2)
                New_Coor  = np.float32(np.reshape(All_Coor[Set_Type][k], New_Shape1))
                New_Data  = np.float32(np.reshape(All_Data[Set_Type][k], New_Shape2))
                Coor_List.append(tf.convert_to_tensor(New_Coor))
                Result_List.append(tf.convert_to_tensor(New_Data))
                #print(All_Data[Set_Type][k].shape, New_Shape2)
            CD.X_C[i] = tf.concat(Coor_List, axis=1)
            CD.U_C[i] = tf.concat(Result_List, axis=1)
            WF.PrintLN(Set_Type, CD.X_C[i].shape, CD.U_C[i].shape)
        elif Set_Type == 2:
            CD.CP_ID = i
            for j in range(1, len(Point_Grouping[i])):
                k = Point_Grouping[i][j]
                New_Shape = [len(All_Coor[Set_Type][k]), -1, 1]
                #print(k, New_Shape)
                New_Coor  = np.float32(np.reshape(All_Coor[Set_Type][k], New_Shape))
                Coor_List.append(tf.convert_to_tensor(New_Coor))
            CD.X_C[i] = tf.concat(Coor_List, axis=1)
            WF.PrintLN(Set_Type, CD.X_C[i].shape)
    WF.PrintLN("-------------------------------------------------------------------------------\n")

# Plot
def M_Plot_Points(CD, Case_Info, WF):
    Plot_Status, Plot_Dim, Plot_Var_ID, Plot_Domain, Fig_Size = Case_Info.Class_RP.Load_Plot_Point_Info()
    _, _, _, Point_Grouping, _ = Case_Info.Load_Weights_List()

    if Plot_Status:
        WF.PrintLN("-------------------------------------------------------------------------------")
        WF.PrintLN("3. Plotting Points")
        WF.PrintLN("-------------------------------------------------------------------------------")
        
        for k in range(len(Plot_Dim)):
            if Plot_Dim[k] == 2:
                fig, ax = plt.subplots(figsize=(Fig_Size[k][0], Fig_Size[k][1]))
                V1 = Plot_Var_ID[k][0]
                V2 = Plot_Var_ID[k][1]
                n_TG = CD.Total_Group

                for i in range(n_TG):
                    WF.PrintLN(Point_Grouping[i][0])
                    WF.PrintLN(np.min(CD.X_C[i][V1]), np.max(CD.X_C[i][V1]))
                    WF.PrintLN(np.min(CD.X_C[i][V2]), np.max(CD.X_C[i][V2]))
                    
                    if (Point_Grouping[i][0] == "C"):
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], marker='.', c='r', alpha=0.5)
                    elif (Point_Grouping[i][0] == "DA"):
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], marker='X', c='cyan')
                    elif (Point_Grouping[i][0] == "BC"):
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], marker='X', c='b')

                for i in range(len(CD.Obj_Set)):
                    if (CD.Obj_Set[i][0] == "Cylinder"):
                        WF.PrintLN("Obj Cylinder")
                        x, y, r = CD.Obj_Set[i][1][0], CD.Obj_Set[i][1][1], CD.Obj_Set[i][1][2]
                        ax.add_patch(patches.Circle([x, y], radius=r, color='black', alpha = 0.5))
                    
                plt.xlabel(CD.Input_Names[V1])
                plt.ylabel(CD.Input_Names[V2])
                plt.xlim(Plot_Domain[k][0][0], Plot_Domain[k][0][1])
                plt.ylim(Plot_Domain[k][1][0], Plot_Domain[k][1][1])
                plt.savefig(CD.MF + 'Point_Distribution' + str(k) + '.png', bbox_inches='tight', dpi=300)
                plt.close()
            elif Plot_Dim[k] == 3:
                fig = plt.figure(figsize=(Fig_Size[0], Fig_Size[1])).add_subplot(projection='3d')
                V1 = Plot_Var_ID[0]
                V2 = Plot_Var_ID[1]
                V3 = Plot_Var_ID[2]
                n_TG = CD.Total_Group
                
                for i in range(n_TG):
                    if (Point_Grouping[i][0] == "C"):
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], CD.X_C[i][V3], marker='.', c='r', alpha=0.5)
                    if (Point_Grouping[i][0] == "DA"):
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], CD.X_C[i][V3], marker='X', c='cyan')
                    else:
                        plt.scatter(CD.X_C[i][V1], CD.X_C[i][V2], CD.X_C[i][V3], marker='X', c='b')

                fig.set_xlabel(CD.Input_Names[V1])
                fig.set_ylabel(CD.Input_Names[V2])
                fig.set_zlabel(CD.Input_Names[V3])
                plt.xlim(Plot_Domain[k][0][0], Plot_Domain[k][0][1])
                plt.ylim(Plot_Domain[k][1][0], Plot_Domain[k][1][1])
                plt.zlim(Plot_Domain[k][2][0], Plot_Domain[k][2][1])
                plt.savefig(CD.MF + 'Point_Distribution' + str(k) + '.png', bbox_inches='tight', dpi=300)
                plt.close()



# Support
def Get_Set_ID(Cur_Label, Names_Conv):
    for i in range(len(Names_Conv)):
        if Names_Conv[i][0] == Cur_Label:
            return Names_Conv[i][1]

class Write_To_File():
    def __init__(self, FN):
        self.File_Holder = open(FN, "w+")
        
    def PrintLN(self, *args):
        if isinstance(args, str):
            self.File_Holder.writelines(args + "\n")
        elif isinstance(args, float):
            self.File_Holder.writelines(args + "\n")
        elif isinstance(args, int):
            self.File_Holder.writelines(args + "\n")
        elif isinstance(args, list) or isinstance(args, np.ndarray) or isinstance(args, tuple):
            str_tot = str(args[0])
            for i in range(1, len(args)):
                str_tot += "\t" + str(args[i])
            self.File_Holder.writelines(str_tot + "\n")
            
    def PrintMinMax(self, Data):
        self.PrintLN("Min", np.min(Data, axis = 1))
        self.PrintLN("Max", np.max(Data, axis = 1))
        
    def StopWrite(self):
        self.File_Holder.close()
