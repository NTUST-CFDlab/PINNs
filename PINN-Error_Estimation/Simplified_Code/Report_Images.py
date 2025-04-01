
# Import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from Points_Generation import Gen_Points_Unif_Box

class Report_Template():
    # Init
    def Generate_Points(self):
        Coor_List, _ = Gen_Points_Unif_Box(self.D_Size, self.D_Sampling)
        New_Shape1   = [len(Coor_List), -1, 1]
        Coor_List    = np.float32(np.reshape(Coor_List, New_Shape1))
        self.TF_Calc = tf.convert_to_tensor(Coor_List)

    def Read_Ref_Data(self, Ref_File, Ref_Type):
        # Ref_File Format Axis1, Axis2, .. AxisN, Value1, Value2, ... ValueN
        # Ref_Type = Combined or Separated
        if Ref_Type == "Combined":
            self.Ref_Data = np.float32(np.loadtxt(Ref_File, unpack=True))
        elif Ref_Type == "Separated":
            Total_Files   = len(Ref_File)
            self.Ref_Data = [[] for i in range(Total_Files)]
            for i in range(Total_Files):
                self.Ref_Data[i] = np.float32(np.loadtxt(Ref_File[i], unpack=True))

    # Update
    def Print_TF_Data(self, Out_File, Out_ID):
        Out_Data = np.concatenate((np.array(self.TF_Plot),
                                   np.array(self.TF_Result_1D[Out_ID[0]][Out_ID[1]])))
        np.savetxt(Out_File, np.transpose(Out_Data), delimiter='\t')

    def Print_Image_Statistics(self, IMG_ID, Out_ID):
        File_Holder = open(self.CD.MF + "Image_Statistics.txt", "a+")

        str_total = IMG_ID[0]
        Stat_Vals = Calc_MinMaxAvg(self.TF_Result_1D[Out_ID[0]][Out_ID[1]])

        for j in range(1, len(IMG_ID)):
            str_total += "\t" + str(IMG_ID[j])
        for j in range(len(Stat_Vals)):
            str_total += "\t" + str(Stat_Vals[j])

        File_Holder.writelines(str_total + "\n")
        File_Holder.close()

    def Update_All(self, Mode, Str_Iter):
        # Calculation
        self.TF_Result_1D = self.GE.Get_Sim_Param(self.TF_Calc, self.NN_Out_Vars)
        
        # Report
        for i in range(len(self.Update_Scr)):
            for j in range(len(self.Out_TF_Index)):
                Std_Name = self.CD.MF + self.Out_Folder[j] + self.Out_Name_H[j] + "-" + Str_Iter
                if self.Update_Scr[i] == "Print_Img":
                    Out_Name = Std_Name + ".png"
                    self.Print_Image(Out_Name, self.Out_TF_Index[j])
                elif self.Update_Scr[i] == "Img_Stat":
                    self.Print_Image_Statistics([Mode, self.Out_Name_H[j], Str_Iter], self.Out_TF_Index[j])
                elif self.Update_Scr[i] == "Write_Img":
                    Out_Name = Std_Name + ".txt"
                    self.Print_TF_Data(Out_Name, self.Out_TF_Index[j])


class Report_xyu_PColorMesh(Report_Template):
    def __init__(self, CD, Img_ID, All_Settings):
        print("Initializing Image ", Img_ID)
        self.CD = CD
        self.GE = CD.GE
        self.Img_ID = Img_ID
        self.Conversion_Layer(All_Settings)
        

        for i in range(len(self.Init_Scr)):
            if self.Init_Scr[i] == "Gen_Points":
                self.Generate_Points()
            elif self.Init_Scr[i] == "Read_Ref":
                self.Read_Ref_Data()    # Setting is not determined

    def Conversion_Layer(self, All_Settings):
        # [Total_Domain, Print_ID, Plot_MinMax, Domain_Info, Sampling, Size] = Case_Info.Class_RP.Load_Image_Setting("Report")
        # All_Settings = Setting_Scr("Report")

        Total_D    = All_Settings[0]
        self.Img_MinMax = All_Settings[2]
        self.Img_Size   = All_Settings[5][self.Img_ID]

        # Vars
        if len(All_Settings[1]) == 2:
            self.NN_Out_Vars  = ["M", "R"]
            self.Out_TF_Index = []
            self.Out_Folder   = []
            self.Out_Name_H   = []
            for i in range(len(All_Settings[1][0])):
                self.Out_TF_Index.append([0, All_Settings[1][0][i]])
                self.Out_Folder.append("Data/")
                self.Out_Name_H.append("M_" + self.CD.Output_Names[i] + str(self.Img_ID))
            for i in range(len(All_Settings[1][1])):
                self.Out_TF_Index.append([1, All_Settings[1][1][i]])
                self.Out_Folder.append("Residual/")
                self.Out_Name_H.append("R_" + self.CD.Residual_Names[i] + str(self.Img_ID))

        # Domain
        self.D_Main_Axis_ID = All_Settings[3][0][self.Img_ID]
        D_Axis1_Val_Range   = All_Settings[3][1][self.Img_ID]
        D_Axis2_Val_Range   = All_Settings[3][2][self.Img_ID]
        D_Other_Axis_Vals   = All_Settings[3][3][self.Img_ID]
        self.D_Mesh_Shape   = np.array(All_Settings[4][self.Img_ID], dtype = int) + 1

        Total_Axis        = len(self.D_Main_Axis_ID) + len(D_Other_Axis_Vals)
        self.D_Sampling   = np.ones(Total_Axis, dtype = int)
        self.D_Size       = np.zeros((Total_Axis, 2))
        Set_Axis_Check    = [False for i in range(Total_Axis)]
        D_MA_Val_Range    = [D_Axis1_Val_Range, D_Axis2_Val_Range]

        for i in range(len(self.D_Main_Axis_ID)):
            j = self.D_Main_Axis_ID[i]
            self.D_Size[j,:]     = np.array(D_MA_Val_Range[i])
            self.D_Sampling[j]   = self.D_Mesh_Shape[i] - 1
            Set_Axis_Check[j]    = True
        j = 0
        for i in range(Total_Axis):
            if not(Set_Axis_Check[i]):
                self.D_Size[i,:]   = np.ones(2) * D_Other_Axis_Vals[j]
                self.D_Sampling[i] = 0
                j += 1
        
        # FIX
        self.D_Mesh_Shape   = np.flip(self.D_Mesh_Shape)
        
        # Scr
        self.Init_Scr   = ["Gen_Points"]            # Gen_Points, Read_Ref
        self.Update_Scr = ["Print_Img", "Img_Stat"] # Print_Img, Write_Img, Img_Stat

    def Print_Image(self, Out_Name, Out_ID):
        fig, ax = plt.subplots(figsize=(self.Img_Size[0], self.Img_Size[1]))

        # Convert to 2D
        x_1D = np.array(self.TF_Calc[self.D_Main_Axis_ID[0]])
        y_1D = np.array(self.TF_Calc[self.D_Main_Axis_ID[1]])
        u_1D = np.array(self.TF_Result_1D[Out_ID[0]][Out_ID[1]])
        x_2D = np.reshape(x_1D, self.D_Mesh_Shape)
        y_2D = np.reshape(y_1D, self.D_Mesh_Shape)
        u_2D = np.reshape(u_1D, self.D_Mesh_Shape)
        #print(Out_Name, x_2D)
        #print(Out_Name, y_2D)

        # Plot
        vmin, vmax = Update_MinMax(self.Img_MinMax[Out_ID[0]][Out_ID[1]], u_1D)
        plt.pcolormesh(x_2D, y_2D, u_2D, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
        for i in range(len(self.CD.Obj_Set)):
            if (self.CD.Obj_Set[i][0] == "Cylinder"):
                x, y, r = self.CD.Obj_Set[i][1][0], self.CD.Obj_Set[i][1][1], self.CD.Obj_Set[i][1][2]
                ax.add_patch(patches.Circle([x, y], radius=r, color='black', alpha=0.5))

        # Label
        font1 = {'family': 'serif', 'size': 20}
        plt.xlabel(self.CD.Input_Names[0], fontdict=font1)
        plt.ylabel(self.CD.Input_Names[1], fontdict=font1)
        plt.tick_params(axis='both', which='major', labelsize=15)

        cbar = plt.colorbar(pad=0.05, aspect=10)
        #cbar.set_label(Axis_Name, fontdict=font1)
        cbar.mappable.set_clim(vmin, vmax)
        cbar.ax.tick_params(labelsize=15)

        # Save
        # plt.axis('equal')
        plt.savefig(Out_Name, bbox_inches='tight', dpi=300);
        plt.close()
        
        
        
        
        
        
        
"""        
class Report_Data():
    def Initialize_Values(self, CD, Domain_Info, Plot_MinMax, Sampling, Size):
        self.CD = CD
        self.GE = CD.GE
        self.Size_x = Size[0]
        self.Size_y = Size[1]
        self.Output_MinMax = Plot_MinMax[0]
        self.Residual_MinMax = Plot_MinMax[1]

        Domain_ID = Domain_Info[0]
        Domain_x = Domain_Info[1]
        Domain_y = Domain_Info[2]
        Domain_Others = Domain_Info[3]
        Total_Points = (Sampling[0]+1) * (Sampling[1]+1)

        # Generate Points
        xspace = tf.linspace(Domain_x[0], Domain_x[1], Sampling[0] + 1)
        yspace = tf.linspace(Domain_y[0], Domain_y[1], Sampling[1] + 1)
        self.X, self.Y = tf.meshgrid(xspace, yspace)
        self.TF_x = tf.reshape(self.X, [Total_Points, 1])
        self.TF_y = tf.reshape(self.Y, [Total_Points, 1])

        # Recap
        self.TF_All = [0 for i in range(CD.n_Input)]
        Check_Dim_ID = [True for i in range(CD.n_Input)]
        self.TF_All[Domain_ID[0]] = self.TF_x
        self.TF_All[Domain_ID[1]] = self.TF_y
        Check_Dim_ID[Domain_ID[0]] = False
        Check_Dim_ID[Domain_ID[1]] = False
        j = 0
        for i in range(CD.n_Input):
            if Check_Dim_ID[i]:
                self.TF_All[i] = self.TF_y * 0. + Domain_Others[j]
                j += 1

    # ----------------------------------------------------------------------------------
    # Calculation (Might need to be adjusted)
    # ----------------------------------------------------------------------------------
    def Calc_Sim_Param(self):
        [Coor, self.Main_Var_NN, Derivative1, Derivative2, Temp_Residuals] = self.GE.Get_Sim_Param(self.TF_All, "A")
        self.Residuals = []
        for i in range(self.CD.n_Eq):
            self.Residuals.append(Temp_Residuals[i].numpy().reshape(self.X.shape))


    # ----------------------------------------------------------------------------------
    # Printing (Save_Result_Image might need some adjustment)
    # ----------------------------------------------------------------------------------
    
    def Print_Result_Data(self, iter, LS_Value, Initial_Time, ID, Print_Res):
        File_Holder = open(self.CD.MF + "Progress.txt", "a+")

        #   ID: (B) Backup  # (L) Level    # (F) Final
        str_total = ID + "\t" + str(iter) + "\t" + str(LS_Value) + "\t" + str(time() - Initial_Time)
        if Print_Res:
            for i in range(self.CD.n_Eq):
                [Min_R, Max_R, Avg_R] = Calc_MinMaxAvg(self.Residuals[i], 2)
                str_total = str_total + "\t" + str(Min_R) + "\t" + str(Max_R) + "\t" + str(Avg_R)

        File_Holder.writelines(str_total + "\n")
        File_Holder.close()
    

    def Print_Result_Image(self, File_ID, Print_Status, Print_ID):
        # Default : Print Main Var & Residual
        # self.Save_Vector_Image([self.TF_x, self.TF_y], [self.Main_Var_NN[0], self.Main_Var_NN[1]], "Data", 'Vector Field-'+File_ID)
        if Print_Status[0]:
            for i in range (len(Print_ID[0])):
                ID = Print_ID[0][i]
                File_Name = self.CD.Output_Names[ID]+"_NN-" + File_ID
                self.Save_Result_Image(self.Main_Var_NN[ID].numpy().reshape(self.X.shape), self.X, self.Y, "Data", File_Name, self.CD.Output_Names[ID], self.Output_MinMax[i])

        if Print_Status[1]:
            for i in range (len(Print_ID[1])):
                ID = Print_ID[1][i]
                File_Name = self.CD.Residual_Names[ID]+"-" + File_ID
                self.Save_Result_Image(self.Residuals[ID], self.X, self.Y, "Residual", File_Name, self.CD.Residual_Names[ID], self.Residual_MinMax[i])


    def Save_Result_Image(self, Main_Data, X, Y, Folder, File_Name, Axis_Name, MinMax_Info = []):
        # Default Axis Follow Case_Details.Input_Names
        # fig = plt.figure(figsize=(self.Size_x, self.Size_y))
        fig, ax = plt.subplots(figsize=(self.Size_x, self.Size_y))
        
        if len(MinMax_Info) == 0:
            vmin, vmax = np.min(np.min(Main_Data)), np.max(np.max(Main_Data))       # <-------- Modify if need costum scale
        else:
            vmin = MinMax_Info[0]
            vmax = MinMax_Info[1]
        
        plt.pcolormesh(X, Y, Main_Data, cmap='rainbow', norm=Normalize(vmin=vmin, vmax=vmax))
        font1 = {'family': 'serif', 'size': 20}

        plt.title(File_Name, fontdict=font1)
        plt.xlabel(self.CD.Input_Names[0], fontdict=font1)          # <-------- Could be modified
        plt.ylabel(self.CD.Input_Names[1], fontdict=font1)          # <-------- Could be modified
        plt.tick_params(axis='both', which='major', labelsize=15)

        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label(Axis_Name, fontdict=font1)
        cbar.mappable.set_clim(vmin, vmax)
        cbar.ax.tick_params(labelsize=15)

        for i in range(len(self.CD.Obj_Set)):
            if (self.CD.Obj_Set[i][0] == "Cylinder"):
                x, y, r = self.CD.Obj_Set[i][1][0], self.CD.Obj_Set[i][1][1], self.CD.Obj_Set[i][1][2]
                ax.add_patch(patches.Circle([x, y], radius=r, color='black', alpha = 0.5))
            #plt.scatter(self.CD.Cylinder_Coor[i][0], self.CD.Cylinder_Coor[i][1], marker='X', alpha = 0.05)
        #ax.add_patch(patches.Circle([22.5, 25.], radius=0.5, color='black', alpha = 0.5))
        #ax.add_patch(patches.Circle([22.5 + 2.165, 25. + 1.25], radius=0.5, color='black', alpha = 0.5))
        #ax.add_patch(patches.Circle([22.5 + 2.165, 25. - 1.25], radius=0.5, color='black', alpha = 0.5))
        plt.axis('equal')
        #plt.xlim([11.8, 12.6])
        #plt.ylim([15.4, 15.7])
        
        plt.savefig(self.CD.MF + Folder + "/" + File_Name + ".png", bbox_inches='tight', dpi=300);
        plt.close()

    def Save_Vector_Image(self, Coor, Vector_Data, Folder, File_Name):
        # Default Axis Follow Case_Details.Input_Names
        fig, ax = plt.subplots(figsize=(self.Size_x, self.Size_y))
        Magnitude = tf.sqrt(tf.square(Vector_Data[0]) + tf.square(Vector_Data[1]))
        colors = Magnitude.numpy()
        norm = Normalize()
        norm.autoscale(colors)
        colormap = plt.cm.rainbow
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])

        plt.quiver(Coor[0], Coor[1], Vector_Data[0], Vector_Data[1], color=colormap(norm(colors)))
        #plt.quiver(X_Coor, Y_Coor, u_Data, v_Data, color=colormap(norm(colors)))
        plt.colorbar(sm, pad=0.05, aspect=10)
        plt.savefig(self.CD.MF + Folder + "/" + File_Name + ".png", bbox_inches='tight', dpi=300);
        plt.close()

class Vel_Profile():
    # This is only to plot @ x,y,z = 0
    def Initialize_Values(self, CD, Case_Info):
        self.CD = CD
        self.GE = CD.GE

        Line_Details, FileNames, self.Image_Details = Case_Info.Class_RP.Load_Vel_Profile_Image_Setting()
        self.L_Samples = Line_Details[0]
        L_Coor = Line_Details[1]
        self.Plot_Names = Line_Details[2]
        self.Plotted_Var_ID = Line_Details[3]

        self.n_Files = len(FileNames)
        self.n_Lines = len(L_Coor)
        self.L_Dim = len(L_Coor[0][0])

        # Coor
        self.TF_Line = []
        for i in range(self.n_Lines):
            Interpolated_Coor = tf.linspace(L_Coor[i][0], L_Coor[i][1], self.L_Samples + 1, axis = 1)
            self.TF_Line.append(tf.reshape(Interpolated_Coor, [self.L_Dim, self.L_Samples + 1, 1]))

        # Reference Data
        self.Ref_Data = []
        for i in range(self.n_Files):
            Data_Per_Line = [0 for j in range(self.n_Lines)]
            for j in range(self.n_Lines):
                Data_Per_Line[j] = np.float32(np.loadtxt(FileNames[i] + "_" + self.Plot_Names[j] +".txt", unpack=True))
            self.Ref_Data.append(Data_Per_Line)

    def Calc_Sim_Param(self):
        self.Result_All = []
        for i in range(self.n_Lines):
            [Temp_Result] = self.GE.Get_Sim_Param(self.TF_Line[i], "M")
            self.Result_All.append(Temp_Result)

    def Print_Result_Data(self, File_ID):
        for i in range(self.n_Lines):
            File_Name = self.CD.MF + "Data/Vel_Profile_" + self.Plot_Names[i]+ "-" + File_ID +".txt"
            File_Holder = open(File_Name, "w+")
            Coor_np = np.asarray(self.TF_Line[i])
            Result_np = np.asarray(self.Result_All[i])
            for j in range(self.L_Samples + 1):
                str_total = str(Coor_np[0][j][0])
                for k in range(1, self.L_Dim):
                    str_total = str_total + "\t" + str(Coor_np[k][j][0])
                str_total = str_total + "\t" + str(Result_np[self.Plotted_Var_ID[i]][j][0])
                File_Holder.writelines(str_total + "\n")
            File_Holder.close()

    def Print_Result_Image(self, File_ID):
        Image_Size = self.Image_Details[0]
        Image_Direction_Axis = self.Image_Details[1]
        Axis_Coor = self.Image_Details[2]
        Image_Legend = self.Image_Details[3]
        Print_Types = self.Image_Details[4]
        Legend_Location = self.Image_Details[5]

        for i in range(self.n_Lines):
            File_Name = self.CD.MF + "Data/Vel_Profile_" + self.Plot_Names[i]+ "-" + File_ID +".png"
            fig = plt.figure(figsize=(Image_Size[0], Image_Size[1]))

            Coor_ID = Axis_Coor[i]
            PINN_Result_ID = self.Plotted_Var_ID[i]
            Data_Result_ID = self.Plotted_Var_ID[i] + self.L_Dim
            Color_List_Dot = ['r', 'black', 'm', 'g']
            Dot_ID = 0

            if Image_Direction_Axis[i] == "x":
                plt.plot(self.TF_Line[i][Coor_ID], self.Result_All[i][PINN_Result_ID].numpy())
                for j in range(self.n_Files):
                    if Print_Types[j+1] == "Line":
                        plt.plot(self.Ref_Data[j][i][Coor_ID], self.Ref_Data[j][i][Data_Result_ID])
                    elif Print_Types[j+1] == "Dot":
                        plt.scatter(self.Ref_Data[j][i][Coor_ID], self.Ref_Data[j][i][Data_Result_ID], marker='o', color = Color_List_Dot[Dot_ID])
                        Dot_ID += 1
                plt.xlabel(self.CD.Input_Names[Coor_ID])
                plt.ylabel(self.CD.Output_Names[PINN_Result_ID])
                plt.legend(Image_Legend, loc=Legend_Location[i])

            elif Image_Direction_Axis[i] == "y":
                plt.plot(self.Result_All[i][PINN_Result_ID].numpy(), self.TF_Line[i][Coor_ID])
                for j in range(self.n_Files):
                    if Print_Types[j+1] == "Line":
                        plt.plot(self.Ref_Data[j][i][Data_Result_ID], self.Ref_Data[j][i][Coor_ID])
                    elif Print_Types[j+1] == "Dot":
                        plt.scatter(self.Ref_Data[j][i][Data_Result_ID], self.Ref_Data[j][i][Coor_ID], marker='o', color = Color_List_Dot[Dot_ID])
                        Dot_ID += 1
                plt.xlabel(self.CD.Output_Names[PINN_Result_ID])
                plt.ylabel(self.CD.Input_Names[Coor_ID])
                plt.legend(Image_Legend, loc=Legend_Location[i])

            plt.savefig(File_Name, bbox_inches='tight', dpi=300)
            plt.close()  
"""           

########################################
# Support
########################################
def Update_MinMax(Img_MinMax, Main_Data):
    if len(Img_MinMax) == 0:
        vmin, vmax = np.min(Main_Data), np.max(Main_Data)
    else:
        vmin, vmax = Img_MinMax[0], Img_MinMax[1]
    return vmin, vmax
    
def Calc_MinMaxAvg(Main_Data):
    Min_Value = np.asarray(tf.reduce_min(Main_Data))
    Max_Value = np.asarray(tf.reduce_max(Main_Data))
    Avg_Value = np.asarray(tf.reduce_mean(Main_Data))
    Abs_Avg   = np.average(np.abs(np.asarray(Main_Data)))

    return [Min_Value, Max_Value, Avg_Value, Abs_Avg]

