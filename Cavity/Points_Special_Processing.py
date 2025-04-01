
# Import
import numpy as np


# Data
def D3_Read_DA_Ref_File(Case_Info):
    #DA_Files, Filter_by_Domain = Case_Info.Load_Self_Defined_Point_Settings("DA")
    #DA_Files = Case_Info.Load_Self_Defined_Point_Settings("DA")
    DA_Files, DA_Format, Max_Format, AP_Status, AP_Type, AP_Value = Case_Info.Load_Data_Info()
    #AddP_Status, AddP_Type, AddP_Value = Case_Info.Load_Additional_Data_Processing()
    [Total_Domain, LB, UB] = Case_Info.Load_Domain_Size()
    Total_Files = len(DA_Files)
    Case_Dim = len(Total_Domain)
    Total_Points = 0

    DA_Coor = []
    DA_Var_Values = []
    for File_ID in range(Total_Files):
        File_Data = np.float32(np.loadtxt(DA_Files[File_ID], unpack=True))
        if AP_Status:
            for i in range(len(AP_Type)):
                if AP_Type[i] == "Filter_by_Domain2D":
                    File_Data = Data_Filter_by_Domain2D(File_Data, AP_Value[i])
                elif AP_Type[i] == "Duplicate_Translate":
                    File_Data = Data_Duplicate_Translation(File_Data, AP_Value[i])
                elif AP_Type[i] == "Add_Noise":
                    File_Data = Data_Value_RandNoise(File_Data, Case_Dim, AP_Value[i])
                elif AP_Type[i] == "Filter_by_ID":
                    File_Data = Data_Filter_by_ID(File_Data, AP_Value[i][0], AP_Value[i][1])

        File_Data = np.float32(File_Data)
        Total_Readed_Var = len(File_Data)
        Total_Points2 = len(File_Data[0])

        Coor_List = [[] for i in range(Max_Format[0])]
        Var_List = [[] for i in range(Max_Format[1])]
        for i in range(len(DA_Format[File_ID][0])):
            Temp_ID = DA_Format[File_ID][0][i]
            Coor_List[Temp_ID] = tf.reshape(tf.convert_to_tensor(File_Data[i]), [Total_Points2, 1])
        for i in range(len(DA_Format[File_ID][1])):
            Temp_ID = DA_Format[File_ID][1][i]
            Var_List[Temp_ID] = tf.reshape(tf.convert_to_tensor(File_Data[i + Max_Format[0]]), [Total_Points2, 1])
        #for i in range(Case_Dim):
        #    Coor_List.append(tf.reshape(tf.convert_to_tensor(File_Data[i]), [Total_Points2, 1]))
        #    Var_List.append(tf.reshape(tf.convert_to_tensor(File_Data[i+Case_Dim]), [Total_Points2, 1]))

        #if Total_Readed_Var > 2*Case_Dim:   # Pressure
        #    Var_List.append(tf.reshape(tf.convert_to_tensor(File_Data[2*Case_Dim]), [Total_Points2, 1]))

        Total_Points += Total_Points2
        DA_Coor.append(Coor_List)
        DA_Var_Values.append(Var_List)

    return DA_Coor, DA_Var_Values, Total_Points

def D3_Filter_DA_by_Domain(File_Data, Total_Domain):
    n_Input = len(Total_Domain)
    n_Var = len(File_Data)
    Filtered_Data = [[] for i in range(n_Var)]

    for j in range(len(File_Data[0])):
        Inside_Domain = True
        for i in range(n_Input):
            if File_Data[i][j] < Total_Domain[i][0] or File_Data[i][j] > Total_Domain[i][1]:
                Inside_Domain = False
        if Inside_Domain:
            for i in range(n_Var):
                Filtered_Data[i].append(File_Data[i][j])

    return Filtered_Data


def Obtain_Pressure_Reference(Coor_List, Val_List, Ref_Coor):
    # Get nearest point
    Distance   = 0.
    for i in range(len(Coor_List)):
        Distance += np.square(Coor_List[i] - Ref_Coor[i])
    ID = np.argmin(Distance)
    
    # Extract data
    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    for j in range(len(Coor_List)):
        Filtered_Coor[j].append(Coor_List[j][ID])
    for j in range(len(Val_List)):
        Filtered_Data[j].append(Val_List[j][ID])
    
    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])

def SP_Filter_Discrete_Error_Estimate(X, f0, EE, Range, WF):
    All_X, All_f0, All_TP  = [], [], []
    
    WF.PrintLN("Total Vars:\t", len(f0))
    WF.PrintLN("Initial Data:\t", len(f0[0]))
    
    for i in range(len(f0)):		# N_Var
        # Create Filter
        Bool_Filter = [False for i in range(len(f0[i]))]
        for j in range(len(f0[i])):	# N_Data
            if EE[i,j] >= Range[0] and EE[i,j] <= Range[1]:
                Bool_Filter[j] = True
                
        # Apply Filter
        Temp_Ref_X = []
        for j in range(len(X)):
            Temp_Ref_X.append(X[j][Bool_Filter])
        Temp_Eval  = f0[i][Bool_Filter]
        All_X.append(np.array(Temp_Ref_X))
        All_f0.append(Temp_Eval)
        All_TP.append(len(Temp_Eval))
        
        # Report
        WF.PrintLN("Var ", i, len(Temp_Eval))
        
    return All_X, All_f0, All_TP
