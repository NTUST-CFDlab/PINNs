##########################################
# Version 1.1	Added the unifrom sampling function
##########################################

# Import
import numpy as np


# Exclusion
def Ex_Points_Poly_2D(Coor_List, Val_List, Poly_Points, Exclusion_Dir):
    # Coor_List     = [Points_x, Points_y]
    # Poly_Points   = [(x1, y1), (x2, y2), ...]
    # Exclusion_Dir = ["x-", "x+", "y-", "y+"]

    # Gen Line Params
    Line_Params = []    # ("x="/"y="/"d", m,c)
    for i in range(len(Poly_Points-1)):
        Line_Params.append(SS_Gen_Line_Equation(Poly_Points[i], Poly_Points[i + 1]))
    Line_Params.append(SS_Gen_Line_Equation(Poly_Points[len(Poly_Points-1)], Poly_Points[0]))

    # Search Range
    Min_Poly_Points = np.min(np.array(Poly_Points), axis=1)
    Max_Poly_Points = np.min(np.array(Poly_Points), axis=1)

    # Exclude
    Filtered_Coor = [[] for i in range(len(Coor_List[0]))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    for i in range(len(Coor_List[0])):
        Safe_From_Exclusion = True
        In_MinMax_Range = True
        for j in range(len(Coor_List)):
            if Coor_List[j, i] < Min_Poly_Points[j] and Coor_List[j, i] > Max_Poly_Points[j]:
                In_MinMax_Range = False

        if In_MinMax_Range:
            Safe_From_Exclusion = False
            for j in range(len(Line_Params)):
                if Line_Params[j][0] == "x=":
                    if Exclusion_Dir == "x+":
                        if Coor_List[0][i] < Line_Params[j][2]:
                            Safe_From_Exclusion = True
                    elif Exclusion_Dir == "x-":
                        if Coor_List[0][i] > Line_Params[j][2]:
                            Safe_From_Exclusion = True
                elif Line_Params[j][0] == "y=":
                    if Exclusion_Dir == "y+":
                        if Coor_List[1][i] < Line_Params[j][2]:
                            Safe_From_Exclusion = True
                    elif Exclusion_Dir == "y-":
                        if Coor_List[1][i] > Line_Params[j][2]:
                            Safe_From_Exclusion = True
                elif Line_Params[j][0] == "d":
                    x_Val = (Coor_List[0, i] - Line_Params[j][2])/Line_Params[j][1]
                    y_Val = Line_Params[j][1] * Coor_List[0, i] + Line_Params[j][2]
                    if Exclusion_Dir == "y+":
                        if Coor_List[1][i] < y_Val:
                            Safe_From_Exclusion = True
                    elif Exclusion_Dir == "y-":
                        if Coor_List[1][i] > y_Val:
                            Safe_From_Exclusion = True
                    elif Exclusion_Dir == "x+":
                        if Coor_List[0][i] < x_Val:
                            Safe_From_Exclusion = True
                    elif Exclusion_Dir == "x-":
                        if Coor_List[0][i] > x_Val:
                            Safe_From_Exclusion = True

        if Safe_From_Exclusion:
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j, i])
                Filtered_Data[j].append(Val_List[j][i])

    return np.array(Filtered_Coor_List), np.array(Filtered_Data), len(Filtered_Coor_List[0])

def Ex_Points_Cylinder_2D(Coor_List, Val_List, Cylinder_Params):
    Rad2 = (np.square(Coor_List[0] - Cylinder_Params[0]) +
            np.square(Coor_List[1] - Cylinder_Params[1]))
    Rad_Ref = np.square(Cylinder_Params[2])

    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    for i in range(len(Coor_List[0])):
        if Rad2[i] > Rad_Ref:
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j][i])
            for j in range(len(Val_List)):
                Filtered_Data[j].append(Val_List[j][i])

    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])

def Ex_Points_Box_MD(Coor_List, Val_List, Domain_Params, Exclusion_Dir):
    Total_Dim = len(Domain_Params)
    #print(Domain_Params)
    #print(np.min(Coor_List, axis = 1))
    #print(np.max(Coor_List, axis = 1))

    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    for i in range(len(Coor_List[0])):
        In_Range = True
        for j in range(Total_Dim):
            if Coor_List[j][i] < Domain_Params[j][0]:
                In_Range = False
            elif Coor_List[j][i] > Domain_Params[j][1]:
                In_Range = False

        if not(In_Range) and Exclusion_Dir == "Exclude":
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j][i])
            for j in range(len(Val_List)):
                Filtered_Data[j].append(Val_List[j][i])
        if In_Range and Exclusion_Dir == "Include":
            for j in range(len(Coor_List)):
                Filtered_Coor[j].append(Coor_List[j][i])
            for j in range(len(Val_List)):
                Filtered_Data[j].append(Val_List[j][i])

    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])

def Ex_Points_Uniform_Sampling(Coor_List, Val_List, Sampling_Freq, Exclusion_Dir):
    # Coor_List     = [Points_x, Points_y]
    # Sampling_Freq = int (e.g. 3 or 5 or 8)
    
    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    Total_Points  = len(Coor_List[0])
    
    # Mask
    if Exclusion_Dir == "Exclude":
        Filter_Array = np.ones(Total_Points, dtype=bool)
        Filter_Array[Sampling_Freq-1::Sampling_Freq] = False
	
    elif Exclusion_Dir == "Include":
        Filter_Array = np.zeros(Total_Points, dtype=bool)
        Filter_Array[Sampling_Freq-1::Sampling_Freq] = True
	
    # Filter
    for i in range(len(Coor_List)):
        Filtered_Coor[i] = Coor_List[i][Filter_Array]
    for i in range(len(Val_List)):
        Filtered_Data[i] = Val_List[i][Filter_Array]
    
    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])

def Ex_Points_Slice_Sturctured(Coor_List, Val_List, Axis_Param):
    # Coor_List     = [Points_x, Points_y, ...]
    # Coor_List     = [Val_u, Val_v, ...]
    # INCLUDE ONLY
    
    Filtered_Coor = [[] for i in range(len(Coor_List))]
    Filtered_Data = [[] for i in range(len(Val_List))]
    Total_Points  = len(Coor_List[0])
    
    Axis_ID  = Axis_Param[0]
    Axis_Val = Axis_Param[1]
    
    # Reshape
    Total_Dim     = len(Coor_List)
    Total_Var     = len(Val_List)
    Ref_Mesh_Comp = []
    Ref_Mesh_Size = []
    for i in range(Total_Dim):
        Ref_Mesh_Comp.append(np.unique(Coor_List[i]))
        Ref_Mesh_Size.append(len(Ref_Mesh_Comp))
    
    Tar_ID      = np.argmin(np.abs(Ref_Mesh_Comp[Axis_ID] - Axis_Val))    
    Temp_Size   = np.insert(Ref_Mesh_Size, 0, Total_Dim)
    Ref_Coor_ND = np.reshape(Coor_List, Temp_Size)
    Temp_Size   = np.insert(Ref_Mesh_Size, 0, Total_Var)
    Ref_Data_ND = np.reshape(Val_List, Temp_Size)
    
    
    # SLICE
    for i in range(Axis_ID):
        Ref_Coor_ND = np.moveaxis(Ref_Coor_ND, 1, -1)
        Ref_Data_ND = np.moveaxis(Ref_Data_ND, 1, -1)
    New_Coor = Ref_Coor_ND[:, Tar_ID]
    New_Data = Ref_Data_ND[:, Tar_ID]
    for i in range(Axis_ID, Total_Dim):
        New_Coor = np.moveaxis(New_Coor, 1, -1)
        New_Data = np.moveaxis(New_Data, 1, -1)
        
    
    # Filter
    for i in range(len(Coor_List)):
        Filtered_Coor[i] = np.reshape(New_Coor[i], (-1))
    for i in range(len(Val_List)):
        Filtered_Data[i] = np.reshape(New_Data[i], (-1))
    
    return np.array(Filtered_Coor), np.array(Filtered_Data), len(Filtered_Coor[0])
    
    
    
    
    

# Support (Very simple functions)
def SS_Gen_Line_Equation(Points1, Points2):
    m = 0
    if Points1[0][0] == Points2[1][0]:
        Type = "x="
        c    = Points1[0][0]
    elif Points1[0][1] == Points2[1][1]:
        Type = "y="
        c    = Points1[0][1]
    else:
        Type = "d"
        m    = (Points2[1][1] - Points2[0][1]) / (Points2[1][0] - Points2[0][0])
        c    = Points2[1][1] - m * Points2[0][1]
    return [Type, m, c]
