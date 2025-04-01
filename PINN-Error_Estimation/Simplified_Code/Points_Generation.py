##########################################
# Version 1.1	Added option for .bak files
##########################################

# Import
import numpy as np


# Support
def Gen_Points_Unif_Box(Domain, Points):
    # Declare Vars
    Dim, LB, UB  = SS_Obtain_Domain_Param(Domain)
    Total_Points = SS_Calc_Defined_Total_Points(Points)
    Coor_List    = np.ones((Dim, Total_Points))
    Delta        = (UB - LB) 
    Temp_Index   = np.zeros((len(Points)))
    
    for i in range(len(Points)):
        if not(Points[i] == 0):
            Delta[i]  = Delta[i] / Points[i]

    # Calc
    for i in range(Total_Points):
        Coor_List[:, i] = LB + Temp_Index * Delta

        Temp_Index[0]   += 1
        for j in range(len(Points) - 1):
            if (Temp_Index[j] == Points[j] + 1):
                Temp_Index[j] = 0
                Temp_Index[j + 1] += 1

    return Coor_List, Total_Points

def Gen_Points_Random_Box(Domain, Points):
    Dim, LB, UB  = SS_Obtain_Domain_Param(Domain)
    Total_Points = SS_Calc_Defined_Total_Points(Points)
    Coor_List    = np.ones((Dim, Total_Points))

    for i in range(Dim):
        Coor_List[i, :] = LB[i] + np.random.rand(Points[i] + 1) * (UB[i] - LB[i])
    return Coor_List, Total_Points
    
def Gen_Points_File(FileName, Coor_Rows = [-1], Value_Rows = [-1]):
    if FileName[-4:len(FileName)] == ".txt":
        Raw_File = np.loadtxt(FileName, unpack=True)
    elif FileName[-4:len(FileName)] == ".npy":
        Raw_File = np.load(FileName)#, unpack=True)
    elif FileName[-2:len(FileName)] == ".x":
        Raw_File, _ = Import_3D_Mesh(FileName, 3, "1D")
    elif FileName[-2:len(FileName)] == ".q":
        Raw_File, _ = Import_3D_Data(FileName, 5, "1D")
    elif FileName[-4:len(FileName)] == ".bak":
        Raw_File, _ = Import_3D_Data(FileName, 5, "1D")

    Total_Data = len(Raw_File[0])
    if Coor_Rows[0] == -1 and len(Coor_Rows) == 1:
        All_Coor = Raw_File
    else:
        All_Coor = np.zeros((len(Coor_Rows), Total_Data))
        for i in range(len(Coor_Rows)):
            if not(Coor_Rows[i] == -1):
                All_Coor[i, :] = Raw_File[Coor_Rows[i], :]

    if Value_Rows[0] == -1 and len(Value_Rows) == 1:
        All_Data = Raw_File
    else:
        All_Data = np.zeros((len(Value_Rows), Total_Data))
        for i in range(len(Value_Rows)):
            if not(Value_Rows[i] == -1):
                All_Data[i, :] = Raw_File[Value_Rows[i], :]
            

        #All_Coor = np.loadtxt(FileName, unpack=True, usecols=Coor_Rows)
        #All_Data = np.loadtxt(FileName, unpack=True, usecols=Value_Rows)
    #print(np.min(All_Coor, axis =1), np.max(All_Coor, axis=1))
    #print(np.min(All_Data, axis =1), np.max(All_Data, axis=1))
    return All_Coor, All_Data, Total_Data
    
def Gen_Points_Obj_Cylinder(Domain, Points):
    Total_Dim = len(Points)
    if Total_Dim == 1:
        ts = np.linspace(0., 2. * np.pi, Points[0], endpoint=False)
        xs = Domain[2] * np.cos(ts) + Domain[0]
        ys = Domain[2] * np.sin(ts) + Domain[1]
        All_Points   = np.array([xs, ys])
        Total_Points = Points[0]

    elif Total_Dim == 2:
        ts = np.linspace(0., 2. * np.pi, Points[0], endpoint=False)
        zs = np.linspace(Domain[1][0], Domain[1][1], Points[1] + 1)
        tm, zm = np.meshgrid(ts, zs)
        ts = np.reshape(tm, (-1))
        zs = np.reshape(zm, (-1))
        xs = Domain[0][2] * np.cos(ts) + Domain[0][0]
        ys = Domain[0][2] * np.sin(ts) + Domain[0][1]
        All_Points   = np.array([xs, ys, zs])
        Total_Points = Points[0] * (Points[1] + 1)

    return All_Points, Total_Points



# Support (Very simple functions)
def SS_Obtain_Domain_Param(Domain):
    Dim = len(Domain)
    LB  = np.array(Domain)[:, 0]
    UB  = np.array(Domain)[:, 1]
    return Dim, LB, UB

def SS_Calc_Defined_Total_Points(Points):
    Total_Points = 1
    for i in range(len(Points)):
        Total_Points = Total_Points * (Points[i] + 1)
    return Total_Points

# Fortran
def Import_3D_Data(Filename, Total_Var, Mode):
    with open(Filename, 'rb') as file:
        # ---------------------------------------------------------
        # Basic params
        # ---------------------------------------------------------
        nblocks = np.fromfile(file, dtype=np.int32, count=1)[0]
        _ = np.fromfile(file, dtype=np.int32, count=3)
        dimensions = np.fromfile(file, dtype=np.int32, count=3)
        _ = np.fromfile(file, dtype=np.int32, count=5)
        time = np.fromfile(file, dtype=np.float32, count=1)

        # ---------------------------------------------------------
        # Data
        # ---------------------------------------------------------
        Total_Cells = dimensions[0] * dimensions[1] * dimensions[2]
        _ = np.fromfile(file, dtype=np.int32, count=2)
        Temp_Val = np.fromfile(file, dtype=np.float32, count=Total_Cells * Total_Var)

        # ---------------------------------------------------------
        # Format
        # Order = Pressure, u, v, w, eta
        # ---------------------------------------------------------
        if Mode == "1D":
            Val = Temp_Val.reshape((Total_Var, Total_Cells))
        elif Mode == "3D":
            Val = Temp_Val.reshape((Total_Var, dimensions[2], dimensions[1], dimensions[0]))

    return Val, dimensions

def Import_3D_Mesh(Filename, Total_Var, Mode):
    with open(Filename, 'rb') as file:
        # ---------------------------------------------------------
        # Basic params
        # ---------------------------------------------------------
        nblocks = np.fromfile(file, dtype=np.int32, count=1)[0]
        _ = np.fromfile(file, dtype=np.int32, count=3)
        dimensions = np.fromfile(file, dtype=np.int32, count=3)
        np.fromfile(file, dtype=np.float32, count=2)

        # ---------------------------------------------------------
        # Data
        # ---------------------------------------------------------
        Total_Cells = dimensions[0] * dimensions[1] * dimensions[2]
        All_Cells = np.fromfile(file, dtype=np.float32, count=Total_Cells * Total_Var)

        # ---------------------------------------------------------
        # Format
        # ---------------------------------------------------------
        if Mode == "1D":
            Val = All_Cells.reshape((Total_Var, Total_Cells))
        elif Mode == "3D":
            Val = All_Cells.reshape((Total_Var, dimensions[2], dimensions[1], dimensions[0]))
    return Val, dimensions
