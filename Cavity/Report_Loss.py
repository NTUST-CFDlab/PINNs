# -----------------------------------------------------
# Version 1.2	Major Change in format
#		        *Still have not been checked for the normal (Total) loss plot
# -----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt


def Plot_LF_Image(Loss_Data, Loss_Names, Mode, File_ID, MF="Reports/"):
    # Mode = Detailed, Total
    # File_ID is only for the Total

    print("Plotting Loss Function")
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')

    if Mode == "Detailed":
        Output_Name = MF + "Detailed_Loss.png"
        for i in range(len(Loss_Data)):
            ax.semilogy(range(len(Loss_Data[i])), Loss_Data[i], label=Loss_Names[i])
        plt.legend()

    elif Mode == "Total":
        Output_Name = MF + "Loss_Function-" + File_ID + ".png"
        ax.semilogy(range(len(Loss_Data)), Loss_Data)

    plt.savefig(Output_Name, bbox_inches='tight', dpi=300)
    plt.close()

def Write_LF_Data(NN_Solver, Mark_Loc, MF="Reports/"):
    File_Holder = open(MF + "Loss_Histogram.txt", "a+")
    if (len(NN_Solver.Temp_Hist) > 0):
        Start_Iter = NN_Solver.iter - len(NN_Solver.Temp_Hist) + 1
        for i in range(len(NN_Solver.Temp_Hist)):
            str_total = str(Start_Iter + i)
            Mark_ID = 0
            for j in range(len(NN_Solver.Temp_Hist[i])):
                str_total += "\t" + str(NN_Solver.Temp_Hist[i][j])
                if j == Mark_Loc[Mark_ID]:
                    str_total += "\t|"
                    Mark_ID += 1
            File_Holder.writelines(str_total + "\n")
    File_Holder.close()
    NN_Solver.Temp_Hist = []

def Plot_Final_LossCurve3(Case_Info, MF="Reports/"):
    print("---------------------------------------------------------")
    print("Printing Final Loss Curve")
    print("---------------------------------------------------------")
    LF_Types, _, LF_PG_Conv, _, LF_Setting = Case_Info.Load_Weights_List()
    Loss_Names, Loss_Format = Obtain_Final_Loss_Names_Format(Case_Info, LF_Types, LF_PG_Conv, LF_Setting)
    All_Loss = Obtain_Final_Loss_Data(MF + "Loss_Histogram.txt", Loss_Format)
    Plot_LF_Image(All_Loss, Loss_Names, "Detailed", "0", MF=MF)
    
    
    
    
# Additional
def Print_Last_Loss(MF = "Reports/"):
    # Take Data 
    Loss_Raw      = np.loadtxt(MF + "Loss_Histogram.txt", dtype = str)
    Last_Loss_Raw = Loss_Raw[-1][1:]

    # Writelines
    File_Holder = open(MF + "Last_Loss.txt", "w+")
    str_tot     = Last_Loss_Raw[0]
    for i in range(1, len(Last_Loss_Raw)):
        if Last_Loss_Raw[i] == "|":
            str_tot += "\n"
        elif Last_Loss_Raw[i-1] == "|":
            str_tot += Last_Loss_Raw[i]
        else:
            str_tot += '\t' + Last_Loss_Raw[i]
    File_Holder.writelines(str_tot + "\n")
    File_Holder.close()
    
def Print_Beta_Loss(Beta_Str, MF = "Reports/"):
    # Take Data 
    Loss_Raw      = np.loadtxt(MF + "Loss_Histogram.txt", dtype = str)
    Last_Loss_Raw = Loss_Raw[-1][1:]

    # Writelines
    File_Holder = open(MF + "Beta_Loss.txt", "a+")
    str_tot     = Beta_Str[1:]
    for i in range(1, len(Last_Loss_Raw)):
        if not(Last_Loss_Raw[i] == "|"):
            str_tot+= "\t" + Last_Loss_Raw[i]
    File_Holder.writelines(str_tot + "\n")
    File_Holder.close()		   
    
    
    
    
# Support
def Obtain_Final_Loss_Names_Format(Case_Info, LF_Types, LF_PG_Conv, LF_Setting):
    Loss_Format = [1]
    Loss_Names  = ["Total"]
    Data_ID     = 0
    
    # NAME
    for i in range(len(LF_Types)):
        if LF_Types[i] == "BC_D":
            Loss_Names.append("BC_D")
        elif LF_Types[i] == "Mov_Wall" or LF_Types[i] == "Mov_Wall3D":
            Loss_Names.append("BC_Mov_Wall")
        elif LF_Types[i] == "BC_N":
            Loss_Names.append("BC_N")
        elif LF_Types[i] == "Avg_p":
            Loss_Names.append("Avg_p")
        elif LF_Types[i] == "Data":
            Loss_Names.append("Data-" + str(Data_ID))
            Data_ID += 1
        elif LF_Types[i] == "NR3":
            for j in range(1, len(Case_Info.Denoise_Param[0])):
                Loss_Names.append("DataO-" + str(j))
                Loss_Names.append("DataA-" + str(j))
            Loss_Names.append("GE")
        elif LF_Types[i] == "GE":
            Loss_Names.append("GE")

    # Format
    for i in range(len(LF_Types)):
        if LF_Types[i] == "NR3":
            for j in range(1, len(Case_Info.Denoise_Param[0])):
                Loss_Format.append(1)
                Loss_Format.append(1)
            Loss_Format.append(1)
        else:
            Total = 0
            for j in range(len(LF_PG_Conv[i])):
                Total += len(LF_Setting[LF_PG_Conv[i][j]]) - 1
            Loss_Format.append(Total)
            
    print("Loss_Names :", Loss_Names)
    print("Loss_Format:", Loss_Format)
    return Loss_Names, Loss_Format
    
def Obtain_Final_Loss_Data(Loss_File, Loss_Format):
    # Read Format (This is more robust then using the loss format directly)
    First_Line = np.loadtxt(Loss_File, dtype=str)[0]
    Available_Cols = []
    for i in range(1, len(First_Line)):
        if not(First_Line[i]) == '|':
            Available_Cols.append(i)

    # Cols
    Col_List, Cur_Index = [], 0
    for i in range(len(Loss_Format)):
        Col_List.append(np.arange(Available_Cols[Cur_Index], Available_Cols[Cur_Index] + Loss_Format[i]))
        Cur_Index += + Loss_Format[i]
    #print(Available_Cols)
    print("Col_List   :", Col_List)

    # Read Data
    print("Reading Data")
    All_Loss = []
    for i in range(len(Loss_Format)):
        Temp_Data = np.loadtxt(Loss_File, unpack=True, usecols=(Col_List[i]))
        if Loss_Format[i]>1:
            All_Loss.append(np.sum(Temp_Data, axis = 0))
        else:
            All_Loss.append(Temp_Data)

    return np.array(All_Loss)

