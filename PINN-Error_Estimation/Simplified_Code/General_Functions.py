
import os
from shutil import copyfile


def Init_Case(MF = 'Reports/'):
    if not(MF == 'Reports/'):
        MF = Check_MF_Name(MF)
    print("Main Folder Name:", MF)
    Init_Folders(MF)
    Clear_Files(MF)
    copyfile('Case_Info.py', MF + 'Case_Info.txt')
    return MF

# -----------------------------------------------------------------------------------------------
# Support
# -----------------------------------------------------------------------------------------------
def Init_Folders(MF):
    SubFolders  = ['Data', 'Residual', 'NN']

    # Create Folders
    print("Creating_Folders")
    if not (os.path.isdir(MF)):
        os.mkdir(MF)
    for i in range(len(SubFolders)):
        Long_Name = MF + SubFolders[i] + '/'
        if not(os.path.isdir(Long_Name)):
            os.mkdir(Long_Name)
            
def Clear_Files(MF):
    # These files are the one that may need be updated during training
    # File that are only updated once is not included (e.g. points)
    Progress_File = open(MF + "Progress.txt", "w+")
    Progress_File.close()
    Hist_File     = open(MF + "Loss_Histogram.txt", "w+")
    Hist_File.close()
    Img_File      = open(MF + "Image_Statistics.txt", "w+")
    Img_File.close()
    Beta_File     = open(MF + "Beta_Loss.txt", "w+")
    Beta_File.close()
    
def Check_MF_Name(Case_Name):
    New_Case_Name = Case_Name
    Folder_Exist  = os.path.isdir(Case_Name)
    Sim_Done_Stat = Check_Sim_Status(Case_Name)
     
    """
    # Add Index
    Add_Index_Val = 0
    while Folder_Exist:
        New_Case_Name = Case_Name + '-' + str(Add_Index_Val)
        Dir_Exist = os.path.isdir(New_Case_Name)
        if Dir_Exist:
            Add_Index_Val += 1
        else:
            Folder_Exist = False
            
    return New_Case_Name 
    """
    
    Add_Index_Val = 0
    while Folder_Exist and Sim_Done_Stat:
        New_Case_Name = Case_Name + '-' + str(Add_Index_Val)
        Folder_Exist  = os.path.isdir(New_Case_Name)
        Sim_Done_Stat = Check_Sim_Status(New_Case_Name)

        if Folder_Exist and Sim_Done_Stat:
            Add_Index_Val += 1

    return New_Case_Name

def Check_Sim_Status(Case_Name):
    Progress_Mode_List = np.loadtxt(Case_Name + "Progress.txt", usecols=0, dtype=str)
    if Progress_Mode_List[-1] == 'F':
        Simulation_is_Done = True
    else:
        Simulation_is_Done = False
    return Simulation_is_Done
    
    
"""   
# -----------------------------------------------------------------------------------------------
# Folder Functions
# -----------------------------------------------------------------------------------------------
def Create_Dir(Case_Info):
    Main_Folder = 'Reports/'
    Trained_Folder = 'Trained_NN'
    Folder_Names = ['Data', 'Residual', 'NN']
    [n_Image, Image_Names] = Case_Info.Class_RP.Load_Image_Setting("Folder")

    # Folder Rank 0
    Folder_Exist = os.path.isdir(Main_Folder)
    if not(Folder_Exist):
        os.mkdir(Main_Folder)

    Folder_Exist = os.path.isdir(Trained_Folder)
    if not(Folder_Exist):
        os.mkdir(Trained_Folder)

    # Folder Rank 1
    for i in range(len(Folder_Names)):
        Long_Name = Main_Folder + Folder_Names[i] + '/'
        Dir_Exist = os.path.isdir(Long_Name)
        if not(Dir_Exist):
            os.mkdir(Long_Name)

    # Folder Rank 2
    #if n_Image > 1:
    #    for i in range(2):
    #        for j in range(n_Image):
    #            Long_Name = Main_Folder + Folder_Names[i] + '/' + Image_Names[j] + '/'
    #            Dir_Exist = os.path.isdir(Long_Name)
    #            if not (Dir_Exist):
    #                os.mkdir(Long_Name)

def Copy_Settings():
    path = os.getcwd()
    copyfile('Case_Info.py', path + '/Reports/Case_Info.txt')
"""

def Rename_Reports_Folder(Case_Name):
    Folder_Exist = True
    Current_Value = 0
    while Folder_Exist:
        New_Folder_Name = Case_Name + '_' + str(Current_Value)
        Dir_Exist = os.path.isdir(New_Folder_Name)
        if Dir_Exist:
            Current_Value += 1
        else:
            os.rename('Reports', New_Folder_Name)
            Folder_Exist = False


