# --------------------------------------------------------------
# Case Info: Contains all of the information
# --------------------------------------------------------------
import tensorflow as tf
import numpy as np


class Case_Info_Class:
    # ----------------------------------------------------------------
    # Basic
    # ----------------------------------------------------------------
    def __init__(self):
        # Special Loss setting
        self.Current_Time = 0.
        #self.Cur_ID       = np.loadtxt('ID.txt')
        #np.savetxt('ID.txt', self.Cur_ID + 1.)
        self.B1_List       = np.flip(np.float32(np.array([0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.4, 0.6, 0.8, 1., 1.2, 1.5, 2., 2.5, 4.])))
        self.Denoise_Param = [
                         [2, 1],  # LF_ID : GE, Data1, Data2, ...
                         [4.5, self.B1_List[0]],  # Beta  : GE, Data1, Data2, ...
                         [1., 0.]]  # Weight: DO1, DA1, DO2, DA2, ...
        self.Special_Loss_Setting = [self.Denoise_Param]

        # Load Class (Default)
        self.Class_RP = Report_Param_Class()
        self.SW_EQ = SW_Equation_Class()
        self.BC_EQ = BC_Equation_Class()
        self.Load_Equation_Info()
        self.Load_Domain_Size()

    # Default
    def Load_Standard_Info(self):
        Data_Type = 'float32'
        Random_Seed = 0
        return [Data_Type, Random_Seed]

    def Load_Equation_Info(self):
        Case_Name = "Cond1DV3C_10_" #+ str(int(self.Cur_ID[0]))
        Governing_Equation = "Cond_1D"  # 2 Dimensional, Navier-Stokes, steady state
        Equation_Constants = [0.002]  # Alpha
        Analytical_Solution_Exist = False
        self.GE_Out_Dev    = [[10., 40.]]

        return [Case_Name, Governing_Equation, Equation_Constants, Analytical_Solution_Exist]

    def Load_Domain_Size(self):
        Total_Domain = [[1., 100.], [-2., 2.]]  # x_min, x_max, ymin, ymax
        LB = np.float32(np.array(Total_Domain)[:, 0])  # Lower Bound
        UB = np.float32(np.array(Total_Domain)[:, 1])  # Upper Bound
        return [Total_Domain, LB, UB]

    # ----------------------------------------------------------------
    # NN
    # ----------------------------------------------------------------
    # Default
    def Load_NN_Size(self):
        Use_Costum_Neurons = False
        Layers = 4
        Neurons = 64
        Costum_Neurons = []

        if Use_Costum_Neurons:
            All_Neurons = [0 for x in range(len(Costum_Neurons))]
            for i in range(len(Costum_Neurons)):
                All_Neurons[i] = Costum_Neurons[i]
        else:
            All_Neurons = [Neurons for x in range(Layers)]

        return All_Neurons

    # Only for TL
    def Load_Loaded_NN_Info(self):
        Load_NN = False
        Loaded_NN_Name = 'Trained_NN/Pre_2SideBC_1'
        return [Load_NN, Loaded_NN_Name]

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    # Default
    def Load_Backup_Info(self):
        # iter Based
        Costum_Backup = [200, 500, 1000, 3000, 5000, 7000, 10000, 10100, 12000, 15000, 20000, 50000, 100000]
        Fixed_Backup = 5000
        Report_Interval = 100
        return [Report_Interval, Costum_Backup, Fixed_Backup]

    # Only for TL
    def Load_Optimizer_Info(self):
        Solver_Order = ["ADAM", "BFGS"]  # ADAM or BFGS
        Solver_Iter = [10000, 300000]

        # Adam
        Default_Adam_Lr = 1e-5
        Initial_Adam_LR = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000, 3000], [1e-2, 1e-3, 1e-4])

        # Filter Adam
        [Load_NN, Loaded_NN_Name] = self.Load_Loaded_NN_Info()
        if Load_NN:
            Adam_Lr = Default_Adam_Lr
        else:
            Adam_Lr = Initial_Adam_LR

        return [Solver_Order, Solver_Iter, Adam_Lr]

    # ----------------------------------------------------------------
    # Geo Points
    # ----------------------------------------------------------------
    def Load_Weights_List(self):
        LF_Types   = ["BC_D", "NR3"]	# NR3
        LF_Weights = [1.,      0.01,    1.]
        LF_PG_Conv = [[0],     [1],    [2]]
        PG_List = [
                   ["BC", 0, 1, 2],  
                   ["DA", 0],
                   ["C" , 0, 1]]  # GE
        LF_Setting = [
                      ["M",  0],	# uv
                      ["M",  0],	# uv
                      ["R",  0]]
        self.Special_Set_ID = [-1, 0]  # Perioidc (0), NR3(1)
        
        return LF_Types, LF_Weights, LF_PG_Conv, PG_List, LF_Setting

    def Load_Special_Weights(self):
        Special_Weights = []
        Special_Coor = []

        #Special_Weights.append([[1., 1., 1.], [1., 1., 1.]])
        Special_Weights.append([[1e+0, 1e+0, 1e+0]])
        Special_Weights.append([[1e+0, 1e+0]])
        #Special_Weights.append([[1e+0, 1e+0]])
        #Special_Weights.append([[1.]])	# pressure
        Special_Weights.append([[1e+0, 1e+0, 1e+0]])
        

        """
        # Default
        LF_Types, LF_Weights, LF_PG_Conv, PG_List, LF_Setting = self.Load_Weights_List()
        for i in range(len(LF_Setting)):
            if LF_Setting[i][0] == "R":
                Special_Weights.append([1., 1., 1.])    # Continuity, Mom-x, Mom-y
            else:
                Temp_Weights1 = []
                print(i, len(LF_PG_Conv[i]))
                for j in range(len(LF_PG_Conv[i])):
                    Temp_Weights2 = []
                    for k in range(1, len(LF_Setting)):
                        Temp_Weights2.append(1.)
                    Temp_Weights1.append(Temp_Weights2)
                Special_Weights.append(Temp_Weights1)
        """

        return [Special_Weights, Special_Coor]

    def Load_Point_Gen_Info(self):
        # Domain Params
        D_Size  = [[1., 100.], [-2. , 2.]]
        D1_Size = [[1., 10.],  [-0.2, 0.2]]
        L_Size  = [ [[ 1.,  1.],   [-2. ,  2.]],
                    [[ 1.,  100.], [-2.,  -2.]],
                    [[ 1.,  100.], [ 2.,   2.]]
                  ]
        
        Scr_BP = [[] for i in range(3)]	# Boundary
        Scr_DP = [[] for i in range(1)] # Data
        Scr_CP = [[] for i in range(2)] # Collocation Point
        Scr_UP = [[] for i in range(0)] # Unsteady

        Scr_BP[0] = ["Gen", "Unif_Box", L_Size[0], [0, 400], "Heat_EQ"]
        Scr_BP[1] = ["Gen", "Unif_Box", L_Size[1], [400, 0], "Heat_EQ"]
        Scr_BP[2] = ["Gen", "Unif_Box", L_Size[2], [400, 0], "Heat_EQ"]
        
        MF        = "Simulation_Data/"
        File_List = [MF + "ResultAll_ID10.txt"]
        
        Scr_DP[0] = ["Gen_File", File_List[0], [0, 1], [2]]
        #Scr_DP[1] = ["SP", "Translate", 0, [100., 0.]]
        #Scr_DP[1] = ["Include", "Unif_Sampling", 0, 100]	# Filter Data

        Scr_CP[0] = ["Gen", "Unif_Box", D_Size,  [200, 50]]
        Scr_CP[1] = ["Gen", "Unif_Box", D1_Size, [40,  50]]
        
        return Scr_BP, Scr_DP, Scr_CP, Scr_UP

	
class Report_Param_Class:
    def Load_Plot_Point_Info(self):
        Plot_Status = True  # if Dimension > 3, it has to use projection.
        Plot_Dim = [2]
        Plot_Var_ID = [[0, 1]]  # Follows the input order in the GE
        Plot_Domain = [[[0., 100.], [-2.5, 2.5]]]
        # Plot_Axis = ["x", "y"]
        Fig_Size = [[8, 3]]
        return Plot_Status, Plot_Dim, Plot_Var_ID, Plot_Domain, Fig_Size

    def Load_Report_Print_Setting(self):
        # This is print the data or not (since it need to be calculated with NN)
        Print_Final = True
        Print_Checkpoint = False
        Print_Backup = True
        Print_Freq = [Print_Final, Print_Checkpoint, Print_Backup]

        Print_Output_Results = False
        Print_Residual = False
        Print_Velocity_Profile = False
        Print_Types = [Print_Output_Results, Print_Residual, Print_Velocity_Profile]
        return Print_Freq, Print_Types

    def Load_Image_Setting(self, Output_Filter):  # Contour
        # Cur_Var = "uvwp"
        Var_ID = [0]
        Res_ID = [0]
        MinMax_Value = [[30., 50.]]

        n_Image = 1
        Image_Names = []
        Print_Output_ID = Var_ID  # Follows the GE Output
        Print_Residual_ID = Res_ID  # Follows the GE Residual

        Domain_ID = [[0, 1]]
        Domain_x = [[1., 100.]]
        Domain_y = [[-2., 2.]]
        Domain_Others = [[]]  # Constants, in order of CD.n_input
        Domain_Info = [Domain_ID, Domain_x, Domain_y, Domain_Others]
        Output_MinMax = MinMax_Value
        Residual_MinMax = [[-0.01, 0.01]]

        Sampling = [[300, 300]]
        Size = [[8, 3]]

        if Output_Filter == "Folder":
            return [n_Image, Image_Names]
        elif Output_Filter == "Report":
            return [n_Image, [Print_Output_ID, Print_Residual_ID], [Output_MinMax, Residual_MinMax], Domain_Info,
                    Sampling, Size]

    def Load_Vel_Profile_Image_Setting(self):
        # Re, CP, DP = Load_Main_Param()

        # Line Information
        Samples = 1000
        Line_Coor = [[[0., -0.5], [0., 0.5]],
                     [[-0.5, 0.], [0.5, 0.]]
                     ]  # Line1 (Start, stop), Line2 (Start, stop)
        # Plotted_Var_ID = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]  # Per line
        Plotted_Var_ID = [0, 1]  # Per line
        Plot_Names = ["x=0", "y=0"]
        Line_Details = [Samples, Line_Coor, Plot_Names, Plotted_Var_ID]

        # Data
        MF = 'Data/Raw/'
        FileNames = []
        Legend = ["PINN", "CFD, Re = " + str(1000), "CFD Data", "Initial Noisy Data", "Current Noisy Data"]
        Print_Types = ["Line", "Line", "Dot", "Dot", "Dot"]  # PINN is hardcoded into LINE

        # Image
        Size = [9, 8]
        Direction_Axis = ["y", "x"]  # x or y only
        Axis_Coor = [1, 0]  # Depends on the input order
        Legend_Location = ["lower right", "upper right"]
        Image_Details = [Size, Direction_Axis, Axis_Coor, Legend, Print_Types, Legend_Location]

        return Line_Details, FileNames, Image_Details


class BC_Equation_Class():
    # BC NOT USED
    def Calc_BC(self, X, Code):
        u, v, w, p = 0. * X[1], 0. * X[1], 0. * X[1], 0. * X[1]
        if (Code == "Wall"):
            u, v, w, p = 0. * X[1], 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Inlet"):
            u = 0. * X[1] + 1.
            v, w, p = 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Outflow"): # du/dx, dv/dx
            u, v, w, p = 0. * X[1], 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Sym"): #du/dy
            u, v, w, p = 0. * X[1], 0. * X[1], 0. * X[1], 0. * X[1]
        elif (Code == "Heat_EQ"):
            A = 20.
            B = 30.
            Temp = A / tf.math.sqrt(X[0]) * tf.math.exp(-tf.square(X[1]) / 4. / 0.002 / X[0]) + B
            u, v, w, p = Temp, Temp, Temp, Temp
        return [u]

# Default (just set as 1)
class SW_Equation_Class():
    def Calc_SW(self, X):
        #W = 100. / X[0] * 1./ tf.math.sqrt(X[0]) * (0.1 + tf.exp(-100. / X[0] * X[1] * X[1]))
        return 0. * X[0] + 1.
