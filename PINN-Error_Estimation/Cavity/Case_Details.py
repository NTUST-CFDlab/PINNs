
from Equation_Database import *

class Case_Details_Class():
    def Initialize_Values(self, Case_Info):
        self.Case_Info = Case_Info

        # Equation
        [Case_Name, Governing_Equation, Equation_Constants, Analytical_Solution_Exist] = Case_Info.Load_Equation_Info()

        #self.MF          = Case_Name + '/'
        self.MF          = 'Reports/'
        self.Case_Name   = Case_Name
        self.GE          = Get_Eq_Class(Governing_Equation)
        self.Analytical_Exist = Analytical_Solution_Exist
        GE_Var_Type_Name = self.GE.Equation_Info()
        Set_Eq_Constants(self.GE, GE_Var_Type_Name[4], Equation_Constants)
        self.GE.Out_Dev  = np.float32(np.array(Case_Info.GE_Out_Dev))
        #Set_GE_Deviation(self.GE, Case_Info, len(GE_Var_Type_Name[0]), len(GE_Var_Type_Name[1]))

        self.Input_Names     = GE_Var_Type_Name[0]
        self.Output_Names    = GE_Var_Type_Name[1]
        self.D1_Output_Names = GE_Var_Type_Name[2]
        self.Residual_Names  = GE_Var_Type_Name[3]
        self.n_Input  = len(self.Input_Names)
        self.n_Output = len(self.Output_Names)
        self.n_Eq     = len(self.Residual_Names)
        self.Cylinder_Coor = []

        # Domain
        [self.Total_Domain, self.Overall_lb, self.Overall_ub] = Case_Info.Load_Domain_Size()
        
        # Additional
        self.Apply_Spatial_Weight = True

        # Point - LF
        # self.LF_Setting
        # self.Total_Group
        # self.X_C
        # self.U_C
        # self.SW
