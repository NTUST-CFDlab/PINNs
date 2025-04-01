
from Equation_Database import *

class Case_Details_Class():
    def Initialize_Values(self, Case_Info):
        self.Case_Info = Case_Info

        # Equation
        [Case_Name, Governing_Equation, Equation_Constants, GE_Out_Dev] = Case_Info.Load_Equation_Info()

        #self.MF          = Case_Name + '/'
        self.MF          = 'Reports/'
        self.Case_Name   = Case_Name
        self.GE          = Get_Eq_Class(Governing_Equation)
        
        GE_Var_Type_Name = self.GE.Equation_Info()
        Set_Eq_Constants(self.GE, GE_Var_Type_Name[4], Equation_Constants)
        self.GE.Out_Dev  = np.float32(np.array(GE_Out_Dev))

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
