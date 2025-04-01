
# Import
import tensorflow as tf
import numpy as np


def Get_Eq_Class(Equation_Set):
    if Equation_Set == "NS_2D_SS":
        return NS_2D_SS()
    if Equation_Set == "NS_2D_SS_MMS":
        return NS_2D_SS_MMS()
    elif Equation_Set == "NS_3D_U":
        return NS_3D_Unsteady()

def Set_Eq_Constants(GE, Constant_Names, Constant_Values):
    for i in range(len(Constant_Values)):
        if Constant_Names[i] == "rho":
            GE.rho = Constant_Values[i]
        elif Constant_Names[i] == "mew":
            GE.mew = Constant_Values[i]
        elif Constant_Names[i] == "U":
            GE.Max_Vel = Constant_Values[i]
        elif Constant_Names[i] == "Radius":
            GE.R = Constant_Values[i]


class NS_2D_SS():
    def Equation_Info(self):
        Input_Names = ["x", "y"]
        Output_Names = ["u", "v", "p"]
        D1_Names = ["u_x", "u_y", "v_x", "v_y"]
        Residual_Names = ["Mass_Imbalance", "Mom-X_Imbalance", "Mom-Y_Imbalance"]
        Constant_Names = ["rho", "mew"]

        return [Input_Names, Output_Names, D1_Names, Residual_Names, Constant_Names]

    def Set_Model(self, NN_Model):
        self.model = NN_Model

    def Call_Calc_Until(self, Filter):
        Calc_Until = 0                  # 0 = Coor, 1 = Main Var, 2 = D1, 3 = D2
        for i in range(len(Filter)):
            if (Filter[i] == "M"):
                if Calc_Until < 1:
                    Calc_Until = 1
            elif (Filter[i] == "D1"):
                if Calc_Until < 2:
                    Calc_Until = 2
            else:
                Calc_Until = 3
        return Calc_Until

    def Get_Sim_Param(self, X_Array, Filter):
        Calc_Until = self.Call_Calc_Until(Filter)
        if Filter == "A":
            Filter = ["C", "M", "D1", "D2", "R"]

        # Calc Gradient
        x = X_Array[0] 
        y = X_Array[1]
        if Calc_Until > 0:
            with tf.GradientTape(persistent=True) as tape:
                # Coor
                tape.watch(x)
                tape.watch(y)

                # Main Var
                All_Var = self.model(tf.stack([x[:, 0], y[:, 0]], axis=1))
                u = All_Var[:, 0:1] * self.Out_Dev[0][0] + self.Out_Dev[0][1]
                v = All_Var[:, 1:2] * self.Out_Dev[1][0] + self.Out_Dev[1][1]
                p = All_Var[:, 2:3] * self.Out_Dev[2][0] + self.Out_Dev[2][1]

                # First Order
                if Calc_Until > 1:
                    u_x = tape.gradient(u, x)
                    u_y = tape.gradient(u, y)
                    v_x = tape.gradient(v, x)
                    v_y = tape.gradient(v, y)
                    p_x = tape.gradient(p, x)
                    p_y = tape.gradient(p, y)

                # Second Order
                if Calc_Until > 2:
                    u_xx = tape.gradient(u_x, x)
                    u_yy = tape.gradient(u_y, y)
                    v_xx = tape.gradient(v_x, x)
                    v_yy = tape.gradient(v_y, y)
            del tape

        # Export Result
        Results = []
        for i in range(len(Filter)):
            if (Filter[i] == "R"):
                nu = self.mew / self.rho

                R1 = self.Mass_Eq(u_x, v_y)
                R2 = self.Mom_x_Eq(u, v, u_x, u_y, p_x, u_xx, u_yy, self.rho, nu)
                R3 = self.Mom_y_Eq(u, v, v_x, v_y, p_y, v_xx, v_yy, self.rho, nu)

                Results.append([R1, R2, R3])
            elif (Filter[i] == "C"):
                Coor = [x, y]
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Main_Var = [u, v, p]
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Derivative1 = [u_x, u_y, v_x, v_y, p_x, p_y]
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Derivative2 = [u_xx, u_yy, v_xx, v_yy]
                Results.append(Derivative2)

        return Results

    def Mass_Eq(self, u_x, v_y):
        return u_x + v_y

    def Mom_x_Eq(self, u, v, u_x, u_y, p_x, u_xx, u_yy, rho, nu):
        Convection_Terms = (u * u_x) + (v * u_y)
        Pressure_Terms = -p_x / rho
        Dissipation_Terms = nu * (u_xx + u_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Mom_y_Eq(self, u, v, v_x, v_y, p_y, v_xx, v_yy, rho, nu):
        Convection_Terms = u * v_x + v * v_y
        Pressure_Terms = -p_y / rho
        Dissipation_Terms = nu * (v_xx + v_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms
        
class NS_2D_SS_MMS():
    def Equation_Info(self):
        Input_Names = ["x", "y"]
        Output_Names = ["u", "v", "p"]
        D1_Names = ["u_x", "u_y", "v_x", "v_y"]
        Residual_Names = ["Mass_Imbalance", "Mom-X_Imbalance", "Mom-Y_Imbalance"]
        Constant_Names = ["rho", "mew"]

        return [Input_Names, Output_Names, D1_Names, Residual_Names, Constant_Names]

    def Set_Model(self, NN_Model):
        self.model = NN_Model

    def Call_Calc_Until(self, Filter):
        Calc_Until = 0                  # 0 = Coor, 1 = Main Var, 2 = D1, 3 = D2
        for i in range(len(Filter)):
            if (Filter[i] == "M"):
                if Calc_Until < 1:
                    Calc_Until = 1
            elif (Filter[i] == "D1"):
                if Calc_Until < 2:
                    Calc_Until = 2
            else:
                Calc_Until = 3
        return Calc_Until

    def Get_Sim_Param(self, X_Array, Filter):
        Calc_Until = self.Call_Calc_Until(Filter)
        if Filter == "A":
            Filter = ["C", "M", "D1", "D2", "R"]

        # Calc Gradient
        x = X_Array[0] 
        y = X_Array[1]
        if Calc_Until > 0:
            with tf.GradientTape(persistent=True) as tape:
                # Coor
                tape.watch(x)
                tape.watch(y)

                # Main Var
                All_Var = self.model(tf.stack([x[:, 0], y[:, 0]], axis=1))
                u = All_Var[:, 0:1] * self.Out_Dev[0][0] + self.Out_Dev[0][1]
                v = All_Var[:, 1:2] * self.Out_Dev[1][0] + self.Out_Dev[1][1]
                p = All_Var[:, 2:3] * self.Out_Dev[2][0] + self.Out_Dev[2][1]

                # First Order
                if Calc_Until > 1:
                    u_x = tape.gradient(u, x)
                    u_y = tape.gradient(u, y)
                    v_x = tape.gradient(v, x)
                    v_y = tape.gradient(v, y)
                    p_x = tape.gradient(p, x)
                    p_y = tape.gradient(p, y)

                # Second Order
                if Calc_Until > 2:
                    u_xx = tape.gradient(u_x, x)
                    u_yy = tape.gradient(u_y, y)
                    v_xx = tape.gradient(v_x, x)
                    v_yy = tape.gradient(v_y, y)
            del tape

        # Export Result
        Results = []
        for i in range(len(Filter)):
            if (Filter[i] == "R"):
                nu = self.mew / self.rho
                Comp1 = 4. * tf.math.sin(2.*x - 2.*y)
                Comp2 = tf.math.sin(x + y) + 4. * tf.math.sin(2. * x + 2. * y)
                Comp3 = -0.25 * tf.math.sin(x - 3. * y)
                Comp4 =  0.25 * tf.math.sin(3. * x - y)
                Sx    = tf.math.sin(4. * x) + Comp2 + Comp1 + 3. * Comp4 + Comp3
                Sy    = tf.math.sin(4. * y) - Comp2 + Comp1 + 3. * Comp3 + Comp4

                R1 = self.Mass_Eq(u_x, v_y)
                R2 = self.Mom_x_Eq(u, v, u_x, u_y, p_x, u_xx, u_yy, Sx, self.rho, nu)
                R3 = self.Mom_y_Eq(u, v, v_x, v_y, p_y, v_xx, v_yy, Sy, self.rho, nu)

                Results.append([R1, R2, R3])
            elif (Filter[i] == "C"):
                Coor = [x, y]
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Main_Var = [u, v, p]
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Derivative1 = [u_x, u_y, v_x, v_y, p_x, p_y]
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Derivative2 = [u_xx, u_yy, v_xx, v_yy]
                Results.append(Derivative2)

        return Results

    def Mass_Eq(self, u_x, v_y):
        return u_x + v_y

    def Mom_x_Eq(self, u, v, u_x, u_y, p_x, u_xx, u_yy, Sx, rho, nu):
        Convection_Terms = (u * u_x) + (v * u_y)
        Pressure_Terms = -p_x / rho
        Dissipation_Terms = nu * (u_xx + u_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms - Sx

    def Mom_y_Eq(self, u, v, v_x, v_y, p_y, v_xx, v_yy, Sy, rho, nu):
        Convection_Terms = u * v_x + v * v_y
        Pressure_Terms = -p_y / rho
        Dissipation_Terms = nu * (v_xx + v_yy)
        return Convection_Terms - Pressure_Terms - Dissipation_Terms - Sy
   
class NS_3D_Unsteady():
    def Equation_Info(self):
        Input_Names  = ["t", "x", "y", "z"]
        Output_Names = ["u", "v", "w", "p"]
        D1_Names     = ["u_t", "v_t", "w_t", "u_x", "u_y", "u_z", "v_x", "v_y", "v_z", "w_x", "w_y", "w_z"]
        Residual_Names = ["Mass_Imbalance", "Mom-X_Imbalance", "Mom-Y_Imbalance", "Mom-Z_Imbalance"]
        Constant_Names = ["rho", "mew"]

        return [Input_Names, Output_Names, D1_Names, Residual_Names, Constant_Names]

    def Set_Model(self, NN_Model):
        self.model = NN_Model

    def Call_Calc_Until(self, Filter):
        Calc_Until = 0                  # 0 = Coor, 1 = Main Var, 2 = D1, 3 = D2
        for i in range(len(Filter)):
            if (Filter[i] == "M"):
                if Calc_Until < 1:
                    Calc_Until = 1
            elif (Filter[i] == "D1"):
                if Calc_Until < 2:
                    Calc_Until = 2
            else:
                Calc_Until = 3
        return Calc_Until

    def Get_Sim_Param(self, X_Array, Filter):
        Calc_Until = self.Call_Calc_Until(Filter)
        if Filter == "A":
            Filter = ["C", "M", "D1", "D2", "R"]

        # Calc Gradient
        t = X_Array[0] 
        x = X_Array[1] 
        y = X_Array[2] 
        z = X_Array[3] 
        if Calc_Until > 0:
            with tf.GradientTape(persistent=True) as tape:
                # Coor
                tape.watch(t)
                tape.watch(x)
                tape.watch(y)
                tape.watch(z)

                # Main Var
                All_Var = self.model(tf.stack([t[:, 0], x[:, 0], y[:, 0], z[:, 0]], axis=1))
                u = All_Var[:, 0:1] * self.Out_Dev[0][0] + self.Out_Dev[0][1]
                v = All_Var[:, 1:2] * self.Out_Dev[1][0] + self.Out_Dev[1][1]
                w = All_Var[:, 2:3] * self.Out_Dev[2][0] + self.Out_Dev[2][1]
                p = All_Var[:, 3:4] * self.Out_Dev[3][0] + self.Out_Dev[3][1]

                # First Order
                if Calc_Until > 1:
                    u_t = tape.gradient(u, t)
                    v_t = tape.gradient(v, t)
                    w_t = tape.gradient(w, t)
                    u_x = tape.gradient(u, x)
                    u_y = tape.gradient(u, y)
                    u_z = tape.gradient(u, z)
                    v_x = tape.gradient(v, x)
                    v_y = tape.gradient(v, y)
                    v_z = tape.gradient(v, z)
                    w_x = tape.gradient(w, x)
                    w_y = tape.gradient(w, y)
                    w_z = tape.gradient(w, z)
                    p_x = tape.gradient(p, x)
                    p_y = tape.gradient(p, y)
                    p_z = tape.gradient(p, z)

                # Second Order
                if Calc_Until > 2:
                    u_xx = tape.gradient(u_x, x)
                    u_yy = tape.gradient(u_y, y)
                    u_zz = tape.gradient(u_z, z)
                    v_xx = tape.gradient(v_x, x)
                    v_yy = tape.gradient(v_y, y)
                    v_zz = tape.gradient(v_z, z)
                    w_xx = tape.gradient(w_x, x)
                    w_yy = tape.gradient(w_y, y)
                    w_zz = tape.gradient(w_z, z)
            del tape

        # Export Result
        Results = []
        for i in range(len(Filter)):
            if (Filter[i] == "R"):
                nu = self.mew / self.rho

                R1 = self.Mass_Eq(u_x, v_y, w_z)
                R2 = self.Mom_x_Eq(u, v, w, u_t, u_x, u_y, u_z, p_x, u_xx, u_yy, u_zz, self.rho, nu)
                R3 = self.Mom_y_Eq(u, v, w, v_t, v_x, v_y, v_z, p_y, v_xx, v_yy, v_zz, self.rho, nu)
                R4 = self.Mom_z_Eq(u, v, w, w_t, w_x, w_y, w_z, p_z, w_xx, w_yy, w_zz, self.rho, nu)

                Results.append([R1, R2, R3, R4])
            elif (Filter[i] == "C"):
                Coor = [x, y]
                Results.append(Coor)
            elif (Filter[i] == "M"):
                Main_Var = [u, v, w, p]
                Results.append(Main_Var)
            elif (Filter[i] == "D1"):
                Derivative1 = [u_t, v_t, w_t, u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z]
                Results.append(Derivative1)
            elif (Filter[i] == "D2"):
                Derivative2 = [u_xx, u_yy, u_zz, v_xx, v_yy, v_zz, w_xx, w_yy, w_zz]
                Results.append(Derivative2)

        return Results

    def Mass_Eq(self, u_x, v_y, w_z):
        return u_x + v_y + w_z

    def Mom_x_Eq(self, u, v, w, u_t, u_x, u_y, u_z, p_x, u_xx, u_yy, u_zz, rho, nu):
        Convection_Terms = (u * u_x) + (v * u_y) + (w * u_z)
        Pressure_Terms = -p_x / rho
        Dissipation_Terms = nu * (u_xx + u_yy + u_zz)
        return u_t + Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Mom_y_Eq(self, u, v, w, v_t, v_x, v_y, v_z, p_y, v_xx, v_yy, v_zz, rho, nu):
        Convection_Terms = (u * v_x) + (v * v_y) + (w * v_z)
        Pressure_Terms = -p_y / rho
        Dissipation_Terms = nu * (v_xx + v_yy + v_zz)
        return v_t + Convection_Terms - Pressure_Terms - Dissipation_Terms

    def Mom_z_Eq(self, u, v, w, w_t, w_x, w_y, w_z, p_z, w_xx, w_yy, w_zz, rho, nu):
        Convection_Terms = (u * w_x) + (v * w_y) + (w * w_z)
        Pressure_Terms = -p_z / rho
        Dissipation_Terms = nu * (w_xx + w_yy + w_zz)
        return w_t + Convection_Terms - Pressure_Terms - Dissipation_Terms

