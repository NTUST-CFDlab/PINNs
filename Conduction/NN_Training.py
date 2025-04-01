
# Import
import tensorflow as tf
import numpy as np

from scipy import optimize as opt
from time import time
import math

class PINN_Solver_Class():
    def Initialize_Training_Info(self, Case_Info, CD, Report_Result, DTYPE):

        # Classes
        self.CD = CD
        self.GE = CD.GE
        self.Report_Result = Report_Result
        self.model = CD.GE.model

        # Load Setting
        [self.Solver_Order, self.Solver_Iter_Limit, self.Adam_Lr] = Case_Info.Load_Optimizer_Info()
        self.LF_Types, self.LF_Weigths, self.LF_PG_Conv, PG_List, LF_Setting = Case_Info.Load_Weights_List()
        #[self.LF_Types, self.LF_Weigths, self.LF_PG_Conv] = Case_Info.Load_LF_Info()
        [self.Special_Weights, self.Special_Coor] = Case_Info.Load_Special_Weights()
        [self.Report_Interval, Costum_Backup, Fixed_Backup] = Case_Info.Load_Backup_Info()
        self.Backup_Interval = Create_Backup_Index(Costum_Backup, Fixed_Backup, self.Solver_Iter_Limit[len(self.Solver_Order)-1])
        self.Special_Loss_Set_ID = Case_Info.Special_Set_ID
        self.Special_Loss_Set_Param = Case_Info.Special_Loss_Setting
        self.B1_List                = Case_Info.B1_List

        # Load Data
        self.LF_Setting = CD.LF_Setting
        self.X_C = CD.X_C
        self.U_C = CD.U_C
        self.n_Group = CD.Total_Group

        # Clear File
        #Progress_File = open("Reports/Progress.txt", "w+")
        #Progress_File.close()
        #Hist_File = open("Reports/Loss_Histogram.txt", "w+")
        #Hist_File.close()

        # Default Values
        self.hist = []
        self.Temp_Hist = []
        self.iter = 0
        self.Backup_Index = 0
        self.DTYPE = DTYPE

    def Begin_Training(self):
        # Set Time
        self.Initial_Time = time()

        # Training
        for Solver_Index in range(len(self.Solver_Order)):
            if self.Solver_Order[Solver_Index]=="ADAM":
                print("Current Solver is ADAM \t it = " + str(self.iter))
                self.solve_with_TFoptimizer(self.Adam_Lr, self.Solver_Iter_Limit[Solver_Index])
            if self.Solver_Order[Solver_Index]=="BFGS":
                print("Current Solver is L-BFGS-B \t it = " + str(self.iter))
                if Solver_Index==0:
                    Temp_Iter = self.Solver_Iter_Limit[0]
                else:
                    Temp_Iter = self.Solver_Iter_Limit[Solver_Index] - self.Solver_Iter_Limit[Solver_Index-1]
                
                self.solve_with_ScipyOptimizer(method='L-BFGS-B',
                                               options={'maxiter': Temp_Iter,
                                                        'maxfun': 50000,
                                                        'maxcor': 50,      # previously 50
                                                        'maxls': 50,       # previously 50
                                                        'ftol': 1.0 * np.finfo(float).eps})
                self.Report_Result.Report_Current_Result(self, "E", "B" + str(self.B1_List[0]))
                for B_Index in range(1, len(self.B1_List)):
                    self.Special_Loss_Set_Param[0][1][1] = self.B1_List[B_Index]
                    self.solve_with_ScipyOptimizer(method='L-BFGS-B',
                                                   options={'maxiter': Temp_Iter,
                                                        'maxfun': 50000,
                                                        'maxcor': 50,      # previously 50
                                                        'maxls': 50,       # previously 50
                                                        'ftol': 1.0 * np.finfo(float).eps})
                    self.Report_Result.Report_Current_Result(self, "E", "B" + str(self.B1_List[B_Index]))                                     

        # End of Training
        print("------------------------------------------------------------------------------")
        print("Finish Training")
        print('Computation time: {} seconds'.format(time()-self.Initial_Time))
        print("------------------------------------------------------------------------------")
        self.Report_Result.Report_Current_Result(self, "F", "FINAL")

    def loss_fn(self):
        Loss_Per_Type = []

        for i in range(len(self.LF_Types)):
            if self.LF_Types[i] == "GE":
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        Weighted_Square = tf.reduce_mean(tf.square(All_Var_Pred[Var_Index]) * self.Special_Weights[i][k][j-1] * self.CD.SW)
                        Loss_Per_Type.append(Weighted_Square * self.LF_Weigths[i])

            elif self.LF_Types[i] == "BC_D":
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        #print(i, self.LF_Setting[C_ID][j], C_ID, Var_Index, tf.reduce_max(self.U_C[C_ID][Var_Index]))
                        Delta_BC_D = tf.square(self.U_C[C_ID][Var_Index] - All_Var_Pred[Var_Index])
                        Loss_Per_Type.append(tf.reduce_mean(Delta_BC_D) * self.LF_Weigths[i] * self.Special_Weights[i][k][j-1])

            elif self.LF_Types[i] == "Mov_Wall3D":
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        W         = (1 - 4. * tf.square(self.X_C[C_ID][1])) * (1 - 4. * tf.square(self.X_C[C_ID][3]))
                        #print(i, self.LF_Setting[C_ID][j], C_ID, Var_Index, tf.reduce_max(self.U_C[C_ID][Var_Index]))
                        Delta_BC_D = tf.square(self.U_C[C_ID][Var_Index] - All_Var_Pred[Var_Index]) * W
                        Loss_Per_Type.append(tf.reduce_mean(Delta_BC_D) * self.LF_Weigths[i] * self.Special_Weights[i][k][j-1])
                        
            elif self.LF_Types[i] == "Mov_Wall":	# 2D
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        W         = 1. - 4. * tf.square(self.X_C[C_ID][0])
                        #print(i, self.LF_Setting[C_ID][j], C_ID, Var_Index, tf.reduce_max(self.U_C[C_ID][Var_Index]))
                        Delta_BC_D = tf.square(self.U_C[C_ID][Var_Index] - All_Var_Pred[Var_Index]) * W
                        Loss_Per_Type.append(tf.reduce_mean(Delta_BC_D) * self.LF_Weigths[i] * self.Special_Weights[i][k][j-1])
            
                        
            elif self.LF_Types[i] == "BC_N":
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        Var_Index2 = 0
                        if Var_Index >= 2:
                            Var_Index2 = 1
                        Delta_BC_N = tf.square(self.U_C[C_ID][Var_Index2] - All_Var_Pred[Var_Index])
                        Loss_Per_Type.append(tf.reduce_mean(Delta_BC_N) * self.LF_Weigths[i] * self.Special_Weights[i][k][j-1])

            elif self.LF_Types[i] == "Data":
                for k in range(len(self.LF_PG_Conv[i])):
                    C_ID = self.LF_PG_Conv[i][k]
                    [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        Delta_BC_D = tf.square(self.U_C[C_ID][Var_Index] - All_Var_Pred[Var_Index])
                        Temp_Loss = tf.reduce_mean(Delta_BC_D)
                        Loss_Per_Type.append(Temp_Loss * self.LF_Weigths[i] * self.Special_Weights[i][k][j-1])

            elif self.LF_Types[i] == "NR3":
                #NR_Points  = self.Special_Loss_Set_Param[self.Special_Loss_Set_ID[i]][0]
                NR_LFID    = self.Special_Loss_Set_Param[self.Special_Loss_Set_ID[i]][0]
                NR_Beta    = self.Special_Loss_Set_Param[self.Special_Loss_Set_ID[i]][1]
                NR_Fweight = self.Special_Loss_Set_Param[self.Special_Loss_Set_ID[i]][2]
                #print(NR_Fweight)

                # GE
                ii    = NR_LFID[0]
                Sum  = 0
                for k in range(len(self.LF_PG_Conv[ii])):
                    C_ID    = self.LF_PG_Conv[ii][k]
                    [R_Raw] = self.GE.Get_Sim_Param(self.X_C[C_ID], ["R"])
                    SW      = self.CD.SW#[self.CD.Inv_CP_ID[C_ID]]
                    for j in range(1,len(self.LF_Setting[C_ID])):
                        Var_Index = self.LF_Setting[C_ID][j]
                        Sum += tf.reduce_mean(tf.square(R_Raw[Var_Index]) * SW * self.Special_Weights[ii][k][j-1])
                GE_Loss = Sum * self.LF_Weigths[ii]

                # Data
                for jj in range(1, len(NR_LFID)):
                    Data_Loss = 0
                    ii = NR_LFID[jj]
                    for k in range(len(self.LF_PG_Conv[ii])):
                        C_ID = self.LF_PG_Conv[ii][k]
                        [All_Var_Pred] = self.GE.Get_Sim_Param(self.X_C[C_ID], [self.LF_Setting[C_ID][0]])
                        for j in range(1, len(self.LF_Setting[C_ID])):
                            Var_Index  = self.LF_Setting[C_ID][j]
                            Delta_BC_D = tf.square(self.U_C[C_ID][Var_Index] - All_Var_Pred[Var_Index])
                            Data_Loss += tf.reduce_mean(Delta_BC_D) * self.LF_Weigths[ii] * self.Special_Weights[ii][k][j-1]

                    # Recap
                    # print(Data_Loss)
                    Alpha1 = 4.2
                    DO_Constant = 2. * tf.math.log(NR_Beta[jj]) / tf.math.log(10.) + 0.5
                    W_DA = tf.sigmoid(4.2 * (tf.math.log(GE_Loss) / tf.math.log(10.) + NR_Beta[0]))
                    W_DO = 2. * tf.sigmoid(Alpha1 * (tf.math.log(Data_Loss) / tf.math.log(10.) - DO_Constant))
                    # tf.print(Data_Loss, W_DO * Data_Loss)
                    Loss_Per_Type.append(W_DO * Data_Loss * NR_Fweight[2 * jj - 2])
                    Loss_Per_Type.append(W_DA * Data_Loss * NR_Fweight[2 * jj - 1])
                Loss_Per_Type.append(GE_Loss)



        loss = 0
        for i in range(len(Loss_Per_Type)):
            loss += Loss_Per_Type[i]

        Loss_Per_Type.insert(0, loss)

        return loss, Loss_Per_Type

    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss, Detailed_loss = self.loss_fn()
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape

        return loss, Detailed_loss, g

    def solve_with_TFoptimizer(self, lr, Iter_Limit):
        # Main Func
        Use_Adam = True
        optim = tf.keras.optimizers.Adam(learning_rate=lr)

        @tf.function
        def train_step():
            #loss, Detailed_loss, grad_theta = self.get_grad()
            #optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            with tf.device('/GPU:0'):
                loss, Detailed_loss, grad_theta = self.get_grad()
                optim.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, Detailed_loss

        while Use_Adam:
            loss, Detailed_loss = train_step()
            self.current_loss = loss.numpy()
            self.Detailed_Loss = np.array(Detailed_loss)
            self.callback()
            if self.iter > Iter_Limit:
                Use_Adam = False

    def solve_with_ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""

            weight_list = []
            #shape_list = []

            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                #shape_list.append(v.shape)
                weight_list.append(tf.reshape(v, (-1)))
                #weight_list.extend(v.numpy().flatten())

            #weight_list = tf.convert_to_tensor(weight_list)
            weight_list = tf.concat(weight_list, axis=0)
            return weight_list
            #return weight_list, shape_list

        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape

                # Weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    new_val = tf.reshape(weight_list[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx + vs[0]]
                    idx += vs[0]

                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1

                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, self.DTYPE))

        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""

            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, Detailed_loss, grad = self.get_grad()

            # Store current loss for callback function
            #loss = loss.numpy().astype(np.float64)
            #Detailed_loss = np.array(Detailed_loss).astype(np.float64)
            self.current_loss = loss.numpy().astype(np.float64)
            self.Detailed_Loss = np.array(Detailed_loss).astype(np.float64)

            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
                #grad_flat.append(tf.reshape(g, (-1)))

            # Gradient list to array
            grad_flat = np.array(grad_flat, dtype=np.float64)
            #grad_flat = np.array(tf.concat(grad_flat, axis=0), dtype=np.float64)

            # Return value and gradient of \phi as tuple
            return self.current_loss, grad_flat
        
            
        #with tf.device('/CPU:0'):
        with tf.device('/GPU:0'):
            x0 = get_weight_tensor()
            Temp_Result = opt.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        return Temp_Result


    def callback(self, xr = None):
        if self.iter % self.Report_Interval == 0:
            print('It {:05d}: loss = {:10.8e}'.format(self.iter, self.current_loss))
        self.hist.append(self.current_loss)
        self.Temp_Hist.append(self.Detailed_Loss)
        self.iter += 1

        if (self.iter==1):
            self.LS_Value_Check = 10 ** math.floor(math.log10(self.hist[0]))
        else:
            if (self.current_loss < self.LS_Value_Check):
                print("Loss Function is below: " + str(self.LS_Value_Check))
                self.Report_Result.Report_Current_Result(self, "C", str(self.LS_Value_Check))
                self.LS_Value_Check = self.LS_Value_Check/10
        if (self.iter == self.Backup_Interval[self.Backup_Index]):
            self.Report_Result.Report_Current_Result(self, "B", str(self.iter))
            self.Backup_Index += 1



# Support 
def Create_Backup_Index(Costum_List, Freq, Total_Iter):
    Backup_List = np.arange(Freq, Total_Iter, Freq)
    Backup_List = np.concatenate((Backup_List, Costum_List))
    Backup_List = np.sort(Backup_List)
    Backup_List = np.unique(Backup_List)

    return Backup_List

