import tensorflow as tf
import numpy as np
#import os
import matplotlib.pyplot as plt
from Report_Loss import *
from Case_Info import *

#Plot_Final_LossCurve([3, 3, 1, 1, 1], [1, 1, 0, 0, 0], ["Total", "BC_D", "BC_N", "Data1", "Data2", "GE"])

Case_Info = Case_Info_Class()
Plot_Final_LossCurve3(Case_Info)
print("DONE :)")

