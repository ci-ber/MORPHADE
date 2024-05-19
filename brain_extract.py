import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np

listt=[]

path="/home/yigit/iml-dl/data/ADNI/MCI_Siem_3T_WM"
for root, dirs, files in os.walk(path):
    for name in files:
        if name.startswith("ADNI") and name.endswith((".nii.gz")) :
            print(name)

            os.system("./ANTs/Scripts/antsBrainExtraction.sh -d 3 -e /home/yigit/ants/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii -m /home/yigit/ants/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_mask.nii -a  "+root+"/"+name+" -o "+root+"/"+name[:name.find(".nii")]+"_skullfree")
