import nibabel as nib
import os
from PIL import Image
import pandas as pd
import numpy as np
import re
listt=[]
path="/home/yigit/iml-dl/data/ADNI/AD_Siem_training"
for root, dirs, files in os.walk(path):
    for name in files:
        if name.startswith("ADNI") and name.endswith(("skullstripped.nii.gz")) :
            print(name)
            os.system("./ANTs/Scripts/antsRegistrationSyNQuick.sh -d 3 -f /home/yigit/ants/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a_skull_stripped.nii.gz -m "+root+"/"+name+" -o "+root+"/"+name[:name.find(".nii")]+"_ -t a")
