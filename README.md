
<h1 align="center">
  <br>
Unsupervised Analysis of  Alzheimer’s Disease Signatures using 3D Deformable Autoencoders
  <br>
</h1>
</h1>



> **Abstract:** *With the increasing incidence of neurodegenerative diseases such as Alzheimer's Disease (AD), there is a need for further research that enhances detection and monitoring of the diseases. We present MORPHADE (Morphological Autoencoders for Alzheimer's Disease Detection), a novel unsupervised learning approach which uses deformations to allow the analysis of 3D T1-weighted brain images. To the best of our knowledge, this is the first use of deformations with deep unsupervised learning to not only detect, but also localize and assess the severity of structural changes in the brain due to AD. We obtain markedly higher anomaly scores in clinically important areas of the brain in subjects with AD compared to healthy controls, showcasing that our method is able to effectively locate AD-related atrophy. We additionally observe a visual correlation between the severity of atrophy highlighted in our anomaly maps and medial temporal lobe atrophy scores evaluated by a clinical expert. Finally, our method achieves an AUROC of 0.80 in detecting AD, out-performing several supervised and unsupervised baselines. We believe our framework shows promise as a tool towards improved understanding, monitoring and detection of AD. *


## Setup and Run


#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository


#### 3). Create a virtual environment with the needed packages (use conda_environment-osx.yaml for macOS)

```
cd ${TARGET_DIR}/MORPHADE
conda env create -f conda_environment.yaml
source activate py308 *or* conda activate py308
```

#### 4). Install PyTorch

> Example installation:

* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
#### 5). Download and preprocess datasets 

> You can download the data from ADNI website according to splits given in ./data/ADNI.
> To apply skull stripping and registration please refer download and install ANTs https://stnava.github.io/ANTs/.
> Run the brain_extract.py for skull stripping and register.py for registering to a MNI template.

#### 6). Run the pipeline

First train the autoencoder with constrained deformer by setting b=1:
```
python core/Main.py --config_path projects/morphade/configs/morphade_3D_from_scratch.yaml
```
Then, train the autoencoder with unconstrained deformer by setting b=0.01 and weights from previous training:
```
python core/Main.py --config_path projects/morphade/configs/morphade_3D_only_deformer.yaml
```
For testing with unconstrained deformer(b=0.01) directly, run this on test mode with shared model(-will make public after review-):
```
python core/Main.py --config_path projects/morphade/configs/morphade_3D_only_deformer.yaml
```

For training plain autoencoder without deformer, run:
```
python core/Main.py --config_path projects/morphade/configs/morphade_3D_no_deformer.yaml
```
> Refer to *.yaml files for experiment configurations.

# That's it, enjoy! :rocket:






