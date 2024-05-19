
<h1 align="center">
  <br>
Deformable Autoencoders for Unsupervised Anomaly Detection: Application to Alzheimer's Disease
  <br>
</h1>
</h1>



> **Abstract:** *The increasing prevalence of neurodegenerative diseases, such as Alzheimer’s Disease (AD), and their impact on cognitive functions underscore the imperative for enhanced diagnostic methodologies. In response, this study introduces MORPHADE (Morphological Autoencoders for Alzheimer's Disease Detection), an innovative method employing unsupervised learning for the detection of AD-related morphological changes in brain magnetic resonance imaging scans. Utilizing 3D deformable autoencoders, MORPHADE advances the field in two key aspects: it incorporates a deformation loss function to improve the reconstruction of individual healthy brain anatomies, using deviations from these reconstructions as indicators of AD, and it assesses deformation field patterns to directly identify pathological markers. Our validation process demonstrates MORPHADE’s effectiveness in increasing the disease classification accuracy. Significantly, the method also facilitates estimation of disease severity, showing a promising correlation with medial temporal lobe atrophy scores determined by an expert neuroradiologist. Additionally, the identification of brain regions most susceptible to AD through MORPHADE aligns with established clinical insights.*


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






