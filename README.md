# LongR²: Longitudinal Rigid Registration for Brain MRI

![Pipeline](./figs/pipeline.png)

This repository contains the source code for the research paper "*Learning Accurate Rigid Registration for Longitudinal Brain MRI from Synthetic Data*". You can find the paper [here](https://ieeexplore.ieee.org/document/10980859). ([arXiv](https://arxiv.org/abs/2501.13010), [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12237398/))

---

## Overview
We provide here a deep learning–based **rigid** registration tool specifically optimized for **longitudinal (within-subject) brain MRIs**. The method is trained using **synthetic longitudinal image pairs**, enabling accurate and robust estimation of rigid transformations across timepoints. The training strategy follows principles of synthetic-data–driven learning (see this [paper](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00337/124867/Synthetic-data-in-generalizable-learning-based) for background). 

The model is trained in an [anatomy-aware and acquisition-agnostic](https://arxiv.org/abs/2301.11329) manner, allowing it to generalize across different MRI contrasts and to operate reliably **with or without skull stripping**.

---

## Table of Contents
- [Instructions](#instructions)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Instructions
### Training Instructions

1. **Environment Setup**
   - Ensure Apptainer (or Singularity) is installed for containerized training.
   - Prepare your configuration file (e.g., `configs/config.yaml`) and training data as described in the paper and repository.

2. **Run Training**
   - Navigate to the main repository directory:

     ```bash
     cd /path/to/longitudinal-rigid-registration
     ```

   - Start training using Apptainer and Python:

     ```bash
     ./setup/run_in_apptainer.sh python -m modules.train
     ```
     This command launches the containerized environment and runs the training script with your configuration YAML file (`configs/config.yaml`).

3. **Monitor Training**
   - Training logs are saved in the `logs/` directory.
   - Model checkpoints are saved in `models/rigid_reg` by default.
   - Adjust configuration parameters in your YAML file as needed for your experiments.

**Troubleshooting:**
Ensure all required data paths and configuration options are correctly set in your config file. For custom experiments, modify the config and rerun the training command.

### Reference / Evaluation

1. **Prepare config**
    - Edit `configs/config.yaml` and set the `eval` section fields: `data` (CSV(s)), `weights` (path template), `run_name`, `labels` (LUT), `save_name`, and optional `out_fig` to save moved images.

2. **Run evaluation**
    - From the repository root run (containerized):

      ```bash
      ./setup/run_in_apptainer.sh python -m modules.eval
      ```

3. **Outputs**
    - Results CSV is written as `save_name`.
    - If `out_fig` is set, moved images are saved in that directory.

4. **Benchmark**
  - Performance note: on a Quadro RTX 6000, with input volumes of size 256×256×256, a single registration takes approximately ~2 seconds (measured end-to-end for model inference and resampling).

## Citation
If you use LongR² in your work, please cite the following paper:
```
@inproceedings{fu2025longitudinalrigid,
  author    = {Fu, Jingru and Dalca, Adrian V. and Fischl, Bruce and Moreno, Rodrigo and Hoffmann, Malte},
  title     = {Learning Accurate Rigid Registration for Longitudinal Brain MRI from Synthetic Data},
  booktitle = {2025 IEEE 22nd International Symposium on Biomedical Imaging (ISBI)},
  year      = {2025},
  pages     = {1--5},
  address   = {Houston, TX, USA},
  doi       = {10.1109/ISBI60581.2025.10980859},
  keywords  = {Training, Neuroimaging, Deep learning, Image registration, Accuracy,
               Magnetic resonance imaging, Transforms, Brain modeling, Synthetic data,
               Rigid image registration, Longitudinal analysis}
}
```

## Acknowledgements:
This repository builds upon ideas and tools from [SynthMorph](https://martinos.org/malte/synthmorph/). 

## TODO
- Add instance-specific optimization
- Add a figure

