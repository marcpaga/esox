# 8-oxo-dG detection using nanopore sequencing

This repository contains the code for the 8-oxo-dG detection model using nanopore sequencing data. This entails two models: 

- Basecalling model: this model is used to basecall 8-oxo-dG as G.
- Modification calling model: this model evaluates any basecalled G to determine if it is 8-oxo-dG.

For more information, please read our [publication](https://www.nature.com/articles/s41467-025-60391-3).

## Limitations

It is important to understand the limitations of this model to avoid misinterpretation of the results, **please do not ignore this section**.

- **Pore chemistry**: the data used to train this model was generated using R9.4.1 flow cells with 4KHz sampling rate. Using this model on other flow cell versions or sampling rates will likely give wrong results.

- **5-mer bias**: the model was trained on a subset of all possible 5-mers, this means that the model will not be able to detect 8-oxo-dG in all possible contexts. Furthermore, the performance of the model is 5-mer specific, meaning that the model will perform better in some contexts than others. Please check `static/kmer_performance.txt` to see the performance per 5-mer, and whether the 5-mer you are interested in is present in the model. 5-mers not in the training dataset will have 0 fp and 0 fn.

- **8-oxo-dG abundance**: 8-oxo-dG is not a very abundant modification, meaning that even a few false positives will reduce the signal-to-noise ratio significantly. Consider what the expected abundance of 8-oxo-dG is in your sample before using this model, and check if this expected abundance is higher than the false positive rate of the model (it should work fine if abundance 8-oxo-dG:G abundance is 1:10000 or higher).

- **Sample preparation**: the standard ONT library prep contains a FFPE repair step, which contains *Fpg*, a DNA glycosylase that is responsible for removing 8-oxo-dG from DNA. If your sample was prepared using this protocol, it is likely that most of the 8-oxo-dG has been removed from the DNA, and this model will not be able to detect it, or it its abundance will be lower than the false positive rate.



## Installation

The 8-oxo-dG calling consists of two steps. Please install the dependencies as follows. 

```bash
conda create -n esox_env python=3.7
conda activate esox_env
conda install -c bioconda ont-tombo  # this might take a while
pip install -r requirements.txt
```

For a full list of dependencies see: `conda.txt`, dependencies in `requirements.txt` are installed via pip. The installation of ont-tombo might be slow.

## Usage

For demo data, here is a small dataset that can be used to test the model. The data is already basecalled using Guppy/Dorado, and the raw data is in the `demo/fast5` folder. The basecalled data is in the `demo/fastq` folder.
Example outputs are in `demo/basecall_out` and `demo/modcall_out`. The processing of these files should take a few minutes when using an up-to-date GPU (year 2020). The code has been tested on the Linux OS.

[Download link](https://surfdrive.surf.nl/files/index.php/s/X2kRYzBOg68eQwc)

### Basecalling

First, we have to basecall the raw data (`.fast5` files) using our basecalling model. You will also need the already basecalled (`.fastq` files) from Guppy/Dorado. This will generate a `.fastq` file with the basecalled sequences, as well as a `.npz` file that can be used as input for our second model.

The scripts expectes equally named `.fast5` and `.fastq` files in the input folders, see the `demo` folder for examples.

```bash
conda activate esox_env

python3 scripts/basecall.py \
--fast5-path demo/fast5 \
--fastq-path demo/fastq \
--output-path demo/basecall_out \
--model-file static/models/bonito.pt \
--progress-bar \
--device cuda:0 \
--demo
```

This is the slowest step, not using a GPU will make this step very slow. If you feel this is too slow, consider dividing the input data into smaller chunks and running them in parallel using a pipeline (e.g. Snakemake).

If you get the following error, see [StackOverflow](https://stackoverflow.com/questions/77939924/importing-pandas-and-cplex-in-a-conda-environment-raises-an-importerror-libstdc/77940023#77940023) on how to solve it:

```
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
```

### Modification calling

After basecalling, we can use the `.fastq` and `.npz` file generated in the previous step to evaluate the basecalled Gs and determine if they are 8-oxo-dG.

```bash
conda activate esox_env # if not already activated

python3 scripts/modcall.py \
--input-path demo/basecall_out \
--output-path demo/modcall_out \
--model-file static/models/remora.pt \
--progress-bar \
--device cuda:0
```
Again, please check `static/kmer_performance.txt` to decide what threshold to use per 5-mer based on the FP rate and your 8-oxo-dG expected abundance. In our work we used a threshold of 0.95 for most 5-mers. 5-mers not in the training dataset will have 0 fp and 0 fn.

## Model training

We provide scripts to process the oligo data from fast5 and fastq files into an adequate input format for model training. We also provide a script to train a model using the processed data with the used configuration in our research. The processing of these demo files should take a about ~30 minutes total on a single CPU.

### Data

In this repository we provide demo files to test the scripts (check `demo/dev`). The oligo dataset can be found in the ENA repository under the accession number [PRJEB46810](https://www.ebi.ac.uk/ena/browser/view/PRJEB76712).

### Mapping oligo repeats

Here we check which oligos compose the repeats in each read and try to determine the random bases. The output will be a table with the read id, the oligos that have been identified, and the random bases based on the basecalling and their quality scores.

```
python scripts/dev/map_oligo_repeats.py \
--fastq-file demo/dev/fastq/demo.fastq \
--ref-file demo/dev/oligo_ref.fasta \
--output-file demo/dev/oligo_repeats/demo.txt
```

#### Making oligo references

Based on the composition of each oligo concatemer read, we define a reference sequence for each read.

```
python scripts/dev/make_oligo_references.py \
--mapped-file demo/dev/oligo_repeats/demo.txt \
--ref-file demo/dev/oligo_ref.fasta \
--output-file demo/dev/oligo_references/demo.fasta
```

#### Resquiggle oligo raw data

We resquiggle the raw data using the references we created in the previous step.

```
python scripts/dev/resquiggle_oligo.py \
--fast5-file demo/dev/fast5/demo.fast5 \
--ref-file demo/dev/oligo_references/demo.fasta \
--output-file demo/dev/oligo_resquiggle/demo.fasta
```

#### Prepare data for training

We prepare the data for training by creating a dataset with the resquiggled data and the references.

```
python scripts/dev/prepare_nn_input.py \
--reference-file demo/dev/oligo_references/demo.fasta \
--fast5-file demo/dev/fast5/demo.fast5 \
--resquiggle-file demo/dev/oligo_resquiggle/demo.fasta \
--output-file demo/dev/nn_input/demo.npz
```

The arrays can be loaded like this:

```python
import numpy as np

data = np.load("demo/dev/nn_input/demo.npz")

x = data["x"] # arrays of shape [n_examples, 100] with the normalized raw signal centered around a guanine/8-oxo-dG
y = data["y"] # arrays of shape [n_examples, ] with the label (0 for G, 1 for 8-oxo-dG)
e = data["e1"] # arrays of shape [n_examples, 100] with the expected normalized signal
s = data["s1"] # arrays of shape [n_examples, 100] with encoded basecalls aligned to the signal
```

#### Training

The model can be trained using the prepared data. The script provides a simple example loop on how to train the model using the prepared demo data.

```
python scripts/dev/train.py \
--data-path demo/dev/nn_input \
--output-path demo/dev/model
```

## Why is it called esox?

Most nanopore tools have fish names, and esox is the [genus of the pike fish](https://en.wikipedia.org/wiki/Esox), which ends in "ox", as in oxidation.

## Citation

If you use our model, please cite our paper:

```
Pagès-Gallego, M., van Soest, D.M.K., Besselink, N.J.M. et al. Direct detection of 8-oxo-dG using nanopore sequencing. Nat Commun 16, 5236 (2025). https://doi.org/10.1038/s41467-025-60391-3
```
