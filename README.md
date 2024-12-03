# Mononucleosis oral disease detection with the help of GANNs

## Abstract

This repo aims to provide an implementation of an ensemble model capable of detectin mononucleosis, pharyngitis, tonsilitis diseases. It also contains the implementation of a GANN capable of generating 
halethy oral cavity images. The dataset used for training as well as the artificially generated images are all located in the repo.

# Table of content

1. [Setting up the environment](#setting-up-the-environment)
2. [Dataset](#dataset)
3. [GANN](#gann)
4. [Ensemble Model](#ensemble-model)
5. [Visualizing training results](#visualizing-training-results)
6. [License](#license)

## Setting up the environment

Install python 3.10 https://www.python.org/downloads/

Create a virtual environment

```ps1
python -m venv .vevn
```
Actibate + install requirements

```ps1
chmod +x ./bin/activate

source ./bin/activate

pip install -r requirements.txt
```

## Dataset

Dataset images are spread across multiple folders of the repo. You can find the images under folder "bucal_cavity_diseases_dataset". 

As for the labeling, it can be found under  "src/dataset_helpers/data.csv" that contains the path to the image and the label considering that the current repo directory is "/home/gmihai00/Repos/TenserflowModelTraining/". The absolute path that makes the csv unportable is a current limitation of the implementation, planning to remove it in the future.


## GANN
TO DO

## Ensemble Model
TO DO

## Visualizing training results

## License

You are free to:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
