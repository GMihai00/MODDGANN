# Mononucleosis oral disease detection with the help of GANNs

## Abstract

This repo aims to provide an implementation of an ensemble model capable of detectin mononucleosis, pharyngitis, tonsilitis diseases. It also contains the implementation of a GANN capable of generating
healthy oral cavity images. The dataset used for training as well as the artificially generated images are all located within the repo.

# Table of content

1. [Setting up the environment](#setting-up-the-environment)
2. [Dataset](#dataset)
3. [GANN](#gann)
4. [State of the art model](#state-of-the-art-model)
5. [Ensemble Model](#ensemble-model)
6. [Visualizing training results](#visualizing-training-results)
7. [License](#license)

## Setting up the environment

Install python 3.10 https://www.python.org/downloads/

Install VS Code https://code.visualstudio.com/

Install code runner + python extensions

Create a virtual environment

```ps1
python -m venv .vevn
```

Activate + install requirements

```ps1
chmod +x ./bin/activate

source ./bin/activate

pip install -r requirements.txt
```

## Dataset

Dataset images are spread across multiple folders of the repo. You can find the images under folder "bucal_cavity_diseases_dataset".

As for the labeling, a csv with the labels can be found under "src/dataset_helpers/data.csv". It contains relative paths to the image locations within the repo.

## GANN

Under ".vscode" folder rename "launch_gan.json" to "launch.json". 

Run the launch file using the vs extension. This will start training the gann from scratch, running 10000 epochs, using a batch size of 400. You can change this paramaeters, please run the following command for further assistence

```ps1
python ./src/model/gann/app.py --help
```

## State of the art model

Under ".vscode" folder rename "launch_state_of_art.json" to "launch.json".

Run the launch file using the vs extension. This will start training an ResNet50 trying to replicate the results from pharyngitis paper using our own dataset, doing 5-fold for validation and outputing tensorboard logs to a newly created "logs" folder. You can change this paramaeters, please run the following command for further assistence

```ps1
python ./src/model/state_of_art.py --help
```

## Ensemble Model

Under ".vscode" folder rename "launch_model.json" to "launch.json".

Run the launch file using the vs extension. This will start training the ensemble from scrath, doing 5-fold for validation and outputing tensorboard logs to a newly created "logs" folder. You can change this paramaeters, please run the following command for further assistence

```ps1
python ./src/model/main.py --help
```

## Visualizing training results

To vizualize the data you will need tensorboard installed. To vizualize your training results you can run:

```ps1
chmod +X ./src/model/display_rez.sh
 ./src/model/display_rez.sh
```

To view the results that the authors achieved please run

```ps1
tensorboard --logdir ./src/model/final_result_logs
```

## License

You are free to:

- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material for any purpose, even commercially.

Under the following terms:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
