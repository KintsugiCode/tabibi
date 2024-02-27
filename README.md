# tabibi
## Audio source separation and tab notation
====================================================================================


## Description

This is the 1st release of tabibi, featuring GRU-based audio source separation and subsequent GRU-based transformation into midi format. The purpose of this project is to train a deep learning model to isolate bass-guitar audio from mixed instrument recordings, and then use a second deep learning model to transform the isolated audio into midi format (which can be opened and viewed in guitar-tab software as tab notation).

There are two ways to run tabibi. The first uses pre-trained models and is described in [General Usage](#general-usage), the second involves training your own models and is described in [Training Specific Usage](#training-specific-usage).


## Setup

1. Clone the repo into a local folder of your choosing.
2. Create a virtual environment and activate it.
3. With the virtual environment active, install the necessary dependencies using pip: `pip install -r requirements.txt`

## General Usage

1. Navigate to `/src/config/constants.py` and update the production constant paths with folders of your own naming.

   **Important:** The `PRODUCTION_INPUT_FOLDER_PATH` and `PRODUCTION_OUTPUT_FOLDER_PATH` constants need to be nested one layer below `PRODUCTION_FOLDER_BASE_PATH`.

   E.g.:
   
       ├── /test_audio 			<-- PRODUCTION_FOLDER_BASE_PATH
              │   ├── /input       		<-- PRODUCTION_INPUT_FOLDER_PATH
              │   ├── /output  		<-- PRODUCTION_OUTPUT_FOLDER_PATH


    If you have trained and saved multiple models, you can select which ones will be used by setting the path to said models as `SEPARATION_MODEL_PATH` and `TRANSCRIPTION_MODEL_PATH`
 
2. Place as many audio files as you'd like in the `/input` folder you specified in the step above. **Important:** At the current development phase, these audio files **must** be in the .wav format.
3. With your virtual environment active, navigate to `/src` in your terminal and run `main.py`
4. Enter **no** in your terminal to process your audio with the pre-existing models.
5. Your audio files will be individually processed, passed through both the separation and the transcription model, and the isolated bass-guitar midi will be saved in your specified `/output` folder.

====================================================================================

## Training Specific Usage 

**Important:** This section is not relevant unless you intend to train/test new model versions yourself.

If you'd like to train/test new models, you'd need to enter **yes** into the terminal when prompted after running main.py -- **However**, to do so, you will need to have datasets in the proper folder structure for tabibi to train/test your models. 

Due to the characteristics of the training datasets that were available at the time, the current code uses **.wav** files as training inputs for the separation model, and **.flac**/**.mid** files as training inputs for the transcription model.

#### For the separation model:
![Screenshot from 2024-02-27 14-03-14](https://github.com/LiamGitGoing/tabibi/assets/41804800/b3621071-fe0f-491e-b0c5-aa18ddd1d7f5)

- Within the `/model1` folder, place all of your data inside a "subset" folder, i.e. `/V1`. You can specify the name of the subset folder in `/src/config/constants.py` as `MODEL1_SUBSET`.
- Each datapoint within your dataset's subset needs to consist of:
    - A parent folder that contains both the .wav mixed-instrument audio file and another folder named `/Bass` that itself contains the .wav bass audio file.


#### For the transcription model:
![Screenshot from 2024-02-27 14-03-40](https://github.com/LiamGitGoing/tabibi/assets/41804800/b75a98f7-e5be-4063-8b41-a6cbf97bd5ca)

- Within the `/model2` folder, place all of your data inside a "subset" folder, i.e. `/combined`. You can specify the name of the subset folder in `/src/config/constants.py` as `MODEL2_SUBSET`.
- Each datapoint within your dataset's subset needs to consist of:
    - A parent folder that contains both the .flac bass-guitar audio file and the .mid bass-guitar midi file. 

## Training Specific Configuration
All parameters are stored in .json files in `/config` for easy manipulation:
- The fourierparameters for each spectrogram transformation can be set in `fourierparameters.json`
- The hyperparameters for each model's training can be set in `hyperparameters_separation.json` and `hyperparameters_transcription.json`

### Pipelines
#### Separation Pipeline
![Screenshot from 2024-02-27 15-21-20](https://github.com/LiamGitGoing/tabibi/assets/41804800/c246e02b-380d-4684-975c-de4f55bfbcd6)

- The dataset is split into train/test groups.
- All mixed audio files and all bass audio files are converted to spectrograms and stored in a dictionary, where the indices allign between mixed audio and its associated bass audio.
- The spectrograms are then normalized and truncated to identical lengths across all tracks in the dictionary.
- This dictionary is then saved as a numpy recarray.

#### Transcription Pipeline
![Screenshot from 2024-02-27 15-21-40](https://github.com/LiamGitGoing/tabibi/assets/41804800/9fb554bf-67e1-47e7-8c37-99f01b3098f2)

- The dataset is split into train/test groups.
- All bass audio files are converted to spectrograms. All bass midi files are transformed into piano rolls and resized to match their associated bass audio spectrograms. All files are then stored in a dictionary, where the indices between bass audio and it associated bass midi allign.
- The bass spectrograms are then normalized (this is not necessary for the bass midi spectrograms, as they were already normalized as part of the piano_roll conversions) and everything is truncated to identical lengths across all tracks in the dictionary.
- This dictionary is then saved as a numpy recarray.

#### Training/Testing Pipelines
![Screenshot from 2024-02-27 15-21-54](https://github.com/LiamGitGoing/tabibi/assets/41804800/effdb5f3-6072-48fd-9644-a96e4f9fd943)

- The training/testing dictionarys are loaded (either for the separation- or the transcription model, both pipelines are the same).
- All data is then truncated once more, to make sure the data matches across dictionaries as well (i.e. to make sure all datapoints across training/testing have the same length too, and not just within a singular dictionary).
- The model runs for the specified number of epochs, outputs loss values for evaluation, saves audio samples and spectrogram samples post-training for manual assessment, and is saved as a now pre-trained model.

#### Post-Training Production Pipeline
![Screenshot from 2024-02-27 15-22-06](https://github.com/LiamGitGoing/tabibi/assets/41804800/67cfbb3a-28de-42be-b5b0-d12044edac83)

The production pipeline is on a loop and runs once per audio track provided in the `/input` folder.
- An audio track is transformed to a spectrogram, structured into a dictionary, normalized, and saved as a numpy recarray.
- It is then converted to a pytorch tensor and passed through the pre-trained separation model.
- The separation model's output is passed through the transcription model.
- The resulting spectrogram is then processed back from a spectrogram to a piano roll, and then to a midi file. It is then saved and can be found in the `/output` folder.

### Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

------------

