# tabibi
## Audio source separation and tab notation
====================================================================================

**Important**: This code uses audio files extensively to train models and run tests simulating data transformations. To reduce repo size, these audio files have not been uploaded to github. If you intend to train your own model, and/or run the testing suite that simulates audio transformations, you will need to place your own audio data within the appropriate folders (more info below in the relevant areas).

## Description

This is the 1st release of tabibi, featuring GRU-based audio source separation and subsequent GRU-based transformation into midi format. The purpose of this project is to train a deep learning model to isolate bass-guitar audio from mixed instrument recordings, and then use a second deep learning model to transform the isolated audio into midi format (which can be opened and viewed in guitar-tab software as tab notation).

There are two ways to run tabibi. The first uses pre-trained models and is described in [General Usage](#general-usage), the second involves training your own models and is described in [Training Specific Usage](#training-specific-usage).

This project was trained using the [DSD100](https://sigsep.github.io/datasets/dsd100.html) and [AAM](https://zenodo.org/records/5794629) datasets.


## Setup

1. Clone the repo into a local folder of your choosing.
2. Create a virtual environment and activate it.
3. With the virtual environment active, install the necessary dependencies using pip: `pip install -r requirements.txt`

## General Usage

1. Navigate to `/src/config/constants.py` and update the production constant paths with folders of your own naming.

   If you have trained and saved multiple models, you can select which ones will be used by setting the path to said models as `SEPARATION_MODEL_PATH` and `TRANSCRIPTION_MODEL_PATH`

   **Important:** The `PRODUCTION_INPUT_FOLDER_PATH` and `PRODUCTION_OUTPUT_FOLDER_PATH` constants need to be nested one layer below `PRODUCTION_FOLDER_BASE_PATH`.

   E.g.:
   
       ├── /test_audio 			<-- PRODUCTION_FOLDER_BASE_PATH
              │   ├── /input       		<-- PRODUCTION_INPUT_FOLDER_PATH
              │   ├── /output  		<-- PRODUCTION_OUTPUT_FOLDER_PATH


 
3. Place as many audio files as you'd like in the `/input` folder you specified in the step above.
**Important:** At the current development phase, these audio files **must** be in the .wav format.
   
5. With your virtual environment active, navigate to `/src` in your terminal and run `main.py`
6. Enter **no** in your terminal to process your audio with the pre-existing models.
7. Your audio files will be individually processed, passed through both the separation and the transcription model, and the isolated bass-guitar midi will be saved in your specified `/output` folder.

====================================================================================

## Training Specific Usage 

**Important:** This section is not relevant unless you intend to train/test new model versions yourself.

If you'd like to train/test new models, you need to enter **yes** into the terminal when prompted after running main.py -- **However**, to do so, you will need to have datasets in the proper folder structure for tabibi to train/test your models. 

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
- The fourierparameters for all spectrogram transformations can be set in `fourierparameters.json`
- The hyperparameters for each model's training can be set in `hyperparameters_separation.json` and `hyperparameters_transcription.json`

### Testing
The project contains basic tests concerning what I percieve to be the most likely points of failure --> The data transformers that convert data into differing data formats. New tests are continually being added.
They can all be run together by navigating to `/src` in your terminal and running `pytest`.

The tests themselves can be found in the code within the `__tests__` folders placed adjacent to the code they are intended to test, together with a `test_data` folder. If you want to run the test suite locally, you need to add audio and midi data to each of the `test_data` folders to provide the test suite with datapoints it can use to test the transformers. 


### Pipelines
#### Separation Pipeline
![Screenshot from 2024-02-27 16-30-00](https://github.com/LiamGitGoing/tabibi/assets/41804800/00a764a7-aa87-4a2d-939f-369787cf3cbc)

- The dataset is split into train/test groups.
- All mixed audio files and all bass audio files are converted to spectrograms and stored in a dictionary where the indices allign between mixed audio and its associated bass audio.
- The spectrograms are then normalized and truncated to identical lengths across all tracks in the dictionary.
- This dictionary is then saved as a numpy recarray.

#### Transcription Pipeline
![Screenshot from 2024-02-27 15-21-40](https://github.com/LiamGitGoing/tabibi/assets/41804800/9fb554bf-67e1-47e7-8c37-99f01b3098f2)

- The dataset is split into train/test groups.
- All bass audio files are converted to spectrograms. All bass midi files are transformed into piano rolls, resized to match their associated bass audio spectrograms, and then converted to spectrograms themselves. All files are then stored in a dictionary where the indices between bass audio and its associated bass midi allign.
- The bass spectrograms are then normalized (this is not necessary for the bass midi spectrograms, as they were already normalized as part of the piano_roll conversions) and everything is truncated to identical lengths across all tracks in the dictionary.
- This dictionary is then saved as a numpy recarray.

#### Training/Testing Pipelines
![Screenshot from 2024-02-27 15-21-54](https://github.com/LiamGitGoing/tabibi/assets/41804800/effdb5f3-6072-48fd-9644-a96e4f9fd943)

- The training/testing dictionaries are loaded (either for the separation- or the transcription model, both pipelines are the same).
- All data is then truncated once more, to make sure the data matches across dictionaries as well (i.e. to make sure all datapoints across training/testing have the same length too, and not just within a singular dictionary).
- The model runs for the specified number of epochs, outputs loss values for evaluation, saves audio samples and spectrogram samples post-training for manual assessment, and is saved as a now pre-trained model.

#### Post-Training Production Pipeline
![Screenshot from 2024-02-27 15-22-06](https://github.com/LiamGitGoing/tabibi/assets/41804800/67cfbb3a-28de-42be-b5b0-d12044edac83)

The production pipeline is on a loop and runs once per audio track provided in the `/input` folder.
- An audio track is transformed to a spectrogram, structured into a dictionary, normalized, and saved as a numpy recarray.
- It is then converted to a pytorch tensor and passed through the pre-trained separation model.
- The separation model's output is passed through the pre-trained transcription model.
- The resulting spectrogram is then processed back from a spectrogram to a piano roll, and then to a midi file. It is then saved and can be found in the `/output` folder.

### Project Organization
------------

    ├── LICENSE
    ├── README.md          <-- The top-level README for developers using this project.
    │
    ├── requirements.txt   <-- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │ 
    ├── src                
    │   │
    │   ├── __helpers__           <-- Utility helper functions
    │   │
    │   ├── config                <-- File-path constants and fourier-/hyper-parameters
    │   │
    │   ├── data                  <-- Processed and raw data used to train/test new models
    │   │
    │   ├── data_manipulation     <-- The processes and helper functions needed to prepare data before 
    │   │                             passing it to a model 
    │   │
    │   ├── models                <-- Model definitions, test/train functions, managers that control entire 
    │   │                             train/test flow, and saved pre-trained models
    │   │
    │   ├── scripts               <-- Scripts used to prepare raw datasets for processing
    │   │
    │   ├── transformers          <-- Functions used to transform data from one datatype format to another
    │   │
    │   ├── visualization         <-- A collection of functions used to visualize input/output, manually 
    │   │                             assess model efficiency, and simulate pipelines for manual evaluation 
    │   │                             of data transformations
    │   │
    │   ├── main.py               <--- The main code to be run, differentiates between training and production 
    │   │                              execution types and deligates accordingly


------------


### To-Do
- Both models currently output poor results. Areas that need improvement are generalization and possibly model complexity (e.g. tracking audio phases, etc.)
- Transcription model outputs are currently lumped into the first few seconds - model fine-tuning should solve this.
- Introduce distortion to the transcription model training to generalize better.
- Superior error handling needs to be implemented.
- Implement batch-sizes in case larger training datasets need to be used.
- Revisit truncation/padding and find a solution that allows for using more time-steps of datapoints for training.
- Expand the testing suite for better coverage, especially regarding file-paths.
- Find a better solution to deal with all the paths needed.
  
