import json
import os
import torch

from src.config.constants import SEPARATION_MODEL_PATH, TRANSCRIPTION_MODEL_PATH
from src.models.audio_separation.gru.gru_separation import GRU_Separation
from src.models.tab_transcription.gru.gru_transcription import GRU_Transcription

dir_path = os.path.dirname(os.path.realpath(__file__))
hyperparameters_separation_path = os.path.join(
    dir_path, "../config/hyperparameters_separation.json"
)
hyperparameters_transcription_path = os.path.join(
    dir_path, "../config/hyperparameters_transcription.json"
)
with open(hyperparameters_separation_path) as hyperparameters_file:
    hyperparameters_separation = json.load(hyperparameters_file)
with open(hyperparameters_transcription_path) as hyperparameters_file:
    hyperparameters_transcription = json.load(hyperparameters_file)


def use_models_on_audio(input_data):

    # Convert to PyTorch Tensor -- Individual conversion before grouped conversion is faster for large datasets
    x_separation = torch.stack([torch.tensor(x) for x in input_data["x"]])
    y_separation = torch.stack([torch.tensor(y) for y in input_data["y"]])

    x_separation = x_separation.float()
    y_separation = y_separation.float()

    # Load trained separation model
    separation_model = GRU_Separation(
        input_size=x_separation.shape[2],
        hidden_dim=hyperparameters_separation["hidden_dim"],
        n_layers=hyperparameters_separation["n_layers"],
        output_size=y_separation.shape[2],
        dropout_rate=hyperparameters_separation["dropout_rate"],
    )
    separation_model.load_state_dict(torch.load(SEPARATION_MODEL_PATH))

    # Make sure the model is in evaluation mode
    separation_model.eval()

    # Apply the model to the input audio
    with torch.no_grad():
        separation_output_tensor, _ = separation_model(x_separation)

    transcription_input_tensor = separation_output_tensor

    # Load trained transcription model
    transcription_model = GRU_Transcription(
        input_size=transcription_input_tensor.shape[2],
        hidden_dim=hyperparameters_separation["hidden_dim"],
        n_layers=hyperparameters_separation["n_layers"],
        output_size=transcription_input_tensor.shape[2],
        dropout_rate=hyperparameters_separation["dropout_rate"],
    )
    transcription_model.load_state_dict(torch.load(TRANSCRIPTION_MODEL_PATH))

    # Make sure the model is in evaluation mode
    transcription_model.eval()

    # Apply the model to the input audio
    with torch.no_grad():
        transcription_output_tensor, _ = transcription_model(separation_output_tensor)

    transcription_output_array = transcription_output_tensor.detach().cpu().numpy()

    return transcription_output_array
