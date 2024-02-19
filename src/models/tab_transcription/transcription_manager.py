from src.data_manipulation.transformers.transform_data import transform_data


def transcription_manager():
    # Transform training/testing data for tab transcription
    transform_data(flag="tab transcription")
