from src.models.audio_separation.separation_manager import separation_manager
from src.models.tab_transcription.transcription_manager import transcription_manager


def main():
    separation_manager()
    transcription_manager()


if __name__ == "__main__":
    main()
