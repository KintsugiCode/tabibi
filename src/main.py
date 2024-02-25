from src.models.audio_separation.separation_manager import separation_manager
from src.models.tab_transcription.transcription_manager import transcription_manager


def main():
    print("@@@@@@ SEPARATION PIPELINE START @@@@@@")
    separation_manager()
    print("@@@@@@ TRANSCRIPTION PIPELINE START @@@@@@")
    transcription_manager()


if __name__ == "__main__":
    main()
