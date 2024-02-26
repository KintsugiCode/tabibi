import json
import os
import pretty_midi
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

fourierparameters_path = os.path.join(dir_path, "../config/fourierparameters.json")

with open(fourierparameters_path) as fourierparameters_file:
    fourierparameters = json.load(fourierparameters_file)


def piano_roll_to_midi(
    piano_roll, output_file_path, flag, mix_names=None, program=34, tag=""
):

    try:
        track_counter = 0

        # Calculate total samples
        total_samples = (
            fourierparameters["track_seconds_considered"]
            * fourierparameters["sample_rate"]
        )

        # Create a time array
        time_array = np.linspace(
            0, fourierparameters["track_seconds_considered"], total_samples
        )

        for track in range(piano_roll.shape[0]):
            track_counter += 1

            # Create a PrettyMIDI object
            midi = pretty_midi.PrettyMIDI()

            # Create an instrument instance for a piano instrument
            instrument = pretty_midi.Instrument(program=program)

            # Iterate over all possible pitch values
            for pitch in range(piano_roll.shape[1]):
                # Retrieve the times at which this pitch is being played
                pitch_events = piano_roll[track, pitch, :]
                scale_factor = int(len(time_array) / len(pitch_events))
                note_start = None
                is_sustained = False
                time = None
                # Iterate over all time steps
                for index, value in enumerate(pitch_events):
                    time = index * scale_factor
                    # If a note is being played at this time step
                    if value > 0:
                        is_sustained = True
                        if note_start is None:
                            note_start = time_array[time].item()
                    # Check if the note has ended
                    elif value == 0 and is_sustained:
                        end_time = time_array[time]
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=pitch,
                            start=note_start,
                            end=end_time,
                        )
                        instrument.notes.append(note)
                        note_start = None
                        is_sustained = False

                # If a note was played but wasn't ended (last event was a non-zero)
                if is_sustained and note_start:
                    end_time = time_array[time]
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=pitch,
                        start=note_start,
                        end=end_time,
                    )
                    instrument.notes.append(note)

            # Add the instrument to the PrettyMIDI object
            midi.instruments.append(instrument)

            if flag == "production":
                file_path_without_extension = os.path.splitext(output_file_path)[0]
                output_file_as_midi = file_path_without_extension + ".mid"
                midi.write(f"{output_file_as_midi}")
            elif flag == "development":
                midi.write(
                    f"{output_file_path}/{track_counter}-{tag}-{mix_names[track]}"
                )

    except Exception as e:
        raise Exception(f"Exception occurred: {e}")
