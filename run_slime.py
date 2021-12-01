import os
from glob import glob

from tqdm import tqdm

from lime import LIME
from spice import SPICE

import numpy as np
import json
import scipy

if os.path.exists("results_data.json"):
    with open('results_data.json') as json_file:
        dictionary_data = json.load(json_file)
else:
    dictionary_data = {"filename": [], "prediction": [], "true": []}

att = -6 # [0, -6, -12]

if att != 0:
  att_db = 10**(att/20)
else:
  att_db = 0.0

config = {"model_load_path": "https://tfhub.dev/google/spice/2",
          "audio_path": "c-scale.wav",
          "num_segments": 100,
          "num_perturb": 150,
          "kernel_width": 0.25,
          "model_type": "spice",
          "pred_precision": None,
          "attenuation": att_db,
          "num_top_features": 4}

spice = SPICE(config)
spice.get_model()

lime = LIME(config, spice)

DATA_PATH = 'nsynth-valid/'

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
OCTAVES = list(range(11))
NOTES_IN_OCTAVE = len(NOTES)

def number_to_note(number: int) -> tuple:
    octave = number // NOTES_IN_OCTAVE - 1
    note = NOTES[number % NOTES_IN_OCTAVE]
    return note+str(octave)

wav_files = glob(DATA_PATH + '/**/*.wav', recursive=True)

for file in tqdm(wav_files[:30]):
    if file in dictionary_data["filename"]:
        save = False
    else:
        save = True
    
    if save:
        dictionary_data["filename"].append(file)
    
    audio_samples, sample_rate = spice.load_audio(file)
    pitch_outputs, confidence_outputs, mean_outputs = spice.get_predictions(audio_samples)
    predicted_notes = spice.get_notes(pitch_outputs, confidence_outputs)

    true_NOTE = number_to_note(int(file.split('/')[-1].split('-')[1]))
    if save:
        dictionary_data["true"].append(true_NOTE)

    if len(predicted_notes) > 1:
        pred_NOTE = max(set(predicted_notes), key=predicted_notes.count)
    else:
        pred_NOTE = predicted_notes[0]
    
    if pred_NOTE == 'Rest':
        if save:
            dictionary_data["prediction"].append([pred_NOTE, None])
        continue

    if not os.path.exists("results/"+file.split("/")[-1][:-4]):
        os.makedirs("results/"+file.split("/")[-1][:-4])
    
    X_top, X_bottom = lime.get_top_bottom(audio_samples, pred_NOTE)
    scipy.io.wavfile.write("results/"+file.split("/")[-1][:-4]+"/top.wav", sample_rate, X_top["audio"])
    scipy.io.wavfile.write("results/"+file.split("/")[-1][:-4]+"/bottom.wav", sample_rate, X_bottom["audio"])
    np.save("results/"+file.split("/")[-1][:-4]+"/top_array", X_top["audio"])
    np.save("results/"+file.split("/")[-1][:-4]+"/bottom_array", X_bottom["audio"])

    pitch_outputs, confidence_outputs, mean_outputs = spice.get_predictions(X_bottom["audio"])
    predicted_notes = spice.get_notes(pitch_outputs, confidence_outputs)

    if len(predicted_notes) > 1:
        new_pred_NOTE = max(set(predicted_notes), key=predicted_notes.count)
    
    else:
        new_pred_NOTE = predicted_notes[0]

    if new_pred_NOTE == 'Rest':
        if save:
            dictionary_data["prediction"].append([pred_NOTE, new_pred_NOTE])
        continue

    if save:
        dictionary_data["prediction"].append([pred_NOTE, new_pred_NOTE])
    
    second_X_top, second_X_bottom = lime.get_top_bottom(X_bottom["audio"], new_pred_NOTE)
    scipy.io.wavfile.write("results/"+file.split("/")[-1][:-4]+"/second_top.wav", sample_rate, second_X_top["audio"])
    scipy.io.wavfile.write("results/"+file.split("/")[-1][:-4]+"/second_bottom.wav", sample_rate, second_X_bottom["audio"])
    np.save("results/"+file.split("/")[-1][:-4]+"/second_top_array", second_X_top["audio"])
    np.save("results/"+file.split("/")[-1][:-4]+"/second_bottom_array", second_X_bottom["audio"])


a_file = open("results_data.json", "w")
json.dump(dictionary_data, a_file)
a_file.close()
