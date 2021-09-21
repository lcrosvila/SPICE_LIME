# coding: utf-8
import os

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import matplotlib.pyplot as plt

import logging
import math
import statistics
import sys

from IPython.display import Audio, Javascript
from scipy.io import wavfile

from base64 import b64decode

from pydub import AudioSegment

EXPECTED_SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768.0

A4 = 440
C0 = A4 * pow(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class SPICE:
    def __init__(self, config):
        self.model_load_path = config["model_load_path"]
        self.original_audio_path = config["audio_path"]
        self.note_names = note_names
        self.C0 = C0
        self.EXPECTED_SAMPLE_RATE = EXPECTED_SAMPLE_RATE
        self.MAX_ABS_INT16 = MAX_ABS_INT16

    def convert_audio_for_model(self, audio_path):
        head, tail = os.path.split(audio_path)
        output_file = os.path.join(head, "converted_"+tail)
        if not os.path.exists(output_file):
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(self.EXPECTED_SAMPLE_RATE).set_channels(1)
            audio.export(output_file, format="wav")
        return output_file

    def load_audio(self, audio_path):
        current_file = self.convert_audio_for_model(audio_path)
        sample_rate, audio_samples = wavfile.read(
            current_file, 'rb')

        audio_samples = audio_samples / float(self.MAX_ABS_INT16)

        if audio_path == self.original_audio_path:
            self.original_samples = audio_samples
        
        return audio_samples, sample_rate

    def get_model(self):
        self.model = hub.load("https://tfhub.dev/google/spice/2")
        self.model.signatures["serving_default"]

    def output2hz(self, pitch_output):
        # Constants taken from https://tfhub.dev/google/spice/2
        PT_OFFSET = 25.58
        PT_SLOPE = 63.07
        FMIN = 10.0
        BINS_PER_OCTAVE = 12.0
        cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
        return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

    def hz2note(self, freq, uncertainty):
        h = round(
            statistics.mean([
                12 * math.log2(freq / self.C0)
            ]))
        octave = h // 12
        n = h % 12
        note_prediction = [0] * len(self.note_names)
        note_prediction[n] = 1.0 - uncertainty.numpy()
        return note_prediction

    def hz2offset(self, freq):
        # This measures the quantization error for a single note.
        if freq == 0:  # Rests always have zero error.
            return None
        # Quantized note.
        h = round(12 * math.log2(freq / self.C0))
        return 12 * math.log2(freq / self.C0) - h

    def quantize_predictions(self, group, ideal_offset):
        # Group values are either 0, or a pitch in Hz.
        non_zero_values = [v for v in group if v != 0]
        zero_values_count = len(group) - len(non_zero_values)

        # Create a rest if 80% is silent, otherwise create a note.
        if zero_values_count > 0.8 * len(group):
            # Interpret as a rest. Count each dropped note as an error, weighted a bit
            # worse than a badly sung note (which would 'cost' 0.5).
            return 0.51 * len(non_zero_values), "Rest"
        else:
            # Interpret as note, estimating as mean of non-rest predictions.
            h = round(
                statistics.mean([
                    12 * math.log2(freq / self.C0) - ideal_offset for freq in non_zero_values
                ]))
            octave = h // 12
            n = h % 12
            note = self.note_names[n] + str(octave)
            # Quantization error is the total difference from the quantized note.
            error = sum([
                abs(12 * math.log2(freq / self.C0) - ideal_offset - h)
                for freq in non_zero_values
            ])
            return error, note

    def get_quantization_and_error(self, pitch_outputs_and_rests, predictions_per_eighth,
                                   prediction_start_offset, ideal_offset):
        # Apply the start offset - we can just add the offset as rests.
        pitch_outputs_and_rests = [0] * prediction_start_offset + \
            pitch_outputs_and_rests
        # Collect the predictions for each note (or rest).
        groups = [
            pitch_outputs_and_rests[i:i + predictions_per_eighth]
            for i in range(0, len(pitch_outputs_and_rests), predictions_per_eighth)
        ]

        quantization_error = 0

        notes_and_rests = []
        for group in groups:
            error, note_or_rest = self.quantize_predictions(
                group, ideal_offset)
            quantization_error += error
            notes_and_rests.append(note_or_rest)

        return quantization_error, notes_and_rests

    def output2note(self, pitch_output, uncertainty_outputs):
        hz_outputs = [self.output2hz(p) for p in pitch_output]
        note_outputs = [self.hz2note(hz, u) for hz, u in zip(
            hz_outputs, uncertainty_outputs)]
        return note_outputs

    def mean_prediction(self, model_output):
        note_outputs = self.output2note(
            model_output["pitch"], model_output["uncertainty"])
        mean_outputs = np.mean(np.array(note_outputs), axis=0)
        return mean_outputs

    def get_predictions(self, audio_samples):
        model_output = self.model.signatures["serving_default"](
            tf.constant(audio_samples, tf.float32))

        pitch_outputs = model_output["pitch"]
        confidence_outputs = 1.0 - model_output["uncertainty"]

        mean_outputs = self.mean_prediction(model_output)
        return pitch_outputs, confidence_outputs, mean_outputs

    def get_notes(self, pitch_outputs, confidence_outputs):
        confidence_outputs = list(confidence_outputs)
        pitch_outputs = [float(x) for x in pitch_outputs]

        indices = range(len(pitch_outputs))

        '''
        (this is needed to plot confidence scores...)

        confident_pitch_outputs = [(i, p)
                                   for i, p, c in zip(indices, pitch_outputs, confidence_outputs) if c >= 0.9]
        
        confident_pitch_outputs_x, confident_pitch_outputs_y = zip(
            *confident_pitch_outputs)

        confident_pitch_values_hz = [self.output2hz(
            p) for p in confident_pitch_outputs_y]
        '''


        pitch_outputs_and_rests = [
            self.output2hz(p) if c >= 0.1 else 0
            for i, p, c in zip(indices, pitch_outputs, confidence_outputs)
        ]

        offsets = [self.hz2offset(p)
                   for p in pitch_outputs_and_rests if p != 0]
        ideal_offset = statistics.mean(offsets)

        best_error = float("inf")
        best_notes_and_rests = None

        for predictions_per_note in range(20, 65, 1):
            for prediction_start_offset in range(predictions_per_note):

                error, notes_and_rests = self.get_quantization_and_error(
                    pitch_outputs_and_rests, predictions_per_note,
                    prediction_start_offset, ideal_offset)

                if error < best_error:
                    best_error = error
                    best_notes_and_rests = notes_and_rests

        if all(e == 'Rest' for e in best_notes_and_rests):
            return 'Rest'
        # At this point, best_notes_and_rests contains the best quantization.
        # Since we don't need to have rests at the beginning, let's remove these:
        while best_notes_and_rests[0] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[1:]
        # Also remove silence at the end.
        while best_notes_and_rests[-1] == 'Rest':
            best_notes_and_rests = best_notes_and_rests[:-1]

        return best_notes_and_rests
