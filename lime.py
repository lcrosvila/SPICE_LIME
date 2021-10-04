# coding: utf-8
import librosa
from librosa import display as librosadisplay
import numpy as np

import skimage.io 
import skimage.segmentation
import sklearn
from sklearn.linear_model import LinearRegression

import copy

EXPECTED_SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768.0

A4 = 440
C0 = A4 * pow(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
note_names_octave = [note+str(i) for i in range(11) for note in note_names]

class LIME:
    def __init__(self, config, model):
        self.model_type = config["model_type"]
        self.model = model

        self.num_segments = config["num_segments"]
        self.num_perturb = config["num_perturb"]
        self.kernel_width = config["kernel_width"]
        self.num_top_features = config["num_top_features"]
        
        self.pred_precision = config["pred_precision"]

        self.model_load_path = config["model_load_path"]
        self.original_audio_path = config["audio_path"]
        self.note_names = note_names
        self.note_names_octave = note_names_octave
        self.C0 = C0
        self.EXPECTED_SAMPLE_RATE = EXPECTED_SAMPLE_RATE
        self.MAX_ABS_INT16 = MAX_ABS_INT16

    def perturb_signal(self, x, perturbation, segments):
        img = librosa.stft(x, n_fft=2048)
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 0.063096 # -12dB
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image*mask
        perturbed_signal = librosa.istft(perturbed_image)
        perturbed_signal = perturbed_signal
        return {"img": np.abs(perturbed_image), "audio": perturbed_signal}

    def load_model(self):
        self.model.model

    def get_stft(self, audio_samples):
        return {"img": np.abs(librosa.stft(audio_samples,  n_fft=2048)), "audio": audio_samples}

    def get_perturbations(self, Xi):
        superpixels = skimage.segmentation.slic(Xi["img"], n_segments=self.num_segments, compactness=.1)
        num_superpixels = np.unique(superpixels).shape[0]
        perturbations = np.random.binomial(1, 0.5, size=(self.num_perturb, num_superpixels))
        predictions = []

        for pert in perturbations:
            perturbed_img = self.perturb_signal(Xi["audio"], pert, superpixels)
            _, _, pred = self.model.get_predictions(perturbed_img["audio"])
            predictions.append(pred)
        
        if "spice" in self.model_type:
            predictions = np.array(predictions)[:,np.newaxis,:]
            
        original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
        distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()

        weights = np.sqrt(np.exp(-(distances**2)/self.kernel_width**2)) #Kernel function

        return superpixels, perturbations, predictions, weights
    
    def find_nearest(array, value):
        return [[value] if abs(p-value) < self.pred_precision else [0] for p in array]
    
    def get_top_bottom(self, audio_samples, name_class_to_explain):
        Xi = self.get_stft(audio_samples)
        superpixels, perturbations, predictions, weights = self.get_perturbations(Xi)
        simpler_model = LinearRegression()
        
        if "spice" in self.model_type:
            class_to_explain = self.note_names_octave.index(name_class_to_explain)
            simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
        else:
            simpler_model.fit(X=perturbations, y=find_nearest(predictions,name_class_to_explain), sample_weight=weights)  
        
        coeff = simpler_model.coef_[0]
        top_features = np.argsort(coeff)[-self.num_top_features:] 

        mask = np.zeros(np.unique(superpixels).shape[0]) 
        mask[top_features]= True #Activate top superpixels

        X_top = self.perturb_signal(Xi["audio"], mask,superpixels)
        X_bottom = self.perturb_signal(Xi["audio"], np.ones(mask.shape)-mask,superpixels)

        return X_top, X_bottom
