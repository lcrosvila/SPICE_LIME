SPICE_LIME
----------

# Introduction

This repository contains the implementation of the LIME algorithm [1] applied to SPICE [2] pitch estimation model in order to explain the model's predictions. Final project for the DT2470 Music Informatics course.

# Requirements

```
pip install -r requirements.txt
```

# Contents

1. `run_slime.py ´: generates a results folder containing a dictionary with `{"filename": [], "prediction": [], "true": []´. It also saves the audio samples of the original, top and bottom signals in both `.wav´ and `.npy´ formats.
2. `SPICE+LIME.ipynb´: interactive notebook that can be run in google colab that allows running the algorithm with any audio sample with the option of recording new ones.

# Dataset and Results

The dataset used for the experiments was the NSynth Dataset [3].
Some results can be found in: [Results folder](https://drive.google.com/drive/folders/1FJkoYyqEGq1x0ITi3wldRPAUGe_u_LeD?usp=sharing)

# References
[1] Ribeiro, Marco Tulio, Sameer Singh, and Carlos Guestrin. “Why should I trust you?: Explaining the predictions of any classifier.” Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. ACM (2016).
[2] Gfeller, Beat, et al. "SPICE: Self-supervised pitch estimation." IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020).
[3] Jesse Engel, Cinjon Resnick, Adam Roberts, Sander Dieleman, Douglas Eck, Karen Simonyan, and Mohammad Norouzi. "Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders." (2017).
