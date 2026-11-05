# Third-party models and code in ClinicalWhisper

ClinicalWhisper (MIT) ships with, or downloads as add-ons, the models below.
Licences are as stated on each model's page at the time of writing.

| Component | Use in the app | Licence |
|---|---|---|
| OpenMOSS-Team/MOSS-Transcribe-Diarize | transcription and speaker labels | Apache-2.0 |
| OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1 | masking identifiers (English) | Apache-2.0 |
| OpenMed/privacy-filter-multilingual-v2-mlx-8bit (Languages add-on) | masking identifiers (other languages) | Apache-2.0 |
| Wespeaker/wespeaker-voxceleb-resnet34-LM | voice embeddings for speaker linking and checks | CC BY 4.0 |
| pyannote/segmentation-3.0 (Hervé Bredin, pyannote) | crosstalk detection in group discussions | MIT |
| mlx-community/Meta-Llama-3-8B-Instruct-4bit (Scoring add-on) | research clinical scores; second name check | Llama 3 Community Licence |
| KintsugiHealth/dam (Kintsugi add-on) | research voice model | Apache-2.0 |
| Praat via praat-parselmouth (Praat add-on, separate) | voice measures | GPL-3.0 (kept out of the MIT app) |

WeSpeaker attribution (CC BY 4.0): voice embeddings use WeSpeaker's ResNet34
model trained on VoxCeleb (wenet-e2e/wespeaker), unchanged.

## pyannote

overlap_detector.py rebuilds the PyanNet network of pyannote.audio in plain
PyTorch to run the published pyannote/segmentation-3.0 weights unchanged
(model card licence: MIT). The network definition follows pyannote.audio:

```
MIT License

Copyright (c) 2020 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The sinc filterbank in overlap_detector.py follows asteroid-filterbanks
(ParamSincFB), used by pyannote.audio:

```
MIT License

Copyright (c) 2019 Pariente Manuel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
