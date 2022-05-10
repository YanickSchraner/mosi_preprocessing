1. Install CMU multimodal SDK https://github.com/A2Zadeh/CMU-MultimodalSDK:
```
git clone git@github.com:A2Zadeh/CMU-MultimodalSDK.git
```
3. Download and unzip the data at: http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip
4. Download glove embeddings: https://nlp.stanford.edu/projects/glove/
5. Set constants in constants.py
6. Install OpenFace: https://github.com/TadasBaltrusaitis/OpenFace/wiki#installation
7. Download Covarep: https://github.com/covarep/covarep/releases
8. Install matlab 2016b
9. Clone P2FA for Python 3.x https://github.com/jaekookang/p2fa_py3 to external/p2fa
10. Register and download HTK source code: http://htk.eng.cam.ac.uk/ and install it (needs Visual Studio with C++ distribution and Perl)
11. Make python virtual environment: python3 -m venv venv
12. Activate the virtual environment: source venv/bin/activate
13. Install python requirements: pip install -r requirements.txt
14. Run the code: python main.py
15. Run OpenFace feature extraction: `sh openface.sh`
16. Run Matlab 2016b, open the Covarep dir
    1. Run startup.m
    2. Set input_dir to the directory containing the audio files like `input_dir='/home/yanick/Code/mosi_preprocessing/data/CMU_MOSI/Raw/Audio/WAV_16000/Aligned/'`
    3. Run `COVAREP_feature_extraction(input_dir)` in Matlab command window