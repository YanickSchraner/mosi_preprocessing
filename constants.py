import os.path
from typing import Optional

# path to the SDK folder
SDK_PATH: Optional[str] = os.path.abspath('external/CMU-MultimodalSDK')

# path to the folder where you want to store data
DATA_PATH: Optional[str] = os.path.abspath('data/CMU_MOSI')

# path to a pretrained word embedding file
# Download from https://nlp.stanford.edu/projects/glove/ and unzip
WORD_EMB_PATH: Optional[str] = './data/glove.6B.300d.txt'

# path to loaded word embedding matrix and corresponding word2id mapping
CACHE_PATH: Optional[str] = './data/embedding_and_mapping.pt'

# path to the openface executables
OPENFACE_FEATURE_EXTRACTION = './external/OpenFace/build/bin/FeatureExtraction'
OPENFACE_LANDMARK_VID = './external/OpenFace/build/bin/FaceLandmarkVid'

# path to COVAREP
COVAREP_PATH = './external/covarep'