# loop through all the videos in the directory
# shellcheck disable=SC2066
for f in ./data/CMU_MOSI/Raw/Video/Aligned/*.mp4; do
    # run the openface script
    ./external/OpenFace/build/bin/FeatureExtraction -2Dfp -aus -f $f -out_dir ./data/CMU_MOSI/Raw/Video/Aligned/
done
