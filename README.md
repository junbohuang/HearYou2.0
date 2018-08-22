# HearYou2.0

Seminar Project for Multimodal Corpus Linguistics seminar in SS18

## Preacquisitions
- python 3.5+
- keras 2
- tf 1.8

## Supported Features

- [x] weighted class weights, attention wrapper,
- [x] train on different data type (scrripted/ improvised/ both)
- [x] train on different speech features (static/ dynamics - deltas and deltasdeltas)
- [x] train on different modalities (speech/ text/ motion/ all/ configure your own combinations in *./configs*)
- [x] train on different ANN architectures (convs/ rnns/ configure your own models in *./models*)
- [x] speech speaker independent configuration

## How to run

```
python HearYou2.0.py -c configs/<model_to_run>.json
```
run all configurations stored in *./configs* if *-c flag* is not given

## Results

| data type         | Scripted| Improvised | Both |
|:-----------------:|:-------:|:----------:|:----:|
| text              | xx%     |    xx%     |  xx% |
| speech            | xx%     |    xx%     |  51% |
| mocap             | xx%     |    xx%     |  xx% |
| text+speech       | xx%     |    xx%     |  xx% |
| speech+mocap      | xx%     |    xx%     |  54.12% |
| text+speech+mocap | xx%     |    xx%     |  xx% |

| feature type      | 34      | MFCC&DD    | 34&DD|
|:-----------------:|:-------:|:----------:|:----:|
| text              | xx%     |    xx%     |  xx% |
| speech            | xx%     |    xx%     |  xx% |
| mocap             | xx%     |    xx%     |  xx% |
| text+speech       | xx%     |    xx%     |  xx% |
| text+speech+mocap | xx%     |    xx%     |  xx% |


## References

IEMOCAP data
- https://sail.usc.edu/iemocap/iemocap_release.htm

Feature Extraction Library
- https://github.com/tyiannak/pyAudioAnalysis/blob/master/pyAudioAnalysis/audioFeatureExtraction.py

Deltas & DeltasDeltas
- https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py

Conceptor
- https://github.com/littleowen/Conceptor

Multimodality
- https://github.com/Samarth-Tripathi/IEMOCAP-Emotion-Detection
