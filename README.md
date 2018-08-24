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
### static speech feature
|data type(all feat)| Scripted| Improvised | Both |
|:-----------------:|:-------:|:----------:|:----:|
| text              | xx%     |    xx%     |  61% |
| speech            | xx%     |    34%     |  51% |
| mocap             | xx%     |    xx%     |  45% |
| text+speech       | xx%     |    xx%     |  67% |
| text+speech+mocap | xx%     |    xx%     |  70% |

### dynamic speech feature (with 1st/2nd derivative)
### improvised data
|feature type     | MFCC&DD    | 34&DD|
|:-----------------:|:----------:|:----:|
| speech            |    57%     |  51% |
| speech+mocap      |    73%     |  xx% |
| text+speech       |    55%     |  xx% |
| text+speech+mocap |    76%     |  xx% |

### scripted data
|feature type    | MFCC&DD    | 34&DD|
|:-----------------:|:----------:|:----:|
| speech            |    53%     |  xx% |
| speech+mocap      |    41%     |  xx% |
| text+speech       |    xx%     |  xx% |
| text+speech+mocap |    25%     |  xx% |

### complete data
|feature type      | MFCC&DD    | 34&DD|
|:-----------------:|:----------:|:----:|
| speech            |    xx%     |  xx% |
| speech+mocap      |    xx%     |  xx% |
| text+speech       |    50%     |  xx% |
| text+speech+mocap |    xx%     |  xx% |


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

