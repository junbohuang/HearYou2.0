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
|feature type       | MFCC       | 34   |
|:-----------------:|:----------:|:----:|
| speech            |    57%     |  51% |
| speech+mocap      |    73%     |  74% |
| text+speech       |    55%     |  50% |
| text+speech+mocap |    76%     |  69% |

### scripted data
|feature type       | MFCC       | 34   |
|:-----------------:|:----------:|:----:|
| speech            |    53%     |  51% |
| speech+mocap      |    41%     |  47% |
| text+speech       |    43%     |  51% |
| text+speech+mocap |    25%     |  38% |

### complete data
|feature type       | MFCC       | 34   |
|:-----------------:|:----------:|:----:|
| speech            |    50%     |  50% |
| speech+mocap      |    61%     |  52% |
| text+speech       |    50%     |  52% |
| text+speech+mocap |    60%     |  50% |


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

