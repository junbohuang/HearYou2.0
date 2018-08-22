# HearYou2.0

```
python HearYou2.0.py -c configs/<model_to_run>.json
```

IDEAS:
1. compare different data types (improvised/ scripted/ full data)
2. compare features (personalized/ non-perrsonalized)

### Results

| data type         | Scripted| Improvised | Both |
|:-----------------:|:-------:|:----------:|:----:|
| text              | 50%     |    50%     |  50% |
| speech            | 50%     |    50%     |  50% |
| mocap             | 50%     |    50%     |  50% |
| text+speech       | 50%     |    50%     |  50% |
| text+speech+mocap | 50%     |    50%     |  50% |

| feature type      | 34      | MFCC&DD    | 34&DD|
|:-----------------:|:-------:|:----------:|:----:|
| text              | 50%     |    50%     |  50% |
| speech            | 50%     |    50%     |  50% |
| mocap             | 50%     |    50%     |  50% |
| text+speech       | 50%     |    50%     |  50% |
| text+speech+mocap | 50%     |    50%     |  50% |