# HearYou2.0

```
python HearYou2.0.py -c configs/<model_to_run>.json
```
## Supported Features

- [x] train on different data type (scrripted/ improvised/ both)
- [x] train on different speech features (static/ dynamics - deltas and deltasdeltas)
- [x] train on different modalities (speech/ text/ motion/ all/ configure your own combinations in *./configs*)
- [x] train on different ANN architectures (convs/ rnns/ configure your own models in *./configs*)
- [ ] speech speaker independent configuration with json(90% done)
- [ ] confusion matrix video visualization showing training process
