# MusicGeneration

## Sequence Extraction

this porject provide following representation for midi format files:

* MIDI-like
* REMI
* MuMIDI
* CP(to do)

 you can transform midi into sequence data as following steps

```shell
cd mg/model/utils
python preprocess_{MIDI_like,REMI,MuMIDI,CP}.py {midi_folder} {result_output_foler} {int:num_workers, 10 is recommand}
```



***

Please see [Installation Instructions](https://github.com/SJTMusicTeam/MusicGeneration/wiki/Installation-Instructions)

