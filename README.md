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

## the structure of representation

the API of representation(like MIDI-like, REMI, MuMIDI)

it should be encapsulate as a EventSeq(or REMI_EventSeq or MuMIDI_EventSeq in sequence.py/REMI.py/MuMIDI.py)

* extract_events(midi_path)->event_seq
* to_array(event_seq)->np.array()
* from_array(np.array)->event_seq() for decoding
* write_midi(event_seq,output_path)
* feats_range()->dict=<feature_name, index>
* dims_feats()->dict=<index , (feat_name, feat_val)>

## the structure of model

the hyper parameter should be load in cmd rather than only yaml file.

you can consider mg/model/MuiscTransformer/train.py or mg/model/Event_Melody/train.py at the begining







***

Please see [Installation Instructions](https://github.com/SJTMusicTeam/MusicGeneration/wiki/Installation-Instructions)

