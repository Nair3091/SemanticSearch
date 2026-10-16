for f in neg1 dog neg2
 do echo -n "$f: "; ffprobe -v error -show_entries format=duration -of csv=p=0 data/raw_videos/$f.mp4; done