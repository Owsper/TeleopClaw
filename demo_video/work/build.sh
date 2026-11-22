#!/bin/bash
# builds every segment at 1920x1080 30fps, one encode at a time, then concat + music
# usage : build.sh [segment names...]   (no args = all)
set -e
cd "$(dirname "$0")"
D=../../data
V=$D/dreamhands/videos/chunk-000
G=gen
O=ov
P=parts
EP=000120
ENC="-r 30 -c:v libx264 -preset medium -crf 16 -pix_fmt yuv420p -an -video_track_timescale 15360"
AI="fps=30,scale=1920:1086:flags=lanczos,crop=1920:1080,setsar=1"
REAL="scale=1920:1080:flags=lanczos,fps=30,setsar=1"
MIRROR=${MIRROR:-v10}

# eval numbers : only tags with 10 clean rows make it on screen
EV=$(../../.venv/Scripts/python.exe -c "
import json
rows = [json.loads(l) for l in open('$D/eval.jsonl') if l.strip()]
out = []
for tag in ('ft','base'):
	r = [x for x in rows if x['tag'] == tag]
	ok = len(r) == 10 and all(x['error'] is None for x in r)
	out.append(str(sum(1 for x in r if x['success'])) if ok else '')
print(' '.join(out if out[0] else []))
")
echo "eval : $EV"
../../.venv/Scripts/python.exe cards.py $EV

want() {
	[ $# -eq 0 ] && return 0
	return 1
}
if [ $# -gt 0 ]; then
	LIST=" $* "
	want() { [[ "$LIST" == *" $1 "* ]]; }
else
	want() { return 0; }
fi

# ai shot with corner tag + caption fading in at $5
aishot() {
	name=$1; src=$2; dur=$3; cap=$4; capt=$5; extra=$6
	ffmpeg -v error -y -i $G/$src.mp4 -loop 1 -framerate 30 -t $dur -i $O/ai_base.png -loop 1 -framerate 30 -t $dur -i $O/$cap.png \
		-filter_complex "[0:v]$AI,trim=duration=$dur,setpts=PTS-STARTPTS[v];[2:v]format=rgba,fade=t=in:st=$capt:d=0.5:alpha=1[c];[v][1:v]overlay[a];[a][c]overlay$extra[o]" \
		-map "[o]" -t $dur $ENC $P/$name.mp4
	echo $name
}

# full screen text card that fades in and out
card() {
	name=$1; png=$2; dur=$3
	ffmpeg -v error -y -loop 1 -framerate 30 -t $dur -i $O/$png.png \
		-filter_complex "[0:v]format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st=$(awk "BEGIN{print $dur - 0.4}"):d=0.4[o]" \
		-map "[o]" -t $dur $ENC $P/$name.mp4
	echo $name
}

if want s1; then aishot s1 v7 5.0 s1_cap 2.6 ",fade=t=in:st=0:d=0.7"; fi
# real screen recording of live webcam teleop : crop the two app panes + status bar, light denoise / sharpen, no speed change
if want live; then
	dur=11.0
	ffmpeg -v error -y -ss 31.4 -t $dur -i src/teleop_rec.mov -loop 1 -framerate 30 -t $dur -i $O/live_base.png -loop 1 -framerate 30 -t $dur -i $O/live1_cap.png -loop 1 -framerate 30 -t $dur -i $O/live2_cap.png 		-filter_complex "color=c=0x05080b:s=1920x1080:r=30:d=$dur[bg];[0:v]fps=30,split=3[a][b][c];[a]crop=955:537:172:498,hqdn3d=1:1:2:2,scale=976:549:flags=lanczos,unsharp=5:5:0.4[sim];[b]crop=717:502:1127:498,hqdn3d=2:2:3:3,scale=784:549:flags=lanczos,unsharp=5:5:0.5,eq=contrast=1.05:saturation=1.06[cam];[c]crop=660:42:172:1038,scale=924:-2:flags=lanczos[bar];[bg][sim]overlay=60:150[x1];[x1][cam]overlay=1076:150[x2];[x2][bar]overlay=60:712,fade=t=in:st=0:d=0.3[x3];[2:v]format=rgba,fade=t=in:st=0.4:d=0.4:alpha=1,fade=t=out:st=4.6:d=0.4:alpha=1[c1];[3:v]format=rgba,fade=t=in:st=5.2:d=0.4:alpha=1[c2];[x3][1:v]overlay[x4];[x4][c1]overlay[x5];[x5][c2]overlay,setsar=1[o]" 		-map "[o]" -an -t $dur $ENC $P/live.mp4
	echo live
fi

if want s2; then card s2 s2 3.8; fi
if want s3; then aishot s3 $MIRROR 3.6 s3_cap 0.3 ""; fi
if want s4; then aishot s4 v3 3.6 s4_cap 0.3 ""; fi

# real training demo, wrist cam then overhead, same episode
if want s5a; then
	dur=2.4
	ffmpeg -v error -y -i $V/observation.images.wrist_r/episode_$EP.mp4 -loop 1 -framerate 30 -t $dur -i $O/realdemo_base.png -loop 1 -framerate 30 -t $dur -i $O/s5a_cap.png \
		-filter_complex "[0:v]trim=duration=$dur,setpts=PTS-STARTPTS,$REAL[v];[2:v]format=rgba,fade=t=in:st=0.2:d=0.4:alpha=1[c];[v][1:v]overlay[a];[a][c]overlay[o]" \
		-map "[o]" -t $dur $ENC $P/s5a.mp4
	echo s5a
fi
if want s5b; then
	dur=5.0
	ffmpeg -v error -y -i $V/observation.images.overhead/episode_$EP.mp4 -loop 1 -framerate 30 -t $dur -i $O/realdemo_base.png -loop 1 -framerate 30 -t $dur -i $O/s5b_cap.png \
		-filter_complex "[0:v]trim=duration=$dur,setpts=PTS-STARTPTS,$REAL[v];[1:v]format=rgba,fade=t=out:st=4.5:d=0.5:alpha=1[b];[2:v]format=rgba,fade=t=in:st=0.3:d=0.4:alpha=1,fade=t=out:st=4.5:d=0.5:alpha=1[c];[v][b]overlay[a];[a][c]overlay[o]" \
		-map "[o]" -t $dur $ENC $P/s5b.mp4
	echo s5b
fi

# pull back from the episode we just watched to all 141 playing at once
if want s6; then
	dur=7.6
	[ -f $P/s6_zoom.mp4 ] || ../../.venv/Scripts/python.exe zoom.py
	ffmpeg -v error -y -i $P/s6_zoom.mp4 -loop 1 -framerate 30 -t $dur -i $O/real_base.png -loop 1 -framerate 30 -t $dur -i $O/s6.png 		-filter_complex "[0:v]setsar=1[v];[1:v]format=rgba,fade=t=in:st=0.3:d=0.6:alpha=1[b];[2:v]format=rgba,fade=t=in:st=3.6:d=0.6:alpha=1[c];[v][b]overlay[a];[a][c]overlay[o]" 		-map "[o]" -t $dur $ENC $P/s6.mp4
	echo s6
fi
if want s6b; then card s6b s6b 2.8; fi
if want s7; then aishot s7 v4 4.4 s7_cap 0.3 ""; fi

# real rollout of the fine tuned model, overhead + wrist cam inset
if want s8; then
	dur=7.6
	ffmpeg -v error -y -ss 0.5 -i $D/rollout_ft.mp4 -ss 0.5 -i $D/rollout_ft_wrist.mp4 -loop 1 -framerate 30 -t $dur -i $O/realft_base.png -loop 1 -framerate 30 -t $dur -i $O/pip.png -loop 1 -framerate 30 -t $dur -i $O/s8_cap.png \
		-filter_complex "[0:v]trim=duration=$dur,setpts=PTS-STARTPTS,$REAL[v];[1:v]trim=duration=$dur,setpts=PTS-STARTPTS,scale=560:315:flags=lanczos,fps=30[w];[4:v]format=rgba,fade=t=in:st=0.4:d=0.5:alpha=1[c];[v][w]overlay=1296:56[a];[a][2:v]overlay[b];[b][3:v]overlay[d];[d][c]overlay[o]" \
		-map "[o]" -t $dur $ENC $P/s8.mp4
	echo s8
fi
if want s8b; then card s8b eval 3.6; fi

if want s9; then
	dur=7.0
	ffmpeg -v error -y -i $G/v5.mp4 -loop 1 -framerate 30 -t $dur -i $O/end.png \
		-filter_complex "[0:v]$AI,trim=duration=$dur,setpts=PTS-STARTPTS[v];[1:v]format=rgba,fade=t=in:st=0.5:d=0.8:alpha=1[c];[v][c]overlay,fade=t=out:st=6.0:d=1.0[o]" \
		-map "[o]" -t $dur $ENC $P/s9.mp4
	echo s9
fi

# concat + music
rm -f $P/list.txt
for s in s1 live s2 s3 s4 s5a s5b s6 s6b s7 s8 s8b s9; do
	echo "file '$s.mp4'" >> $P/list.txt
done
ffmpeg -v error -y -f concat -safe 0 -i $P/list.txt -c copy $P/silent.mp4
T=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $P/silent.mp4)
FO=$(awk "BEGIN{print $T - 3}")
ffmpeg -v error -y -i $P/silent.mp4 -i $G/music.mp3 -filter_complex "[1:a]atrim=0:$T,afade=t=in:st=0:d=0.6,afade=t=out:st=$FO:d=3,loudnorm=I=-15:TP=-1.5:LRA=11[a]" -map 0:v -map "[a]" -c:v copy -c:a aac -b:a 192k -ar 48000 -movflags +faststart -t $T ../dreamhands_demo.mp4
echo "done $T"
