#!/bin/bash
# 12x12 mosaic of all 141 overhead episodes, built row by row to keep memory low
set -e
V=data/dreamhands/videos/chunk-000/observation.images.overhead
W=demo_video/work/parts
D=8
for r in $(seq 0 11); do
	ins=""
	fl=""
	lab=""
	for c in $(seq 0 11); do
		i=$((r*12+c))
		if [ $i -lt 141 ]; then
			f=$(printf "$V/episode_%06d.mp4" $i)
			ins="$ins -stream_loop 2 -i $f"
		else
			ins="$ins -f lavfi -i color=c=0x0a0e12:s=640x360:r=10"
		fi
		fl="$fl[$c:v]fps=10,scale=320:180:flags=lanczos,trim=duration=$D,setpts=PTS-STARTPTS[v$c];"
		lab="$lab[v$c]"
	done
	ffmpeg -v error -y $ins -filter_complex "${fl}${lab}hstack=inputs=12[o]" -map "[o]" -t $D -r 10 -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p $W/row_$r.mp4
	echo "row $r"
done
ins=""
lab=""
for r in $(seq 0 11); do
	ins="$ins -i $W/row_$r.mp4"
	lab="$lab[$r:v]"
done
ffmpeg -v error -y $ins -filter_complex "${lab}vstack=inputs=12[o]" -map "[o]" -r 10 -c:v libx264 -preset veryfast -crf 14 -pix_fmt yuv420p $W/mosaic4k.mp4
ffprobe -v error -show_entries format=duration:stream=width,height,nb_frames -of csv=p=0 $W/mosaic4k.mp4
