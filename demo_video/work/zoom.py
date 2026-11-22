# pull back from one mosaic tile to the whole 12x12 grid, streamed frame by frame
import av,subprocess,sys,os
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here,"parts","mosaic4k.mp4")
dst = os.path.join(here,"parts","s6_zoom.mp4")
W,H = 3840,2160
start,dur,fps = 5.0,8.0,30
tx,ty = 160,1890
zmax = 12.0
hold,ramp = 9,96

enc = subprocess.Popen(["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","1920x1080","-r",str(fps),"-i","-","-c:v","libx264","-preset","medium","-crf","12","-pix_fmt","yuv420p",dst],stdin = subprocess.PIPE)
c = av.open(src)
st = c.streams.video[0]
frames = c.decode(st)
img = None
cur = -1
n = int(dur * fps)
for i in range(n):
	want = int(round(start * 10)) + i * 10 // fps
	while cur < want:
		f = next(frames)
		cur += 1
		if cur == want:
			img = f.to_image()
	p = min(max((i - hold) / ramp,0),1)
	e = 1 - (1 - p) ** 3
	z = zmax ** (1 - e)
	k = (z - 1) / (zmax - 1)
	cx = W / 2 + (tx - W / 2) * k
	cy = H / 2 + (ty - H / 2) * k
	w,h = W / z,H / z
	out = img.resize((1920,1080),Image.BICUBIC,box = (cx - w / 2,cy - h / 2,cx + w / 2,cy + h / 2))
	enc.stdin.write(out.tobytes())
enc.stdin.close()
enc.wait()
c.close()
print("zoom ok",n)
