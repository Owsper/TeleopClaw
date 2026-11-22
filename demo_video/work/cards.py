# renders text overlays as transparent 1920x1080 pngs
import sys,json,os,re
from PIL import Image,ImageDraw,ImageFont,ImageFilter

W,H = 1920,1080
out = os.path.join(os.path.dirname(__file__),"ov")
os.makedirs(out,exist_ok = True)
TEAL = (64,224,208)
AMBER = (255,176,64)
WHITE = (255,255,255)
GREY = (200,212,216)

def font(size,var = "SemiBold"):
	f = ImageFont.truetype("C:/Windows/Fonts/bahnschrift.ttf",size)
	f.set_variation_by_name(var)
	return f

def shadowText(img,xy,txt,f,fill,anchor = "la"):
	sh = Image.new("RGBA",img.size,(0,0,0,0))
	d = ImageDraw.Draw(sh)
	d.text(xy,txt,font = f,fill = (0,0,0,210),anchor = anchor)
	sh = sh.filter(ImageFilter.GaussianBlur(8))
	img.alpha_composite(sh)
	img.alpha_composite(sh)
	ImageDraw.Draw(img).text(xy,txt,font = f,fill = fill + (255,),anchor = anchor)

def grad(img,top,alpha):
	g = Image.new("L",(1,H),0)
	for y in range(top,H):
		g.putpixel((0,y),int(alpha * ((y - top) / (H - top)) ** 1.3))
	g = g.resize((W,H))
	blk = Image.new("RGBA",(W,H),(0,0,0,255))
	blk.putalpha(g)
	img.alpha_composite(blk)

def runText(img,xy,txt,f,fill):
	x,y = xy
	fb = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",f.size)
	for part in re.split("(→)",txt):
		ff = fb if part == "→" else f
		shadowText(img,(x,y),part,ff,fill,"ls")
		x += ImageDraw.Draw(img).textlength(part,font = ff)

# base layer : bottom gradient + corner tag, on screen for the whole shot
def base(name,tag,col,g = True):
	img = Image.new("RGBA",(W,H),(0,0,0,0))
	if g:
		grad(img,560,215)
	if tag:
		f = font(26,"SemiBold")
		d = ImageDraw.Draw(img)
		tw = d.textlength(tag,font = f)
		x,y = 64,56
		pill = Image.new("RGBA",(W,H),(0,0,0,0))
		ImageDraw.Draw(pill).rounded_rectangle((x,y,x + tw + 62,y + 48),24,fill = (0,0,0,150),outline = col + (230,),width = 2)
		img.alpha_composite(pill)
		d = ImageDraw.Draw(img)
		d.ellipse((x + 18,y + 17,x + 32,y + 31),fill = col + (255,))
		d.text((x + 44,y + 25),tag,font = f,fill = WHITE + (255,),anchor = "lm")
	return img,os.path.join(out,name + "_base.png")

# lower third caption : main line + optional sub line
def cap(name,main,sub = None,size = 86):
	img = Image.new("RGBA",(W,H),(0,0,0,0))
	x = 120
	y = 880 if sub else 930
	ImageDraw.Draw(img).rectangle((x - 30,y - size + 14,x - 22,y + 6),fill = TEAL + (255,))
	shadowText(img,(x,y),main,font(size,"Bold"),WHITE,"ls")
	if sub:
		runText(img,(x,y + 62),sub,font(38,"Regular"),GREY)
	img.save(os.path.join(out,name + "_cap.png"))

# centred card : small line, big line, optional extra lines
def centre(name,lines,panel = False,bg = None):
	img = Image.new("RGBA",(W,H),bg or (0,0,0,0))
	tot = sum(l[1] * 1.45 for l in lines)
	y = (H - tot) / 2
	if panel:
		p = Image.new("RGBA",(W,H),(0,0,0,0))
		ImageDraw.Draw(p).rectangle((0,y - 70,W,y + tot + 50),fill = (4,8,12,200))
		p = p.filter(ImageFilter.GaussianBlur(30))
		img.alpha_composite(p)
	for txt,size,col,var in lines:
		y += size * 1.45
		shadowText(img,(W / 2,y - size * 0.35),txt,font(size,var),col,"ms")
	img.save(os.path.join(out,name + ".png"))

for a in [("ai","AI CONCEPT SHOT",AMBER),("realdemo","REAL SIM RECORDING  ·  TRAINING DEMO",TEAL),("realft","REAL SIM RECORDING  ·  MODEL ROLLOUT",TEAL)]:
	img,path = base(*a)
	img.save(path)

# mosaic : no gradient, label sits on the 3 empty slots of the 12x12 grid
img,path = base("real","REAL SIM RECORDINGS",TEAL,False)
d = ImageDraw.Draw(img)
d.rectangle((1440,990,W,H),fill = (6,10,14,255))
d.text((1680,1035),"ALL 141, PLAYING AT ONCE",font = font(26,"SemiBold"),fill = TEAL + (255,),anchor = "mm")
img.save(path)

# frame + label for the wrist cam picture in picture
PX,PY,PW,PH = 1296,56,560,315
img = Image.new("RGBA",(W,H),(0,0,0,0))
d = ImageDraw.Draw(img)
d.rectangle((PX - 3,PY - 3,PX + PW + 2,PY + PH + 2),outline = TEAL + (255,),width = 3)
d.rectangle((PX,PY + PH - 40,PX + 210,PY + PH),fill = (0,0,0,170))
d.text((PX + 14,PY + PH - 20),"WRIST CAMERA",font = font(22,"SemiBold"),fill = WHITE + (255,),anchor = "lm")
img.save(os.path.join(out,"pip.png"))

# live teleop screen recording : frames and labels around the two real panes
img,path = base("live","REAL SCREEN RECORDING",TEAL,False)
d = ImageDraw.Draw(img)
for x,w,lab in ((60,976,"ROBOT   ·   YAM arm in MuJoCo, overhead camera"),(1076,784,"HAND   ·   laptop webcam + hand tracking")):
	d.rectangle((x - 2,148,x + w + 1,150 + 549 + 1),outline = (255,255,255,70),width = 2)
	d.rectangle((x - 2,148,x + 90,150),fill = TEAL + (255,))
	d.text((x,136),lab,font = font(24,"SemiBold"),fill = GREY + (255,),anchor = "ls")
img.save(path)
cap("live1","This part is real.","Live webcam teleop  ·  screen recording, real speed")
cap("live2","Real hand. Real-time control.","Live webcam teleop  ·  screen recording, real speed")

cap("s1","Teach a robot with a webcam.","Hand tracking  ·  21 landmarks  ·  one laptop camera")
cap("s3","Your hand moves. The arm follows.","MediaPipe hand tracking  →  teleop  →  YAM arm in MuJoCo")
cap("s4","Pinch to grasp.","Fist = clutch   ·   Pinch = close the gripper")
cap("s5a","Wrist camera","Training demo  ·  pick up the apple",size = 70)
cap("s5b","Reach. Grasp. Lift.","Training demo  ·  a simulated hand drives the real teleop code",size = 70)
cap("s7","Fine-tune Dream-1.","500 training steps   ·   $100 of compute")
cap("s8","Fine-tuned Dream-1, running in the sim",None,size = 64)

BG = (5,8,11,255)
centre("s2",[("Robots learn from demonstrations.",44,GREY,"Regular"),("Collecting them usually takes a second robot.",76,WHITE,"Bold")],bg = BG)
centre("s6",[("141 episodes",190,WHITE,"Bold"),("About 15 minutes of data   ·   8,913 frames   ·   3 cameras",46,TEAL,"SemiBold"),("Demos generated in simulation through the teleop stack",36,GREY,"Regular")],panel = True)
centre("s6b",[("Quality-filtered.",76,WHITE,"Bold"),("Written as a LeRobot dataset.",76,WHITE,"Bold")],bg = BG)

# end card
img = Image.new("RGBA",(W,H),(0,0,0,0))
dark = Image.new("RGBA",(W,H),(2,5,8,120))
img.alpha_composite(dark)
shadowText(img,(W / 2,430),"DreamHands",font(190,"Bold"),WHITE,"ms")
ImageDraw.Draw(img).rectangle((W / 2 - 60,476,W / 2 + 60,482),fill = TEAL + (255,))
shadowText(img,(W / 2,580),"Any laptop is a robot data station.",font(64,"SemiBold"),WHITE,"ms")
shadowText(img,(W / 2,660),"HackLab #2 x Dream Labs  ·  Berlin",font(36,"Regular"),GREY,"ms")
shadowText(img,(W / 2,985),"Concept shots are AI-generated. Sim clips are real recordings.",font(30,"Regular"),GREY,"ms")
shadowText(img,(W / 2,1028),"The webcam teleop clip is a real screen recording at real speed.",font(26,"Regular"),GREY,"ms")
img.save(os.path.join(out,"end.png"))

# eval card, numbers come from the caller, base only when confirmed
if len(sys.argv) > 1:
	ft = sys.argv[1]
	if len(sys.argv) > 2:
		b = sys.argv[2]
		centre("eval",[("Sim eval  ·  pick up the apple  ·  same 10 apple positions",40,GREY,"Regular"),("Base Dream-1: " + b + "/10.",96,GREY,"SemiBold"),("Fine-tuned: " + ft + "/10.",150,WHITE,"Bold")],bg = BG)
	else:
		centre("eval",[(ft + "/10",260,WHITE,"Bold"),("apple picks",80,TEAL,"SemiBold"),("Fine-tuned Dream-1  ·  sim eval, 10 episodes",40,GREY,"Regular")],bg = BG)
print("ok")
