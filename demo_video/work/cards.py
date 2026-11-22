# renders text overlays as transparent 1920x1080 pngs
import sys,json,os
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

# base layer : bottom gradient + corner tag, on screen for the whole shot
def base(name,tag,col):
	img = Image.new("RGBA",(W,H),(0,0,0,0))
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
	img.save(os.path.join(out,name + "_base.png"))

# lower third caption : main line + optional sub line
def cap(name,main,sub = None,size = 86):
	img = Image.new("RGBA",(W,H),(0,0,0,0))
	x = 120
	y = 880 if sub else 930
	ImageDraw.Draw(img).rectangle((x - 30,y - size + 14,x - 22,y + 6),fill = TEAL + (255,))
	shadowText(img,(x,y),main,font(size,"Bold"),WHITE,"ls")
	if sub:
		shadowText(img,(x,y + 62),sub,font(38,"Regular"),GREY,"ls")
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

base("ai","AI CONCEPT SHOT",AMBER)
base("real","REAL SIM RECORDING",TEAL)
base("realdemo","REAL SIM RECORDING  ·  TRAINING DEMO",TEAL)
base("realft","REAL SIM RECORDING  ·  MODEL ROLLOUT",TEAL)
base("none",None,TEAL)

cap("s1","Teach a robot with a webcam.","Hand tracking  ·  21 landmarks  ·  one laptop camera")
cap("s3","Your hand moves. The arm follows.","MediaPipe hand tracking  →  teleop  →  YAM arm in MuJoCo")
cap("s4","Pinch to grasp.","Fist = clutch   ·   Pinch = close the gripper")
cap("s5a","Wrist camera","Training demo  ·  task : pick up the apple",size = 70)
cap("s5b","Reach. Grasp. Lift.","Training demo  ·  a simulated hand drives the real teleop code",size = 70)
cap("s7","Fine-tune Dream-1.","500 training steps   ·   $100 of compute")
cap("s8","Fine-tuned Dream-1, running in the sim",None,size = 64)

BG = (5,8,11,255)
centre("s2",[("Robots learn from demonstrations.",44,GREY,"Regular"),("Collecting them usually takes a second robot.",76,WHITE,"Bold")],bg = BG)
centre("s6",[("141 episodes",190,WHITE,"Bold"),("≈ 15 minutes of data   ·   8,913 frames   ·   3 cameras",46,TEAL,"SemiBold"),("Demos generated in simulation through the teleop stack",36,GREY,"Regular")],panel = True)
centre("s6b",[("Quality-filtered.",76,WHITE,"Bold"),("Written as a LeRobot dataset.",76,WHITE,"Bold")],bg = BG)

# end card
img = Image.new("RGBA",(W,H),(0,0,0,0))
dark = Image.new("RGBA",(W,H),(2,5,8,120))
img.alpha_composite(dark)
shadowText(img,(W / 2,430),"DreamHands",font(190,"Bold"),WHITE,"ms")
ImageDraw.Draw(img).rectangle((W / 2 - 60,476,W / 2 + 60,482),fill = TEAL + (255,))
shadowText(img,(W / 2,580),"Any laptop is a robot data station.",font(64,"SemiBold"),WHITE,"ms")
shadowText(img,(W / 2,660),"HackLab #2 x Dream Labs  ·  Berlin",font(36,"Regular"),GREY,"ms")
shadowText(img,(W / 2,1010),"Concept shots are AI-generated. Sim clips are real recordings.",font(30,"Regular"),GREY,"ms")
img.save(os.path.join(out,"end.png"))

# optional eval line, only called with real numbers
if len(sys.argv) > 2:
	b,f = sys.argv[1],sys.argv[2]
	cap("s8e","Fine-tuned Dream-1, running in the sim","Sim eval, pick up the apple  ·  base " + b + "/10  vs  fine-tuned " + f + "/10",size = 64)
	centre("eval",[("Sim eval  ·  pick up the apple  ·  10 episodes each",40,GREY,"Regular"),("Base " + b + "/10   vs   fine-tuned " + f + "/10",110,WHITE,"Bold")],bg = BG)
print("ok")
