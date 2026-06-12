# -*- coding: utf-8 -*-
"""Hand-built SVG reproduction of the Beijing v4 architecture figure (PaperBanana raster -> editable SVG)."""
import html

W, H = 1672, 980
S = []
def raw(s): S.append(s)
def esc(s): return html.escape(s, quote=True)

def rect(x,y,w,h,fill="#ffffff",stroke="none",sw=1.0,rx=8,dash=None,op=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>')

def txt(x,y,s,size=13,fill="#222222",weight="normal",anchor="start",style="normal",family="Arial",spacing=None):
    sp = f' letter-spacing="{spacing}"' if spacing else ""
    s = s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    raw(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
        f'font-weight="{weight}" font-style="{style}" text-anchor="{anchor}"{sp}>{s}</text>')

def tline(x,y,parts,size=13,fill="#222",weight="normal",anchor="start"):
    """parts: list of (text, kind) kind in normal/sub/sup/it"""
    out=[f'<text x="{x}" y="{y}" font-family="Arial" font-size="{size}" fill="{fill}" font-weight="{weight}" text-anchor="{anchor}">']
    for t,k in parts:
        if k=="sub": out.append(f'<tspan baseline-shift="-25%" font-size="{int(size*0.72)}">{esc(t)}</tspan>')
        elif k=="sup": out.append(f'<tspan baseline-shift="40%" font-size="{int(size*0.72)}">{esc(t)}</tspan>')
        elif k=="it": out.append(f'<tspan font-style="italic">{esc(t)}</tspan>')
        else: out.append(f'<tspan>{esc(t)}</tspan>')
    out.append('</text>'); raw("".join(out))

def line(x1,y1,x2,y2,stroke="#555",sw=1.4,dash=None,marker=None):
    d=f' stroke-dasharray="{dash}"' if dash else ""
    m=f' marker-end="url(#{marker})"' if marker else ""
    raw(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}"{d}{m}/>')

def path(d,stroke="#555",sw=1.4,fill="none",dash=None,marker=None):
    da=f' stroke-dasharray="{dash}"' if dash else ""
    m=f' marker-end="url(#{marker})"' if marker else ""
    raw(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{da}{m}/>')

def circle(cx,cy,r,fill="#fff",stroke="#333",sw=1.2):
    raw(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def poly(pts,fill="#ccc",stroke="#555",sw=1.0):
    p=" ".join(f"{a},{b}" for a,b in pts)
    raw(f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def bars(x,y,vals,bw=7,gap=3,maxh=26,fill="#e0975a"):
    for i,v in enumerate(vals):
        h=maxh*v; raw(f'<rect x="{x+i*(bw+gap)}" y="{y-h}" width="{bw}" height="{h}" fill="{fill}"/>')

# ---- colors ----
C = dict(
  inp="#ededed", inpbd="#c6c6c6", inphd="#8a8a8a",
  mi_bg="#ffffff", mi_row="#f6efe0", mi_hd="#e6e6e6",
  org="#fdf0e3", orgbd="#e3a05f", orghd="#cf7a28",
  grn="#eef6e9", grnbd="#8bbf8b", grnhd="#3f8f4f", grnbox="#dcecd6",
  orgbox="#fbe5cd", orgboxbd="#e0975a", pink="#f7d4cf", pinkbd="#d06b62",
  blu="#e9f0f9", blubd="#8fb0d8", bluhd="#3f6fb0", blubox="#cfe0f3", bluboxbd="#5e7891",
  stat="#d7e6cf", statbd="#7aa37a", purp="#e7ddf1", purpbd="#9a86c0",
  yel="#fdf8e7", yelbd="#e0c558", yelhd="#bf8a2a", yelbox="#fbf3da",
  tr="#ececec", trbd="#bfbfbf", trblue="#d6e4f5", trpurp="#e7ddf1",
  ink="#222", mute="#777", arr="#5a5a5a")

raw(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Arial">')
# markers
for name,col in [("ah","#5a5a5a"),("ahg","#3f8f4f"),("ahb","#3f6fb0"),("aho","#cf7a28"),("ahgray","#888888")]:
    raw(f'<defs><marker id="{name}" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 z" fill="{col}"/></marker></defs>')
# colorbar gradient
raw('<defs><linearGradient id="cb" x1="0" y1="1" x2="0" y2="0">'
    '<stop offset="0" stop-color="#eef3fb"/><stop offset="0.5" stop-color="#8fb6e0"/>'
    '<stop offset="1" stop-color="#2f6fb0"/></linearGradient></defs>')
rect(0,0,W,H,fill="#ffffff",rx=0)

# =================================================================== INPUT
rect(14,12,222,322,fill=C["inp"],stroke=C["inpbd"],sw=1.4,rx=10,dash="5 4")
txt(28,40,"INPUT",16,C["inphd"],"bold",spacing="1")
# map grid
gx,gy,gw,gh,n=34,52,182,210,7
rect(gx,gy,gw,gh,fill="#fbfbfb",stroke="#dcdcdc",sw=1,rx=4)
for i in range(n+1):
    line(gx,gy+i*gh/n,gx+gw,gy+i*gh/n,stroke="#e3e3e3",sw=0.8)
    line(gx+i*gw/n,gy,gx+i*gw/n,gy+gh,stroke="#e3e3e3",sw=0.8)
# candidates (green) + small bars
import math
cand=[(70,80),(150,95),(60,150),(170,170),(95,210),(120,120),(185,235)]
ox,oy=98,150
for (cx,cy) in cand:
    rect(gx+cx-9,gy+cy-9,18,18,fill="#cfe6c4",stroke="#8bbf8b",sw=0.8,rx=2)
    for k in range(3):
        raw(f'<rect x="{gx+cx-7+k*5}" y="{gy+cy+1-(3+k*3)}" width="3" height="{3+k*3}" fill="#7fae6b"/>')
    line(gx+ox,gy+oy,gx+cx,gy+cy,stroke="#b6b6b6",sw=0.8,dash="3 3")
rect(gx+ox-11,gy+oy-11,22,22,fill="#cdcdcd",stroke="#9a9a9a",sw=1,rx=2)
circle(gx+ox,gy+oy,4,fill="#7d7d7d",stroke="#7d7d7d")
txt(125,288,"choice set C(o):",12,C["ink"],"normal","middle")
txt(125,304,"~2,700 reachable candidates",11,C["mute"],"normal","middle")
txt(125,318,"per origin (not all cells)",11,C["mute"],"normal","middle")

# =================================================================== MODEL INPUTS
mx,my,mw,mh=14,398,216,344
rect(mx,my,mw,mh,fill="#ffffff",stroke=C["inpbd"],sw=1.3,rx=10)
rect(mx,my,mw,34,fill=C["mi_hd"],stroke="none",rx=10)
rect(mx,my+18,mw,16,fill=C["mi_hd"],stroke="none",rx=0)
txt(mx+mw/2,my+23,"Model inputs",14,"#444","bold","middle")
rows=[("clock","Travel time"),("ruler","Distance"),("jobs","Jobs"),
      ("coin","Average wage"),("person","Competition"),("sun","Time period")]
ry=my+44
for icon,lab in rows:
    rect(mx+10,ry,mw-20,44,fill=C["mi_row"],stroke="#ece3cf",sw=0.8,rx=7)
    ix,iy=mx+34,ry+22
    if icon=="clock":
        circle(ix,iy,11,fill="#fff",stroke="#6a6a6a",sw=1.4); line(ix,iy,ix,iy-7,stroke="#6a6a6a",sw=1.4); line(ix,iy,ix+5,iy+2,stroke="#6a6a6a",sw=1.4)
    elif icon=="ruler":
        raw(f'<g transform="rotate(35 {ix} {iy})"><rect x="{ix-13}" y="{iy-5}" width="26" height="10" fill="#fff" stroke="#6a6a6a" stroke-width="1.3" rx="1"/>'
            + "".join(f'<line x1="{ix-9+k*6}" y1="{iy-5}" x2="{ix-9+k*6}" y2="{iy-1}" stroke="#6a6a6a" stroke-width="1"/>' for k in range(4))+'</g>')
    elif icon=="jobs":
        rect(ix-11,iy-6,22,14,fill="#fff",stroke="#6a6a6a",sw=1.3,rx=2); rect(ix-5,iy-11,10,6,fill="#fff",stroke="#6a6a6a",sw=1.3,rx=1); line(ix-11,iy+1,ix+11,iy+1,stroke="#6a6a6a",sw=1)
    elif icon=="coin":
        circle(ix,iy,11,fill="#f3d488",stroke="#c79a3a",sw=1.3); txt(ix,iy+4,"$",12,"#9a7320","bold","middle")
    elif icon=="person":
        circle(ix,iy-5,5,fill="#6a6a6a"); raw(f'<path d="M{ix-9},{iy+9} q9,-12 18,0 z" fill="#6a6a6a"/>')
    elif icon=="sun":
        circle(ix-3,iy,7,fill="#f4cf6a",stroke="#c79a3a",sw=1); raw(f'<path d="M{ix+4},{iy-6} a7,7 0 1,0 0,12 a9,9 0 0,1 0,-12 z" fill="#9fb6d8"/>')
    txt(mx+58,ry+27,lab,13,"#333","normal")
    ry+=48

# =================================================================== POPULATION GROUPS (orange)
px,py,pw,ph=268,18,176,274
rect(px,py,pw,ph,fill=C["org"],stroke=C["orgbd"],sw=1.4,rx=10)
txt(px+pw/2,py+24,"Population groups",13.5,C["orghd"],"bold","middle")
txt(px+pw/2,py+39,"(education × occupation)",11,C["orghd"],"normal","middle")
def groupcard(gx,gy,lab,vals):
    rect(gx,gy,72,64,fill="#ffffff",stroke="#ecc59a",sw=1,rx=6)
    txt(gx+36,gy+15,lab,11,"#9a6a28","bold","middle")
    bars(gx+12,gy+54,vals,bw=8,gap=4,maxh=30,fill="#e0975a")
groupcard(px+12,py+52,"Group A",[0.5,0.8,0.6,1.0])
groupcard(px+92,py+52,"Group B",[0.9,0.6,0.4,0.7])
groupcard(px+12,py+128,"Group C",[0.4,0.7,1.0,0.6])
rect(px+92,py+128,72,64,fill="#fbeede",stroke="#ecc59a",sw=1,rx=6,dash="3 3")
txt(px+128,py+165,". . .",16,"#c79a6a","bold","middle")
txt(px+pw/2,py+212,"a separate utility function",10.5,"#9a6a28","normal","middle")
txt(px+pw/2,py+226,"per group",10.5,"#9a6a28","normal","middle")
# group-specific arrow (horizontal, into RUM panel)
path(f"M{px+pw-2},150 L474,150",stroke=C["orghd"],sw=9,marker="aho")
raw(f'<text x="461" y="232" font-family="Arial" font-size="10.5" fill="{C["orghd"]}" '
    f'font-weight="bold" text-anchor="middle" transform="rotate(-90 461 232)">group-specific coefficients</text>')

# =================================================================== RANDOM UTILITY MODEL (green)
rx0,ry0,rw,rh=478,18,608,392
rect(rx0,ry0,rw,rh,fill=C["grn"],stroke=C["grnbd"],sw=1.5,rx=10)
txt(rx0+rw/2,ry0+26,"Random utility model",16,C["grnhd"],"bold","middle")
# main equation row of boxes
ey=ry0+78
txt(rx0+30,ey+6,"",12)
tline(rx0+22,ey+6,[("V",""),("od","sub"),(" =","")],18,C["ink"],"bold")
def termbox(x,w,fill,bd,parts,sz=13):
    rect(x,ey-18,w,36,fill=fill,stroke=bd,sw=1.2,rx=6); tline(x+w/2,ey+5,parts,sz,C["ink"],"bold","middle")
termbox(rx0+78,96,C["grnbox"],C["grnbd"],[("γ",""),("M","sub"),(" · log M",""),("d","sub"),("o","sup")])
txt(rx0+180,ey+5,"+",15,C["ink"],"bold","middle")
termbox(rx0+196,86,C["grnbox"],C["grnbd"],[("a",""),("W","sub"),(" · log W",""),("d","sub"),("a","sup")])
txt(rx0+288,ey+5,"+",15,C["ink"],"bold","middle")
termbox(rx0+304,78,C["grnbox"],C["grnbd"],[("v",""),("D","sub"),(" · log D",""),("d","sub")])
txt(rx0+388,ey+5,"+",15,C["ink"],"bold","middle")
termbox(rx0+404,74,C["orgbox"],C["orgboxbd"],[("λ · IV",""),("od","sub")])
txt(rx0+484,ey+5,"+",15,C["ink"],"bold","middle")
termbox(rx0+500,92,C["grnbox"],C["grnbd"],[("s",""),("(o,t)","sub"),(" · I(d=o)","")])
# dashed callouts below term boxes
def callout(cx,labels):
    line(cx,ey+18,cx,ey+44,stroke="#888",sw=1,dash="3 3")
    for i,l in enumerate(labels): txt(cx,ey+58+i*13,l,10,C["mute"],"normal","middle")
callout(rx0+126,["occupation-","matched jobs"])
callout(rx0+239,["wage"])
callout(rx0+343,["competition"])
callout(rx0+546,["very short","within-cell trips"])
# logsum box
lsx,lsy=rx0+360,ry0+210
rect(lsx,lsy,128,34,fill=C["pink"],stroke=C["pinkbd"],sw=1.2,rx=16)
tline(lsx+64,lsy+22,[("logsum (V",""),("od","sub"),(")","")],14,C["ink"],"bold","middle")
# arrow logsum -> lambda IV
path(f"M{rx0+441},{lsy} L{rx0+441},{ey+20}",stroke=C["orgboxbd"],sw=1.4,dash="4 3",marker="aho")
# lower mode equation
my2=ry0+322
tline(rx0+22,my2,[("V",""),("od","sub"),("m","sup"),(" =","")],15,C["ink"],"bold")
tline(rx0+88,my2,[("asc",""),("m","sub"),("   + (β",""),("0","sub"),("m","sup"),(" + β",""),
   ("1","sub"),("m","sup"),(" · log dist",""),("od","sub"),(") · TT",""),("od","sub"),("m","sup"),
   ("   + θ",""),("m","sup"),(" · inc",""),("o","sub")],14,C["ink"])
txt(rx0+560,my2-6,"m: car,",11,C["ink"],"normal")
txt(rx0+560,my2+8,"transit, walk",11,C["ink"],"normal")
# callouts under lower eq
for cx,labs in [(rx0+108,["mode","constant"]),(rx0+250,["time sensitivity","varies with distance"]),(rx0+470,["income","effect"])]:
    line(cx,my2+10,cx,my2+30,stroke="#888",sw=1,dash="3 3")
    for i,l in enumerate(labs): txt(cx,my2+44+i*13,l,10,C["mute"],"normal","middle")
# arrows from lower eq up to logsum, and logsum up to main row
path(f"M{rx0+300},{my2-14} C {rx0+300},{lsy+60} {lsx-20},{lsy+50} {lsx},{lsy+24}",stroke="#7a7a7a",sw=1.2,marker="ah")
path(f"M{lsx+128},{lsy+10} C {lsx+170},{lsy} {lsx+180},{lsy-40} {lsx+150},{lsy-30}",stroke="#7a7a7a",sw=1.2)

# =================================================================== MERGE + and U=V+eps
plusx,plusy=1108,118
circle(plusx,plusy,16,fill="#fff",stroke="#444",sw=1.6); txt(plusx,plusy+6,"+",20,"#333","bold","middle")
# green arrow from RUM to plus
path(f"M{rx0+rw},{ry0+90} C {rx0+rw+30},{ry0+90} {plusx-40},{plusy} {plusx-16},{plusy}",stroke=C["grnhd"],sw=2,marker="ahg")
# blue arrow from GNN W up to plus
path(f"M{1010},{560} C {1080},{560} {plusx},{plusy+120} {plusx},{plusy+16}",stroke=C["bluhd"],sw=2,marker="ahb")
# U box
ubx,uby=1150,92
rect(ubx,uby,232,68,fill=C["yelbox"],stroke=C["yelbd"],sw=1.3,rx=8)
tline(ubx+116,uby+28,[("U",""),("od","sub"),(" = V",""),("od","sub"),(" + w",""),("nn","sub"),(" · ε",""),("od","sub")],15,C["ink"],"bold","middle")
txt(ubx+116,uby+50,"for every candidate destination",11,C["mute"],"normal","middle")
path(f"M{plusx+16},{plusy} L{ubx},{plusy}",stroke=C["arr"],sw=1.6,marker="ah")
# green U=V+eps
tline(1486,116,[("→  U = V + ε","")],17,C["grnhd"],"bold")
path(f"M{ubx+232},{plusy} L{1480},{plusy}",stroke=C["grnhd"],sw=2,marker="ahg")

# =================================================================== SPATIO-TEMPORAL GNN (blue)
bx,by,bw,bh=255,428,790,316
rect(bx,by,bw,bh,fill=C["blu"],stroke=C["blubd"],sw=1.5,rx=10)
txt(bx+bw/2,by+26,"Spatio-Temporal GNN",16,C["bluhd"],"bold","middle")
# rotated input labels
txt(bx+22,by+150,"dynamic",10,C["mute"],"normal","middle")
# tensor cube
def cube(cx,cy,s,fill,bd):
    poly([(cx,cy),(cx+s,cy),(cx+s+12,cy-12),(cx+12,cy-12)],fill=fill,stroke=bd,sw=1)
    poly([(cx+s,cy),(cx+s+12,cy-12),(cx+s+12,cy+s-12),(cx+s,cy+s)],fill=fill,stroke=bd,sw=1)
    rect(cx,cy,s,s,fill=fill,stroke=bd,sw=1,rx=0)
    for k in (1,2):
        line(cx+k*s/3,cy,cx+k*s/3,cy+s,stroke=bd,sw=0.5)
        line(cx,cy+k*s/3,cx+s,cy+k*s/3,stroke=bd,sw=0.5)
cube(bx+46,by+78,44,C["blubox"],C["bluboxbd"])
txt(bx+74,by+150,"[T, N, C]",11,C["ink"],"normal","middle")
def gnnbox(x,y,w,h,lab1,lab2,fill,bd,gicon="blue"):
    rect(x,y,w,h,fill="#fff",stroke=bd,sw=1.2,rx=8)
    # mini graph
    cxx,cyy=x+w/2,y+22
    pts=[(cxx-14,cyy-8),(cxx+14,cyy-6),(cxx-6,cyy+10),(cxx+12,cyy+12),(cxx,cyy)]
    col= "#7aa9d8" if gicon=="blue" else "#8bbf8b"
    for a,b in [(0,4),(1,4),(2,4),(3,4),(0,1)]:
        line(pts[a][0],pts[a][1],pts[b][0],pts[b][1],stroke=col,sw=0.7)
    for p in pts: circle(p[0],p[1],3.4,fill=col,stroke=col)
    txt(x+w/2,y+h-16,lab1,10.5,C["ink"],"normal","middle")
    if lab2: txt(x+w/2,y+h-4,lab2,10.5,C["ink"],"normal","middle")
gnnbox(bx+128,by+74,96,72,"GraphSAGE on","k-NN graph",C["blubox"],C["blubd"])
# GRU+TCN box
rect(bx+244,by+74,110,72,fill="#fff",stroke=C["blubd"],sw=1.2,rx=8)
for i in range(4):
    rect(bx+254+i*22,by+86,16,16,fill=C["blubox"],stroke=C["blubd"],sw=0.7,rx=2)
    rect(bx+254+i*22,by+106,16,10,fill="#cfd9ec",stroke=C["blubd"],sw=0.6,rx=2)
txt(bx+299,by+136,"GRU +",10.5,C["ink"],"normal","middle")
txt(bx+299,by+148,"multi-scale TCN",10.5,C["ink"],"normal","middle")
# temporal pooling
rect(bx+374,by+74,96,72,fill="#fff",stroke=C["blubd"],sw=1.2,rx=8)
for i in range(4):
    hh=[18,10,22,14][i]; rect(bx+386+i*18,by+118-hh,12,hh,fill=C["blubox"],stroke=C["blubd"],sw=0.7,rx=1)
txt(bx+422,by+136,"Temporal",10.5,C["ink"],"normal","middle")
txt(bx+422,by+148,"pooling over time",10.5,C["ink"],"normal","middle")
# static row
txt(bx+22,by+250,"static",10,C["mute"],"normal","middle")
cube(bx+46,by+218,40,C["stat"],C["statbd"])
txt(bx+72,by+286,"Static features",11,C["ink"],"normal","middle")
txt(bx+72,by+300,"[N, F]",11,C["ink"],"normal","middle")
gnnbox(bx+150,by+214,96,72,"GraphSAGE ×2","",C["stat"],C["statbd"],gicon="green")
# green output blocks
for i in range(4): rect(bx+300+i*16,by+238,12,24,fill="#bcd6ad",stroke=C["statbd"],sw=0.6,rx=1)
# arrows in GNN top row
for x1,x2 in [(bx+114,bx+128),(bx+228,bx+244),(bx+358,bx+374)]:
    line(x1,by+110,x2,by+110,stroke=C["arr"],sw=1.4,marker="ah")
line(bx+90,by+238,bx+150,by+250,stroke=C["arr"],sw=1.4,marker="ah")
line(bx+246,by+250,bx+296,by+250,stroke=C["arr"],sw=1.4,marker="ah")
# z_o, z_d stacks
zx=bx+600
for i in range(4): rect(zx+i*7,by+90,6,40,fill=C["blubox"],stroke=C["bluboxbd"],sw=0.5)
tline(zx+34,by+86,[("z",""),("o","sub")],12,C["ink"],"italic")
for i in range(4): rect(zx+i*7,by+220,6,40,fill="#bcd6ad",stroke=C["statbd"],sw=0.5)
tline(zx+34,by+216,[("z",""),("d","sub")],12,C["ink"],"italic")
# W box (purple)
wx,wy=bx+700,by+150
rect(wx,wy,64,64,fill=C["purp"],stroke=C["purpbd"],sw=1.4,rx=8)
txt(wx+32,wy+42,"W",26,"#5a4a8a","bold","middle")
path(f"M{zx+30},{by+110} C {zx+90},{by+110} {wx},{wy+10} {wx},{wy+24}",stroke="#9a9a9a",sw=1.2)
path(f"M{zx+30},{by+240} C {zx+90},{by+240} {wx},{wy+54} {wx},{wy+40}",stroke="#9a9a9a",sw=1.2)
# epsilon eq
tline(wx-30,wy+96,[("ε",""),("od","sub"),(" = z",""),("o","sub"),("T","sup"),(" · W · z",""),("d","sub")],13,C["ink"])
tline(wx+78,wy+40,[("ε",""),("od","sub")],13,C["ink"],"italic")
line(wx+64,wy+32,wx+74,wy+32,stroke="#9a9a9a",sw=1.2,marker="ahgray")
txt(wx+8,wy+118,"low-rank",9.5,C["mute"],"italic"); txt(wx+8,wy+130,"bilinear contrast",9.5,C["mute"],"italic")

# =================================================================== CONSIDER / DEST / MODE (yellow)
yx,yy,yw,yh=1140,188,300,540
rect(yx,yy,yw,yh,fill=C["yel"],stroke=C["yelbd"],sw=1.4,rx=10)
# 1. consider
txt(yx+16,yy+26,"1. Consider (soft screening)",12.5,C["yelhd"],"bold")
# funnel of dots
fcx=yx+58
for r in range(4):
    for c in range(7-r):
        circle(fcx-((7-r)*8)/2+c*8+4,yy+44+r*9,3,fill="#a9b6c9",stroke="none")
poly([(fcx-22,yy+82),(fcx+22,yy+82),(fcx+6,yy+104),(fcx+6,yy+118),(fcx-6,yy+118),(fcx-6,yy+104)],fill="#fff",stroke="#c9b06a",sw=1.2)
circle(fcx,yy+126,3,fill="#bf8a2a")
# legend icons
lx=yx+108
circle(lx,yy+52,7,fill="#8bbf8b",stroke="none"); txt(lx+14,yy+56,"occupation match",10.5,C["ink"],"normal")
circle(lx,yy+76,8,fill="#cfe0c8",stroke="#8bbf8b",sw=1); txt(lx,yy+80,"S",10,"#3f8f4f","bold","middle"); txt(lx+14,yy+80,"travel cost",10.5,C["ink"],"normal")
circle(lx,yy+100,7,fill="#fff",stroke="#7a7a7a",sw=1.2); line(lx,yy+100,lx,yy+96,stroke="#7a7a7a",sw=1); line(lx,yy+100,lx+3,yy+101,stroke="#7a7a7a",sw=1)
txt(lx+14,yy+98,"commute-time tolerance",10.5,C["ink"],"normal"); txt(lx+14,yy+110,"(by education)",9.5,C["mute"],"italic")
tline(yx+16,yy+150,[("w",""),("d","sub"),(" = σ(k",""),("occ","sub"),(") · σ(k",""),("cost","sub"),(") · σ(k",""),("time","sub"),(") ∈ [0,1]","")],12,C["ink"])
# 2. destination
txt(yx+16,yy+196,"2. Destination choice",12.5,C["yelhd"],"bold")
for i,h in enumerate([34,20,12]): rect(yx+24+i*16,yy+244-h,12,h,fill="#8fb0d8",stroke="none")
tline(yx+96,yy+232,[("P(d) ∝ w",""),("d","sub"),(" · exp(U",""),("od","sub"),(")","")],12.5,C["ink"])
# 3. mode
txt(yx+16,yy+292,"3. Mode choice (within destination)",12,C["yelhd"],"bold")
# tree
tcx=yx+58
circle(tcx,yy+318,7,fill="#7aa9d8",stroke="none")
for dx in (-30,0,30):
    line(tcx,yy+325,tcx+dx,yy+352,stroke="#9a9a9a",sw=1)
    circle(tcx+dx,yy+360,9,fill="#fff",stroke="#9aa",sw=1)
txt(tcx-30,yy+364,"🚗",10,"#333","normal","middle")
txt(tcx,yy+364,"🚆",10,"#333","normal","middle")
txt(tcx+30,yy+364,"🚶",10,"#333","normal","middle")
txt(yx+120,yy+322,"P(m | d):",12,C["ink"],"normal")
txt(yx+120,yy+338,"from the mode nest",11,C["mute"],"italic")
tline(yx+16,yy+392,[("P(m, d)","")],12,C["ink"])
tline(yx+120,yy+392,[("P(d,m) = P(d) · P(m | d)","")],12,C["ink"])

# =================================================================== from groups to observed flows
fx,fy,fw,fh=1486,196,176,150
rect(fx,fy,fw,fh,fill="#ffffff",stroke="#cfcfcf",sw=1.2,rx=8)
txt(fx+fw/2,fy+24,"from groups to",12,C["ink"],"bold","middle")
txt(fx+fw/2,fy+40,"observed flows",12,C["ink"],"bold","middle")
tline(fx+18,fy+74,[("F",""),("od,m","sub"),(" = Σ",""),("g","sub"),(" s",""),("g","sub"),("(o) ·","")],12,C["ink"])
tline(fx+60,fy+98,[("P(d,m | o,g)","")],12,C["ink"])
tline(fx+18,fy+126,[("s",""),("g","sub"),("(o):","")],11,C["ink"])
txt(fx+18,fy+140,"group shares (census)",10.5,C["mute"],"normal")
# arrows into it
path(f"M{yx+yw},{yy+30} C {fx-30},{yy+30} {fx-20},{fy+60} {fx},{fy+60}",stroke="#9a9a9a",sw=1.2,marker="ahgray")
path(f"M{fx+fw/2},{fy+fh} L{fx+fw/2},{fy+fh+30}",stroke="#888",sw=1.2,marker="ah")

# =================================================================== predicted choice probabilities
txt(fx+fw/2,fy+fh+50,"Predicted choice probabilities",11.5,C["ink"],"bold","middle")
txt(fx+fw/2,fy+fh+66,"(destination, mode)",10.5,C["mute"],"normal","middle")
pmx,pmy,pms,pn=1492,470,156,4
rect(pmx,pmy,pms,pms,fill="#fbfbfb",stroke="#d6d6d6",sw=1,rx=2)
for i in range(pn+1):
    line(pmx,pmy+i*pms/pn,pmx+pms,pmy+i*pms/pn,stroke="#e6e6e6",sw=0.8)
    line(pmx+i*pms/pn,pmy,pmx+i*pms/pn,pmy+pms,stroke="#e6e6e6",sw=0.8)
import random
cells=[(0,0,"#8fb0d8"),(2,0,"#9b86c0"),(1,1,"#7fae8a"),(3,1,"#8fb0d8"),(0,2,"#9b86c0"),(2,3,"#7fae8a"),(3,3,"#8fb0d8")]
for (cc,rr,col) in cells:
    bx0=pmx+cc*pms/pn+6; by0=pmy+rr*pms/pn+pms/pn-6
    for k,h in enumerate([10,18,8]):
        raw(f'<rect x="{bx0+k*7}" y="{by0-h}" width="5" height="{h}" fill="{col}"/>')
rect(pmx+1.2*pms/pn,pmy+1.2*pms/pn,pms/pn-4,pms/pn-4,fill="#cfcfcf",stroke="#a8a8a8",sw=1)
# colorbar
rect(pmx+pms+12,pmy+10,10,pms-20,fill="url(#cb)",stroke="#bbb",sw=0.5)
txt(pmx+pms+30,pmy+18,"1.0",9,C["mute"],"normal"); txt(pmx+pms+30,pmy+pms/2,"0.5",9,C["mute"],"normal"); txt(pmx+pms+30,pmy+pms-10,"0.0",9,C["mute"],"normal")
# arrow mode-panel -> map
path(f"M{yx+yw},{yy+430} C {pmx-30},{yy+430} {pmx-20},{pmy+40} {pmx},{pmy+40}",stroke="#9a9a9a",sw=1.2,marker="ahgray")

# =================================================================== TRAINING & LOSSES
tx,ty,tw,th=14,786,1648,178
rect(tx,ty,tw,th,fill=C["tr"],stroke=C["trbd"],sw=1.4,rx=10)
txt(tx+24,ty+50,"Training & losses",17,"#333","bold")
line(tx+210,ty+50,tx+280,ty+50,stroke="#666",sw=1.4,marker="ah")
rect(tx+288,ty+30,150,40,fill="#e2e2e2",stroke="#bdbdbd",sw=1,rx=6)
tline(tx+363,ty+55,[("Main objective L",""),("flow","sub")],12.5,C["ink"],"bold","middle")
# dashed group box
rect(tx+470,ty+18,340,140,fill="none",stroke="#9a9a9a",sw=1.2,rx=8,dash="5 4")
# smart card
rect(tx+486,ty+34,150,40,fill="#fff",stroke="#cfcfcf",sw=1,rx=6)
rect(tx+498,ty+46,18,16,fill="#8fb0d8",stroke="none",rx=2)
txt(tx+560,ty+59,"Smart card",11,C["ink"],"normal","middle"); txt(tx+560,ty+71,"data",11,C["ink"],"normal","middle")
rect(tx+664,ty+34,128,40,fill=C["trblue"],stroke="#9fb6d8",sw=1,rx=6)
tline(tx+728,ty+59,[("w",""),("1","sub"),(" · L",""),("transit","sub")],12.5,C["ink"],"bold","middle")
# survey shares
rect(tx+486,ty+102,150,42,fill="#fff",stroke="#cfcfcf",sw=1,rx=6)
circle(tx+508,ty+123,10,fill="#b9a7e0",stroke="none"); raw(f'<path d="M{tx+508},{ty+123} L{tx+508},{ty+113} A10,10 0 0,1 {tx+517},{ty+126} z" fill="#9b86c0"/>')
txt(tx+560,ty+120,"Survey shares",10.5,C["ink"],"normal","middle"); txt(tx+560,ty+132,"(yield, mode, etc.)",9.5,C["mute"],"normal","middle")
rect(tx+664,ty+102,128,42,fill=C["trpurp"],stroke="#b9a7e0",sw=1,rx=6)
tline(tx+728,ty+128,[("w",""),("2","sub"),(" · L",""),("share","sub")],12.5,C["ink"],"bold","middle")
line(tx+636,ty+54,tx+664,ty+54,stroke="#888",sw=1.2,marker="ah")
line(tx+636,ty+123,tx+664,ty+123,stroke="#888",sw=1.2,marker="ah")
# brace to total loss
path(f"M{tx+792},{ty+54} C {tx+840},{ty+54} {tx+840},{ty+90} {tx+880},{ty+90}",stroke="#888",sw=1.2)
path(f"M{tx+792},{ty+123} C {tx+840},{ty+123} {tx+840},{ty+90} {tx+880},{ty+90}",stroke="#888",sw=1.2,marker="ah")
# total loss
rect(tx+886,ty+60,300,60,fill="#e2e2e2",stroke="#bdbdbd",sw=1.2,rx=8)
txt(tx+1036,ty+84,"Total loss:",13,C["ink"],"bold","middle")
tline(tx+1036,ty+106,[("L = L",""),("flow","sub"),(" + w",""),("1","sub"),(" · L",""),("transit","sub"),(" + w",""),("2","sub"),(" · L",""),("share","sub")],12.5,C["ink"],"normal","middle")
# trained end-to-end dashed
txt(tx+1075,ty+38,"trained",10.5,C["mute"],"italic","middle"); txt(tx+1075,ty+50,"end-to-end",10.5,C["mute"],"italic","middle")
path(f"M{tx+1186},{ty+90} C {tx+1300},{ty+90} {tx+1300},{ty-40} {bx+560},{by+bh+10}",stroke="#9a9a9a",sw=1,dash="4 4",marker="ahgray")
txt(1015,772,"trained end-to-end",10.5,C["mute"],"italic","middle")
path(f"M{460},{by+bh} C {460},{ty} {tx+300},{ty} {tx+300},{ty+30}",stroke="#9a9a9a",sw=1,dash="4 4")

raw('</svg>')

with open(r"D:\GIT\mesa Gsoc\GSoC-learning-space\models\05_beijing_nested_logit_v4\report\architecture_replica.svg","w",encoding="utf-8") as f:
    f.write("\n".join(S))
print("saved architecture_replica.svg", len(S), "elements")
