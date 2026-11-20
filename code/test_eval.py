import sys; sys.path.insert(0, "/home/team22/llvip_eval/code")
import numpy as np, pandas as pd, eval as E
gt = pd.DataFrame([("000001",100,100,150,250),   # q1 (h150)
                   ("000001",130,100,180,250),   # overlaps with first (IoA 0.4)
                   ("000001",500,100,600,400),   # normal
                   ("000002",10,10,60,110)], columns=["image_id","x1","y1","x2","y2"])
gt = E.add_gt_flags(gt, ["000001","000002","000003"])
print(gt)
c = pd.DataFrame([("000001",100,100,150,250,0.9),   # TP gt0
                  ("000001",101,100,151,250,0.8),   # dup -> gt1? IoU with gt1 low -> FP
                  ("000001",500,100,600,400,0.1),   # low conf on gt2
                  ("000003",0,0,10,10,0.5),          # FP on empty image
                  ("000002",300,300,350,400,0.9)], columns=E.CAND_COLS)
r = E.fppi_recall(c, gt, 0.25); print(r)
assert r["tp"]==1 and r["fp"]==3 and abs(r["fppi"]-1.0)<1e-9 and r["recall"]==0.25
print(E.miss_causes(c, gt)[["image_id","cause"]])
print(E.miss_cause_table(E.miss_causes(c, gt)))
r0 = E.fppi_recall(c, gt, 0.05); assert r0["tp"]==2
# prefix property check on random data
rng=np.random.default_rng(0); rows=[]
for i in range(50):
    for _ in range(30):
        x,y=rng.uniform(0,1000,2); w,h=rng.uniform(20,200,2); rows.append((f"{i:06d}",x,y,x+w,y+h,rng.uniform()))
cr=pd.DataFrame(rows,columns=E.CAND_COLS); g2=E.add_gt_flags(cr.sample(400,random_state=1)[["image_id","x1","y1","x2","y2"]].assign(x1=lambda d:d.x1+5),[f"{i:06d}" for i in range(50)])
full=E.match(cr,g2,0.0)
for t in [0.2,0.5,0.8]:
    m=E.match(cr,g2,t); assert (m["cand_tp"]==(full["cand_tp"]&(cr.conf>=t))).all()
print("all tests ok")
