import base64, io
from typing import Dict, List
import cv2, numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title='LunaSafe AI Prototype')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:4173',
        'http://127.0.0.1:4173',
        'https://lunasafe-ai.vercel.app',
    ],
    allow_credentials=False,
    allow_methods=['GET', 'POST', 'OPTIONS'],
    allow_headers=['*'],
)

def data_url(img: np.ndarray, cmap=None) -> str:
    if cmap is not None: img = cv2.applyColorMap(img, cmap)
    ok, buff = cv2.imencode('.png', img)
    return 'data:image/png;base64,' + base64.b64encode(buff).decode() if ok else ''

def zones_from_risk(risk: np.ndarray) -> List[Dict]:
    h,w=risk.shape; size=max(28,min(h,w)//5); candidates=[]
    for y in range(0,h-size,max(8,size//4)):
        for x in range(0,w-size,max(8,size//4)):
            candidates.append((float(risk[y:y+size,x:x+size].mean()),x,y))
    picked=[]
    for value,x,y in sorted(candidates):
        if all((x-px)**2+(y-py)**2>(size*1.25)**2 for _,px,py in picked):
            picked.append((value,x,y))
        if len(picked)==3: break
    names=['ALPHA','BRAVO','CHARLIE']; out=[]
    for i,(risk_value,x,y) in enumerate(picked):
        # Both displayed values are derived from the exact same mean fused-risk window.
        zone_risk = round(risk_value)
        out.append({'id':f'ZONE {names[i]}','location':f'{round(100*x/w)}% E / {round(100*y/h)}% N',
                    'x':x, 'y':y, 'size_px':size, 'score':100-zone_risk, 'risk':zone_risk})
    return out

@app.get('/api/health')
def health(): return {'status':'ok','mode':'prototype heuristics'}

@app.post('/api/analyze')
async def analyze(file: UploadFile = File(...)):
    raw=await file.read()
    if len(raw)>20*1024*1024: raise HTTPException(413,'Image exceeds 20 MB')
    try:
        image=np.array(Image.open(io.BytesIO(raw)).convert('RGB'))
    except Exception: raise HTTPException(400,'Please upload a valid image')
    if min(image.shape[:2])<32: raise HTTPException(400,'Image must be at least 32 pixels wide and high')
    max_dim=900; scale=min(1,max_dim/max(image.shape[:2])); image=cv2.resize(image,(int(image.shape[1]*scale),int(image.shape[0]*scale)))
    bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR); gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    # Prototype SR fallback: local contrast enhancement + unsharp filtering, no learned model.
    clahe=cv2.createCLAHE(clipLimit=2.2,tileGridSize=(8,8)); enhanced=clahe.apply(gray); enhanced=cv2.addWeighted(enhanced,1.35,cv2.GaussianBlur(enhanced,(0,0),2.1),-.35,0)
    # Explainable heuristic risk proxies: dark pixels, local edge texture, gradient strength, low detail uncertainty.
    shadow=np.clip((82-gray)/82,0,1)
    edges=cv2.Canny(enhanced,45,120); rough=cv2.GaussianBlur(edges.astype(np.float32)/255,(0,0),5)
    gx=cv2.Sobel(enhanced,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(enhanced,cv2.CV_32F,0,1,ksize=3); slope=cv2.normalize(cv2.magnitude(gx,gy),None,0,1,cv2.NORM_MINMAX)
    local_std=np.sqrt(np.maximum(0,cv2.blur(enhanced.astype(np.float32)**2,(31,31))-cv2.blur(enhanced.astype(np.float32),(31,31))**2)); uncertainty=1-cv2.normalize(local_std,None,0,1,cv2.NORM_MINMAX)
    # Robust display normalisation prevents a few extreme pixels from washing the map into yellow.
    low, high = np.percentile(uncertainty, (5, 95))
    uncertainty_display=np.clip((uncertainty-low)/max(high-low, 1e-6),0,1)
    circles=cv2.HoughCircles(cv2.GaussianBlur(enhanced,(9,9),2),cv2.HOUGH_GRADIENT,1.2,max(20,min(gray.shape)//7),param1=80,param2=23,minRadius=max(5,min(gray.shape)//45),maxRadius=max(12,min(gray.shape)//6))
    crater=np.zeros_like(gray,dtype=np.float32)
    n_circles=0
    if circles is not None:
        for x,y,r in np.round(circles[0]).astype(int)[:35]: cv2.circle(crater,(x,y),r,1,-1); n_circles+=1
        crater=cv2.GaussianBlur(crater,(0,0),max(2,min(gray.shape)//90))
    risk=np.clip(100*(.27*shadow+.22*rough+.18*slope+.18*crater+.15*uncertainty),0,100).astype(np.uint8)
    hazard=cv2.applyColorMap(risk,cv2.COLORMAP_JET)
    # highlight ranked square sites, with score calculated before annotations.
    landing_zones=zones_from_risk(risk)
    for z in landing_zones:
        cv2.rectangle(hazard,(z['x'],z['y']),(z['x']+z['size_px'],z['y']+z['size_px']),(100,255,150),2)
    # One rounded global fused-risk value drives every global score/status value returned to the UI.
    global_risk=round(float(risk.mean()))
    safety=100-global_risk
    readiness='CONDITIONAL GO' if safety>=60 else ('REVIEW REQUIRED' if safety>=40 else 'NO-GO — HIGH SURFACE RISK')
    return {'safety_score':safety,'fused_risk':global_risk,'mission_readiness':readiness,'readiness_note':'Prototype image-only assessment. Requires independent mission validation.', 'images':{'original':data_url(bgr),'enhanced':data_url(enhanced),'uncertainty':data_url((uncertainty_display*255).astype(np.uint8),cv2.COLORMAP_OCEAN),'hazard_map':data_url(hazard)},'hazard_summary':{'shadow_hazards':{'value':round(float(shadow.mean()*100)),'unit':'%','percent':round(float(shadow.mean()*100))},'crater_candidates':{'value':n_circles,'unit':'sites','percent':min(100,n_circles*8)},'rough_terrain':{'value':round(float(rough.mean()*100)),'unit':'%','percent':round(float(rough.mean()*100))},'slope_proxy':{'value':round(float(slope.mean()*100)),'unit':'risk','percent':round(float(slope.mean()*100))},'detail_uncertainty':{'value':round(float(uncertainty.mean()*100)),'unit':'%','percent':round(float(uncertainty.mean()*100))}},'landing_zones':landing_zones,'explanation':{'title':f'{landing_zones[0]["id"]} has the lowest fused image risk','text':'This candidate combines lower shadow coverage, reduced edge texture and a calmer local gradient than surrounding regions. The ranking is a transparent image-processing heuristic, not a flight navigation decision.','metrics':[f'{landing_zones[0]["score"]}/100 zone score','Image-only proxy','Prototype heuristic']}}
