from pathlib import Path
import gc
import urllib.request
import warnings
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from astropy.io import fits
from astropy.wcs import WCS

BASE_URL="https://raw.githubusercontent.com/gear66me-ui/JWST/d056523edefdd1deb874e3c2e66b60d1a2fcedf3/JWST_0224_MODEL_E_MANUAL_MASK.py"
source=urllib.request.urlopen(BASE_URL,timeout=60).read().decode("utf-8")
source=source.replace('if __name__=="__main__":\n    main()','')
ns={"__name__":"jwst_0224_base"}
exec(compile(source,"JWST_0224_MODEL_E_MANUAL_MASK.py","exec"),ns)

VERSION="JWST_0225"
ns["VERSION"]=VERSION
ns["MODEL"]="MODEL_E_TABLET_APERTURE"

class DummySelector:
    def set_active(self,value):
        pass

def ellipse_vertices(cx,cy,rx,ry,angle_deg,n=96):
    t=np.linspace(0,2*np.pi,n,endpoint=False)
    a=np.deg2rad(angle_deg)
    x=rx*np.cos(t)
    y=ry*np.sin(t)
    xr=x*np.cos(a)-y*np.sin(a)+cx
    yr=x*np.sin(a)+y*np.cos(a)+cy
    return np.column_stack((xr,yr))

def main():
    warnings.filterwarnings("ignore")
    CUBE=ns["CUBE"]
    if not CUBE.exists():
        raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header
        data=hdul[0].data
        fax=ns["spec_axis"](h)
        freq=ns["axis_values"](h,fax)
        np_spec=data.ndim-fax
        aw=WCS(h).celestial
        tx0,ty0=aw.world_to_pixel_values(ns["RA_DEG"],ns["DEC_DEG"])
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0))
        pix=abs(float(h["CDELT2"]))*3600
        bmaj=float(h.get("BMAJ",0))*3600
        bmin=float(h.get("BMIN",0))*3600
        half=64
        ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
        ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq)))*1000
        k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-ns["TARGET_GHZ"])))
        lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim
            sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(ns["collapse"](np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1:
                print(f"5 MHz cube: {j+1:3d}/{len(groups):3d}")
        del data
        gc.collect()

    nearest_group=int(np.argmin(np.abs(centers-ns["TARGET_GHZ"])))
    preview=np.nanmean(cube[max(0,nearest_group-1):min(len(cube),nearest_group+2)],axis=0)

    cx=widgets.IntSlider(value=64,min=20,max=108,step=1,description="Center X")
    cy=widgets.IntSlider(value=64,min=20,max=108,step=1,description="Center Y")
    rx=widgets.IntSlider(value=6,min=2,max=30,step=1,description="Radius X")
    ry=widgets.IntSlider(value=6,min=2,max=30,step=1,description="Radius Y")
    ang=widgets.IntSlider(value=0,min=-90,max=90,step=5,description="Angle°")
    runb=widgets.Button(description="Run Model E",button_style="success",icon="play")
    status=widgets.HTML("<b>Adjust the aperture sliders, then tap Run Model E.</b>")
    out=widgets.Output()

    def redraw(*_):
        with out:
            clear_output(wait=True)
            verts=ellipse_vertices(cx.value,cy.value,rx.value,ry.value,ang.value)
            fig,ax=plt.subplots(figsize=(7,7))
            ax.imshow(preview,origin="lower",cmap="viridis",interpolation="nearest")
            ax.plot(np.r_[verts[:,0],verts[0,0]],np.r_[verts[:,1],verts[0,1]],lw=2)
            ax.scatter([txc],[tyc],marker="+",s=200,c="white",linewidths=2)
            ax.set_title("JWST_0225 TABLET APERTURE PREVIEW")
            plt.show()

    state={
        "completed":False,
        "cube":cube,
        "centers":centers,
        "preview":preview,
        "pix":pix,
        "txc":txc,
        "tyc":tyc,
        "bmaj":bmaj,
        "bmin":bmin,
        "selector_figure":plt.figure(),
        "selector":DummySelector()
    }
    plt.close(state["selector_figure"])

    def run_clicked(_):
        if state["completed"]:
            return
        runb.disabled=True
        verts=ellipse_vertices(cx.value,cy.value,rx.value,ry.value,ang.value)
        ns["run_analysis"](verts,state,status)

    for w in (cx,cy,rx,ry,ang):
        w.observe(redraw,names="value")
    runb.on_click(run_clicked)
    display(widgets.VBox([status,widgets.HBox([cx,cy]),widgets.HBox([rx,ry]),ang,runb,out]))
    redraw()

if __name__=="__main__":
    main()
