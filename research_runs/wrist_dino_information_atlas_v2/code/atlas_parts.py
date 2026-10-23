"""Explicit 80-image annotation/review workflow. Unreviewed masks are never truth."""
from __future__ import annotations
import base64
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from atlas_common import ROOT,atomic_json,atomic_text,config,digest,load_json,require,sha256_file
from atlas_math import balanced_token_indices,crop_map,grouped_folds,patch_reduce,cluster_interval


HTML=r'''<!doctype html><meta charset="utf-8"><title>DINO 部件核对</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:20px auto}canvas{width:448px;height:448px;image-rendering:auto;border:1px solid #888}button,select{margin:8px;padding:8px}label{display:inline-block;margin:5px}</style>
<h1>80 张部件核对</h1><p>点击顶点画多边形，点击“完成区域”。为每类画区域，或标记不可见/无法辨认。未标注像素保持未知，不作背景负例。</p>
<div id="name"></div><canvas id="c" width="224" height="224"></canvas><br>
<select id="cls"></select><button onclick="finish()">完成区域</button><button onclick="undo()">撤销区域</button>
<div id="missing"></div><label><input id="review" type="checkbox" onchange="save()">本图已核对</label><br>
<button onclick="go(-1)">上一张</button><button onclick="go(1)">下一张</button><button onclick="download()">下载标注 JSON</button>
<input type="file" accept="application/json" onchange="restore(this)"><p id="count"></p>
<script>const DATA=__DATA__;let i=0,points=[],img=new Image();const canvas=document.getElementById('c'),ctx=canvas.getContext('2d');
const colors=['#000','#ff5050','#ffb000','#48dc65','#358bff','#b063ff','#dddddd'];
const out=DATA.images.map(x=>({id:x.id,reviewed:false,polygons:[],unobservable:[]}));
DATA.classes.slice(1).forEach((x,k)=>{let o=document.createElement('option');o.value=k+1;o.textContent=x;document.getElementById('cls').append(o)});
function save(){out[i].reviewed=document.getElementById('review').checked;out[i].unobservable=Array.from(document.querySelectorAll('#missing input:checked')).map(x=>Number(x.value));localStorage.setItem(DATA.signature,JSON.stringify(out));count()}
function count(){document.getElementById('count').textContent=out.filter(x=>x.reviewed).length+'/80 张已核对；'+(i+1)+'/80'}
function draw(){ctx.clearRect(0,0,224,224);ctx.drawImage(img,0,0,224,224);for(const p of out[i].polygons){ctx.beginPath();p.points.forEach((v,k)=>k?ctx.lineTo(...v):ctx.moveTo(...v));ctx.closePath();ctx.fillStyle=colors[p.class_id]+'55';ctx.strokeStyle=colors[p.class_id];ctx.fill();ctx.stroke()}if(points.length){ctx.beginPath();points.forEach((v,k)=>k?ctx.lineTo(...v):ctx.moveTo(...v));ctx.strokeStyle=colors[Number(document.getElementById('cls').value)];ctx.stroke()}}
function load(){points=[];document.getElementById('name').textContent=DATA.images[i].id;img.onload=draw;img.src=DATA.images[i].image;document.getElementById('review').checked=out[i].reviewed;const e=document.getElementById('missing');e.innerHTML='';DATA.classes.slice(1).forEach((n,k)=>{const l=document.createElement('label'),a=document.createElement('input');a.type='checkbox';a.value=k+1;a.checked=out[i].unobservable.includes(k+1);a.onchange=save;l.append(a,document.createTextNode(n+' 不可见/无法辨认'));e.append(l)});count()}
canvas.onclick=e=>{const b=canvas.getBoundingClientRect();points.push([Math.max(0,Math.min(223,(e.clientX-b.left)*224/b.width)),Math.max(0,Math.min(223,(e.clientY-b.top)*224/b.height))]);draw()};
function finish(){if(points.length<3)return;out[i].polygons.push({class_id:Number(document.getElementById('cls').value),points});points=[];out[i].reviewed=false;document.getElementById('review').checked=false;save();draw()}
function undo(){points=[];out[i].polygons.pop();out[i].reviewed=false;document.getElementById('review').checked=false;save();draw()}
function go(d){save();i=Math.max(0,Math.min(79,i+d));load()}
function download(){save();const b=new Blob([JSON.stringify({signature:DATA.signature,annotations:out},null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='parts_annotations.json';a.click()}
function restore(input){const f=input.files[0];if(!f)return;const r=new FileReader();r.onload=()=>{const j=JSON.parse(r.result);if(j.signature!==DATA.signature||j.annotations.length!==80){alert('标注不属于这批图');return}j.annotations.forEach((a,k)=>out[k]=a);save();load()};r.readAsText(f)}
try{const s=JSON.parse(localStorage.getItem(DATA.signature));if(s&&s.length===80)s.forEach((a,k)=>out[k]=a)}catch(e){}load();</script>'''


def export_review(prep,rows):
    folder=ROOT/'parts'; folder.mkdir(exist_ok=True); lookup={r['state_id']:r for r in rows}; images=[]
    for item in prep['parts_selection']:
        row=lookup[item['state_id']]
        with np.load(ROOT/f"data/task_{row['task_id']:02d}/states/{row['state_id']}.npz",allow_pickle=False) as z:
            rgb=z[item['view']+'_rgb']
        crop=np.clip(crop_map(rgb).round(),0,255).astype(np.uint8)
        buffer=io.BytesIO(); Image.fromarray(crop).save(buffer,format='PNG')
        images.append({**item,'id':item['state_id']+'__'+item['view'],
                       'image_sha256':__import__('hashlib').sha256(crop.tobytes()).hexdigest(),
                       'image':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()})
    manifest={'images':[{k:v for k,v in im.items() if k!='image'} for im in images],
              'capture_signature':prep['signature'],'classes':config()['parts']['classes']}
    sid=digest(manifest); manifest['signature']=sid
    atomic_json(manifest,folder/'REVIEW_MANIFEST.json')
    atomic_text(HTML.replace('__DATA__',json.dumps({'images':images,'classes':manifest['classes'],'signature':sid})),folder/'review.html')
    return manifest


def validate_annotations(value,manifest):
    require(value.get('signature')==manifest['signature'],'Annotation image/signature mismatch')
    rows=value.get('annotations',[]); expected={r['id'] for r in manifest['images']}
    require(len(rows)==80 and {r['id'] for r in rows}==expected,'Need exactly 80 unique image reviews')
    result={}
    for row in rows:
        require(row.get('reviewed') is True,'Unreviewed image: '+row['id'])
        unavailable=set(row.get('unobservable',[])); polygons=row.get('polygons',[])
        require(unavailable<=set(range(1,7)),'Unknown part class')
        seen=set(); image=Image.new('I',(224,224),0); draw=ImageDraw.Draw(image)
        for polygon in polygons:
            klass=polygon['class_id']; points=np.asarray(polygon['points'],dtype=float)
            require(klass in range(1,7) and points.ndim==2 and points.shape[1]==2 and len(points)>=3,
                    'Invalid part polygon')
            require(np.isfinite(points).all() and (points>=0).all() and (points<=223).all(),'Polygon outside image')
            require(klass not in unavailable,'Part is both annotated and unobservable')
            draw.polygon([tuple(p) for p in points],fill=klass); seen.add(klass)
        require(seen|unavailable==set(range(1,7)),'Each part needs a polygon or an explicit unobservable flag')
        result[row['id']]=np.asarray(image,dtype=np.int16)
    return result


def import_annotations(path):
    manifest=load_json(ROOT/'parts/REVIEW_MANIFEST.json'); value=load_json(path)
    masks=validate_annotations(value,manifest); folder=ROOT/'parts/reviewed'
    require(not folder.exists(),'Reviewed annotations are immutable; use a new explicit revision')
    folder.mkdir()
    for key,mask in masks.items(): np.save(folder/(key+'.npy'),mask,allow_pickle=False)
    atomic_json(value,folder/'ANNOTATIONS.json')
    atomic_json({'status':'REVIEWED_80','signature':manifest['signature'],'images':80,
                 'annotation_sha256':sha256_file(folder/'ANNOTATIONS.json'),
                 'masks':{str(p.relative_to(ROOT)):sha256_file(p) for p in folder.glob('*.npy')}},folder/'STATUS.json')


def review_status(prep):
    path=ROOT/'parts/reviewed/STATUS.json'
    if not path.exists(): return {'status':'PENDING_REVIEW','expected_images':80,'reviewed_images':0}
    r=load_json(path); manifest=load_json(ROOT/'parts/REVIEW_MANIFEST.json')
    require(r['signature']==manifest['signature'] and manifest['capture_signature']==prep['signature'],
            'Reviewed parts belong to different inputs')
    require(all(sha256_file(ROOT/p)==h for p,h in r['masks'].items()),'Reviewed masks changed')
    require(sha256_file(ROOT/'parts/reviewed/ANNOTATIONS.json')==r['annotation_sha256'],'Annotations changed')
    return r


def parts_readouts(rows,signature):
    from atlas_analysis import panel_paths
    from atlas_local import audit_tokens
    from atlas_common import commit,receipt_valid,prepared
    status=review_status(prepared())
    if status['status']!='REVIEWED_80': return []
    manifest=load_json(ROOT/'parts/REVIEW_MANIFEST.json'); lookup={r['state_id']:r for r in rows}; output=[]
    annotation_signature=digest([signature,status['annotation_sha256']])
    for view in config()['views']:
        selected=[lookup[r['state_id']] for r in manifest['images'] if r['view']==view]
        train=[r for r in selected if r['split']=='discovery']; test=[r for r in selected if r['split']=='validation']
        for block in config()['detail_blocks']:
            key=f'{view}__b{block:02d}'; folder=panel_paths(key,'parts')
            if receipt_valid(folder/'COMPLETE.json',annotation_signature):
                output.extend(load_json(folder/'SUMMARY.json')); continue
            arrays={}; x=[]; targets=[]; owner=[]; weights=[]; token_ids=[]; validation=[]; truths=[]; valids=[]
            for split,items in [('discovery',train),('validation',test)]:
                for i,row in enumerate(items):
                    mask=np.load(ROOT/f"parts/reviewed/{row['state_id']}__{view}.npy",allow_pickle=False)
                    counts=np.stack([patch_reduce((mask==c).astype(float)) for c in range(1,7)],1)
                    valid=counts.sum(1)>0; labels=np.where(valid,counts.argmax(1),-1)
                    task=row['task_id']
                    if task not in arrays:
                        arrays[task]=np.load(ROOT/f'data/task_{task:02d}/dense/{view}__dense_b{block:02d}.npy',mmap_mode='r')
                    features=np.asarray(arrays[task][row['row_index']],dtype=np.float32)
                    if split=='discovery':
                        ids=balanced_token_indices(labels,32,42+i)
                        target=np.zeros((256,6)); target[valid]=np.eye(6)[labels[valid]]; targets.append(target)
                        x.append(features[ids]); weights.extend(counts.sum(1)[ids]); owner.extend([i]*len(ids)); token_ids.extend(ids)
                    else:
                        validation.append(features); truths.append(labels); valids.append(valid)
            full=np.asarray(targets); seen=np.flatnonzero(full.sum((0,1))>0)
            if len(seen)<2 or not np.asarray(valids).any() or len(np.unique(grouped_folds(train)[np.asarray(owner,dtype=int)]))<2:
                summary=[{'site':key,'status':'INSUFFICIENT_REVIEWED_CLASS_VARIATION_OR_FOLDS',
                          'train_classes':len(seen),'train_images':len(train),'test_images':len(test),
                          'labelled_train_tokens':len(owner),'labelled_test_tokens':int(np.asarray(valids).sum())}]
            else:
                summary=audit_tokens(folder,key,np.concatenate(x),full,np.asarray(owner),np.asarray(token_ids),
                                     np.concatenate(validation),train,test,np.asarray(truths),np.asarray(valids),
                                     config()['parts']['classes'][1:],np.asarray(weights))
                for r in summary: r.update(train_images=len(train),test_images=len(test),reviewed_images_total=80)
            atomic_json(summary,folder/'SUMMARY.json')
            commit(folder/'COMPLETE.json',[p for p in folder.glob('*') if p.is_file()],annotation_signature,
                   annotation_sha256=status['annotation_sha256'])
            output.extend(summary)
    return output
