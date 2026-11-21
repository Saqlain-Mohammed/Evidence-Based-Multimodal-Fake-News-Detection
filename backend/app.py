"""Veritas local research server. All inference uses real-data v2 checkpoints."""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from accounts import OWNER, account_db, authenticate
from fastapi.staticfiles import StaticFiles
from workbench import router, registry

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='Veritas content verification',version='3.0.0',
    description='Independent models trained from scratch. Research signals, not proof of authenticity. Local single-user server; do not expose publicly without authentication.')
app.add_middleware(CORSMiddleware,
    allow_origin_regex=r'^(https?://(localhost|127\.0\.0\.1)(:\d+)?|chrome-extension://[a-p]+|moz-extension://[a-zA-Z0-9-]+)$',
    allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type','Authorization'])
@app.middleware('http')
async def identify(request, call_next):
    path = account_db()
    if path is None or not request.url.path.startswith('/api/v2/') or request.method == 'OPTIONS':
        return await call_next(request)
    owner = authenticate(path,request.headers.get('Authorization',''))
    if owner is None:
        return JSONResponse({'detail':'A valid workspace access token is required.'},status_code=401,
                            headers={'WWW-Authenticate':'Bearer'})
    handle=OWNER.set(owner)
    try:
        return await call_next(request)
    finally:
        OWNER.reset(handle)


app.include_router(router)

@app.get('/api/health')
def health():
    rows=registry()
    return {'status':'ok','version':'3.0.0','device':'CPU','trained_models':sum(r['status']=='ready' for r in rows),
        'models':{r['id']:r['status'] for r in rows},'synthetic_models_used':False}

@app.get('/api/stats',include_in_schema=False)
def stats(): return RedirectResponse('/api/v2/models')

@app.get('/')
def index(): return FileResponse(ROOT/'frontend/index.html')

app.mount('/static',StaticFiles(directory=ROOT/'frontend'),name='static')

if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=8000)
