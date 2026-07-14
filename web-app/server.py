import os, sys, json, io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "research-agent"))
sys.path.insert(0, str(BASE / "rag-qa"))

import database as db
from multi_agent import MultiAgentSystem
from review_ai import STYLES, generate_reply
from loader import load_file, chunk_text
from vector_store import VectorStore

app = FastAPI(title="AI 工具箱 SaaS")
kb_stores = {}
rag_chains = {}

def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "): raise HTTPException(401)
    user = db.get_user_by_token(authorization[7:])
    if not user: raise HTTPException(401)
    return user

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8"))

# Auth
@app.post("/api/auth/register")
async def register(username: str = Form(...), password: str = Form(...)):
    r = db.create_user(username, password)
    return JSONResponse(r, 400) if "error" in r else r
@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    r = db.login(username, password)
    return JSONResponse(r, 400) if "error" in r else r
@app.get("/api/auth/me")
async def me():
    try: return get_current_user()
    except HTTPException: return JSONResponse({"error":"未登录"},401)

# KB
@app.post("/api/kb/create")
async def kb_create(name: str = Form(...), user=Header(None)):
    u = get_current_user(user); return db.create_kb(u["id"], name)
@app.get("/api/kb/list")
async def kb_list(user=Header(None)):
    u = get_current_user(user); return db.list_kbs(u["id"])
@app.delete("/api/kb/{kb_id}")
async def kb_delete(kb_id: int, user=Header(None)):
    u = get_current_user(user); db.delete_kb(u["id"], kb_id)
    kb_stores.pop(kb_id, None); rag_chains.pop(kb_id, None); return {"ok":True}
@app.post("/api/kb/{kb_id}/upload")
async def kb_upload(kb_id: int, file: UploadFile = File(...), user=Header(None)):
    u = get_current_user(user)
    d = Path(__file__).parent / "uploads" / str(u["id"]) / str(kb_id); d.mkdir(parents=True, exist_ok=True)
    (d / file.filename).write_bytes(await file.read())
    text = load_file(str(d / file.filename))
    chunks = chunk_text(text, 500)
    store = VectorStore(mode="semantic"); store.create_collection("kb_"+str(kb_id))
    store.add_documents(chunks, [{"source":file.filename,"chunk":i,"kb_id":kb_id} for i in range(len(chunks))])
    kb_stores[kb_id] = store
    return {"status":"ok","filename":file.filename,"chunks":len(chunks),"chars":len(text)}

# RAG ask
@app.post("/api/kb/{kb_id}/ask")
async def kb_ask(kb_id: int, question: str = Form(...), api_key: str = Form(""), user=Header(None)):
    u = get_current_user(user)
    store = kb_stores.get(kb_id)
    if not store: return {"status":"error","msg":"请先上传文档"}
    key = api_key or os.getenv("OPENAI_API_KEY","")
    if not key:
        results = store.search(question,3)
        return {"status":"ok","answer":"(需要 API Key)","retrieved":[{"text":d[:300],"source":m.get("source",""),"similarity":s} for d,s,m in results]}
    from qa_chain import RAGChain
    chain = RAGChain(store, api_key=key)
    result = chain.ask(question, top_k=3, verbose=False)
    db.save_chat(u["id"],"user",question,kb_id)
    db.save_chat(u["id"],"assistant",result["answer"],kb_id)
    return {"status":"ok","answer":result["answer"],"sources":result.get("sources",[])}

# Agent
@app.post("/api/agent/research")
async def agent_research(topic: str = Form(...), api_key: str = Form(""), mode: str = Form("single"), user=Header(None)):
    try: u = get_current_user(user); uid = u["id"]
    except: uid = 0
    key = api_key or os.getenv("OPENAI_API_KEY","")
    async def gen():
        yield f"data: {json.dumps({'type':'start','topic':topic,'mode':mode},ensure_ascii=False)}\n\n"
        if not key: yield f"data: {json.dumps({'type':'error','msg':'请设置 API Key'},ensure_ascii=False)}\n\n"; return
        buf = io.StringIO(); old = sys.stdout; sys.stdout = buf; log = ""; report = ""
        try:
            if mode == "multi":
                mas = MultiAgentSystem(api_key=key); log, report = mas.run(topic)
            else:
                from agent import ResearchAgent
                a = ResearchAgent(api_key=key); a.max_steps = 5; report = a.run(topic)
        finally: sys.stdout = old
        log = buf.getvalue()
        yield f"data: {json.dumps({'type':'log','content':log},ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type':'result','report':report},ensure_ascii=False)}\n\n"
        if uid and report: db.save_agent_log(uid, topic, log, report)
    return StreamingResponse(gen(), media_type="text/event-stream")

# History
@app.get("/api/history/agent")
async def hist_agent(user=Header(None)): return db.get_agent_logs(get_current_user(user)["id"])
@app.get("/api/history/agent/{lid}")
async def hist_detail(lid: int, user=Header(None)):
    r = db.get_agent_log(get_current_user(user)["id"], lid)
    if not r: raise HTTPException(404); return r
@app.get("/api/history/chat/{kb_id}")
async def hist_chat(kb_id: int, user=Header(None)): return db.get_chat_history(get_current_user(user)["id"], kb_id)

# ==================== 口碑助手 ====================
@app.get("/api/review/styles")
async def review_styles():
    return [{"key":k,"name":v["name"],"icon":v["icon"],"category":"差评回复" if k in ["sincere","compensate","improve","professional","gentle"] else "好评回复"} for k,v in STYLES.items()]

@app.post("/api/review/generate")
async def review_gen(review: str = Form(...), style: str = Form("sincere"), rating: int = Form(3), api_key: str = Form(""), user=Header(None)):
    key = api_key or os.getenv("OPENAI_API_KEY","")
    if not key: return {"status":"error","msg":"请设置 API Key"}
    if style not in STYLES: return {"status":"error","msg":"未知风格"}
    return {"status":"ok","reply":generate_reply(key,review,style,rating),"style":STYLES[style]["name"]}

if __name__ == "__main__":
    port = int(os.getenv("PORT","8080"))
    print(f"\n  AI 工具箱 SaaS 已启动: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
