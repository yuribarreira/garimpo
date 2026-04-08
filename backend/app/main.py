import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .ai_rules import AIRuleGenerator
from .pipeline import LeadPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
TMP_DIR = Path(tempfile.gettempdir()) / "garimpo"
ACTIVE = {"iniciando", "carregando", "processando", "treinando", "predizendo"}


class AIRuleRequest(BaseModel):
    requirement: str
    data_context: Optional[Dict] = None


app = FastAPI(title="Garimpo ETL API", version="1.0.0")

origins = [o.strip() for o in os.getenv("GARIMPO_ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

state = {"pipelines": {}, "rule_generator": None,
         "stats": {"pipelines": 0, "records": 0, "rules": 0}}

if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.on_event("startup")
def startup() -> None:
    try:
        state["rule_generator"] = AIRuleGenerator()
        logger.info("gerador de regras IA pronto")
    except RuntimeError as e:
        logger.warning(f"IA indisponível: {e}")


@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"service": "Garimpo ETL API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "garimpo", "timestamp": datetime.now().isoformat()}


@app.post("/pipeline/execute")
async def execute_pipeline(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    mode: str = Form(...),
    target_column: str = Form("qualidade_lead"),
    model_dir: str = Form("./production_models/"),
    enable_enrichment: bool = Form(False),
    output_filename: str = Form("resultado.csv"),
):
    if mode not in ("train", "batch-predict"):
        raise HTTPException(400, "mode deve ser 'train' ou 'batch-predict'")

    pid = f"pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    src = TMP_DIR / f"{pid}_{file.filename}"
    content = await file.read()
    src.write_bytes(content)

    state["pipelines"][pid] = {
        "status": "iniciando", "progress": 0.0, "message": "fila",
        "start_time": datetime.now().isoformat(),
        "file": {"name": file.filename, "size": len(content)},
    }
    background.add_task(_run, pid, str(src), mode, target_column, model_dir,
                        enable_enrichment, output_filename)
    return {"pipeline_id": pid, "status": "iniciada"}


def _run(pid, src, mode, target_column, model_dir, enrich, out_name) -> None:
    def upd(status, progress, message, details=None):
        state["pipelines"][pid].update(
            status=status, progress=progress, message=message,
            last_update=datetime.now().isoformat())
        if details:
            state["pipelines"][pid]["details"] = details

    try:
        upd("carregando", 20.0, "carregando dados")
        pipe = LeadPipeline(model_dir=model_dir, enable_api_calls=enrich)
        df = pipe.run_etl(src)

        stats = {"records": len(df), "columns": len(df.columns)}
        upd("processando", 60.0, f"{len(df)} registros processados", stats)

        out_dir = TMP_DIR / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{pid}_{out_name}"

        if mode == "train":
            upd("treinando", 80.0, "treinando modelo")
            pipe.train_model(df, target_column=target_column)
            df.to_csv(out_file, index=False, sep=";", encoding="utf-8-sig")
        else:
            upd("predizendo", 80.0, "gerando predições")
            df = pipe.batch_predict(df)
            df.to_csv(out_file, index=False, sep=";", encoding="utf-8-sig")

        state["stats"]["pipelines"] += 1
        state["stats"]["records"] += len(df)
        upd("concluida", 100.0, "concluído", {"output_file": str(out_file), **stats})
        logger.info(f"pipeline {pid} concluída")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"pipeline {pid} falhou: {e}")
        upd("erro", 0.0, str(e), {"error_type": type(e).__name__})
    except Exception as e:
        logger.exception(f"erro inesperado na pipeline {pid}")
        upd("erro", 0.0, f"erro interno: {e}", {"error_type": type(e).__name__})


@app.get("/pipeline/status/{pid}")
def pipeline_status(pid: str):
    if pid not in state["pipelines"]:
        raise HTTPException(404, "pipeline não encontrada")
    return state["pipelines"][pid]


@app.get("/pipeline/list")
def list_pipelines():
    return {"pipelines": state["pipelines"], "stats": state["stats"]}


@app.post("/ai/generate-rule")
def generate_ai_rule(req: AIRuleRequest):
    gen = state["rule_generator"]
    if not gen:
        raise HTTPException(503, "gerador de regras IA indisponível, confira as API keys")
    try:
        rule = gen.generate_transformation_rule(req.requirement, req.data_context)
    except ValueError as e:
        raise HTTPException(422, str(e))
    state["stats"]["rules"] += 1
    return {"success": True, "rule": rule}


@app.get("/download/{pid}")
def download(pid: str):
    p = state["pipelines"].get(pid)
    if not p:
        raise HTTPException(404, "pipeline não encontrada")
    if p["status"] != "concluida":
        raise HTTPException(400, "pipeline ainda não concluída")
    out = Path(p.get("details", {}).get("output_file", ""))
    if not out.exists():
        raise HTTPException(404, "arquivo de resultado não encontrado")
    return FileResponse(str(out), media_type="application/octet-stream",
                        filename=f"garimpo_{pid}.csv")


@app.get("/stats")
def stats():
    return {
        "stats": state["stats"],
        "active": sum(1 for p in state["pipelines"].values() if p["status"] in ACTIVE),
        "total": len(state["pipelines"]),
        "ai_available": state["rule_generator"] is not None,
    }
