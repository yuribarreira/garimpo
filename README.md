# Garimpo

Ferramenta de ETL para extrair valor de dados bagunçados. A versão web roda 100% no navegador; o backend FastAPI adiciona pipelines com Machine Learning e geração de regras por IA.

[garimpo.netlify.app](https://garimpo.netlify.app) — ou abra `frontend/index.html` localmente.

## O que faz

- Extração de CSV, TSV, Excel, JSON/JSONL, APIs REST e SQLite
- Limpeza, normalização, deduplicação e transformação de colunas
- Validação de dados brasileiros: CPF, CNPJ, telefone, CEP, moeda
- Qualificação de leads com RandomForest (modo train/predict)
- Geração de regras de transformação a partir de linguagem natural (Claude ou OpenAI), validadas por AST antes de rodar

A interface web cobre extração, filtros, deduplicação e exportação sem instalar nada. O backend é para pipelines mais pesados.

## Web

Abra [garimpo.netlify.app](https://garimpo.netlify.app) ou `frontend/index.html`. Aceita CSV, TSV, Excel e JSON, exporta CSV, JSON ou Excel.

## Backend

```bash
docker compose up
```

Frontend em `http://localhost:3000`, API em `http://localhost:8000` (docs em `/docs`).

Sem Docker:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Para a geração de regras por IA, defina `ANTHROPIC_API_KEY` ou `OPENAI_API_KEY`. As origens liberadas no CORS vêm de `GARIMPO_ALLOWED_ORIGINS`.

CLI do pipeline:

```bash
python -m app.pipeline --mode train --input data/leads.csv --target qualidade_lead
python -m app.pipeline --mode predict --input novos.csv --output saida.csv
```

## Estrutura

```
garimpo/
├── backend/app/
│   ├── main.py            API FastAPI
│   ├── pipeline.py        ETL + treino/predição de leads
│   ├── ai_rules.py        regras de transformação via IA (validação AST)
│   ├── inference.py       predição a partir de modelo salvo
│   ├── brazilian_utils.py validação e formatação BR
│   ├── base.py / schemas.py
│   └── extractors/        CSV, Excel, JSON, API, SQLite + factory
├── frontend/index.html    interface web standalone
├── data/sample/
├── config.example.yaml
└── docker-compose.yml
```

## API

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/pipeline/execute` | executa o pipeline (train ou batch-predict) |
| `GET`  | `/pipeline/status/{id}` | status de execução |
| `POST` | `/ai/generate-rule` | gera regra de transformação com IA |
| `GET`  | `/download/{id}` | baixa o resultado |
| `GET`  | `/health` | health check |

## Licença

MIT — veja [LICENSE](LICENSE).

Por [@yuribarreira](https://github.com/yuribarreira).
