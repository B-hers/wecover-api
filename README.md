# WeCover API

Backend pour l'outil de mesure de toiture WeCover.

Frontend : https://b-hers.github.io/We-Cover---Mesures-Auto/

## Endpoints

- `GET /` — Healthcheck
- `GET /api/cadastre?lat=&lng=&region=` — Proxy PICC/GRB
- `POST /api/analyze` — Analyse IA via Claude
- `POST /api/detect-edges` — (à venir) Détection d'arêtes OpenCV
- `POST /api/roof-solver` — (à venir) Reconstruction topologique

## Local dev

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --reload
```

Accessible sur http://localhost:8000

## Déploiement Render

Auto-deploy depuis ce repo via `render.yaml`.

Variable d'environnement à configurer dans Render Dashboard :
- `ANTHROPIC_API_KEY` — clé API Anthropic
