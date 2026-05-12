"""
WeCover API — Backend pour l'outil de mesure de toiture
Déployé sur Render.com

Endpoints :
  GET  /                    Healthcheck
  GET  /api/cadastre        Récupère le bâti PICC/GRB (proxy CORS)
  POST /api/analyze         Analyse IA via Claude (clé côté serveur)
  POST /api/detect-edges    Detection d'arêtes OpenCV (Phase 3)
  POST /api/roof-solver     Reconstruction topologique (Phase 4)
"""

import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic

app = FastAPI(title="WeCover API", version="0.1.0")

# CORS : autoriser le frontend GitHub Pages + localhost dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://b-hers.github.io",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",  # VS Code Live Server
        "null",  # local file:// for testing
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Healthcheck ──────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "WeCover API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": ["/api/cadastre", "/api/analyze"],
    }


# ────────────────────────────────────────────────────────────
# /api/cadastre — Proxy PICC (Wallonie) / GRB (Flandre)
# ────────────────────────────────────────────────────────────
@app.get("/api/cadastre")
async def get_cadastre(
    lat: float = Query(..., description="Latitude WGS84"),
    lng: float = Query(..., description="Longitude WGS84"),
    region: str = Query("wallonia", description="wallonia | flanders | brussels"),
    delta: float = Query(0.0008, description="Half-extent in degrees (default ~80m)"),
):
    """
    Récupère les bâtiments cadastraux autour d'un point.
    Wallonia → PICC Constructions Principales (layer 3)
    Flanders/Brussels → GRB Gbg layer
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            if region == "wallonia":
                # PICC = Plan de vue en Croix - couche 3 = bâtiments principaux
                url = "https://geoservices.wallonie.be/arcgis/rest/services/TOPOGRAPHIE/PICC_VDIFF/MapServer/3/query"
                params = {
                    "geometry": f"{lng-delta},{lat-delta},{lng+delta},{lat+delta}",
                    "geometryType": "esriGeometryEnvelope",
                    "inSR": "4326",
                    "spatialRel": "esriSpatialRelIntersects",
                    "outFields": "*",
                    "f": "geojson",
                    "outSR": "4326",
                }
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                return {"type": "wallon-picc", "data": data}

            elif region in ("flanders", "brussels"):
                # GRB Gbg = Gebouw aan de Grond (building footprint)
                url = "https://geo.api.vlaanderen.be/GRB/wfs"
                params = {
                    "service": "WFS",
                    "version": "2.0.0",
                    "request": "GetFeature",
                    "typeNames": "GRBgis:Gbg",
                    "outputFormat": "application/json",
                    "srsName": "EPSG:4326",
                    "bbox": f"{lat-delta},{lng-delta},{lat+delta},{lng+delta},EPSG:4326",
                }
                r = await client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                return {"type": "grb", "data": data}

            else:
                raise HTTPException(status_code=400, detail=f"Unknown region: {region}")

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Upstream {region} error: {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────
# /api/analyze — Claude vision pour analyse toiture
# ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = ""
    region: Optional[str] = "wallonia"
    image_base64: Optional[str] = None  # JPEG base64 sans préfixe data:
    solar_segments: Optional[List[dict]] = None
    building_footprint: Optional[List[dict]] = None  # [{lat,lng},...]
    zoom: int = 21
    image_width: int = 640
    image_height: int = 640


@app.post("/api/analyze")
async def analyze_roof(req: AnalyzeRequest):
    """
    Envoie l'image satellite + contexte à Claude pour identifier les pans.
    La clé API Anthropic est côté serveur (jamais exposée au frontend).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    # Build context
    solar_ctx = "Pas de données Solar API."
    if req.solar_segments:
        lines = [
            f"  Pan {i+1}: pente={s.get('pitch',0):.0f}°, azimut={s.get('azimuth',0):.0f}°, surface={s.get('area',0):.0f}m²"
            for i, s in enumerate(req.solar_segments)
        ]
        solar_ctx = "Solar API (pentes/azimuts fiables) :\n" + "\n".join(lines)

    cadastre_ctx = "Pas de cadastre disponible."
    if req.building_footprint:
        coords = ", ".join(f"({p['lat']:.6f},{p['lng']:.6f})" for p in req.building_footprint[:10])
        suffix = " ..." if len(req.building_footprint) > 10 else ""
        cadastre_ctx = (
            f"EMPRISE CADASTRALE PICC/GRB ({len(req.building_footprint)} sommets) :\n"
            f"{coords}{suffix}\n"
            f"→ Tous les pans DOIVENT être dans cette emprise. Couvre toute l'emprise."
        )

    has_image = bool(req.image_base64)

    if has_image:
        import math
        px_per_m = (256 * (2 ** req.zoom)) * math.cos(math.radians(req.lat)) / 40075016
        prompt = f"""Tu mesures une toiture belge depuis une image satellite pour un devis ITE.

ADRESSE : {req.label or f"{req.lat},{req.lng}"}
IMAGE : {req.image_width}×{req.image_height}px, zoom {req.zoom}, centre = pixel ({req.image_width//2},{req.image_height//2})
ÉCHELLE : 1 mètre = {px_per_m:.1f} pixels

{solar_ctx}

{cadastre_ctx}

MISSION : Identifie TOUS les pans de toiture du bâtiment principal.

RÈGLES :
1. Si emprise cadastrale fournie : reste à l'intérieur, couvre l'ensemble.
2. Pas de limite arbitraire : 2 à 10 pans selon complexité réelle.
3. Fusionne les pans de même pente ET azimut (±10°).
4. Ignore annexes, garages, abris hors emprise, lucarnes <5m².
5. Trace les polygones sur les arêtes visibles (faîtages, gouttières, noues).

TYPOLOGIES BELGES :
- Mitoyenne 2 façades : 2 pans
- 3 façades : 2 pans + croupe/pignon
- 4 façades simple : 2 ou 4 pans
- Villa croupes : 4 pans
- Décrochement L-shape : 4-6+ pans
- Toit plat + extension : 1 plat + 2-4 inclinés

Pour chaque pan :
- name : "Pan Sud", "Croupe Est", etc.
- polygon : 4-8 points [[x,y]] pixels
- pitch_deg : pente
- azimuth_deg : direction du bas (0=N, 90=E, 180=S, 270=O)

Globalement : building_eave_height_m, building_type, notes.

JSON UNIQUEMENT, sans autre texte :
{{"planes":[{{"name":"...","polygon":[[x,y],...],"pitch_deg":35,"azimuth_deg":180}}],"building_eave_height_m":6.0,"building_type":"...","notes":""}}"""
    else:
        prompt = f"""Tu reconstruis la géométrie d'une toiture belge pour un devis ITE.

ADRESSE : {req.label or f"{req.lat},{req.lng}"}
CENTRE GPS : {req.lat:.6f}, {req.lng:.6f}

{solar_ctx}

{cadastre_ctx}

RÈGLES :
1. Si emprise cadastrale fournie, dessine dans cette emprise.
2. Pentes/azimuts Solar API = valeurs exactes à utiliser.
3. Fusionne les pans de même azimut (±15°).
4. Architecture belge : 2 pans mitoyen, 4 pans villa, plus si décrochements.

Pour chaque pan :
- name : "Pan Sud", "Pan Nord", etc.
- offsets_m : [[est_m, nord_m]] mètres depuis centre GPS
- pitch_deg : valeur Solar API
- azimuth_deg : valeur Solar API

JSON UNIQUEMENT :
{{"planes":[{{"name":"Pan Sud","offsets_m":[[-4,-3],[4,-3],[4,3],[-4,3]],"pitch_deg":35,"azimuth_deg":180}}],"building_eave_height_m":6.0,"building_type":"...","notes":""}}"""

    # Build message content
    content = []
    if has_image:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": req.image_base64,
            },
        })
    content.append({"type": "text", "text": prompt})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2500,
            messages=[{"role": "user", "content": content}],
        )
        text = message.content[0].text if message.content else ""

        # Parse JSON from response
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise HTTPException(status_code=502, detail=f"AI returned non-JSON: {text[:200]}")

        import json
        result = json.loads(m.group(0))
        result["_coord_mode"] = "pixels" if has_image else "meters"
        result["_image_used"] = has_image
        return result

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Invalid JSON from AI: {e.msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────
# /api/detect-edges — Phase 3 (sera implémenté plus tard)
# ────────────────────────────────────────────────────────────
@app.post("/api/detect-edges")
async def detect_edges():
    return {"status": "not_implemented_yet", "phase": 3}


# ────────────────────────────────────────────────────────────
# /api/roof-solver — Phase 4 (sera implémenté plus tard)
# ────────────────────────────────────────────────────────────
@app.post("/api/roof-solver")
async def roof_solver():
    return {"status": "not_implemented_yet", "phase": 4}
