"""
WeCover API — Backend pour l'outil de mesure de toiture
Déployé sur Render.com
"""

import os
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import anthropic

app = FastAPI(title="WeCover API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://b-hers.github.io",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "null",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "WeCover API",
        "version": "0.2.0",
        "status": "running",
        "endpoints": ["/api/cadastre", "/api/analyze"],
    }


# ────────────────────────────────────────────────────────────
# /api/cadastre — OSM Overpass (couvre toute la Belgique)
# ────────────────────────────────────────────────────────────
@app.get("/api/cadastre")
async def get_cadastre(
    lat: float = Query(...),
    lng: float = Query(...),
    region: str = Query("auto"),
    delta: float = Query(0.0008),
):
    south, west = lat - delta, lng - delta
    north, east = lat + delta, lng + delta

    overpass_query = f"""
    [out:json][timeout:15];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out body;
    >;
    out skel qt;
    """

    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            r = await client.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": overpass_query},
                headers={"User-Agent": "WeCover/0.2"},
            )
            r.raise_for_status()
            osm_data = r.json()
            geojson = osm_to_geojson(osm_data)
            return {
                "type": "osm-buildings",
                "data": geojson,
                "count": len(geojson["features"]),
            }
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"OSM error: {e.response.status_code}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cadastre error: {str(e)}")


def osm_to_geojson(osm_data: dict) -> dict:
    nodes = {n["id"]: (n["lon"], n["lat"]) for n in osm_data.get("elements", []) if n.get("type") == "node"}
    features = []
    for el in osm_data.get("elements", []):
        if el.get("type") != "way" or "building" not in el.get("tags", {}):
            continue
        node_ids = el.get("nodes", [])
        if len(node_ids) < 4:
            continue
        coords = [nodes.get(nid) for nid in node_ids if nid in nodes]
        coords = [c for c in coords if c is not None]
        if len(coords) < 4:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        features.append({
            "type": "Feature",
            "properties": {
                "osm_id": el["id"],
                "building": el["tags"].get("building", "yes"),
                "name": el["tags"].get("name", ""),
                "addr_street": el["tags"].get("addr:street", ""),
                "addr_housenumber": el["tags"].get("addr:housenumber", ""),
            },
            "geometry": {"type": "Polygon", "coordinates": [coords]},
        })
    return {"type": "FeatureCollection", "features": features}


# ────────────────────────────────────────────────────────────
# /api/analyze — Claude vision
# ────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = ""
    region: Optional[str] = "wallonia"
    image_base64: Optional[str] = None
    solar_segments: Optional[List[dict]] = None
    building_footprint: Optional[List[dict]] = None
    zoom: int = 21
    image_width: int = 640
    image_height: int = 640


@app.post("/api/analyze")
async def analyze_roof(req: AnalyzeRequest):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    solar_ctx = "Pas de données Solar API."
    if req.solar_segments:
        lines = [
            f"  Pan {i+1}: pente={s.get('pitch',0):.0f}°, azimut={s.get('azimuth',0):.0f}°, surface={s.get('area',0):.0f}m²"
            for i, s in enumerate(req.solar_segments)
        ]
        solar_ctx = "Solar API (pentes/azimuts fiables) :\n" + "\n".join(lines)

    cadastre_ctx = "Pas de cadastre disponible — devine l'emprise du bâtiment principal."
    if req.building_footprint:
        coords = ", ".join(f"({p['lat']:.6f},{p['lng']:.6f})" for p in req.building_footprint[:10])
        suffix = " ..." if len(req.building_footprint) > 10 else ""
        cadastre_ctx = (
            f"EMPRISE CADASTRALE OSM ({len(req.building_footprint)} sommets) :\n"
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
5. Trace les polygones sur les arêtes visibles.

TYPOLOGIES BELGES :
- Mitoyenne 2 façades : 2 pans
- 3 façades : 2 pans + croupe/pignon
- 4 façades simple : 2 ou 4 pans
- Villa croupes : 4 pans
- Décrochement L-shape : 4-6+ pans

Pour chaque pan :
- name : "Pan Sud", "Croupe Est", etc.
- polygon : 4-8 points [[x,y]] pixels
- pitch_deg : pente
- azimuth_deg : direction du bas (0=N, 90=E, 180=S, 270=O)

JSON UNIQUEMENT :
{{"planes":[{{"name":"...","polygon":[[x,y],...],"pitch_deg":35,"azimuth_deg":180}}],"building_eave_height_m":6.0,"building_type":"...","notes":""}}"""
    else:
        prompt = f"""Tu reconstruis la géométrie d'une toiture belge.

ADRESSE : {req.label or f"{req.lat},{req.lng}"}
CENTRE GPS : {req.lat:.6f}, {req.lng:.6f}

{solar_ctx}
{cadastre_ctx}

Pour chaque pan :
- name, offsets_m: [[est_m, nord_m]], pitch_deg, azimuth_deg

JSON UNIQUEMENT :
{{"planes":[{{"name":"Pan Sud","offsets_m":[[-4,-3],[4,-3],[4,3],[-4,3]],"pitch_deg":35,"azimuth_deg":180}}],"building_eave_height_m":6.0,"building_type":"...","notes":""}}"""

    content = []
    if has_image:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": req.image_base64}})
    content.append({"type": "text", "text": prompt})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            messages=[{"role": "user", "content": content}],
        )
        text = message.content[0].text if message.content else ""
        import re, json
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise HTTPException(status_code=502, detail=f"AI returned non-JSON: {text[:200]}")
        result = json.loads(m.group(0))
        result["_coord_mode"] = "pixels" if has_image else "meters"
        result["_image_used"] = has_image
        return result
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-edges")
async def detect_edges():
    return {"status": "not_implemented_yet", "phase": 3}


@app.post("/api/roof-solver")
async def roof_solver():
    return {"status": "not_implemented_yet", "phase": 4}
