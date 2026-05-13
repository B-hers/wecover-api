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
        "version": "0.3.0",
        "status": "running",
        "endpoints": ["/api/cadastre", "/api/analyze", "/api/detect-edges"],
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


class EdgeDetectRequest(BaseModel):
    lat: float
    lng: float
    zoom: Optional[int] = 21
    source: Optional[str] = "auto"          # auto | wallonia | google
    canny_low: Optional[int] = 50
    canny_high: Optional[int] = 150
    hough_threshold: Optional[int] = 80
    min_line_px: Optional[int] = 40
    max_line_gap: Optional[int] = 10
    min_length_m: Optional[float] = 2.5


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
async def detect_edges(req: EdgeDetectRequest):
    """
    Détection d'arêtes de toiture via OpenCV Canny + Hough.
    Sources : orthophoto IGN Wallonie / Geopunt Flandre / Google Static Maps fallback.
    Filtre les panneaux solaires (lignes en grille régulière) mais GARDE les lucarnes et chiens-assis.
    """
    import io, math
    from PIL import Image
    import numpy as np
    import cv2

    lat, lng = req.lat, req.lng
    zoom = req.zoom or 21
    W, H = 1024, 1024

    # Bbox de l'image en degrés (approximation locale Belgique)
    latM = 111320
    lngM = 111320 * math.cos(math.radians(lat))
    dlat = (H / 2) / latM
    dlng = (W / 2) / lngM

    # ── Sélection source imagery ──────────────────────────────────────
    # Auto-détection région par bounds approximatives
    region = req.source
    if region == "auto":
        # Bruxelles : lat 50.76-50.92, lng 4.24-4.49
        if 50.76 < lat < 50.92 and 4.24 < lng < 4.49:
            region = "brussels"
        # Flandre : nord de 50.78, ouest de 5.92
        elif lat > 50.78 and lng < 5.92:
            region = "flanders"
        else:
            region = "wallonia"

    img_url = None
    source_used = None
    if region == "wallonia":
        img_url = (
            "https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_2021/MapServer/WMSServer"
            f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=0&STYLES=&FORMAT=image/jpeg"
            f"&CRS=EPSG:4326&BBOX={lat-dlat},{lng-dlng},{lat+dlat},{lng+dlng}"
            f"&WIDTH={W}&HEIGHT={H}"
        )
        source_used = "wallonia-ign-25cm"
    elif region == "flanders":
        img_url = (
            "https://geo.api.vlaanderen.be/OMWRGBMRVL/wms"
            f"?service=WMS&version=1.3.0&request=GetMap&layers=omwrgbmrvl"
            f"&styles=&format=image/jpeg&CRS=EPSG:4326"
            f"&BBOX={lat-dlat},{lng-dlng},{lat+dlat},{lng+dlng}&WIDTH={W}&HEIGHT={H}"
        )
        source_used = "flanders-geopunt-25cm"

    # Fallback Google Static Maps (Bruxelles ou échec WMS)
    if not img_url or region == "brussels":
        gkey = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        if not gkey:
            raise HTTPException(
                status_code=400,
                detail=f"Région {region} : pas d'orthophoto WMS et GOOGLE_MAPS_API_KEY non configuré"
            )
        img_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lng}&zoom={zoom}&size={W//2}x{H//2}&scale=2"
            f"&maptype=satellite&key={gkey}"
        )
        source_used = source_used or "google-satellite"

    # ── Téléchargement avec fallback Google si WMS échoue ─────────────
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(img_url)
            r.raise_for_status()
            img_bytes = r.content
        except Exception as e:
            # Fallback automatique sur Google si WMS national échoue
            gkey = os.environ.get("GOOGLE_MAPS_API_KEY", "")
            if gkey and "google" not in source_used:
                fallback_url = (
                    f"https://maps.googleapis.com/maps/api/staticmap?"
                    f"center={lat},{lng}&zoom={zoom}&size={W//2}x{H//2}&scale=2"
                    f"&maptype=satellite&key={gkey}"
                )
                try:
                    r = await client.get(fallback_url)
                    r.raise_for_status()
                    img_bytes = r.content
                    source_used = source_used + "+fallback-google"
                except Exception as e2:
                    raise HTTPException(status_code=502, detail=f"Image fetch failed (WMS + Google): {e2}")
            else:
                raise HTTPException(status_code=502, detail=f"Image fetch failed: {e}")

    # ── Décodage et prétraitement ─────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        np_img = np.array(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image decode failed: {e}")

    # CLAHE pour rehausser le contraste local
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    np_img = clahe.apply(np_img)

    # Flou gaussien anti-bruit
    blurred = cv2.GaussianBlur(np_img, (5, 5), 1.0)

    # Canny
    canny_low = req.canny_low or 50
    canny_high = req.canny_high or 150
    edges = cv2.Canny(blurred, canny_low, canny_high, apertureSize=3)

    # Hough probabiliste
    hough_thresh = req.hough_threshold or 70
    min_line_px = req.min_line_px or 25      # ~5-6m à zoom 21 en Belgique
    max_gap_px = req.max_line_gap or 10

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_px,
        maxLineGap=max_gap_px,
    )

    if lines is None:
        return {"lines": [], "panels": [], "count": 0, "source": source_used, "image_size": [W, H]}

    # ── Conversion pixels → lat/lng ───────────────────────────────────
    def px_to_ll(px, py):
        latv = lat + dlat - (py / H) * 2 * dlat
        lngv = lng - dlng + (px / W) * 2 * dlng
        return latv, lngv

    min_length_m = req.min_length_m or 1.0   # ↓ Garde lucarnes/chiens-assis (>1m)
    raw_lines = []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        lat1, lng1 = px_to_ll(x1, y1)
        lat2, lng2 = px_to_ll(x2, y2)
        dx = (lng2 - lng1) * lngM
        dy = (lat2 - lat1) * latM
        length_m = math.sqrt(dx * dx + dy * dy)
        if length_m < min_length_m:
            continue
        angle = (math.degrees(math.atan2(dy, dx)) + 180) % 180
        # Centre en mètres pour clustering
        cx = ((lng1 + lng2) / 2) * lngM
        cy = ((lat1 + lat2) / 2) * latM
        raw_lines.append({
            "lat1": round(lat1, 7), "lng1": round(lng1, 7),
            "lat2": round(lat2, 7), "lng2": round(lng2, 7),
            "length_m": round(length_m, 2),
            "angle_deg": round(angle, 1),
            "_cx": cx, "_cy": cy,
        })

    # ── Filtre panneaux solaires : grille parallèle régulière ─────────
    # Heuristique : un panneau solaire = ligne courte (<2.5m) ayant 3+ voisins
    # parallèles (±8°) à moins de 2.5m. Les lucarnes/chiens-assis sont isolés.
    panel_indices = set()
    PANEL_MAX_LEN = 2.5
    PARALLEL_TOL_DEG = 8
    NEIGHBOR_RADIUS_M = 2.5
    MIN_NEIGHBORS = 3

    for i, li in enumerate(raw_lines):
        if li["length_m"] > PANEL_MAX_LEN:
            continue  # ligne longue = pas un panneau
        cnt = 0
        for j, lj in enumerate(raw_lines):
            if i == j or lj["length_m"] > PANEL_MAX_LEN:
                continue
            # Parallélisme
            adiff = abs(li["angle_deg"] - lj["angle_deg"])
            adiff = min(adiff, 180 - adiff)
            if adiff > PARALLEL_TOL_DEG:
                continue
            # Distance entre centres
            ddx = li["_cx"] - lj["_cx"]
            ddy = li["_cy"] - lj["_cy"]
            if math.sqrt(ddx * ddx + ddy * ddy) > NEIGHBOR_RADIUS_M:
                continue
            cnt += 1
        if cnt >= MIN_NEIGHBORS:
            panel_indices.add(i)

    # Séparer résultats
    keep_lines, panel_lines = [], []
    for i, li in enumerate(raw_lines):
        out = {k: v for k, v in li.items() if not k.startswith("_")}
        if i in panel_indices:
            panel_lines.append(out)
        else:
            keep_lines.append(out)

    # Tri par longueur décroissante
    keep_lines.sort(key=lambda x: -x["length_m"])
    panel_lines.sort(key=lambda x: -x["length_m"])

    return {
        "lines": keep_lines,           # arêtes principales + lucarnes/chiens-assis
        "panels": panel_lines,         # lignes identifiées comme panneaux solaires
        "count": len(keep_lines),
        "panel_count": len(panel_lines),
        "raw_count": len(raw_lines),
        "source": source_used,
        "image_size": [W, H],
        "filters": {
            "min_length_m": min_length_m,
            "canny": [canny_low, canny_high],
            "panel_detection": {
                "max_length_m": PANEL_MAX_LEN,
                "parallel_tol_deg": PARALLEL_TOL_DEG,
                "neighbor_radius_m": NEIGHBOR_RADIUS_M,
                "min_neighbors": MIN_NEIGHBORS,
            },
        },
    }


@app.post("/api/roof-solver")
async def roof_solver():
    return {"status": "not_implemented_yet", "phase": 4}
