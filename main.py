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

app = FastAPI(title="WeCover API", version="0.4.0")

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
        "version": "0.4.0",
        "status": "running",
        "endpoints": ["/api/cadastre", "/api/analyze", "/api/detect-edges", "/api/refine-footprint"],
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
    source: Optional[str] = "auto"
    canny_low: Optional[int] = 50
    canny_high: Optional[int] = 150
    hough_threshold: Optional[int] = 80
    min_line_px: Optional[int] = 40
    max_line_gap: Optional[int] = 10
    min_length_m: Optional[float] = 2.5
    building_footprint: Optional[List[dict]] = None   # [{lat,lng}] — filtre les lignes au bâtiment


class RefineFootprintRequest(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = ""
    region: Optional[str] = "auto"
    osm_footprint: List[dict]    # [{lat,lng}] — polygone OSM à affiner
    zoom: Optional[int] = 21
    image_base64: Optional[str] = None   # image satellite optionnelle (sinon backend la fetch)


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
            continue
        cnt = 0
        for j, lj in enumerate(raw_lines):
            if i == j or lj["length_m"] > PANEL_MAX_LEN:
                continue
            adiff = abs(li["angle_deg"] - lj["angle_deg"])
            adiff = min(adiff, 180 - adiff)
            if adiff > PARALLEL_TOL_DEG:
                continue
            ddx = li["_cx"] - lj["_cx"]
            ddy = li["_cy"] - lj["_cy"]
            if math.sqrt(ddx * ddx + ddy * ddy) > NEIGHBOR_RADIUS_M:
                continue
            cnt += 1
        if cnt >= MIN_NEIGHBORS:
            panel_indices.add(i)

    # ── Filtre par proximité au bâtiment cadastral (si fourni) ────────
    # On garde uniquement les lignes dont au moins UNE extrémité est à
    # moins de FOOTPRINT_MARGIN_M du polygone cadastral.
    FOOTPRINT_MARGIN_M = 5.0
    far_indices = set()
    if req.building_footprint and len(req.building_footprint) >= 3:
        # Convertir le footprint en mètres relatifs au centre
        fp_m = [
            ((p["lng"] - lng) * lngM, (p["lat"] - lat) * latM)
            for p in req.building_footprint
        ]

        def dist_point_to_polygon_m(px_m, py_m):
            """Distance min d'un point au polygone (mètres). 0 si à l'intérieur."""
            # Point-in-polygon (ray casting)
            inside = False
            n = len(fp_m)
            j = n - 1
            for k in range(n):
                xi, yi = fp_m[k]
                xj, yj = fp_m[j]
                if ((yi > py_m) != (yj > py_m)) and \
                   (px_m < (xj - xi) * (py_m - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = k
            if inside:
                return 0.0
            # Distance min aux segments
            min_d = float("inf")
            for k in range(n):
                ax, ay = fp_m[k]
                bx, by = fp_m[(k + 1) % n]
                dx, dy = bx - ax, by - ay
                ll = dx * dx + dy * dy
                if ll == 0:
                    d = math.hypot(px_m - ax, py_m - ay)
                else:
                    t = max(0, min(1, ((px_m - ax) * dx + (py_m - ay) * dy) / ll))
                    proj_x = ax + t * dx
                    proj_y = ay + t * dy
                    d = math.hypot(px_m - proj_x, py_m - proj_y)
                if d < min_d:
                    min_d = d
            return min_d

        for i, li in enumerate(raw_lines):
            cx_m = li["_cx"]
            cy_m = li["_cy"]
            # Convertir extrémités en mètres relatifs au centre
            p1x = (li["lng1"] - lng) * lngM
            p1y = (li["lat1"] - lat) * latM
            p2x = (li["lng2"] - lng) * lngM
            p2y = (li["lat2"] - lat) * latM
            # Au moins une extrémité ou le centre doit être proche
            d1 = dist_point_to_polygon_m(p1x, p1y)
            d2 = dist_point_to_polygon_m(p2x, p2y)
            dc = dist_point_to_polygon_m(cx_m, cy_m)
            if min(d1, d2, dc) > FOOTPRINT_MARGIN_M:
                far_indices.add(i)

    # Séparer résultats
    keep_lines, panel_lines = [], []
    excluded_count = len(far_indices)
    for i, li in enumerate(raw_lines):
        if i in far_indices:
            continue  # trop loin du bâtiment
        out = {k: v for k, v in li.items() if not k.startswith("_")}
        if i in panel_indices:
            panel_lines.append(out)
        else:
            keep_lines.append(out)

    keep_lines.sort(key=lambda x: -x["length_m"])
    panel_lines.sort(key=lambda x: -x["length_m"])

    return {
        "lines": keep_lines,
        "panels": panel_lines,
        "count": len(keep_lines),
        "panel_count": len(panel_lines),
        "raw_count": len(raw_lines),
        "filtered_far_count": excluded_count,
        "source": source_used,
        "image_size": [W, H],
        "filters": {
            "min_length_m": min_length_m,
            "canny": [canny_low, canny_high],
            "footprint_margin_m": FOOTPRINT_MARGIN_M if req.building_footprint else None,
            "panel_detection": {
                "max_length_m": PANEL_MAX_LEN,
                "parallel_tol_deg": PARALLEL_TOL_DEG,
                "neighbor_radius_m": NEIGHBOR_RADIUS_M,
                "min_neighbors": MIN_NEIGHBORS,
            },
        },
    }


# ────────────────────────────────────────────────────────────
# /api/refine-footprint — Claude affine le polygone cadastral OSM
# ────────────────────────────────────────────────────────────
@app.post("/api/refine-footprint")
async def refine_footprint(req: RefineFootprintRequest):
    """
    Prend le polygone OSM cadastral (70-80% précis) + image satellite,
    demande à Claude de raffiner le contour pour qu'il colle EXACTEMENT
    à la toiture visible.
    """
    import io, math, json, re
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    if not req.osm_footprint or len(req.osm_footprint) < 3:
        raise HTTPException(status_code=400, detail="osm_footprint requis (>= 3 sommets)")

    lat, lng = req.lat, req.lng
    zoom = req.zoom or 21
    W, H = 640, 640
    latM = 111320
    lngM = 111320 * math.cos(math.radians(lat))

    # Capture image si pas fournie
    img_b64 = req.image_base64
    if not img_b64:
        gkey = os.environ.get("GOOGLE_MAPS_API_KEY", "")
        if not gkey:
            raise HTTPException(status_code=400, detail="image_base64 ou GOOGLE_MAPS_API_KEY requis")
        img_url = (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lng}&zoom={zoom}&size={W}x{H}&scale=2"
            f"&maptype=satellite&key={gkey}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(img_url)
            r.raise_for_status()
            import base64
            img_b64 = base64.b64encode(r.content).decode("ascii")

    # Convertir le footprint en offsets mètres pour Claude
    fp_meters = [
        {
            "east_m": round((p["lng"] - lng) * lngM, 2),
            "north_m": round((p["lat"] - lat) * latM, 2),
        }
        for p in req.osm_footprint
    ]

    prompt = f"""Tu affines un polygone cadastral OSM pour qu'il colle EXACTEMENT à la toiture visible sur l'image satellite.

ADRESSE : {req.label or f"{lat},{lng}"}
IMAGE : {W}×{H}px centrée sur ({lat:.6f}, {lng:.6f})

POLYGONE OSM ACTUEL (à affiner — {len(fp_meters)} sommets, ~70-80% précis) :
{json.dumps(fp_meters, indent=2)}

MISSION : Retourne un polygone affiné dont les sommets coïncident PRÉCISÉMENT avec les arêtes externes de la toiture (gouttières, débords, pignons) visibles dans l'image.

RÈGLES :
1. Garde la forme générale du polygone OSM (même topologie L, U, rectangle, etc.)
2. Ajuste chaque sommet pour qu'il tombe sur une vraie arête de toiture visible
3. Ajoute des sommets supplémentaires si la toiture a des décrochements non capturés par OSM
4. Supprime les sommets aberrants (sortie OSM imprécise)
5. Conserve les angles ~90° quand le bâtiment est orthogonal
6. Le polygone final doit être fermé (premier sommet = dernier ignoré, le système ferme automatiquement)

Réponds UNIQUEMENT en JSON valide :
{{
  "refined_footprint": [
    {{"east_m": 0.0, "north_m": 0.0}},
    ...
  ],
  "confidence": 0.85,
  "notes": "Ajustement de 4 sommets sur la façade rue, ajout d'un décrochement arrière"
}}"""

    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": prompt},
    ]

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = message.content[0].text if message.content else ""
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise HTTPException(status_code=502, detail=f"AI non-JSON: {text[:200]}")
        result = json.loads(m.group(0))

        # Convertir refined_footprint en lat/lng pour le frontend
        refined_ll = []
        for p in result.get("refined_footprint", []):
            refined_ll.append({
                "lat": lat + p["north_m"] / latM,
                "lng": lng + p["east_m"] / lngM,
            })

        return {
            "refined_footprint": refined_ll,
            "refined_count": len(refined_ll),
            "original_count": len(req.osm_footprint),
            "confidence": result.get("confidence"),
            "notes": result.get("notes", ""),
        }

    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/roof-solver")
async def roof_solver():
    return {"status": "not_implemented_yet", "phase": 4}
