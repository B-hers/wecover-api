# WeCover Backend API v0.5.0
# Belgian-only orthophoto sources (Wallonia, Flanders) — no Google Static Maps dependency
# Endpoints: /api/cadastre, /api/analyze, /api/detect-edges, /api/refine-footprint

import os
import json
import httpx
import anthropic
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(title="WeCover API", version="0.5.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════

class CadastreRequest(BaseModel):
    lat: float
    lng: float


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
    hough_threshold: Optional[int] = 70
    min_line_px: Optional[int] = 25
    max_line_gap: Optional[int] = 10
    min_length_m: Optional[float] = 1.0
    building_footprint: Optional[List[dict]] = None


class RefineFootprintRequest(BaseModel):
    lat: float
    lng: float
    label: Optional[str] = ""
    region: Optional[str] = "auto"
    osm_footprint: List[dict]
    zoom: Optional[int] = 21
    image_base64: Optional[str] = None


# ════════════════════════════════════════════════════════════════════════
# HELPER: Fetch Belgian WMS orthophoto
# ════════════════════════════════════════════════════════════════════════

async def fetch_belgian_orthophoto(lat: float, lng: float, width: int, height: int, region: str = "auto") -> bytes:
    """
    Télécharge une orthophoto depuis les WMS belges (Wallonie ou Flandre).
    Retourne les bytes JPEG de l'image.
    Lève HTTPException si échec.
    """
    import math
    
    # Compute bbox - target 50m coverage for better resolution (~0.08m/px at 640px)
    # instead of previous 640m coverage (~1m/px)
    target_width_m = 50  # meters of real-world coverage
    latM = 111320
    lngM = 111320 * math.cos(math.radians(lat))
    dlat = (target_width_m / 2) / latM
    dlng = (target_width_m / 2) / lngM
    
    # Auto-detect region
    if region == "auto":
        if 50.76 < lat < 50.92 and 4.24 < lng < 4.49:
            region = "brussels"
        elif lat > 50.78 and lng < 5.92:
            region = "flanders"
        else:
            region = "wallonia"
    
    # Select WMS source
    if region == "wallonia":
        img_url = (
            "https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_2021/MapServer/WMSServer"
            f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=0&STYLES=&FORMAT=image/jpeg"
            f"&CRS=EPSG:4326&BBOX={lat-dlat},{lng-dlng},{lat+dlat},{lng+dlng}"
            f"&WIDTH={width}&HEIGHT={height}"
        )
        source = "wallonia-ign-25cm"
    elif region == "flanders":
        img_url = (
            "https://geo.api.vlaanderen.be/OMWRGBMRVL/wms"
            f"?service=WMS&version=1.3.0&request=GetMap&layers=omwrgbmrvl"
            f"&styles=&format=image/jpeg&CRS=EPSG:4326"
            f"&BBOX={lat-dlat},{lng-dlng},{lat+dlat},{lng+dlng}&WIDTH={width}&HEIGHT={height}"
        )
        source = "flanders-geopunt-25cm"
    elif region == "brussels":
        # UrbIS Brussels Orthophoto WMS (officiel Bruxelles Mobilité)
        # Source: https://data.mobility.brussels/info/Ortho
        # CRITICAL: Layer name is "Ortho" not "ortho2021"
        img_url = (
            "https://geoservices-urbis.irisnet.be/geoserver/urbisgrid/ows"
            f"?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap&LAYERS=urbisgrid:Ortho"
            f"&STYLES=&FORMAT=image/jpeg&CRS=EPSG:4326"
            f"&BBOX={lat-dlat},{lng-dlng},{lat+dlat},{lng+dlng}"
            f"&WIDTH={width}&HEIGHT={height}"
        )
        source = "brussels-urbis-ortho"
    else:
        raise HTTPException(status_code=400, detail=f"Région '{region}' non supportée")
    
    # Download
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.get(img_url)
            r.raise_for_status()
            img_bytes = r.content
            if len(img_bytes) < 1000:
                raise HTTPException(status_code=502, detail=f"WMS {source} image invalide (trop petite)")
            
            # Quick check: if image is all white/blank (common WMS error), reject it
            # Sample first 100 bytes - if all are 0xFF (white JPEG), it's likely blank
            if img_bytes[:100].count(0xFF) > 95:
                raise HTTPException(status_code=502, detail=f"WMS {source} image blanche (hors couverture)")
            
            return img_bytes
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=502,
                detail=f"WMS {source} HTTP {e.response.status_code} — service indisponible"
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"WMS {source} erreur : {str(e)[:100]}")


# ════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════════════

@app.get("/")
async def healthcheck():
    return {
        "service": "WeCover API",
        "version": "0.5.0",
        "status": "running",
        "endpoints": ["/api/cadastre", "/api/analyze", "/api/detect-edges", "/api/refine-footprint"],
    }


@app.get("/api/cadastre")
async def get_cadastre(lat: float, lng: float):
    """OSM Overpass query for building footprints in Belgium with fallback."""
    radius_m = 80
    query = f"""
    [out:json][timeout:15];
    (
      way["building"](around:{radius_m},{lat},{lng});
      relation["building"](around:{radius_m},{lat},{lng});
    );
    out geom;
    """
    
    # Try primary Overpass server, fallback to secondary if it fails
    overpass_servers = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    
    last_error = None
    async with httpx.AsyncClient(timeout=20.0) as client:
        for server_url in overpass_servers:
            try:
                r = await client.post(
                    server_url,
                    data={"data": query},
                    headers={"User-Agent": "WeCoverAPI/0.5.0 (roof measurement tool)"},
                )
                r.raise_for_status()
                data = r.json()
                
                # Convert OSM to GeoJSON format expected by frontend
                features = []
                for elem in data.get("elements", []):
                    if elem["type"] == "way" and "geometry" in elem:
                        coords = [[pt["lon"], pt["lat"]] for pt in elem["geometry"]]
                        features.append({
                            "type": "Feature",
                            "id": elem["id"],
                            "geometry": {"type": "Polygon", "coordinates": [coords]},
                            "properties": {}
                        })
                    elif elem["type"] == "relation" and "members" in elem:
                        for member in elem["members"]:
                            if member["role"] == "outer" and "geometry" in member:
                                coords = [[pt["lon"], pt["lat"]] for pt in member["geometry"]]
                                features.append({
                                    "type": "Feature",
                                    "id": f"{elem['id']}-{member.get('ref', 0)}",
                                    "geometry": {"type": "Polygon", "coordinates": [coords]},
                                    "properties": {}
                                })
                
                return {
                    "type": "FeatureCollection",
                    "data": {
                        "type": "FeatureCollection",
                        "features": features
                    }
                }
            
            except Exception as e:
                last_error = e
                continue  # Try next server
        
        # All servers failed
        raise HTTPException(
            status_code=502,
            detail=f"OSM Overpass error (tried {len(overpass_servers)} servers): {str(last_error)[:150]}"
        )


@app.post("/api/analyze-deep")
async def analyze_roof_deep(req: AnalyzeRequest):
    """
    Deep analysis with 3 sequential phases:
    1. Inventory: identify all buildings in cadastre
    2. Detail: analyze each building separately with dedicated Claude call
    3. Validate: reconcile all results and verify consistency
    
    Takes 1-3 minutes but produces professional-grade results.
    """
    try:
        # Fetch WMS image
        region = req.region or detect_region(req.lat, req.lng)
        img_base64 = await fetch_wms_image(
            req.lat, req.lng, region, 50  # 50m target width
        )
        
        # PHASE 1: Inventory all buildings
        cadastre_json = json.dumps(req.building_footprint) if req.building_footprint else "Non fournie"
        inventory_prompt = f"""Tu es un expert en analyse de toitures depuis images aériennes.

MISSION : Inventorie TOUS les bâtiments distincts visibles dans l'emprise cadastrale fournie.

EMPRISE CADASTRALE (polygone orange) :
{cadastre_json}

Pour CHAQUE bâtiment distinct à l'intérieur de l'emprise :
1. Assigne un ID unique (1, 2, 3...)
2. Classe le type : "principal", "annexe", "garage", "extension", "autre"
3. Décris l'orientation dominante : "N-S", "E-O", "NE-SO", "NO-SE"
4. Estime la complexité de la toiture :
   - "simple" : 2 pans symétriques
   - "moyenne" : 3-4 pans, quelques décrochements
   - "complexe" : 5+ pans, forme en L/U, multiples faîtages
5. Note toute particularité visible : lucarnes, toiture terrasse partielle, décrochements

RÈGLES :
- Ignore les structures HORS emprise cadastrale
- Sépare les bâtiments clairement distincts (différents volumes, orientations différentes)
- Un garage accolé = bâtiment séparé si sa toiture est indépendante
- Toiture continue sur L ou U = 1 seul bâtiment

RETOURNE UNIQUEMENT un objet JSON valide (pas de markdown, pas de texte avant/après) :
{{
  "buildings": [
    {{
      "id": 1,
      "type": "principal",
      "orientation": "E-O",
      "complexity": "simple",
      "notes": "Toiture 2 pans classique, tuiles orangées"
    }}
  ],
  "total_buildings": 1
}}"""

        inventory_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {"type": "text", "text": inventory_prompt}
                ]
            }]
        )
        
        inventory_text = inventory_response.content[0].text.strip()
        # Remove markdown fences if present
        if inventory_text.startswith("```"):
            inventory_text = inventory_text.split("```")[1]
            if inventory_text.startswith("json"):
                inventory_text = inventory_text[4:]
        inventory = json.loads(inventory_text)
        
        # PHASE 2: Detailed analysis per building
        all_results = []
        for building in inventory["buildings"]:
            detail_prompt = f"""Tu es un expert en analyse de toitures depuis images aériennes.

MISSION : Analyse approfondie du bâtiment ID {building['id']} UNIQUEMENT.

CONTEXTE DU BÂTIMENT :
- Type : {building['type']}
- Orientation : {building['orientation']}
- Complexité : {building['complexity']}
- Notes : {building.get('notes', 'Aucune')}

EMPRISE CADASTRALE (polygone orange) :
{cadastre_json}

RÈGLES STRICTES :
1. Concentre-toi UNIQUEMENT sur ce bâtiment, ignore complètement les autres structures visibles
2. IMPÉRATIF : TOUS les sommets de TOUS les polygones DOIVENT être à l'intérieur de l'emprise cadastrale. Vérifie chaque coin.
3. Prends le temps d'identifier chaque pan de toiture avec précision
4. Orientation : utilise les arêtes visibles (faîtages, gouttières). Le Nord est en haut de l'image.
5. Azimut : 0°=Nord, 90°=Est, 180°=Sud, 270°=Ouest. Précision ±5°.
6. Si un pan fait <3m², c'est probablement une lucarne → ignore
7. Fusionne les pans de même pente ET azimut (±10°)
8. Vérifie 3 fois que les polygones ne sortent PAS du cadastre

RETOURNE UNIQUEMENT un objet JSON valide (pas de markdown, pas de texte avant/après) :
{{
  "building_id": {building['id']},
  "planes": [
    {{
      "id": "pan_1",
      "orientation_deg": 178,
      "slope_deg": 32,
      "area_m2": 87.3,
      "polygon": [[lon1, lat1], [lon2, lat2], ...]
    }}
  ],
  "total_area_m2": 145.6,
  "confidence": 0.92,
  "warnings": []
}}

Les coordonnées des polygones doivent être en longitude/latitude (système GPS).
"""

            detail_response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {"type": "text", "text": detail_prompt}
                    ]
                }]
            )
            
            detail_text = detail_response.content[0].text.strip()
            if detail_text.startswith("```"):
                detail_text = detail_text.split("```")[1]
                if detail_text.startswith("json"):
                    detail_text = detail_text[4:]
            detail_result = json.loads(detail_text)
            all_results.append(detail_result)
            
            # Small delay between requests to avoid rate limits
            await asyncio.sleep(0.5)
        
        # PHASE 3: Global validation
        validation_prompt = f"""Tu es un expert en analyse de toitures depuis images aériennes.

MISSION : Validation globale des analyses de toitures.

RÉSULTATS PAR BÂTIMENT :
{json.dumps(all_results, indent=2)}

EMPRISE CADASTRALE :
{cadastre_json}

TÂCHES DE VALIDATION :
1. Vérifie qu'aucun polygone ne sort de l'emprise cadastrale
2. Détecte les chevauchements entre toitures (si 2 polygones de bâtiments différents se superposent)
3. Vérifie la cohérence des surfaces totales (somme plausible pour la zone visible)
4. Signale les anomalies (orientations improbables, pentes extrêmes >60°, surfaces aberrantes)
5. Si des corrections sont nécessaires, applique-les

RETOURNE UNIQUEMENT un objet JSON valide :
{{
  "validated": true,
  "buildings": [...],  // Liste complète avec corrections éventuelles
  "total_area_m2": 255.3,
  "warnings": [],
  "corrections_applied": []
}}"""

        validation_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {"type": "text", "text": validation_prompt}
                ]
            }]
        )
        
        validation_text = validation_response.content[0].text.strip()
        if validation_text.startswith("```"):
            validation_text = validation_text.split("```")[1]
            if validation_text.startswith("json"):
                validation_text = validation_text[4:]
        final_result = json.loads(validation_text)
        
        return final_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze")
async def analyze_roof(req: AnalyzeRequest):
    """
    Claude roof analysis with Belgian WMS imagery fallback.
    If frontend doesn't provide image_base64, backend fetches from WMS.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    
    # Solar API context
    solar_ctx = ""
    if req.solar_segments:
        segs = [f"Pan {i+1}: {s.get('area_m2',0):.1f}m², pente {s.get('pitch',0):.0f}°, az {s.get('azimuth',0):.0f}°"
                for i, s in enumerate(req.solar_segments[:10])]
        solar_ctx = f"SOLAR API GOOGLE ({len(req.solar_segments)} pans) :\n" + "\n".join(segs)
    
    # Cadastre context
    cadastre_ctx = ""
    if req.building_footprint:
        coords = ", ".join(f"({p['lat']:.6f},{p['lng']:.6f})" for p in req.building_footprint[:10])
        suffix = " ..." if len(req.building_footprint) > 10 else ""
        cadastre_ctx = (
            f"EMPRISE CADASTRALE OSM ({len(req.building_footprint)} sommets) :\n"
            f"{coords}{suffix}\n"
            f"→ Tous les pans DOIVENT être dans cette emprise. Couvre toute l'emprise."
        )
    
    # Fetch image if not provided
    if not req.image_base64:
        try:
            img_bytes = await fetch_belgian_orthophoto(
                req.lat, req.lng, req.image_width, req.image_height, req.region
            )
            import base64
            req.image_base64 = base64.b64encode(img_bytes).decode("ascii")
        except HTTPException:
            # WMS failed → GPS-only mode
            pass
    
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

RÈGLES STRICTES :
1. IMPÉRATIF : Si emprise cadastrale fournie, TOUS les sommets de TOUS les polygones DOIVENT être à l'intérieur de cette emprise. AUCUN sommet ne peut sortir. Vérifie chaque coin.
2. Orientation : utilise les arêtes visibles (faîtages, gouttières) pour déterminer les directions réelles. Le Nord est en haut de l'image.
3. Pas de limite arbitraire : 2 à 10 pans selon complexité réelle.
4. Fusionne les pans de même pente ET azimut (±10°).
5. Ignore annexes, garages, abris hors emprise, lucarnes <5m².

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
            model="claude-sonnet-4-6",  # Available with API key (verified via /v1/models)
            max_tokens=3000,
            messages=[{"role": "user", "content": content}],
        )
        text = message.content[0].text if message.content else ""
        
        import json, re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise HTTPException(status_code=502, detail=f"AI non-JSON: {text[:200]}")
        result = json.loads(m.group(0))
        
        # Add coordinate mode flag for frontend
        result["_coord_mode"] = "pixels" if has_image else "meters"
        result["_has_image"] = has_image
        result["_source"] = "belgian-wms" if (has_image and not req.image_base64) else ("frontend" if has_image else "gps-only")
        
        return result
    
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect-edges")
async def detect_edges(req: EdgeDetectRequest):
    """
    Détection d'arêtes OpenCV Canny + Hough sur orthophoto WMS belge.
    Filtre par building_footprint si fourni.
    """
    import io, math
    from PIL import Image
    import numpy as np
    import cv2
    
    lat, lng = req.lat, req.lng
    zoom = req.zoom or 21
    W, H = 1024, 1024
    
    # CRITICAL: Image covers 50m real-world (set in fetch_belgian_orthophoto)
    # NOT 1024m as previously calculated
    target_width_m = 50
    latM = 111320
    lngM = 111320 * math.cos(math.radians(lat))
    dlat = (target_width_m / 2) / latM
    dlng = (target_width_m / 2) / lngM
    
    # Fetch image
    img_bytes = await fetch_belgian_orthophoto(lat, lng, W, H, req.source)
    
    # Decode
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    np_img = np.array(img)
    
    # CLAHE + blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    np_img = clahe.apply(np_img)
    blurred = cv2.GaussianBlur(np_img, (5, 5), 1.0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, req.canny_low or 50, req.canny_high or 150, apertureSize=3)
    
    # CRITICAL: Apply cadastre mask BEFORE Hough to reduce noise
    # Create mask from building_footprint polygon (white inside, black outside)
    if req.building_footprint and len(req.building_footprint) >= 3:
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Convert cadastre lat/lng to pixel coordinates
        pts = []
        for p in req.building_footprint:
            px = int((p["lng"] - (lng - dlng)) / (2 * dlng) * W)
            py = int(((lat + dlat) - p["lat"]) / (2 * dlat) * H)
            pts.append([px, py])
        
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)  # White inside polygon
        
        # Expand mask by 15 pixels to catch edges on boundary (was 5, too restrictive)
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Apply mask: keep only edges inside/near cadastre
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Hough line detection (lowered threshold to 50 to catch more roof edges)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=req.hough_threshold or 50,  # Was 70, now 50 for better detection
        minLineLength=req.min_line_px or 20,  # Was 25, now 20 to catch smaller features
        maxLineGap=req.max_line_gap or 10,
    )
    
    if lines is None:
        return {"lines": [], "panels": [], "count": 0, "source": req.source}
    
    # Convert to lat/lng
    def px_to_ll(px, py):
        latv = lat + dlat - (py / H) * 2 * dlat
        lngv = lng - dlng + (px / W) * 2 * dlng
        return latv, lngv
    
    min_length_m = req.min_length_m or 1.0
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
        cx = ((lng1 + lng2) / 2) * lngM
        cy = ((lat1 + lat2) / 2) * latM
        raw_lines.append({
            "lat1": round(lat1, 7), "lng1": round(lng1, 7),
            "lat2": round(lat2, 7), "lng2": round(lng2, 7),
            "length_m": round(length_m, 2),
            "angle_deg": round(angle, 1),
            "_cx": cx, "_cy": cy,
        })
    
    # Panel detection (solar panels = short parallel lines)
    panel_indices = set()
    for i, li in enumerate(raw_lines):
        if li["length_m"] > 2.5:
            continue
        cnt = 0
        for j, lj in enumerate(raw_lines):
            if i == j or lj["length_m"] > 2.5:
                continue
            adiff = abs(li["angle_deg"] - lj["angle_deg"])
            adiff = min(adiff, 180 - adiff)
            if adiff > 8:
                continue
            ddx = li["_cx"] - lj["_cx"]
            ddy = li["_cy"] - lj["_cy"]
            if math.sqrt(ddx * ddx + ddy * ddy) > 2.5:
                continue
            cnt += 1
        if cnt >= 3:
            panel_indices.add(i)
    
    # Footprint filter
    FOOTPRINT_MARGIN_M = 5.0
    far_indices = set()
    if req.building_footprint and len(req.building_footprint) >= 3:
        fp_m = [((p["lng"] - lng) * lngM, (p["lat"] - lat) * latM) for p in req.building_footprint]
        
        def dist_point_to_polygon_m(px_m, py_m):
            # Point-in-polygon
            inside = False
            n = len(fp_m)
            j = n - 1
            for k in range(n):
                xi, yi = fp_m[k]
                xj, yj = fp_m[j]
                if ((yi > py_m) != (yj > py_m)) and (px_m < (xj - xi) * (py_m - yi) / (yj - yi + 1e-12) + xi):
                    inside = not inside
                j = k
            if inside:
                return 0.0
            # Distance to edges
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
            p1x = (li["lng1"] - lng) * lngM
            p1y = (li["lat1"] - lat) * latM
            p2x = (li["lng2"] - lng) * lngM
            p2y = (li["lat2"] - lat) * latM
            cx_m = li["_cx"]
            cy_m = li["_cy"]
            d1 = dist_point_to_polygon_m(p1x, p1y)
            d2 = dist_point_to_polygon_m(p2x, p2y)
            dc = dist_point_to_polygon_m(cx_m, cy_m)
            if min(d1, d2, dc) > FOOTPRINT_MARGIN_M:
                far_indices.add(i)
    
    # Separate results
    keep_lines, panel_lines = [], []
    for i, li in enumerate(raw_lines):
        if i in far_indices:
            continue
        out = {k: v for k, v in li.items() if not k.startswith("_")}
        if i in panel_indices:
            panel_lines.append(out)
        else:
            keep_lines.append(out)
    
    keep_lines.sort(key=lambda x: -x["length_m"])
    panel_lines.sort(key=lambda x: -x["length_m"])
    
    # Limit to top 100 lines to avoid UI overload (thousands of lines)
    MAX_LINES = 100
    if len(keep_lines) > MAX_LINES:
        keep_lines = keep_lines[:MAX_LINES]
    
    return {
        "lines": keep_lines,
        "panels": panel_lines,
        "count": len(keep_lines),
        "panel_count": len(panel_lines),
        "filtered_far_count": len(far_indices),
        "raw_count": len(raw_lines),
        "source": req.source,
    }


@app.post("/api/refine-footprint")
async def refine_footprint(req: RefineFootprintRequest):
    """
    Claude affine le polygone cadastral OSM pour qu'il colle exactement à la toiture.
    Utilise Belgian WMS orthophoto.
    """
    import base64, json, re, math
    
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    
    if not req.osm_footprint or len(req.osm_footprint) < 3:
        raise HTTPException(status_code=400, detail="osm_footprint requis (>= 3 sommets)")
    
    lat, lng = req.lat, req.lng
    latM = 111320
    lngM = 111320 * math.cos(math.radians(lat))
    
    # Fetch image if not provided
    if not req.image_base64:
        img_bytes = await fetch_belgian_orthophoto(lat, lng, 640, 640, req.region)
        req.image_base64 = base64.b64encode(img_bytes).decode("ascii")
    
    # Convert footprint to meters
    fp_meters = [
        {
            "east_m": round((p["lng"] - lng) * lngM, 2),
            "north_m": round((p["lat"] - lat) * latM, 2),
        }
        for p in req.osm_footprint
    ]
    
    prompt = f"""Tu affines un polygone cadastral OSM pour qu'il colle EXACTEMENT à la toiture visible sur l'image satellite.

ADRESSE : {req.label or f"{lat},{lng}"}
IMAGE : 640×640px centrée sur ({lat:.6f}, {lng:.6f})

POLYGONE OSM ACTUEL (à affiner — {len(fp_meters)} sommets, ~70-80% précis) :
{json.dumps(fp_meters, indent=2)}

MISSION : Retourne un polygone affiné dont les sommets coïncident PRÉCISÉMENT avec les arêtes externes de la toiture (gouttières, débords, pignons) visibles dans l'image.

RÈGLES STRICTES :
1. Garde la forme générale du polygone OSM (même topologie L, U, rectangle, etc.)
2. Ajuste chaque sommet pour qu'il tombe EXACTEMENT sur une vraie arête de toiture visible (gouttière, débord, pignon)
3. Utilise les lignes rectilignes des bords de toiture pour placer les sommets avec une précision ±20cm
4. Ajoute des sommets supplémentaires si la toiture a des décrochements non capturés par OSM
5. Supprime les sommets aberrants (sortie OSM imprécise)
6. Conserve les angles ~90° quand le bâtiment est orthogonal
7. Le polygone final doit être fermé (premier sommet = dernier ignoré, le système ferme automatiquement)
8. Priorité : PRÉCISION > proximité OSM. Si un sommet OSM est à 2m du vrai bord, corrige-le de 2m.

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
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": req.image_base64}},
        {"type": "text", "text": prompt},
    ]
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",  # Available with API key (verified via /v1/models)
            max_tokens=2000,
            messages=[{"role": "user", "content": content}],
        )
        text = message.content[0].text if message.content else ""
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            raise HTTPException(status_code=502, detail=f"AI non-JSON: {text[:200]}")
        result = json.loads(m.group(0))
        
        # Convert to lat/lng
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
