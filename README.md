# LunaSafe AI — Lunar Landing Intelligence

A visual proof-of-concept for SIH lunar-lander navigation review. It uses transparent OpenCV/NumPy image-processing heuristics; it does **not** use trained AI models or make safety-validation claims.

## What it demonstrates

- Upload a lunar surface image and run a complete visible assessment workflow.
- Prototype SR fallback: CLAHE local contrast enhancement and unsharp filtering (explicitly not learned super-resolution).
- Heuristic risk layers: dark/shadow pixels, Hough-circle crater candidates, edge-density roughness, Sobel gradient terrain-risk proxy, and low-local-detail uncertainty proxy.
- Fused red/yellow/green hazard map, explainable score, mission-readiness indicator and spatially separated top three square zones.

## Run locally

Requires Node 18+ and Python 3.9+.

```bash
# Terminal 1 — API
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

```bash
# Terminal 2 — frontend
npm install
npm run dev
```

Open the displayed Vite URL (usually http://localhost:5173), upload a PNG/JPG/TIFF image and press **Run Landing Analysis**.

## Honesty notice

Research prototype — not flight-qualified navigation software. All output is from image-based prototype heuristics and must be independently validated; no trained-model accuracy or real-world landing safety is claimed.
