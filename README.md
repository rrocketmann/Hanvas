# Hanvas

Hanvas is a browser-based hand-tracking paint app. It uses webcam input and MediaPipe hand landmarks to let you draw in the air, choose colors, clear strokes, and export the canvas as PNG.
It supports multiple simultaneous hands, with independent stroke trails for each detected hand.

## How It Works

1. The webcam feed is started with `getUserMedia`.
2. MediaPipe Tasks Vision runs hand landmark detection on each video frame.
3. A hand state heuristic (open, partial, fist) controls whether drawing is active.
4. Midpoints are sampled and interpolated between nearby points to smooth gaps.
5. Strokes render on the canvas and can be exported as image output.

## Libraries and External Dependencies

- `@mediapipe/tasks-vision` (via CDN import in `script.js`)
	- Provides `HandLandmarker` and `FilesetResolver`.
	- Handles the ML inference pipeline for real-time hand tracking.
- `@mediapipe/drawing_utils` (via CDN script in `index.html`)
	- Renders landmark connectors and keypoints for debug/visual feedback.
- `@mediapipe/hands` (via CDN script in `index.html`)
	- Supplies shared hand constants (`HAND_CONNECTIONS`) used during overlay drawing.
- Google Fonts: `Roboto` and `JetBrains Mono`
	- `Roboto` is used for readable UI text.
	- `JetBrains Mono` is used for technical output/status text.

## Controls

- `ENABLE WEBCAM` / `DISABLE WEBCAM`: starts or stops live tracking.
- `CLEAR DRAWING`: removes all painted points.
- `EXPORT`: downloads the current drawing as PNG.
- Color selector: changes paint color for new points.

## Running Locally

Use a local web server (recommended for webcam permissions):

```bash
python3 -m http.server 8000
```

Then open:

`http://localhost:8000`

## Deploy on Render

This repo includes a `render.yaml` blueprint for a static web service.

1. Push this project to GitHub/GitLab.
2. In Render, choose **New +** → **Blueprint**.
3. Connect your repository and deploy.
4. Render will read `render.yaml` and create a static service named `hanvas`.

If Blueprint import fails, create the service manually:

1. In Render, choose **New +** → **Static Site**.
2. Connect the repository.
3. Set **Build Command** to empty.
4. Set **Publish Directory** to `.`.
5. Deploy.

After deploy, open your Render URL over HTTPS and allow camera access in the browser.

### If you see: "Couldn't find a package.json file"

Render is building this as a Node service. This repo now includes `package.json` so that mode works too.

Use these service settings in Render:

- **Build Command**: `yarn install && yarn build`
- **Start Command**: `yarn start`

Or keep using **Static Site** mode (no build command, publish directory `.`).

## License

- This project source code is licensed under the MIT License (see `LICENSE`).
- MediaPipe components are provided by Google under Apache License 2.0 terms.
## Architecture

### Data Pipeline

```
Browser → Webcam API → MediaPipe HandLandmarker → 21 landmarks per hand
    ↓
Hand State Inference (fist/partial/open) → Drawing Gate
    ↓
Midpoint Calculation (average of 21 points) → Paint Point
    ↓
Interpolation (fill gaps between nearby samples) → Trail Array
    ↓
Canvas Rendering (ellipses at trail points) → Visual Preview
    ↓
Export (canvas → PNG download)
```

### Key Pipeline Stages

**1. Webcam Input & Model Loading**
- `getUserMedia` captures video stream and binds to `<video>` element.
- MediaPipe `FilesetResolver` initializes WASM runtime and loads the hand landmarker model.
- Model tries GPU delegate first, falls back to CPU if unavailable.

**2. Hand Landmark Detection**
- Every frame, `HandLandmarker.detectForVideo()` returns up to 2 hands with 21 normalized landmarks each.
- Landmarks include position (x, y, z in 0–1 range) and visibility scores.
- Landmarks represent joint positions: wrist, fingers, palm center.

**3. Hand State Inference**
- `inferHandState()` compares fingertip distance to base distance for each of 5 fingers.
- If tip is >1.15× farther from wrist than base, finger is extended.
- States: 4+ extended = "Open hand", 0–1 extended = "Fist", else = "Partial hand".
- Drawing is enabled only for "Fist" or "Partial hand" states.

**4. Paint Point Generation**
- `getLandmarksMidpoint()` averages all 21 landmarks into a single normalized (x, y, z) point.
- Each midpoint is stored with its current paint color and size (radiusX, radiusY = 10).
- Midpoints are added to a `midpointTrail` array.

**5. Gap Interpolation (Smoothing)**
- Between consecutive midpoints, distance is calculated in pixels.
- If distance is within 80 pixels and > 0, intermediate points are created.
- Intermediate points split the gap into ≤10 pixel segments for smooth visual appearance.
- Interpolated points preserve color and size from current brush settings.

**6. Canvas Rendering**
- On each frame, the canvas is cleared and the entire `midpointTrail` is redrawn.
- `drawMidpoint()` renders each point as a filled ellipse with stored color.
- Hand skeleton overlays (connectors + landmarks) are drawn in a separate color if enabled.

**7. Export & Download**
- `exportDrawing()` creates a temporary canvas, flips it horizontally (to match on-screen orientation), and saves as PNG.
- Browser triggers automatic download with timestamped filename.

### Color & Drawing State

- Current paint color is stored in `currentPaintColor` (default: soft blue `#6fa8e6`).
- Color can be changed via dropdown selector without affecting already-drawn points.
- Each point in the trail stores its own color, allowing mixed-color drawings.
- Trail persists until **CLEAR DRAWING** is clicked.

## Implementation Details

### Coordinate Systems

- **Normalized coordinates** (from MediaPipe): range 0–1, origin at top-left.
- **Canvas pixels** (screen): multiply normalized by canvas width/height.
- **Mirrored display**: both video and output canvas use `scaleX(-1)` to match user expectation (camera flips horizontally).

### Performance Considerations

- Frame-by-frame detection uses `requestAnimationFrame` for smooth 30–60 FPS pacing.
- Drawing updates only when hand state is active; reduces painting overhead on every frame.
- Interpolation limits max gap size (10 px) and distance threshold (80 px) to balance smoothness vs. memory.

### Browser Compatibility

- Requires `getUserMedia` support (modern browsers).
- WebGL recommended for GPU acceleration; CPU fallback available.
- Webcam access must be allowed (HTTPS or localhost required).
