# ENM 5320 Project 3 — Windows Omniverse Kit App (Milestone 5)

## Project Overview

This is the **Windows-side** interactive visualization component of the Fluid Digital Twin project.
The Kit extension receives real-time Navier-Stokes flow field predictions from the WSL2 inference
server and renders them as an interactive OpenUSD scene in NVIDIA Omniverse.

**User experience goal**: Drag a cylinder prim in the viewport → the flow field updates in < 100 ms → colour visualization hot-reloads in Omniverse (30+ FPS during streaming).

**WSL2 counterpart**: `/home/jimwell/ENM5320_Project3` — contains the FNO model, training code, and FastAPI server.

---

## Architecture

```
Windows (this project)                        WSL2
──────────────────────────────────            ─────────────────────────────────
Omniverse Kit App                             FastAPI server (localhost:8000)
  └─ FluidTwinExtension (omni.ext)    ──────▶  POST /predict
       ├─ CylinderManipulator               {cx, cy, re}
       │    └─ listens to prim Xform                │
       ├─ ServerClient (requests)     ◀──────  {u, v, p, usd_path}
       └─ StageUpdater                               │
            ├─ direct primvar update          USD file written to:
            └─ OR open_stage(usd_path)       C:\Users\jimwell\omniverse_assets\
```

**Two rendering modes** (choose one per session):

| Mode | Latency | Code path |
|------|---------|-----------|
| **Direct** (recommended) | ~50 ms | Update `custom:u_field` / `displayColor` primvars in-place via `pxr` |
| **File reload** | ~200 ms | Server exports `.usda`, Kit calls `open_stage(usd_path)` |

---

## Environment

**Platform**: Windows 10/11 + NVIDIA Omniverse Launcher  
**Kit SDK**: Omniverse Kit 106.x (bundled Python 3.10)  
**Template**: "Digital Twins for Fluid Simulation" blueprint (Kit App Template)

### Key packages (available inside Kit Python)

| Package | Purpose |
|---------|---------|
| `omni.ext` | Extension lifecycle (`IExt`) |
| `omni.ui` | UI widgets (sliders, buttons, frames) |
| `omni.usd` | USD context and stage access |
| `omni.kit.commands` | Undo-able stage operations |
| `pxr` | OpenUSD Python API (`Usd`, `UsdGeom`, `Sdf`, `Vt`, `Gf`) |
| `requests` | HTTP calls to WSL2 server (bundled with Kit) |
| `carb` | Settings, logging (`carb.log_info`) |
| `asyncio` + `omni.kit.async_engine` | Non-blocking server calls |

### Inference Server (WSL2)

- **URL**: `http://localhost:8000/predict`
- **Method**: `POST`
- **Request JSON**:
  ```json
  { "cx": 0.4, "cy": 0.5, "re": 100.0, "export_usd": true }
  ```
- **Response JSON**:
  ```json
  {
    "u": [[...128 rows × 128 cols...]],
    "v": [[...]],
    "p": [[...]],
    "usd_path": "C:/Users/jimwell/omniverse_assets/flow_scene.usda"
  }
  ```

**Start the server in WSL2 before launching this app:**
```powershell
# In WSL2 terminal (or Windows Terminal with WSL profile)
conda activate fluid-twin
cd /home/jimwell/ENM5320_Project3
python -m src.inference.server --checkpoint models/fno_baseline/best.pt
```

---

## Directory Structure

```
windows_app/           ← this project root (copy to C:\Users\jimwell\omniverse_app\)
├── CLAUDE.md
├── exts/
│   └── fluid_twin/
│       ├── config/
│       │   └── extension.toml          # Extension manifest
│       └── fluid_twin/
│           ├── __init__.py
│           ├── extension.py            # IExt entry point + lifecycle
│           ├── ui.py                   # omni.ui panel (sliders, button)
│           ├── server_client.py        # HTTP client for /predict
│           └── stage_updater.py        # USD primvar update or open_stage
├── apps/
│   └── fluid_twin.kit                  # Kit app config (lists extensions)
└── omniverse_assets/                   # Shared with WSL2 via C:\...
    └── flow_scene.usda                 # Written by WSL2 server
```

---

## Key Extension Files

### `extension.toml`

```toml
[package]
title = "Fluid Digital Twin"
version = "0.1.0"
description = "Real-time FNO flow field visualization"

[dependencies]
"omni.ui" = {}
"omni.usd" = {}
"omni.kit.commands" = {}
```

### `extension.py` — Entry Point

```python
import omni.ext
import omni.ui as ui
from .ui import FluidTwinWindow

class FluidTwinExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._window = FluidTwinWindow()

    def on_shutdown(self):
        if self._window:
            self._window.destroy()
            self._window = None
```

### `ui.py` — Control Panel

```python
import omni.ui as ui
import asyncio
import omni.kit.async_engine
from .server_client import predict_flow
from .stage_updater import update_stage_direct

class FluidTwinWindow(ui.Window):
    def __init__(self):
        super().__init__("Fluid Digital Twin", width=300, height=200)
        self._cx = 0.4
        self._cy = 0.5
        self._re = 100.0
        self._build_ui()

    def _build_ui(self):
        with self.frame:
            with ui.VStack(spacing=8):
                ui.Label("Cylinder Position & Reynolds Number")

                with ui.HStack():
                    ui.Label("cx", width=40)
                    cx_slider = ui.FloatSlider(min=0.2, max=0.7, step=0.01)
                    cx_slider.model.set_value(self._cx)
                    cx_slider.model.add_value_changed_fn(
                        lambda m: self._on_param_changed("cx", m.get_value_as_float())
                    )

                with ui.HStack():
                    ui.Label("cy", width=40)
                    cy_slider = ui.FloatSlider(min=0.3, max=0.7, step=0.01)
                    cy_slider.model.set_value(self._cy)
                    cy_slider.model.add_value_changed_fn(
                        lambda m: self._on_param_changed("cy", m.get_value_as_float())
                    )

                with ui.HStack():
                    ui.Label("Re", width=40)
                    re_slider = ui.FloatSlider(min=20.0, max=200.0, step=1.0)
                    re_slider.model.set_value(self._re)
                    re_slider.model.add_value_changed_fn(
                        lambda m: self._on_param_changed("re", m.get_value_as_float())
                    )

                ui.Button("Predict", clicked_fn=self._run_predict)

    def _on_param_changed(self, key: str, value: float):
        setattr(self, f"_{key}", value)

    def _run_predict(self):
        asyncio.ensure_future(self._predict_async())

    async def _predict_async(self):
        import carb
        carb.log_info(f"Predicting: cx={self._cx:.2f} cy={self._cy:.2f} re={self._re:.1f}")
        result = await predict_flow(self._cx, self._cy, self._re, export_usd=True)
        if result:
            update_stage_direct(result)
```

### `server_client.py` — HTTP Client

```python
import asyncio
import aiohttp                    # or use requests in a thread
import carb
from typing import Optional

SERVER_URL = "http://localhost:8000/predict"


async def predict_flow(
    cx: float, cy: float, re: float, export_usd: bool = True
) -> Optional[dict]:
    """Call the WSL2 inference server asynchronously.

    Returns the parsed JSON dict {u, v, p, usd_path} or None on error.
    """
    payload = {"cx": cx, "cy": cy, "re": re, "export_usd": export_usd}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(SERVER_URL, json=payload, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                if resp.status == 200:
                    return await resp.json()
                carb.log_warn(f"Server returned {resp.status}: {await resp.text()}")
    except Exception as e:
        carb.log_error(f"Inference server unreachable: {e}")
    return None
```

### `stage_updater.py` — USD Scene Update

Two modes — choose based on latency target:

```python
import numpy as np
import carb

# ── Mode A: direct primvar update (fastest, ~50 ms) ──────────────────────────

def update_stage_direct(result: dict) -> None:
    """Update flow field primvars in-place without reloading the stage."""
    try:
        from pxr import Usd, UsdGeom, Sdf, Vt, Gf
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/FlowField/VelocityMesh")
        if not prim.IsValid():
            carb.log_warn("FlowField prim not found — run File > Open to load flow_scene.usda first")
            return

        u = np.array(result["u"], dtype=np.float32)
        v = np.array(result["v"], dtype=np.float32)
        p = np.array(result["p"], dtype=np.float32)
        h, w = u.shape

        # Update custom field attributes
        prim.GetAttribute("custom:u_field").Set(Vt.FloatArray(u.ravel().tolist()))
        prim.GetAttribute("custom:v_field").Set(Vt.FloatArray(v.ravel().tolist()))
        prim.GetAttribute("custom:p_field").Set(Vt.FloatArray(p.ravel().tolist()))

        # Recompute display colour (v_max=2.0 matches server constant)
        V_MAX = 2.0
        vel_mag = np.sqrt(u ** 2 + v ** 2)
        mag_norm = np.clip(vel_mag / V_MAX, 0.0, 1.0).ravel()
        mesh = UsdGeom.Mesh(prim)
        UsdGeom.PrimvarsAPI(mesh).GetPrimvar("displayColor").Set(
            Vt.Vec3fArray([Gf.Vec3f(float(m), 0.0, 1.0 - float(m)) for m in mag_norm])
        )
        carb.log_info(f"Stage updated — u_max={u.max():.3f}")

    except Exception as e:
        carb.log_error(f"Stage update failed: {e}")


# ── Mode B: reload stage from USD file (~200 ms, simpler) ────────────────────

def update_stage_from_file(result: dict) -> None:
    """Reload the USD stage from the file written by the WSL2 server."""
    usd_path = result.get("usd_path")
    if not usd_path:
        carb.log_warn("No usd_path in server response — set export_usd=True")
        return
    import omni.usd
    omni.usd.get_context().open_stage(usd_path)
    carb.log_info(f"Stage reloaded from {usd_path}")
```

---

## Data Flow (Interactive Loop)

```
1. User moves cx/cy slider (or drags cylinder prim)
2. ui.py: _run_predict() → asyncio.ensure_future(_predict_async())
3. server_client.py: POST localhost:8000/predict {cx, cy, re, export_usd=true}
4. WSL2 server:
     a. build_input(cx, cy, re) → FNO model → denormalize → (u, v, p)
     b. export_flow_to_usd(u, v, p) → writes C:\...\flow_scene.usda
     c. returns {u, v, p, usd_path}
5. stage_updater.py: update_stage_direct(result)
     - update custom:u_field, custom:v_field, custom:p_field attributes
     - recompute displayColor primvar (velocity magnitude, blue→red)
6. Omniverse viewport re-renders updated mesh colours
```

---

## USD Scene Layout

The scene loaded from `flow_scene.usda` has this prim hierarchy:

```
/FlowField  (UsdGeom.Xform)
└── /FlowField/VelocityMesh  (UsdGeom.Mesh)
    ├── points              — 128×128 = 16 384 vertices, flat XZ plane
    ├── faceVertexCounts    — 127×127 = 16 129 quads, all [4]
    ├── faceVertexIndices   — CCW winding: BL, BR, TR, TL
    ├── primvars:displayColor  — per-vertex Color3f (blue→red = slow→fast)
    ├── custom:u_field      — FloatArray[16 384], x-velocity (physical units)
    ├── custom:v_field      — FloatArray[16 384], y-velocity (physical units)
    ├── custom:p_field      — FloatArray[16 384], pressure (physical units)
    └── custom:field_shape  — Int2 (128, 128)
```

**Coordinate convention**: X = flow direction (left→right), Z = transverse (bottom→top), Y = 0 (flat plane).

**Colour scale**: `GLOBAL_V_MAX = 2.0` — fixed per session. Blue = 0 m/s, Red = 2.0 m/s. Values are clipped, not rescaled, so the colour scale is stable across cylinder positions.

---

## Cylinder Prim Manipulation (Optional Enhancement)

To enable viewport dragging (move cylinder directly in the 3D viewport):

1. Add a separate cylinder prim to the scene:
   ```python
   # In stage_updater.py or a setup script
   from pxr import Usd, UsdGeom, Gf
   import omni.usd
   stage = omni.usd.get_context().get_stage()
   cyl = UsdGeom.Cylinder.Define(stage, "/Cylinder")
   cyl.GetRadiusAttr().Set(0.05)
   cyl.GetHeightAttr().Set(0.01)
   cyl.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.5))  # (cx, 0, cy)
   ```

2. Subscribe to prim transform changes and extract (cx, cy):
   ```python
   import omni.usd
   from pxr import Usd

   def _on_objects_changed(notice, stage):
       for path in notice.GetChangedInfoOnlyPaths():
           if str(path) == "/Cylinder.xformOp:translate":
               prim = stage.GetPrimAtPath("/Cylinder")
               xform = UsdGeom.Xformable(prim)
               transform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
               cx = transform.ExtractTranslation()[0]   # X → cx
               cy = transform.ExtractTranslation()[2]   # Z → cy
               asyncio.ensure_future(predict_flow(cx, cy, current_re))

   stage = omni.usd.get_context().get_stage()
   listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, _on_objects_changed, stage)
   ```

---

## Key Commands

```powershell
# Launch the Kit app (from Omniverse Launcher or CLI)
# Replace <kit_path> with your actual Omniverse installation path
"C:\Users\jimwell\AppData\Local\ov\pkg\kit-106.0\kit.exe" apps\fluid_twin.kit

# Check server health before launching
curl http://localhost:8000/docs

# Manually trigger a prediction (PowerShell)
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"cx":0.4,"cy":0.5,"re":100.0,"export_usd":false}' |
  Select-Object -ExpandProperty u | Measure-Object -Property Count
```

---

## Notes & Gotchas

- **Server must be running in WSL2** before the Kit app makes any prediction calls. The extension should gracefully show a warning (not crash) if the server is unreachable — see `server_client.py` error handling.
- **`aiohttp` may need installation** inside the Kit Python environment: use Kit's bundled `pip` at `<kit_install>/python/pip`. Alternatively, use `requests` in a `concurrent.futures.ThreadPoolExecutor` to avoid blocking the UI thread.
- **USD path on Windows**: The server returns paths like `C:/Users/jimwell/omniverse_assets/flow_scene.usda` (forward slashes). Kit's `open_stage()` accepts both `/` and `\`.
- **V_MAX consistency**: The `V_MAX = 2.0` constant in `stage_updater.py` must match `GLOBAL_V_MAX` in the WSL2 `src/inference/usd_exporter.py`. If you change it on the server side, update it here too.
- **Coordinate mapping** — server domain is `[0,1]×[0,1]`; USD mesh X/Z are also `[0,1]`. Map cx/cy directly from slider values to server payload.
- **Kit extension hot-reload**: Press `Ctrl+S` on any `.py` file in the `exts/` folder while Kit is running to reload the extension without restarting Kit.
- **Performance**: Direct primvar update (Mode A) should achieve 30+ FPS for static scenes. For animated/streaming use, call the server at 10–30 Hz from a timer and update primvars each frame; the FNO inference takes ~10–20 ms on GPU.

## Reference Links

- [Omniverse Kit Extensions Manual](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_advanced.html)
- [omni.ui Documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/)
- [OpenUSD Python API](https://openusd.org/release/api/index.html)
- [Digital Twins for Fluid Simulation Blueprint](https://docs.omniverse.nvidia.com/blueprints/latest/digital-twins-fluid.html)
- [WSL2 Inference Server API](http://localhost:8000/docs) — FastAPI auto-docs (requires server running)
