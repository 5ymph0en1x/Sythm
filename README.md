# Sythm

[![Alt text](https://img.youtube.com/vi/dvqdNY_nohM/0.jpg)](https://www.youtube.com/watch?v=dvqdNY_nohM)

**Real-time, audio-reactive 3D particle visualizer** — Python + GLSL, built for an **RTX 4090**.

Sythm captures, in *loopback*, the sound coming out of your speakers (Spotify, YouTube, a game…),
extracts spectral *features* **on the GPU**, and animates **tens of millions of particles** that
react to bass, mids, highs and beats — finished with a real-time denoiser and HDR tone mapping.

No microphone is used: only the default audio-output stream is read.

It can also output **true stereoscopic 3D** — genuine two-camera depth packed into the **HDMI
frame-packing** format for a 3D display or projector (see *Stereoscopic 3D* below).

---

## The idea

- Particles **fill the visible space** and are **advected by a 3D velocity field (an ABC flow)
  whose coefficients ARE a hidden Lorenz system** (the reduced model of atmospheric convection).
  The Lorenz attractor is **never drawn** — it is the hidden mathematical generator that drives
  the chaotic, ever-changing motion.
- Each "origin" particle continuously **emits short-lived trail particles**, launched
  ballistically along its instantaneous velocity → flowing trails that reveal the field. A GPU
  ring buffer recycles them at the end of their lifetime, with a smooth fade-out.
- **Color is driven by the frequency bands** — bass → red, mids → green, highs → blue/violet,
  shifting live with the spectrum — and the **motion surges on the bass**.

---

## Architecture

The program is split into independent slices, wired together by `main.py`:

| Module            | Role |
|-------------------|------|
| `audio_engine.py` | Loopback capture (WASAPI/PulseAudio) + GPU FFT → `AudioFeatures` (bands, beat, centroid, raw waveform, and `spectrum_gpu` that **stays in VRAM**). |
| `particles.py`    | Space-filling cloud advected by an **ABC field whose coefficients ARE a HIDDEN Lorenz system**; each origin **emits short-lived ballistic trails** via a ring buffer → **tens of millions** of particles. Color by frequency band, motion keyed to the bass. CUDA kernels (CuPy) + **zero-copy** CUDA↔OpenGL interop. |
| `renderer.py`     | OpenGL 4.6 (moderngl): a single `GL_POINTS` draw call, additive gaussian sprites into an HDR (RGBA16F) framebuffer, perspective camera. `render(eye=…)` also builds **off-axis stereo** projections for 3D. |
| `postfx.py`       | Fullscreen post-processing: **à-trous edge-aware denoise**, separable bloom, history-buffer motion blur, ACES/Uncharted2 tone mapping, Lanczos downscale. |
| `stereo.py`       | **Stereoscopic 3D**: draws the one simulation through two **off-axis** cameras and packs the eyes into the HDMI **frame-packing 1080p** layout (1920×2205 @ 24 Hz). Two independent post-FX chains, one per eye (no inter-eye ghosting). |
| `window.py`       | GLFW window + OpenGL 4.6 core context, vsync, keyboard, resize. Borderless fullscreen targets the monitor **under the window** — so `F` follows you to a second display / 3D projector. |
| `config_window.py`| **Launch-time settings window** (themed Tk via `TKinterModernThemes`): ~35+ options across seven groups (including **Stereoscopic 3D**), **five UI languages** (EN/DE/FR/IT/ES) and one-click **presets**. Holds `DEFAULTS` — the **single source of truth** for every tunable (`main.py`/`particles.py` read from it) — and persists choices to `sythm_config.json`. |

**Per-frame flow** (`main.py`):
`audio.get_features()` → `particles.update(dt, features)` → `renderer.render()` (HDR texture)
→ `postfx.process(hdr, screen)` → `swap_buffers()`.
In **3D mode** the render + post-FX step runs **twice** (one off-axis eye each) and the two images are
packed top/bottom into a single frame — the simulation itself still runs **once**.

**Key performance point:** the spectrum and the particle attributes never leave the GPU. The
simulation writes straight into the OpenGL VBOs through **zero-copy CUDA interop** (the driver API
`cuGraphicsGLRegisterBuffer` called via `ctypes`, inside CuPy's primary context — no PyCUDA).

---

## Requirements

- **NVIDIA GPU** (tuned for an RTX 4090; runs on other RTX cards too — the particle count
  **auto-caps to your VRAM**, see *Configuration window*).
- **Recent NVIDIA driver + CUDA Toolkit 13.x.**
  ⚠️ The CuPy wheel must match the toolkit: **`cupy-cuda13x`** for CUDA 13.x (use `cupy-cuda12x`
  if you are still on CUDA 12.x). See *Troubleshooting* below.
- **Python 3.11+** (tested on 3.13).
- A **loopback audio source**: WASAPI (Windows) or a PulseAudio/PipeWire monitor (Linux).

## Installation

```bash
# with uv (recommended)
uv pip install -r requirements.txt

# or plain pip
pip install -r requirements.txt
```

On the reference machine (Windows 11, CUDA 13.1), the **`CUDA_PATH`** environment variable must
point at the toolkit so CuPy can find the CUDA runtime it compiles kernels with — **NVRTC** and
**cudart**. (Sythm deliberately avoids cuFFT/cuBLAS: the audio FFT runs on the CPU.)

```powershell
# Persistent (user scope) — do this once:
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1", "User")
```

## Run

```bash
uv run python main.py
# or
python main.py
```

On launch, a **settings window opens first** (themed, five languages, presets) — adjust
anything and click **Launch**; your choices are saved to `sythm_config.json` for next time,
and closing the window cancels the launch. See *Configuration window* below.

Play some music on your default output and enjoy.

---

## Controls

| Key   | Action |
|-------|--------|
| `ESC` | Quit |
| `B`   | Toggle **bloom** |
| `M`   | Toggle **motion blur** |
| `C`   | Cycle **camera mode** (`fixed` → `auto_rotate` → `beat_reactive`) |
| `F`   | Toggle **borderless fullscreen** on the monitor under the window (HDR-preserving; **pixel-exact** in 3D) |
| `R`   | Start/stop **recording** (**H.265/HEVC** → `.mp4`) |

The window title shows the live FPS, particle count, effect state, and a `● REC` marker while recording.

**Fullscreen (`F`)** uses a **borderless** window covering the monitor **under the window** — *not*
exclusive fullscreen. On Windows, an exclusive OpenGL app with no HDR surface forces the display into
SDR (it looks like "HDR turned off"); borderless keeps the DWM compositor in charge, so the screen
stays in HDR. Drag the window to a second display (or a 3D projector) first, then press `F` to go
fullscreen there.

**Recording (`R`)** encodes HEVC through the `ffmpeg` binary bundled with `imageio-ffmpeg` — no system
ffmpeg needed. Files go to `RECORD_DIR` as `sythm-<timestamp>.mp4`. Capture is **frame-paced** at a
fixed `RECORD_FPS`, and writing happens on a **background thread** so the encoder never stalls the
render loop (if it can't keep up, frames are dropped rather than freezing the app).

Two encoders, via `RECORD_ENCODER`:

- **`"nvenc"` (default)** — hardware `hevc_nvenc`, **guaranteed real-time** on the GPU and safe at any
  resolution; the right default for one-key capture. Quality via `RECORD_QUALITY` (NVENC CQ).
- **`"x265"`** — software **libx265**, **higher quality** on this **dark, fine-grained, high-frequency**
  content: `psy-rd`/`psy-rdoq` preserve the grain and the thin filaments instead of smearing them into a
  washed-out blur, and `aq-mode=3` steers bits toward the dark areas. Quality via `RECORD_QUALITY`
  (CRF, ~18 = excellent). It is CPU-bound — if frames drop at the end, lower the resolution (or pick a
  faster x265 preset in code; `RECORD_PRESET` is not exposed in the settings window).

Both encode **10-bit** by default (`RECORD_PIXFMT = "yuv420p10le"` → HEVC *Main10*, which kills the
banding 8-bit produces in dark gradients) in standard (limited-range bt709) color. For pixel-perfect
colored filaments set `RECORD_PIXFMT = "yuv444p10le"` (full-chroma 4:4:4; plays in mpv/VLC).

---

## Stereoscopic 3D (frame packing)

Sythm can output **genuine stereoscopic 3D** — designed at the render stage, *not* faked afterwards.
Since the whole image comes from one GPU simulation, the **same particle cloud is drawn twice** from
two **off-axis** cameras (parallel axes + an asymmetric frustum — the geometrically correct method:
**no vertical parallax**, and **zero parallax at the convergence plane**). The two eyes are packed into
the **HDMI 1.4a frame-packing 1080p** layout:

```
 left eye    1920 × 1080   (top)
 active gap          45 lines   (blanking — stays black)
 right eye   1920 × 1080   (bottom)
 ───────────────────────────────────
 = 1920 × 2205  @ 24 Hz
```

**Turn it on** in the settings window's **🥽 Stereoscopic 3D** panel:

| Setting | Meaning |
|---------|---------|
| **3D depth (frame packing)** | Master on/off. |
| **Eye separation** | Interocular distance in world units — larger = stronger depth (default `0.22`). |
| **Convergence** | Scales the zero-parallax plane: `< 1` makes the cloud **pop out** of the screen, `> 1` pushes it **behind** (default `1.0`). |
| **Swap L/R** | Flip the eyes if the depth looks inverted on your hardware. |

**The workflow:** launch with 3D on — the window opens **movable** (a portrait preview of the two
stacked eyes). **Drag it onto your 3D display/projector** (set to its frame-packing mode, where the
desktop becomes 1920×2205), then press **`F`** → **pixel-exact fullscreen** on that screen, and the
depth pops. `ESC` returns to the settings window. The frame rate is capped to **24 Hz** (the
frame-packing cadence), and recording (`R`) captures the whole packed frame.

> **About the standard.** True HDMI 1.4 frame-packing *timing* is negotiated at the display/driver
> level — a desktop window can't force it by itself. Sythm produces the **spec-exact packed surface**
> (1080 / 45 / 1080); feed it to a display already in frame-packing mode and you get correct,
> comfortable depth. The windowed preview scales gracefully (over/under) at any size, so you can
> position it before going fullscreen.

---

## Configuration window

Launching Sythm opens a **settings window first** (themed, dark) — adjust, then **Launch**:

- **~35+ settings** across seven groups: **Cloud**, **Window & rendering**, **Post-FX**,
  **Rhythm & flow**, **Color & harmony**, **Camera & capture**, and **Stereoscopic 3D**.
- **Five UI languages** — English, Deutsch, Français, Italiano, Español — switch live from the
  dropdown (top-right); the choice is remembered.
- **Presets** — one click for a whole look: *Ambient · Minimal · Energetic · Cosmic · Percussive*.
- Choices persist to **`sythm_config.json`** (beside the executable, or the working dir when run
  from source) and reload next time. Closing the window cancels the launch.

`config_window.DEFAULTS` is the **single source of truth** for every tunable — `main.py` and
`particles.py` read their constants from it — so there's exactly one place to change a factory default.

---

## Tuning

Tune from the **configuration window** above (no code, no rebuild), or edit the factory defaults
in **`config_window.DEFAULTS`**. Main knobs:

- `N_PARTICLES` — number of **origin** particles. Total ≈ `N_PARTICLES × (1 + EMIT_RATE × EMITTED_LIFETIME)`.
  The total is **automatically capped** to what your GPU can hold — free VRAM **and** the 32-bit GL
  buffer limit (≈ 134 M points) — so a heavy preset won't crash a smaller card; it just renders fewer
  points (the cap is logged to stderr).
- `EMIT_RATE` / `EMITTED_LIFETIME` — emission rate and trail lifetime (drive the total count and trail length).
- `PARTICLE_SIZE` — apparent particle size in pixels (smaller = fine grain, each particle distinct).
- `CLOUD_RADIUS` — half-size of the particle box (the filled visible space).
- `EXPOSURE` — global brightness. **Important:** additive accumulation scales with the particle
  count, so **lower the exposure as you raise the count** (≈ 0.6 at 5M, ≈ 0.15 at 35M+).
  Brightness is now **resolution-independent**: exposure is auto-scaled to the render resolution
  (calibrated at 720p), so the same value looks equally bright at 1080p / 1440p / 4K and in fullscreen.
- `ENABLE_DENOISE` / `DENOISE_SIGMA` / `DENOISE_ITERS` — à-trous denoiser (lower sigma keeps more detail).
- `ENABLE_BLOOM`, `ENABLE_MOTION_BLUR`, `BLOOM_INTENSITY`, `BLOOM_THRESHOLD` — cinematic look.
- `CAMERA_MODE`, `FULLSCREEN`, `WINDOW_W/H`, `SUPERSAMPLE_FACTOR`, `VSYNC`.
- `STEREO_3D`, `STEREO_EYE_SEP`, `STEREO_CONVERGENCE`, `STEREO_SWAP_EYES` — **stereoscopic 3D** output
  (frame packing); see *Stereoscopic 3D* above.

Fine color/motion tuning lives in `particles.py`: the frequency→hue mapping in `spectral_color`,
and the `*bass` coefficients that key the motion to the low frequencies.

---

## Troubleshooting

**`ImportError: DLL load failed while importing cufft`** (or `cublas`, `nvrtc`…)
The CuPy wheel does not match the installed CUDA Toolkit, or `CUDA_PATH` is unset.
- Check the toolkit version: `nvcc --version`.
- Install the matching wheel: `cupy-cuda13x` (CUDA 13.x) or `cupy-cuda12x` (CUDA 12.x).
- Set `CUDA_PATH` (see *Installation*). The `…\CUDA\vXX.Y\bin\x64` folder must be on `PATH`.

**No audio reaction / features stay at zero**
`soundcard` found no loopback device; the visualizer still runs "at rest".
- Windows: make sure audio is playing on the **default output**. As a fallback, `pyaudiowpatch`
  (commented in `requirements.txt`) is a WASAPI alternative.
- Linux: capture uses the default sink's PulseAudio/PipeWire monitor.

**`[particles] … fallback (upload CuPy→GL)` at startup**
Normally **zero-copy** interop is active (`ZERO-COPIE (driver CUDA/GL)`): the CUDA kernels write
directly into the OpenGL VBOs via the driver API (`cuGraphicsGLRegisterBuffer`), no PyCUDA. If you
see "fallback", registration failed (driver not found, GPU ≠ the GL context's GPU…) — a
`CUresult=…` message appears just above. It **still runs** via a per-frame VRAM→RAM→VRAM upload
(slower); check the NVIDIA driver and `CUDA_PATH`.

**It's slow**
Lower `N_PARTICLES` (or `EMIT_RATE` / `EMITTED_LIFETIME`), lower `SUPERSAMPLE_FACTOR`, or disable
bloom (`B`).
