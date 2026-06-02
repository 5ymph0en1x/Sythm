# Sythm

**Real-time, audio-reactive 3D particle visualizer** — Python + GLSL, built for an **RTX 4090**.

Sythm captures, in *loopback*, the sound coming out of your speakers (Spotify, YouTube, a game…),
extracts spectral *features* **on the GPU**, and animates **tens of millions of particles** that
react to bass, mids, highs and beats — finished with a real-time denoiser and HDR tone mapping.

No microphone is used: only the default audio-output stream is read.

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
| `renderer.py`     | OpenGL 4.6 (moderngl): a single `GL_POINTS` draw call, additive gaussian sprites into an HDR (RGBA16F) framebuffer, perspective camera. |
| `postfx.py`       | Fullscreen post-processing: **à-trous edge-aware denoise**, separable bloom, history-buffer motion blur, ACES/Uncharted2 tone mapping, Lanczos downscale. |
| `window.py`       | GLFW window + OpenGL 4.6 core context, vsync, keyboard, resize. |

**Per-frame flow** (`main.py`):
`audio.get_features()` → `particles.update(dt, features)` → `renderer.render()` (HDR texture)
→ `postfx.process(hdr, screen)` → `swap_buffers()`.

**Key performance point:** the spectrum and the particle attributes never leave the GPU. The
simulation writes straight into the OpenGL VBOs through **zero-copy CUDA interop** (the driver API
`cuGraphicsGLRegisterBuffer` called via `ctypes`, inside CuPy's primary context — no PyCUDA).

---

## Requirements

- **NVIDIA GPU** (tuned for an RTX 4090; works on other RTX cards with a lower `N_PARTICLES`).
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
point at the toolkit so CuPy can find cuFFT/cuBLAS:

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

Play some music on your default output and enjoy.

---

## Controls

| Key   | Action |
|-------|--------|
| `ESC` | Quit |
| `B`   | Toggle **bloom** |
| `M`   | Toggle **motion blur** |
| `C`   | Cycle **camera mode** (`fixed` → `auto_rotate` → `beat_reactive`) |

The window title shows the live FPS, particle count and effect state.

---

## Tuning

**Everything is at the top of `main.py`** (the tunable-parameter header), the only place you need
to touch. Main knobs:

- `N_PARTICLES` — number of **origin** particles. Total ≈ `N_PARTICLES × (1 + EMIT_RATE × EMITTED_LIFETIME)`.
- `EMIT_RATE` / `EMITTED_LIFETIME` — emission rate and trail lifetime (drive the total count and trail length).
- `PARTICLE_SIZE` — apparent particle size in pixels (smaller = fine grain, each particle distinct).
- `CLOUD_RADIUS` — half-size of the particle box (the filled visible space).
- `EXPOSURE` — global brightness. **Important:** additive accumulation scales with the particle
  count, so **lower the exposure as you raise the count** (≈ 0.6 at 5M, ≈ 0.15 at 35M+).
- `ENABLE_DENOISE` / `DENOISE_SIGMA` / `DENOISE_ITERS` — à-trous denoiser (lower sigma keeps more detail).
- `ENABLE_BLOOM`, `ENABLE_MOTION_BLUR`, `BLOOM_INTENSITY`, `BLOOM_THRESHOLD` — cinematic look.
- `CAMERA_MODE`, `FULLSCREEN`, `WINDOW_W/H`, `SUPERSAMPLE_FACTOR`, `VSYNC`.

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
