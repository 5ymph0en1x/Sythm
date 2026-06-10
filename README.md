# Sythm

> *Real-time, audio-reactive 3D particle visualizer — Python + CUDA + GLSL, built for an RTX 4090.*

![license](https://img.shields.io/badge/license-MIT-green) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![cuda](https://img.shields.io/badge/CUDA-13.x-76b900) ![gpu](https://img.shields.io/badge/built%20for-RTX%204090-76b900)

[![Watch the demo](https://img.youtube.com/vi/dvqdNY_nohM/0.jpg)](https://www.youtube.com/watch?v=dvqdNY_nohM)

Sythm turns whatever is playing on your speakers into a living storm of light. It listens, in *loopback*, to the sound your system is already producing — Spotify, a film, a game — without ever touching a microphone, and spends that sound on **tens of millions of particles** drifting through a three-dimensional current.

That current is never random. It is steered, in secret, by a **Lorenz attractor** — the reduced model of atmospheric convection — that Sythm computes on every frame and **never draws**. You see the weather; you never see the butterfly that makes it.

And since version 1.2, Sythm no longer merely *reacts* to sound — it **listens to the music**: it finds the beat, anticipates the downbeat, feels a build tighten and a drop break, and even reads the key. It can also render in **true stereoscopic 3D** — genuine two-camera depth, packed into the HDMI frame-packing format for a 3D display or projector.

Version 1.3 gives that storm **two new worlds to inhabit**: an endless **hyperspace tunnel** that snakes through space — its bends steered by the hidden attractor, the rhythm arriving as rings of light — and a **living Mandelbulb**, a 3D fractal made not of polygons but of tens of millions of particles condensed onto its surface, whose very shape **mutates with the music**.

---

## The idea — a storm with a hidden conductor

Three ingredients make the picture, and a fourth makes it musical.

**A hidden engine.** The particles fill the visible box and are carried by a 3-D velocity field — an **ABC flow**. An ABC flow is a *Beltrami* flow: it has zero divergence, which is exactly why the cloud spreads to fill space instead of collapsing into clumps, and its streamlines are already chaotic. Sythm takes the field's three coefficients — the A, B and C — and feeds them the **live state of a Lorenz system**, integrated each frame with **RK4** (robust to the audio-variable timestep, and free of the slow energy drift that plain Euler injects into Lorenz). So two chaoses are stacked on top of each other: the particles wander the field, and the field itself slowly reorganises as the hidden Lorenz state drifts across its butterfly. A second, divergence-free layer of **curl noise** adds fine turbulence without ever breaking the space-filling property. The attractor conducts the whole storm — and it is never on stage.

**Living trails.** Every *origin* particle continuously **emits short-lived trail particles**, launched ballistically along its instantaneous velocity. These are the filaments you actually see streaming through the volume, and there are tens of millions of them. A GPU **ring buffer** recycles them: Sythm emits new trails at exactly the rate at which old ones die, so the slot it overwrites is always one whose particle has just faded out — recycling with no search and no hard cut-off. Each trail dims as it ages, so it dies simply by going dark.

**Senses.** Color is driven by the **frequency bands** — bass leans red, mids green, highs blue-violet — and the **motion surges on the bass**. That is the part most visualizers stop at. Sythm keeps going.

---

## What Sythm hears

Sythm's ear is frugal where it can be and lavish where it counts. A capture thread reads the default output's loopback stream into a lock-light ring buffer; once per rendered frame, the newest window is windowed (Hann) and transformed by a **CPU** real FFT — NumPy's `rfft`, a few microseconds on ~4096 samples. Only the magnitude crosses onto the GPU; everything heavier — band grouping, the spectrum, the centroid — stays there, and the spectrum **never leaves VRAM**. Running the FFT on the CPU is a choice, not a compromise: it means a standalone build never has to embed cuFFT, which alone would add ~284 MB.

From that magnitude, Sythm extracts far more than a few bars.

The **512-bin spectrum** is **log-spaced from 30 Hz to 18 kHz** — about 21 cents per bin, each octave occupying roughly 11 % of the range — with **per-bin normalization** on top. That balance matters downstream: when the spectrum is later mapped onto radius, every octave gets equal visual real estate, and a quiet hi-hat still registers next to a loud kick.

It detects **onsets per drum voice** — kick, snare and hat tracked separately — so each can trigger its own distinct gesture. It runs a **predictive tempo-and-phase model**: an adaptive oscillator locks onto the pulse and *predicts* the next downbeat, which lets the visuals act *before* the beat rather than merely after it; on music with no clear pulse, the lock simply melts away. It watches the **macro shape of the track** — a slow *build* of energy, then the *drop* — and it estimates the **key and mode**, so major and minor can feel different.

All of it is published once per frame in a single `AudioFeatures` snapshot, read straight by the particle kernels.

---

## What Sythm shows

Every one of those heard things becomes something you can *see*.

**Rhythm becomes weather.** Each percussive onset spawns a **spherical shockwave** that travels outward through the cloud at finite speed — and its epicenter is the hidden Lorenz state, so the rhythm is literally born wherever the attractor happens to be standing. The three drum voices feel different on purpose: the kick is a slow, thick shell with a heavy outward shove; the snare is a fast, thin shell that adds a tangential *shear*, a twist; the hat is a quick, tenuous flash. You watch the beat propagate through matter.

**The flow's violence becomes light.** Each particle computes its own **material acceleration**, |a| = Dv/Dt — the change in its velocity from one frame to the next, which is precisely the quantity that spikes where the field bends most sharply. Compressed through a `tanh` and added as a per-particle sparkle, it makes the **turbulent knots of the field glow**, picking out structure that mere speed would miss.

**Sustained harmony becomes terrain.** The log-spaced spectrum is smoothed slowly — a ~0.7 s time constant, so only *held* notes count — into a **radial relief**: low frequencies near the core, highs toward the rim, one octave per ~11 % of the radius. The cloud drifts gently up that relief into **concentric striations**, a standing landscape sculpted by the harmony and animated by the rhythm — kept deliberately gentle and capped, so it biases density without ever collapsing the space-filling cloud.

**The pulse becomes breath.** Carried by the tempo model's confidence, the cloud **inhales** — drawing slightly toward its heart — on the rising anticipation just *before* the predicted downbeat, then **exhales** outward on the beat itself. A track with no groove simply doesn't breathe.

**Structure becomes drama.** During a **build**, the flow speeds up, darkens, and draws inward — the calm before the storm. On the **drop**, it detonates: a massive radial bloom, a giant **central shockwave** sweeping the whole box, and a flash.

**Key becomes temperature.** Mode and key shift the whole **palette** slowly — minor cools the hue, major warms it, each key carrying a faint house tint — while the spectral centroid nudges the color note by note: a bright timbre runs cooler, a dull one warmer.

---

## Architecture

Sythm is cut into independent slices, wired together by `main.py`. Each does one job and hands off cleanly to the next.

| Module | Role |
|--------|------|
| `audio_engine.py` | Loopback capture (WASAPI / PulseAudio-PipeWire). A **CPU** real FFT (NumPy `rfft`) feeds GPU DSP that produces the `AudioFeatures` snapshot: frequency bands, RMS, spectral centroid, **per-drum onsets** (kick/snare/hat), a **predictive tempo & phase** model, **phrase** cues (build/drop), **key & mode**, and a 512-bin log-spaced spectrum that **stays in VRAM** (`spectrum_gpu`). Only ~a dozen scalars ever return to the CPU. |
| `particles.py` | The space-filling cloud, advected by the **ABC field whose coefficients are a hidden, RK4-integrated Lorenz system**; ballistic **trail emission** through a GPU ring buffer (tens of millions of points); percussive **shockwaves**, **shear sparkle**, **tonal relief**, **breathing**, **build/drop** and **harmonic** color — plus two whole-world modes: the **snaking hyperspace tunnel** and the **particle Mandelbulb** (per-particle distance-field evaluation). CUDA kernels (CuPy) writing into the GL buffers through **zero-copy** interop. |
| `renderer.py` | OpenGL 4.6 (moderngl): one `GL_POINTS` draw call, additive Gaussian sprites into an HDR (RGBA16F) framebuffer, MSAA resolve, perspective camera. `render(eye=…)` also builds the **off-axis stereo** projections. |
| `postfx.py` | Fullscreen post-processing: **à-trous edge-aware denoise**, separable bloom, history-buffer motion blur, ACES / Uncharted-2 tone mapping, Lanczos downscale. |
| `stereo.py` | **Stereoscopic 3D**: draws the one simulation through two **off-axis** cameras and packs the eyes into the HDMI **frame-packing 1080p** layout. Two independent post-FX chains — one per eye — so motion-blur history never bleeds across. |
| `window.py` | GLFW window + OpenGL 4.6 core context, vsync, keyboard, resize. Borderless fullscreen targets the monitor **under the window**, so `F` follows you to a projector. |
| `config_window.py` | The **launch-time settings window** (themed Tk): ~45 options in eight groups across **four compact tabs** (fits a 1366×768 laptop), five UI languages, one-click presets. Holds `DEFAULTS`, the **single source of truth** for every tunable, and persists choices to `sythm_config.json`. |

**The per-frame flow** is a short pipeline: `audio.get_features()` → `particles.update(dt, features)` → `renderer.render()` (an HDR texture) → `postfx.process(hdr, screen)` → `swap_buffers()`. In 3D mode the render-and-post-FX step runs **twice** — once per off-axis eye — and the two images are packed top-and-bottom into one frame; the simulation itself still runs only **once**.

**The one performance idea that matters:** the spectrum and the particle attributes never make a round trip to the CPU. The CUDA kernels write straight into the OpenGL vertex buffers through **zero-copy interop** — the driver-API call `cuGraphicsGLRegisterBuffer`, reached via `ctypes` inside CuPy's own primary context, with no PyCUDA in the loop. If that registration ever fails (wrong GPU, missing driver), Sythm falls back to a slower per-frame VRAM→RAM→VRAM upload and keeps running.

---

## Stereoscopic 3D (frame packing)

Sythm's 3D is **designed at the render stage, not faked afterwards**. Because the whole image already comes from one GPU simulation, the **same particle cloud is simply drawn twice**, from two cameras. Crucially, those cameras use the geometrically correct **off-axis** method (Paul Bourke's): their optical axes stay **parallel** — no *toe-in*, which would introduce uncomfortable *vertical* parallax — and all the depth comes from an **asymmetric frustum**, sheared horizontally just enough to put **zero parallax exactly at the convergence plane**. The result is comfortable depth with no vertical disparity.

The two eyes are then packed into the HDMI 1.4a **frame-packing 1080p** layout:

```
 left eye    1920 × 1080   (top)
 active gap        45 lines (blanking — stays black)
 right eye   1920 × 1080   (bottom)
 ─────────────────────────────────
   = 1920 × 2205  @ 24 Hz
```

Two independent post-FX chains run the two sub-images, so each eye keeps its own motion-blur history and **neither eye ghosts onto the other**.

**Turn it on** in the settings window's *Stereoscopic 3D* panel:

| Setting | Meaning |
|---------|---------|
| **3D depth (frame packing)** | Master on / off. |
| **Eye separation** | Interocular distance in world units — larger = stronger depth (default `0.22`). |
| **Convergence** | Scales the zero-parallax plane: `< 1` pops the cloud **out** of the screen, `> 1` pushes it **behind** (default `1.0`). |
| **Swap L/R** | Flip the eyes if depth looks inverted on your hardware. |

**The workflow:** launch with 3D on — the window opens *movable*, showing a portrait preview of the two stacked eyes. **Drag it onto your 3D display or projector** (set to its frame-packing mode, where the desktop becomes 1920×2205), then press **`F`** for pixel-exact fullscreen, and the depth pops. `ESC` returns to the settings window. The frame rate is capped to the **24 Hz** frame-packing cadence, and `R` records the whole packed frame.

> **About the standard.** True HDMI frame-packing *timing* is negotiated at the display/driver level — a desktop window can't force it alone. Sythm produces the **spec-exact packed surface** (1080 / 45 / 1080); feed it to a display already in frame-packing mode and you get correct, comfortable depth. The windowed preview scales gracefully (over/under) at any size, so you can position it before going fullscreen.

---

## Requirements

You'll need an **NVIDIA GPU** — Sythm is tuned for an RTX 4090, but it runs on other RTX cards too, because the particle count **auto-caps to your VRAM** (and to the 32-bit GL buffer ceiling of ~134 M points), so a heavy preset quietly renders fewer points instead of crashing. You'll also need a **recent NVIDIA driver and CUDA Toolkit 13.x**, with a CuPy wheel that matches it — `cupy-cuda13x` for CUDA 13.x, or `cupy-cuda12x` if you're still on 12.x — plus **Python 3.11+** (tested on 3.13) and a **loopback audio source** (WASAPI on Windows, a PulseAudio/PipeWire monitor on Linux). A packaged standalone build carries its own CUDA runtime, so end users need only the driver, not the full toolkit.

## Installation

```bash
# with uv (recommended)
uv pip install -r requirements.txt

# or plain pip
pip install -r requirements.txt
```

On the reference machine (Windows 11, CUDA 13.1), point CuPy at the toolkit so it can find the runtime it compiles kernels with:

```powershell
# CuPy needs NVRTC + cudart to JIT the CUDA kernels. (Sythm runs its FFT on the CPU
# on purpose, so cuFFT/cuBLAS are never required — that alone keeps a standalone
# build ~284 MB lighter.) Set this once, persistently, at user scope:
[Environment]::SetEnvironmentVariable("CUDA_PATH", "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1", "User")
```

## Run

```bash
uv run python main.py
# or
python main.py
```

A **settings window opens first** — themed, five languages, presets. Adjust anything and click **Launch**; your choices are saved to `sythm_config.json` for next time, and closing the window cancels the launch. Then play some music on your default output and enjoy.

---

## Controls

The window title shows live FPS, particle count, effect state, and a `● REC` marker while recording.

| Key | Action |
|-----|--------|
| `ESC` | Quit (in 3D, return to the settings window) |
| `B` | Toggle **bloom** |
| `M` | Toggle **motion blur** |
| `C` | Cycle **camera mode** (`fixed` → `auto_rotate` → `beat_reactive` → `tunnel`) |
| `F` | **Borderless fullscreen** on the monitor under the window (HDR-preserving; pixel-exact in 3D) |
| `R` | Start / stop **recording** (H.265 / HEVC → `.mp4`) |

**Fullscreen (`F`)** uses a *borderless* window covering the monitor under it — not exclusive fullscreen. On Windows, an exclusive OpenGL app with no HDR surface forces the display into SDR (it looks like "HDR turned off"); a borderless window leaves the DWM compositor in charge, so the screen **stays in HDR**. Drag Sythm to a second display or projector first, then press `F`.

**Recording (`R`)** encodes HEVC through the `ffmpeg` binary bundled with `imageio-ffmpeg` — no system ffmpeg required. Capture is **frame-paced**, and writing happens on a **background thread**, so the encoder can never stall the render loop; if it falls behind, frames are dropped rather than freezing the picture. Two encoders are available via `RECORD_ENCODER`:

**`nvenc` (default)** is hardware `hevc_nvenc` — guaranteed real-time on the GPU and safe at any resolution, which makes it the right default for one-key capture.

**`x265`** is software `libx265` — higher quality on this dark, fine-grained, high-frequency content, because `psy-rd` / `psy-rdoq` preserve the grain and the thin filaments instead of smearing them, and `aq-mode=3` steers bits toward the dark areas. It's CPU-bound, so lower the resolution if frames drop.

Both encode **10-bit** by default (HEVC *Main10*, which kills the banding that 8-bit produces in dark gradients). For pixel-perfect colored filaments, switch `RECORD_PIXFMT` to full-chroma `yuv444p10le`.

---

## The configuration window

Launching Sythm opens a **settings window first** — themed, dark, and translated into **five languages** (English, Deutsch, Français, Italiano, Español), switchable live. Around **45 settings** are grouped into eight panels, laid out across **four compact tabs** — Cloud & effects · Rhythm & flow · Colour & camera · Window, 3D & fractal — so the whole window fits comfortably on a **1366×768 laptop screen**, and a row of one-click **presets** (*Ambient, Minimal, Energetic, Cosmic, Percussive, Cognitive, Tunnel, Mandelbulb*) dials in a whole look at once. *Mandelbulb* renders a **living 3D fractal**: every particle evaluates the Mandelbulb's distance field on the GPU and the cloud condenses onto its surface, where the ABC flow glides along the shell — a fractal made of current, not stone — while a continuous rain of re-seeded comets keeps the coating fresh, the **power *n* mutates with the music** (the drop precipitates a visible metamorphosis), color follows the orbit trap (each lobe its own hue), and the percussive shockwaves ripple across the shell. *Tunnel* deserves a word: it turns the box into an **endless, snaking hyperspace tube** — the bends steered by the smoothed state of the hidden Lorenz attractor, the camera flying down the curved axis and banking into the turns — where the rhythm arrives as **rings of light** sweeping toward you (a kick inflates the wall, a snare wrenches it, a hat sparkles), and the drop fires a wall of light from the far end while the tunnel straightens and the field of view dilates. Adjust, then click **Launch**; your choices are saved to **`sythm_config.json`** and reloaded next time.

Under the hood, `config_window.DEFAULTS` is the **single source of truth** for every tunable — `main.py` and `particles.py` read their constants from it — so there is exactly one place to change a factory default.

---

## Tuning

Tune from the configuration window (no code, no rebuild), or edit the factory defaults in `config_window.DEFAULTS`. The headline knobs are the **particle count** (the total is roughly `N_PARTICLES × (1 + EMIT_RATE × EMITTED_LIFETIME)`, automatically capped to your card), the **emission rate** and **trail lifetime** (which set the count and the length of the streaks), the **cloud radius** and **particle size**, and **exposure**.

A word on exposure: because the blending is additive, brightness scales with the particle count, so you lower exposure as you raise the count — roughly 0.6 at 5 M, 0.15 at 35 M and beyond. Brightness is now **resolution-independent** (auto-scaled to the render resolution, calibrated at 720p), so the same value looks equally bright at 1080p, 1440p, 4K, and in fullscreen. Finer color and motion live in `particles.py`: the frequency-to-hue mapping in `spectral_color`, and the `*bass` coefficients that key the motion to the low end.

---

## Troubleshooting

**`ImportError: DLL load failed while importing nvrtc`** (or `cudart`). The CuPy wheel doesn't match the installed CUDA Toolkit, or `CUDA_PATH` is unset. Check the toolkit version with `nvcc --version`, install the matching wheel (`cupy-cuda13x` for CUDA 13.x, `cupy-cuda12x` for 12.x), and set `CUDA_PATH` so that `…\CUDA\vXX.Y\bin\x64` is on `PATH`. *(Sythm never imports cuFFT or cuBLAS — its FFT is on the CPU — so a `cufft` error specifically should not appear.)*

**No audio reaction / features stay at zero.** `soundcard` found no loopback device, and the visualizer is running "at rest". On Windows, make sure audio is actually playing on the **default output**; `pyaudiowpatch` (commented in `requirements.txt`) is a WASAPI fallback. On Linux, capture uses the default sink's PulseAudio/PipeWire monitor.

**`[particles] … fallback (upload CuPy→GL)` at startup.** Zero-copy interop couldn't register the GL buffers (driver not found, or the GPU differs from the one holding the GL context) — look for the `CUresult=…` line just above. Sythm still runs, via the slower per-frame upload path; check the driver and `CUDA_PATH`.

**It's slow.** Lower `N_PARTICLES` (or `EMIT_RATE` / `EMITTED_LIFETIME`), drop `SUPERSAMPLE_FACTOR`, or toggle bloom off with `B`.

---

## License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
