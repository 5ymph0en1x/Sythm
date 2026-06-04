# -*- coding: utf-8 -*-
"""
recorder.py
===========
Enregistrement de la diffusion en H.265/HEVC, via deux encodeurs au choix :

  - "x265"  : encodeur LOGICIEL libx265 (qualité MAX, plus lent). Bien meilleur
              que NVENC sur notre contenu sombre + granuleux + filaments fins :
              psy-rd/psy-rdoq préservent l'énergie haute-fréquence (le grain et
              les filaments) au lieu de la lisser en une bouillie délavée, et
              aq-mode=3 redistribue les bits vers les zones sombres.
  - "nvenc" : encodeur MATÉRIEL hevc_nvenc (GPU, temps réel garanti).

On capture le framebuffer FINAL (ce qui est affiché : `ctx.screen`, après le
post-traitement) et on pousse les pixels bruts (RGB) dans `ffmpeg` (fourni par
imageio-ffmpeg -> aucune install système). Colorimétrie laissée STANDARD (YUV
limité bt709, comme un flux HD classique) : pas de bascule full-range qui
délavait l'image.

CADENCE FIXE (frame pacing) : on capture à `fps` constant, piloté par l'horloge
murale (duplication si le rendu est plus lent). La vidéo a un timing CORRECT.

THREAD D'ÉCRITURE : l'écriture vers ffmpeg se fait dans un thread dédié, via une
file bornée. La boucle de rendu n'est donc JAMAIS bloquée par l'encodeur (crucial
avec x265, logiciel). Si l'encodeur ne suit pas, des frames sont abandonnées
(comptées dans `self.dropped`) plutôt que de figer l'application.

Lève RuntimeError si imageio-ffmpeg est absent ; l'appelant gère proprement.
"""
from __future__ import annotations

import os
import sys
import time
import queue
import wave
import datetime
import threading
import subprocess

import numpy as np

try:
    import imageio_ffmpeg
    _HAS_FFMPEG = True
except Exception:
    imageio_ffmpeg = None
    _HAS_FFMPEG = False


_NVENC_PRESETS = {"p1", "p2", "p3", "p4", "p5", "p6", "p7"}
_X265_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow", "placebo"}

# Réglages x265 taillés pour notre visuel (sombre, granuleux, filaments fins) :
#   aq-mode=3            -> quantif. adaptative biaisée vers les zones sombres
#   psy-rd / psy-rdoq    -> conserve l'énergie HF (grain + filaments nets)
#   no-sao               -> pas de filtre SAO (qui floute les détails)
#   strong-intra-smoothing=0 -> pas de lissage des aplats (garde le grain)
# (Pas de 'deblock=a:b' ici : le ':' casserait le parsing de -x265-params.)
_X265_PARAMS_DEFAULT = (
    "aq-mode=3:aq-strength=1.0:psy-rd=2.0:psy-rdoq=1.0:rdoq-level=2:"
    "no-sao=1:strong-intra-smoothing=0:bframes=4:rc-lookahead=40:log-level=error"
)


def _hevc_profile_for(pix_fmt):
    """Profil HEVC déduit du format de pixels (pour NVENC ; x265 le déduit seul)."""
    p = pix_fmt.lower()
    if "444" in p:
        return "rext"
    if "p010" in p or "p016" in p or "10le" in p or "16le" in p:
        return "main10"
    return "main"


class Recorder:
    """Encodeur HEVC (x265 logiciel ou NVENC matériel) alimenté par le framebuffer
    final, avec thread d'écriture découplé.

    Usage :
        rec = Recorder(width, height, fps, out_dir, encoder="x265")
        ...  rec.maybe_capture(ctx.screen)   # à chaque frame (pacing interne)
        path, n = rec.close()                # finalise le fichier
    """

    def __init__(self, width, height, fps=60, out_dir=".",
                 encoder="x265", quality=18, preset="medium",
                 pix_fmt="yuv420p10le", full_range=False,
                 x265_params=None, nvenc_aq=8, nvenc_multipass="fullres",
                 queue_size=12, audio_queue=None, audio_rate=48000,
                 audio_bitrate="192k"):
        if not _HAS_FFMPEG:
            raise RuntimeError(
                "imageio-ffmpeg absent : `uv pip install imageio-ffmpeg` "
                "(fournit ffmpeg avec libx265 et NVENC).")
        # Dimensions arrondies à des valeurs PAIRES (requis par le 4:2:0).
        self.width = int(width) & ~1
        self.height = int(height) & ~1
        self.fps = float(max(1.0, fps))
        self.encoder = "nvenc" if str(encoder).lower() == "nvenc" else "x265"
        self._row_bytes = self.width * self.height * 3        # rgb24, lignes serrées
        self._interval = 1.0 / self.fps
        self._next_t = None
        self._frames = 0
        self.dropped = 0
        self.last_error = None
        self._closed = False
        self._pipe_broken = False

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(out_dir, f"sythm-{ts}.mp4")
        self._logpath = os.path.join(out_dir, ".sythm-ffmpeg.log")

        # --- Audio (piste muxée à la fermeture) ------------------------------
        # Si `audio_queue` est fourni (dérivation de AudioEngine.start_recording_
        # tap()), on capture l'audio BRUT en parallèle dans un WAV temporaire,
        # puis on MUXE vidéo (copy) + audio (AAC stéréo 192k 48 kHz) à la
        # fermeture. Sans audio_queue -> comportement d'origine (vidéo muette).
        self._audio_q = audio_queue
        self._audio_rate = int(audio_rate) if audio_rate else 48000
        self._audio_bitrate = str(audio_bitrate or "192k")
        self._audio_enabled = audio_queue is not None
        _base_noext = os.path.splitext(self.path)[0]
        # La vidéo va dans un fichier temporaire quand on doit muxer, sinon
        # directement dans le fichier final.
        self._video_out = (_base_noext + ".tmpvideo.mp4"
                           if self._audio_enabled else self.path)
        self._audio_wav = _base_noext + ".tmpaudio.wav"
        self._wav = None
        self._audio_channels = 0
        self._audio_frames = 0
        self._audio_stop = False
        self._audio_thread = None

        # -vf vflip : OpenGL lit le framebuffer de bas en haut -> on le remet à
        # l'endroit. Pas de conversion de range par défaut (colorimétrie standard).
        vf = "vflip"
        if full_range:
            vf = "vflip,scale=w=iw:h=ih:out_range=pc"

        base = [
            exe, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", f"{self.fps:g}",
            "-i", "pipe:0", "-an",
            "-vf", vf,
        ]

        if self.encoder == "nvenc":
            p = preset if preset in _NVENC_PRESETS else "p7"
            self.profile = _hevc_profile_for(str(pix_fmt))
            enc = [
                "-c:v", "hevc_nvenc",
                "-preset", str(p),
                "-tune", "hq",
                "-rc", "constqp", "-qp", str(quality),    # quality=18 par défaut
                "-rc-lookahead", "32",
                "-multipass", "fullres",
                "-spatial-aq", "1",
                "-temporal-aq", "1",
                "-aq-strength", "10",                     # ou 10 si vous préférez
                "-b_ref_mode", "1",
                "-bf", "4",
                "-profile:v", "main10",
                "-pix_fmt", str(pix_fmt),
            ]
            if nvenc_multipass and str(nvenc_multipass) != "0":
                enc += ["-multipass", str(nvenc_multipass)]
            enc += ["-pix_fmt", str(pix_fmt), "-profile:v", self.profile]
        else:
            p = preset if preset in _X265_PRESETS else "medium"
            self.profile = None  # x265 déduit le profil du pix_fmt
            xp = x265_params or _X265_PARAMS_DEFAULT
            enc = [
                "-c:v", "libx265", "-preset", p, "-crf", str(quality),
                "-x265-params", xp,
                "-pix_fmt", str(pix_fmt),
            ]

        color = []
        if full_range:
            color = ["-color_range", "pc", "-colorspace", "bt709",
                     "-color_primaries", "bt709", "-color_trc", "bt709"]

        cmd = base + enc + color + ["-tag:v", "hvc1", self._video_out]
        self.cmd = cmd
        self.pix_fmt = str(pix_fmt)
        self.preset = p

        # stderr -> fichier log (et non DEVNULL) : si un réglage est refusé, la
        # raison est lisible dans le .log et remontée via self.last_error.
        self._logfh = open(self._logpath, "wb")
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self._logfh)

        # Thread d'écriture : draine la file vers ffmpeg sans bloquer le rendu.
        self._q = queue.Queue(maxsize=int(max(2, queue_size)))
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()
        self._next_t = time.perf_counter()

        # Thread audio : draine la dérivation AudioEngine -> WAV temporaire.
        if self._audio_enabled:
            self._audio_thread = threading.Thread(
                target=self._audio_loop, name="RecAudioThread", daemon=True)
            self._audio_thread.start()

    # ------------------------------------------------------------------ #
    #  Thread d'écriture                                                  #
    # ------------------------------------------------------------------ #
    def _writer_loop(self):
        """Pompe la file et écrit vers ffmpeg (hors thread de rendu)."""
        while True:
            data = self._q.get()
            if data is None:                       # sentinelle de fin
                self._q.task_done()
                break
            try:
                self._proc.stdin.write(data)
            except (BrokenPipeError, OSError):
                self._pipe_broken = True
                self._q.task_done()
                break
            self._q.task_done()

    def _alive(self):
        return (not self._pipe_broken and self._proc is not None
                and self._proc.poll() is None)

    # ------------------------------------------------------------------ #
    #  Thread audio : dérivation AudioEngine -> WAV temporaire (PCM 16b)  #
    # ------------------------------------------------------------------ #
    def _audio_loop(self):
        """Draine la file de la dérivation audio et écrit un WAV PCM 16 bits.
        Le nombre de canaux est déduit du PREMIER bloc (canaux natifs de la
        capture : stéréo en loopback). S'arrête quand `_audio_stop` est posé et
        la file est vide, ou sur une sentinelle None -> tout l'audio est écrit."""
        while True:
            try:
                block = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                if self._audio_stop:
                    break
                continue
            if block is None:
                break
            try:
                self._write_audio_block(block)
            except Exception as exc:
                self.last_error = f"audio: {exc}"

    def _write_audio_block(self, block):
        """Convertit un bloc float32 [-1,1] (numframes, canaux) en PCM int16 et
        l'ajoute au WAV (ouvert paresseusement au premier bloc)."""
        a = np.asarray(block, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        if a.size == 0:
            return
        if self._wav is None:
            self._audio_channels = int(a.shape[1])
            self._wav = wave.open(self._audio_wav, "wb")
            self._wav.setnchannels(self._audio_channels)
            self._wav.setsampwidth(2)                  # PCM 16 bits
            self._wav.setframerate(self._audio_rate)
        elif a.shape[1] != self._audio_channels:
            # Canaux changeants (rare) : on ramène au format du premier bloc.
            if a.shape[1] > self._audio_channels:
                a = a[:, :self._audio_channels]
            else:
                a = np.repeat(a[:, :1], self._audio_channels, axis=1)
        i16 = (np.clip(a, -1.0, 1.0) * 32767.0).astype("<i2")
        self._wav.writeframes(np.ascontiguousarray(i16).tobytes())
        self._audio_frames += int(a.shape[0])

    # ------------------------------------------------------------------ #
    #  Capture (thread de rendu)                                          #
    # ------------------------------------------------------------------ #
    def maybe_capture(self, screen_fbo, fb_size=None):
        """À appeler CHAQUE frame (après le post-traitement, avant swap_buffers).
        Ne capture qu'aux échéances de la cadence fixe ; lit le framebuffer final
        et l'envoie au thread d'écriture. Ne bloque jamais le rendu : si l'encodeur
        est en retard, la frame est abandonnée (comptée dans self.dropped).

        `fb_size` (w, h) : taille COURANTE du framebuffer, fournie par l'appelant
        (screen_fbo.size est périmé sous moderngl). Si elle diffère de la taille
        d'enregistrement (resize / sortie de plein écran en cours de capture), on
        saute la frame pour ne pas lire une région incohérente."""
        if self._closed or not self._alive():
            return
        now = time.perf_counter()
        if now < self._next_t:
            return
        if fb_size is not None:
            cw = int(fb_size[0]) & ~1
            ch = int(fb_size[1]) & ~1
            if cw != self.width or ch != self.height:
                self._next_t = now + self._interval
                return
        try:
            # Viewport EXPLICITE (0,0,W,H) : on lit TOUT le backbuffer réel sans
            # dépendre de screen_fbo.size, que moderngl laisse périmé au resize
            # (sinon capture tronquée à un coin en plein écran).
            data = screen_fbo.read(viewport=(0, 0, self.width, self.height),
                                   components=3, alignment=1, dtype="f1")
        except Exception:
            return
        if len(data) != self._row_bytes:
            # Taille changée (resize/plein écran) -> on saute pour ne pas corrompre.
            self._next_t = now + self._interval
            return
        # Tient la cadence : 1 à 3 frames selon le retard (duplique si rendu lent).
        writes = 0
        while now >= self._next_t and writes < 3:
            try:
                self._q.put_nowait(data)
                self._frames += 1
            except queue.Full:
                self.dropped += 1          # encodeur en retard -> on abandonne
            self._next_t += self._interval
            writes += 1
        # Gros décrochage -> resync sans accumuler de retard.
        if now - self._next_t > 0.5:
            self._next_t = now + self._interval

    # ------------------------------------------------------------------ #
    #  Fermeture                                                          #
    # ------------------------------------------------------------------ #
    def close(self):
        """Ferme le flux et finalise le fichier. Renvoie (chemin, nb_frames).
        En cas d'échec ffmpeg (réglage refusé, 0 frame), `self.last_error`
        contient la fin du log pour diagnostic ; `self.dropped` = frames
        abandonnées faute d'encodeur assez rapide."""
        self._closed = True

        # --- Arrêt du thread audio (draine le reste de la dérivation) --------
        self._audio_stop = True
        if self._audio_q is not None:
            try:
                self._audio_q.put_nowait(None)     # sentinelle (débloque le get)
            except Exception:
                pass
        if self._audio_thread is not None:
            try:
                self._audio_thread.join(timeout=15)
            except Exception:
                pass
        if self._wav is not None:
            try:
                self._wav.close()
            except Exception:
                pass

        # Réveille / termine le thread d'écriture (sentinelle), même file pleine.
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        try:
            self._writer.join(timeout=30)
        except Exception:
            pass

        rc = None
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                rc = self._proc.wait(timeout=60)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

        fh = getattr(self, "_logfh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            self._logfh = None

        if (rc not in (0, None)) or self._frames == 0:
            self._read_log_tail()

        # --- Muxage : vidéo (copy) + audio capturé (AAC stéréo 192k 48 kHz) --
        if self._audio_enabled:
            self._mux_audio()

        return (self.path, self._frames)

    # ------------------------------------------------------------------ #
    #  Muxage audio (à la fermeture)                                      #
    # ------------------------------------------------------------------ #
    def _mux_audio(self):
        """Combine la vidéo HEVC temporaire et le WAV capturé en un MP4 final
        avec une piste AAC stéréo 192 kbps 48 kHz. Replis robustes : sans audio
        ou si le muxage échoue, on garde au minimum la vidéo (renommée en final)."""
        have_audio = (self._audio_frames > 0
                      and os.path.exists(self._audio_wav)
                      and os.path.exists(self._video_out))
        if not have_audio:
            self._promote_video()          # pas d'audio -> vidéo seule en final
            return
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            mux = [
                exe, "-y", "-loglevel", "error",
                "-i", self._video_out,                 # vidéo HEVC (sans audio)
                "-i", self._audio_wav,                 # audio capturé (WAV PCM)
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy",                        # vidéo NON ré-encodée
                "-c:a", "aac", "-b:a", self._audio_bitrate,
                "-ar", "48000", "-ac", "2",            # AAC stéréo 48 kHz
                "-movflags", "+faststart", "-shortest",
                self.path,
            ]
            with open(self._logpath, "ab") as lf:
                rc = subprocess.call(mux, stdout=subprocess.DEVNULL, stderr=lf)
            if rc == 0 and os.path.exists(self.path):
                self._safe_remove(self._video_out)     # succès -> ménage
                self._safe_remove(self._audio_wav)
            else:
                self._read_log_tail()
                self._promote_video()                  # échec -> vidéo muette
        except Exception as exc:
            self.last_error = f"mux: {exc}"
            self._promote_video()

    def _promote_video(self):
        """Replie la vidéo temporaire vers le chemin final (audio absent ou
        muxage en échec)."""
        if self._video_out != self.path and os.path.exists(self._video_out):
            try:
                if os.path.exists(self.path):
                    self._safe_remove(self.path)
                os.replace(self._video_out, self.path)
            except Exception:
                self.path = self._video_out            # dernier recours
        self._safe_remove(self._audio_wav)

    @staticmethod
    def _safe_remove(p):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

    def _read_log_tail(self):
        try:
            with open(self._logpath, "rb") as lf:
                tail = lf.read()[-800:].decode("utf-8", "replace").strip()
            if tail:
                self.last_error = tail
        except Exception:
            pass
