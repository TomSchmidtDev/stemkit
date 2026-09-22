# Synced Lyrics Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, fully local feature that transcribes the isolated vocals stem into a line-synced LRC file using OpenAI Whisper, and displays it as a scrolling, click-to-seek lyrics panel in the player.

**Architecture:** Nearly all feature logic lives in three new files — `python/transcribe.py` (Whisper transcription, NDJSON progress protocol matching the existing scripts), `src/main/lyrics.ts` (model download/caching/orchestration on the main process side), and `src/renderer/src/components/Lyrics.tsx` (the UI panel). Existing files (`pipeline.ts`, `types.ts`, `settings.ts`, `index.ts`, `preload/index.ts`, `Player.tsx`, `Settings.tsx`, `App.tsx`) get small, additive, mechanical touches only, so the change stays easy to review and to upstream.

**Tech Stack:** `openai-whisper` (PyPI package, import name `whisper`) — torch-based, reuses the app's existing torch/GPU install instead of a separate runtime. `torchaudio` (already a dependency) for 44.1kHz→16kHz resampling. Output format: standard `.lrc` (line-synced lyrics).

**Spec:** `docs/superpowers/specs/2026-09-22-synced-lyrics-design.md`

## Global Constraints

- Lyrics extraction must be **100% local** — no external services, no paid APIs. Whisper model weights are downloaded once from OpenAI's own public CDN via the `whisper` package's built-in downloader (a weights download, same category as the existing vocals-engine/fine-tune-engine downloads — not an API call).
- Feature is **opt-in** (`AppSettings.extractLyrics`, default `false`) with a **global** model-size setting (`AppSettings.lyricsModel`), not per-song.
- Reuses the existing `gpuSplit` GPU toggle/torch build — no separate GPU engine management for lyrics.
- A lyrics extraction failure must **never fail the containing split job** — it's purely additive.
- All new fachlich (domain) code goes in the three new files listed above; existing files get additive-only diffs (new fields, one new import + one new call site, one new UI block).

---

## File Structure

**New files:**
- `python/transcribe.py` — standalone script, same NDJSON stdout protocol as `separate.py`/`roformer.py`.
- `src/main/lyrics.ts` — model download/caching, path helpers, `maybeExtractLyrics()` orchestration, IPC-facing `readLyrics()`, smoke-test helper.
- `src/renderer/src/components/Lyrics.tsx` — the lyrics panel component + LRC parser.

**Modified files (additive touches only):**
- `src/shared/types.ts` — new fields/types.
- `src/main/settings.ts` — sanitize two new settings fields.
- `src/main/pipeline.ts` — one import, one call after `runSeparation` (×2: `startJob`, `startLocalJob`), one field in `finalizeJob`'s `upsertSong` call.
- `src/main/index.ts` — one new IPC handler, two existing handlers get one new case each.
- `src/preload/index.ts` — one new bridge method.
- `src/main/smoke.ts` — one new smoke-test block.
- `src/renderer/src/App.tsx` — one new `switch` case in `stageLabel`.
- `src/renderer/src/components/Player.tsx` — one new conditional panel + minor layout wrap.
- `src/renderer/src/components/Settings.tsx` — one new section (toggle + model picker).

---

### Task 1: Shared types and settings plumbing

**Files:**
- Modify: `src/shared/types.ts`
- Modify: `src/main/settings.ts`

**Interfaces:**
- Produces: `AppSettings.extractLyrics: boolean`, `AppSettings.lyricsModel: 'small' | 'medium' | 'large-v3'`, `Song.lyrics?: boolean`, `JobStage` including `'lyrics'`, `StemKitApi.getLyrics(videoId: string): Promise<string | null>`, `StemKitApi.fetchEngine(which: 'vocals' | 'ft' | 'gpu' | 'lyrics'): Promise<void>` (widened), `EngineStatus.lyricsDownloading: boolean`, `EngineStatus.lyricsReady: boolean`.

- [ ] **Step 1: Extend `src/shared/types.ts`**

In `AppSettings`, add two fields after `hideVideo`:

```ts
export interface AppSettings {
  shifts: 1 | 2
  htdemucsFt: boolean
  roformerVocals: boolean
  gpuSplit: boolean
  hideVideo: boolean
  // opt-in local lyrics transcription (whisper, run on the isolated vocals
  // stem); the model choice is global, not per-song
  extractLyrics: boolean
  lyricsModel: 'small' | 'medium' | 'large-v3'
}
```

Update `DEFAULT_SETTINGS`:

```ts
export const DEFAULT_SETTINGS: AppSettings = {
  shifts: 1,
  htdemucsFt: false,
  roformerVocals: false,
  gpuSplit: false,
  hideVideo: false,
  extractLyrics: false,
  lyricsModel: 'medium'
}
```

In `Song`, add after `source`:

```ts
export interface Song {
  videoId: string
  title: string
  duration: number
  addedAt: number
  model?: string
  stems?: string[]
  took?: number
  source?: 'local'
  // whether a synced lyrics.lrc exists for this song
  lyrics?: boolean
}
```

In `EngineStatus`, add after `gpuReady`:

```ts
export interface EngineStatus {
  vocalsDownloading: boolean
  vocalsReady: boolean
  ftDownloading: boolean
  ftVerified: boolean
  gpuDownloading: boolean
  gpuReady: boolean
  lyricsDownloading: boolean
  lyricsReady: boolean
}
```

Change `JobStage`:

```ts
export type JobStage = 'metadata' | 'download' | 'convert' | 'separate' | 'lyrics' | 'finalize'
```

In `StemKitApi`, widen `fetchEngine` and add `getLyrics`:

```ts
  fetchEngine(which: 'vocals' | 'ft' | 'gpu' | 'lyrics'): Promise<void>
  getLyrics(videoId: string): Promise<string | null>
```//  (add getLyrics anywhere in the interface body, e.g. right after getThumb)

- [ ] **Step 2: Extend `src/main/settings.ts`**

Replace the body of `saveSettings` with a version that also sanitizes the two new fields:

```ts
const LYRICS_MODELS: AppSettings['lyricsModel'][] = ['small', 'medium', 'large-v3']

export function saveSettings(patch: Partial<AppSettings>): AppSettings {
  const merged = { ...loadSettings(), ...patch }
  const next: AppSettings = {
    shifts: merged.shifts === 2 ? 2 : 1,
    htdemucsFt: !!merged.htdemucsFt,
    roformerVocals: !!merged.roformerVocals,
    gpuSplit: !!merged.gpuSplit,
    hideVideo: !!merged.hideVideo,
    extractLyrics: !!merged.extractLyrics,
    lyricsModel: LYRICS_MODELS.includes(merged.lyricsModel) ? merged.lyricsModel : 'medium'
  }
  writeFileSync(settingsFile(), JSON.stringify(next, null, 2))
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('settings:changed', next)
  }
  return next
}
```

Place the `LYRICS_MODELS` constant above the function, after the existing imports.

- [ ] **Step 3: Verify with typecheck**

Run: `npm run typecheck`
Expected: exits 0. (It will report errors in files that consume `AppSettings`/`Song`/`JobStage`/`StemKitApi` until later tasks fill them in — for this step specifically, only `types.ts` and `settings.ts` should compile clean; other files' errors are expected and resolved by later tasks. Confirm this by running `npm run typecheck:node` and `npm run typecheck:web` and checking that the only reported errors are in files this plan touches later: `pipeline.ts`, `index.ts`, `preload/index.ts`, `App.tsx`, `Player.tsx`, `Settings.tsx`.)

- [ ] **Step 4: Commit**

```bash
git add src/shared/types.ts src/main/settings.ts
git commit -m "feat: add lyrics settings/types plumbing"
```

---

### Task 2: `python/transcribe.py`

**Files:**
- Create: `python/transcribe.py`

**Interfaces:**
- Consumes: nothing from other tasks (standalone CLI script).
- Produces: CLI contract consumed by Task 3's `runTranscribe()`: `python transcribe.py --input <vocals.wav> --out <dir> --model <small|medium|large-v3> --ckpt-dir <dir> --device <auto|cpu|cuda|mps>`. Writes `<out>/lyrics.lrc` (only if any non-hallucinated segment was found) and emits NDJSON lines on stdout: `{"type":"progress","stage":"lyrics","pct":N,"message"?:str}`, `{"type":"done","lines":N,"out":str|null}`, `{"type":"error","message":str}`.

- [ ] **Step 1: Write the script**

```python
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

# whisper hallucinates text during silence/instrumental passages; both
# fields come back on every segment, so filter on them before writing
# anything to disk instead of trusting whisper's own text output blindly
NO_SPEECH_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0


def emit(**kwargs):
    print(json.dumps(kwargs), flush=True)


def fail(message):
    emit(type="error", message=str(message))
    sys.exit(1)


def load_wav(path):
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            width = w.getsampwidth()
            frames = w.readframes(w.getnframes())
    except Exception as e:
        fail(f"cannot read wav {path}: {e}")
    if width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype="<f4").astype(np.float32)
    else:
        fail(f"unsupported sample width {width}")
    if channels == 0:
        fail("empty wav")
    return audio.reshape(-1, channels).T, sr


def fmt_lrc_time(seconds):
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="medium")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    import torchaudio
    import whisper

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        fail("GPU engine not available (no supported NVIDIA/AMD GPU, or the GPU build of torch is not installed)")

    audio, sr = load_wav(args.input)
    mono = audio.mean(axis=0).astype(np.float32)
    if sr != 16000:
        t = torch.from_numpy(mono)
        t = torchaudio.functional.resample(t, sr, 16000)
        mono = t.numpy().astype(np.float32)

    # whisper's own audio loader shells out to ffmpeg on PATH, which the app
    # doesn't rely on elsewhere (ffmpeg is bundled and invoked by full path);
    # feeding a pre-resampled numpy array bypasses that shell-out entirely
    emit(type="progress", stage="lyrics", pct=0, message=f"loading lyrics engine on {device}")
    try:
        # whisper's mps kernels are incomplete for some ops; cpu is the safe
        # fallback there, same reasoning as demucs's mps->cpu fallback
        load_device = "cpu" if device == "mps" else device
        model = whisper.load_model(args.model, device=load_device, download_root=args.ckpt_dir)
    except Exception as e:
        fail(f"lyrics engine load failed: {e}")

    emit(type="progress", stage="lyrics", pct=20, message="transcribing")
    try:
        result = model.transcribe(
            mono,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
        )
    except Exception as e:
        fail(f"transcription failed: {e}")

    emit(type="progress", stage="lyrics", pct=90, message="writing lyrics")

    lines = []
    for seg in result.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        if seg.get("no_speech_prob", 0.0) >= NO_SPEECH_THRESHOLD:
            continue
        if seg.get("avg_logprob", 0.0) <= AVG_LOGPROB_THRESHOLD:
            continue
        lines.append((float(seg["start"]), text))

    os.makedirs(args.out, exist_ok=True)
    if lines:
        lrc_path = os.path.join(args.out, "lyrics.lrc")
        with open(lrc_path, "w", encoding="utf-8") as f:
            for start, text in lines:
                f.write(f"[{fmt_lrc_time(start)}]{text}\n")
        emit(type="done", lines=len(lines), out=lrc_path)
    else:
        emit(type="done", lines=0, out=None)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail(str(e)[:400] or e.__class__.__name__)
```

- [ ] **Step 2: Manual smoke-check the script in isolation**

This script needs a Python environment with `torch`, `torchaudio` and `openai-whisper`. Use the app's own dev venv (created by `npm run dev` once) plus a one-off manual install of whisper for this manual check (Task 3 automates that install in-app):

Run (macOS example path — adjust for your platform's Electron `userData` dir, printed by the app or found under `~/Library/Application Support/StemKit/venv`):
```bash
"$HOME/Library/Application Support/StemKit/venv/bin/pip" install -q openai-whisper
"$HOME/Library/Application Support/StemKit/venv/bin/python" -c "
import wave, struct
with wave.open('/tmp/silence.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(44100)
    w.writeframes(struct.pack('<h', 0) * 2 * 44100)
"
"$HOME/Library/Application Support/StemKit/venv/bin/python" python/transcribe.py \
  --input /tmp/silence.wav --out /tmp/lyrics-test --model small \
  --ckpt-dir /tmp/whisper-models --device cpu
```
Expected: NDJSON `progress` lines, then a `done` line with `"lines": 0, "out": null` (pure silence must not hallucinate lyrics), exit code 0, and no `/tmp/lyrics-test/lyrics.lrc` file created.

- [ ] **Step 3: Commit**

```bash
git add python/transcribe.py
git commit -m "feat: add local whisper-based lyrics transcription script"
```

---

### Task 3: `src/main/lyrics.ts`

**Files:**
- Create: `src/main/lyrics.ts`

**Interfaces:**
- Consumes: `venvPython()`, `modelsDir()` from `./env` (both already exported, unchanged); `songDir()`, `stemsDir()` from `./library` (already exported, unchanged); `AppSettings` from `../shared/types` (Task 1).
- Produces (consumed by Tasks 4–9):
  - `export type LyricsModel = 'small' | 'medium' | 'large-v3'`
  - `export function ensureLyricsEngine(model: LyricsModel, onProgress?: (pct: number) => void): Promise<boolean>`
  - `export function lyricsEngineStatus(model: LyricsModel): { lyricsDownloading: boolean; lyricsReady: boolean }`
  - `export function lyricsPath(videoId: string): string`
  - `export function lyricsPresent(videoId: string): boolean`
  - `export function readLyrics(videoId: string): string | null`
  - `export function maybeExtractLyrics(videoId: string, settings: AppSettings, device: string, onProgress: (pct: number, message?: string) => void): Promise<void>`
  - `export function smokeTestTranscribe(vocalsPath: string, outDir: string): Promise<{ ok: boolean; lines: number; error?: string }>`

- [ ] **Step 1: Write the file**

```ts
import { spawn, execFile } from 'child_process'
import { createInterface } from 'readline'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs'
import { join } from 'path'
import { app, BrowserWindow } from 'electron'
import { venvPython, modelsDir } from './env'
import { songDir, stemsDir } from './library'
import type { AppSettings } from '../shared/types'

export type LyricsModel = AppSettings['lyricsModel']

function sendEnvEvent(message: string, level: 'info' | 'error' | 'success' = 'info'): void {
  for (const win of BrowserWindow.getAllWindows()) {
    win.webContents.send('env:event', { message, level })
  }
}

function runCapture(cmd: string, args: string[], timeout = 20000): Promise<string> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { timeout }, (err, stdout) => {
      if (err) reject(err)
      else resolve(stdout)
    })
  })
}

function lyricsEngineDir(): string {
  return join(modelsDir(), 'whisper')
}

function readyMarkerPath(model: LyricsModel): string {
  return join(lyricsEngineDir(), `${model}.ready`)
}

function transcribeScript(): string {
  if (app.isPackaged) {
    return join(process.resourcesPath, 'python', 'transcribe.py')
  }
  return join(app.getAppPath(), 'python', 'transcribe.py')
}

export function lyricsPath(videoId: string): string {
  return join(songDir(videoId), 'lyrics.lrc')
}

function lyricsModelMarkerPath(videoId: string): string {
  return join(songDir(videoId), 'lyrics.model')
}

export function lyricsPresent(videoId: string): boolean {
  return existsSync(lyricsPath(videoId))
}

export function readLyrics(videoId: string): string | null {
  const p = lyricsPath(videoId)
  if (!existsSync(p)) return null
  try {
    return readFileSync(p, 'utf8')
  } catch {
    return null
  }
}

/* lazily installs the openai-whisper package into the venv, mirroring
   ensureEngineDeps() in env.ts for the roformer engine's extra deps —
   kept out of the base bootstrap so users who never enable lyrics don't
   pay for the extra download */
let lyricsDepsReady = false

export async function ensureLyricsDeps(): Promise<boolean> {
  if (lyricsDepsReady) return true
  try {
    await runCapture(venvPython(), ['-c', 'import whisper'], 20000)
    lyricsDepsReady = true
    return true
  } catch {}
  sendEnvEvent('Preparing lyrics components…')
  try {
    await new Promise<void>((resolve, reject) => {
      const child = spawn(venvPython(), ['-m', 'pip', 'install', '-q', 'openai-whisper'], {
        env: { ...process.env }
      })
      child.on('close', (code) =>
        code === 0 ? resolve() : reject(new Error(`pip install failed (${code})`))
      )
      child.on('error', reject)
    })
    lyricsDepsReady = true
    sendEnvEvent('Lyrics components ready', 'success')
    return true
  } catch (err) {
    sendEnvEvent(
      `Lyrics components failed: ${err instanceof Error ? err.message : String(err)}`,
      'error'
    )
    return false
  }
}

/* downloads the chosen whisper checkpoint once, via whisper's own
   downloader (parsing its tqdm stderr for progress, same pattern as
   torchHubFetch() in env.ts for the htdemucs_ft download). A marker file
   is written on success so readiness survives app restarts without
   depending on whisper's internal cache file naming */
let lyricsEnginePromise: Promise<boolean> | null = null
const lyricsProgressListeners = new Set<(pct: number) => void>()

function whisperDownloadFetch(model: LyricsModel, onProgress?: (pct: number) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    mkdirSync(lyricsEngineDir(), { recursive: true })
    const child = spawn(
      venvPython(),
      [
        '-c',
        `import whisper; whisper.load_model(${JSON.stringify(model)}, device="cpu", download_root=${JSON.stringify(lyricsEngineDir())})`
      ],
      { env: { ...process.env } }
    )
    let lastPct = 0
    let stderrTail = ''
    child.stderr?.on('data', (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString()).slice(-1000)
      for (const piece of chunk.toString().split(/[\r\n]/)) {
        const m = piece.match(/(\d{1,3})%/)
        if (!m) continue
        const pct = Math.min(99, parseInt(m[1], 10))
        if (pct <= lastPct) continue
        lastPct = pct
        sendEnvEvent(`lyrics engine: ${pct}%`)
        onProgress?.(pct)
      }
    })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        onProgress?.(100)
        resolve()
      } else {
        reject(
          new Error(
            stderrTail.split('\n').filter(Boolean).slice(-1).join('') ||
              `lyrics engine download exited ${code}`
          )
        )
      }
    })
  })
}

export function ensureLyricsEngine(
  model: LyricsModel,
  onProgress?: (pct: number) => void
): Promise<boolean> {
  if (onProgress) lyricsProgressListeners.add(onProgress)
  const detach = (): boolean => {
    if (onProgress) lyricsProgressListeners.delete(onProgress)
    return true
  }
  if (existsSync(readyMarkerPath(model))) {
    detach()
    return Promise.resolve(true)
  }
  if (!lyricsEnginePromise) {
    lyricsEnginePromise = (async () => {
      sendEnvEvent(`Downloading the lyrics engine (${model}, one time)`)
      await whisperDownloadFetch(model, (pct) => {
        for (const listener of lyricsProgressListeners) listener(pct)
      })
      writeFileSync(readyMarkerPath(model), '')
      sendEnvEvent('Lyrics engine ready', 'success')
      return true
    })()
      .catch((err) => {
        sendEnvEvent(
          `Lyrics engine download failed: ${err instanceof Error ? err.message : String(err)}`,
          'error'
        )
        return false
      })
      .finally(() => {
        lyricsEnginePromise = null
      })
  }
  return lyricsEnginePromise.then(detach)
}

export function lyricsEngineStatus(
  model: LyricsModel
): { lyricsDownloading: boolean; lyricsReady: boolean } {
  return {
    lyricsDownloading: lyricsEnginePromise !== null,
    lyricsReady: existsSync(readyMarkerPath(model))
  }
}

interface TranscribeResult {
  lines: number
  error?: string
}

function runTranscribe(
  vocalsPath: string,
  outDir: string,
  model: LyricsModel,
  device: string,
  onProgress?: (pct: number) => void
): Promise<TranscribeResult> {
  return new Promise((resolve, reject) => {
    const child = spawn(
      venvPython(),
      [
        transcribeScript(),
        '--input',
        vocalsPath,
        '--out',
        outDir,
        '--model',
        model,
        '--ckpt-dir',
        lyricsEngineDir(),
        '--device',
        device
      ],
      { env: { ...process.env } }
    )
    let lines = 0
    let scriptError: string | undefined
    let lastPct = 0
    if (child.stdout) {
      createInterface({ input: child.stdout }).on('line', (line) => {
        let parsed: Record<string, unknown>
        try {
          parsed = JSON.parse(line)
        } catch {
          return
        }
        if (parsed.type === 'progress') {
          const pct = Math.max(lastPct, Number(parsed.pct ?? 0))
          lastPct = pct
          onProgress?.(pct)
        } else if (parsed.type === 'error') {
          scriptError = String(parsed.message)
        } else if (parsed.type === 'done') {
          lines = Number(parsed.lines ?? 0)
        }
      })
    }
    let stderrTail = ''
    child.stderr?.on('data', (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString()).slice(-1000)
    })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ lines, error: scriptError })
      } else {
        reject(
          new Error(
            scriptError ||
              stderrTail.split('\n').filter(Boolean).slice(-2).join(' — ') ||
              `transcribe.py exited with code ${code}`
          )
        )
      }
    })
  })
}

/* called from pipeline.ts right after a job's stem separation finishes.
   Purely additive: any failure here is logged as an env event but never
   throws, so it can never fail the containing split job. Caching is
   decoupled from the stem cache in library.ts — switching the lyrics
   model re-transcribes without re-splitting the stems */
export async function maybeExtractLyrics(
  videoId: string,
  settings: AppSettings,
  device: string,
  onProgress: (pct: number, message?: string) => void
): Promise<void> {
  if (!settings.extractLyrics) return
  const vocalsPath = join(stemsDir(videoId), 'vocals.wav')
  if (!existsSync(vocalsPath)) return

  const model = settings.lyricsModel
  const markerPath = lyricsModelMarkerPath(videoId)
  const existingModel = existsSync(markerPath) ? readFileSync(markerPath, 'utf8').trim() : null
  if (lyricsPresent(videoId) && existingModel === model) return

  try {
    if (!(await ensureLyricsDeps())) {
      onProgress(0, 'Could not prepare the lyrics engine components')
      return
    }
    const engineReady = await ensureLyricsEngine(model, (pct) =>
      onProgress(Math.round(pct * 0.3), `Downloading lyrics engine: ${pct}%`)
    )
    if (!engineReady) {
      onProgress(0, 'Could not download the lyrics engine')
      return
    }
    onProgress(30, 'Extracting lyrics')
    const result = await runTranscribe(vocalsPath, songDir(videoId), model, device, (pct) =>
      onProgress(30 + Math.round(pct * 0.7))
    )
    writeFileSync(markerPath, model)
    if (result.lines === 0) {
      onProgress(100, 'No lyrics found')
    } else {
      onProgress(100, 'Lyrics ready')
    }
  } catch (err) {
    sendEnvEvent(
      `Lyrics extraction failed: ${err instanceof Error ? err.message : String(err)}`,
      'error'
    )
  }
}

/* used by src/main/smoke.ts: runs the whole lyrics pipeline (deps, engine
   download, transcription) against an arbitrary wav and reports whether it
   ran cleanly and how many lines it produced — the smoke test asserts 0
   lines on a pure sine-tone mix, proving the hallucination filter works */
export async function smokeTestTranscribe(
  vocalsPath: string,
  outDir: string
): Promise<{ ok: boolean; lines: number; error?: string }> {
  if (!(await ensureLyricsDeps())) return { ok: false, lines: 0, error: 'lyrics deps install failed' }
  if (!(await ensureLyricsEngine('small'))) {
    return { ok: false, lines: 0, error: 'lyrics engine download failed' }
  }
  try {
    const result = await runTranscribe(vocalsPath, outDir, 'small', 'cpu')
    return { ok: true, lines: result.lines }
  } catch (err) {
    return { ok: false, lines: 0, error: err instanceof Error ? err.message : String(err) }
  }
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck:node`
Expected: no errors originating from `src/main/lyrics.ts`. (Other files that will reference `lyrics.ts` exports — `pipeline.ts`, `index.ts`, `smoke.ts` — still error until Tasks 4–6; that's expected at this point.)

- [ ] **Step 3: Manual end-to-end check against the dev venv**

With the same dev venv from Task 2 (whisper already installed there), run a tiny throwaway Node script to exercise `maybeExtractLyrics` directly:

```bash
node -e "
process.env.ELECTRON_RUN_AS_NODE = '1'
" 2>/dev/null || true
```

Since `lyrics.ts` imports Electron's `app`/`BrowserWindow`, it can only run inside the Electron process. Instead, verify it end-to-end through the running app once Task 4 wires it into the pipeline (flip `extractLyrics` on via a manual `settings.json` edit in the app's userData folder, then split a short song and confirm `songDir/<id>/lyrics.lrc` appears). Note this explicitly as the real verification point for this task; defer it to Task 4's manual check to avoid a redundant throwaway harness.

- [ ] **Step 4: Commit**

```bash
git add src/main/lyrics.ts
git commit -m "feat: add main-process lyrics orchestration module"
```

---

### Task 4: Wire `pipeline.ts`

**Files:**
- Modify: `src/main/pipeline.ts:1-40` (imports), `startJob` (around line 304-306), `startLocalJob` (around line 369-371), `finalizeJob` (around line 192-210)

**Interfaces:**
- Consumes: `maybeExtractLyrics`, `lyricsPresent` from `./lyrics` (Task 3); `plan.settings`, `plan.deviceArg()` from the existing `EnginePlan` (unchanged, already present in `pipeline.ts`).

- [ ] **Step 1: Add the import**

In `src/main/pipeline.ts`, add to the top-level imports (near the other local imports):

```ts
import { maybeExtractLyrics, lyricsPresent } from './lyrics'
```

- [ ] **Step 2: Call it after separation in `startJob`**

Find:
```ts
    const producedStems = await runSeparation(job, plan, stems)
    if (job.cancelled || !jobs.has(videoId)) return
    finalizeJob(job, { title: meta!.title, duration: meta!.duration, addedAt, startedAt }, producedStems)
```
Replace with:
```ts
    const producedStems = await runSeparation(job, plan, stems)
    if (job.cancelled || !jobs.has(videoId)) return
    await maybeExtractLyrics(videoId, plan.settings, plan.deviceArg(), (pct, msg) =>
      progress(job, 'lyrics', pct, msg)
    )
    finalizeJob(job, { title: meta!.title, duration: meta!.duration, addedAt, startedAt }, producedStems)
```

- [ ] **Step 3: Call it after separation in `startLocalJob`**

Find:
```ts
    const producedStems = await runSeparation(job, plan, stems)
    if (job.cancelled || !jobs.has(videoId)) return
    finalizeJob(job, { title, duration, addedAt, startedAt, source: 'local' }, producedStems)
```
Replace with:
```ts
    const producedStems = await runSeparation(job, plan, stems)
    if (job.cancelled || !jobs.has(videoId)) return
    await maybeExtractLyrics(videoId, plan.settings, plan.deviceArg(), (pct, msg) =>
      progress(job, 'lyrics', pct, msg)
    )
    finalizeJob(job, { title, duration, addedAt, startedAt, source: 'local' }, producedStems)
```

- [ ] **Step 4: Record lyrics presence in `finalizeJob`**

Find:
```ts
  const songs = upsertSong({
    videoId: job.videoId,
    title: info.title,
    duration: info.duration,
    addedAt: info.addedAt,
    model: job.model,
    stems: producedStems,
    took,
    source: info.source
  })
```
Replace with:
```ts
  const songs = upsertSong({
    videoId: job.videoId,
    title: info.title,
    duration: info.duration,
    addedAt: info.addedAt,
    model: job.model,
    stems: producedStems,
    took,
    source: info.source,
    lyrics: lyricsPresent(job.videoId)
  })
```

- [ ] **Step 5: Typecheck**

Run: `npm run typecheck:node`
Expected: no errors in `pipeline.ts`. `index.ts` and `smoke.ts` may still error until Tasks 5–6.

- [ ] **Step 6: Manual end-to-end verification**

```bash
npm run dev
```
In the running app: open Settings, enable "Extract lyrics" (added in Task 9 — if Task 9 isn't done yet, instead edit `settings.json` directly in the app's userData folder to set `"extractLyrics": true, "lyricsModel": "small"`, then restart the app), then split a short song with clear vocals. Watch the main-process console for `lyrics engine: N%` / `Lyrics engine ready` events, then confirm `<userData>/songs/<videoId>/lyrics.lrc` exists and contains `[mm:ss.xx]text` lines.

- [ ] **Step 7: Commit**

```bash
git add src/main/pipeline.ts
git commit -m "feat: run lyrics extraction after stem separation"
```

---

### Task 5: IPC handlers and preload bridge

**Files:**
- Modify: `src/main/index.ts` (imports near top, `engines:status`/`engines:fetch` handlers, add `lyrics:get` handler)
- Modify: `src/preload/index.ts`

**Interfaces:**
- Consumes: `ensureLyricsEngine`, `lyricsEngineStatus`, `readLyrics` from `./lyrics` (Task 3); `loadSettings` from `./settings` (already imported in `index.ts`).
- Produces: `getLyrics` IPC channel `'lyrics:get'` consumed by Task 7/8's renderer code.

- [ ] **Step 1: Import lyrics functions in `src/main/index.ts`**

Add near the other local imports:

```ts
import { ensureLyricsEngine, lyricsEngineStatus, readLyrics } from './lyrics'
```

- [ ] **Step 2: Extend the `engines:status` handler**

Find:
```ts
  ipcMain.handle('engines:status', () => {
    // warm the cuda probe so gpuReady flips without waiting for a split
    if (getStatus().ready) void hasGpuAcceleration()
    return engineStatus()
  })
```
Replace with:
```ts
  ipcMain.handle('engines:status', () => {
    // warm the cuda probe so gpuReady flips without waiting for a split
    if (getStatus().ready) void hasGpuAcceleration()
    return { ...engineStatus(), ...lyricsEngineStatus(loadSettings().lyricsModel) }
  })
```

- [ ] **Step 3: Extend the `engines:fetch` handler**

Find:
```ts
  ipcMain.handle('engines:fetch', (_e, which: 'vocals' | 'ft' | 'gpu') => {
    if (which === 'vocals') void ensureVocalsEngine()
    else if (which === 'ft') void ensureFtWeights()
    else void ensureGpuEngine()
  })
```
Replace with:
```ts
  ipcMain.handle('engines:fetch', (_e, which: 'vocals' | 'ft' | 'gpu' | 'lyrics') => {
    if (which === 'vocals') void ensureVocalsEngine()
    else if (which === 'ft') void ensureFtWeights()
    else if (which === 'gpu') void ensureGpuEngine()
    else void ensureLyricsEngine(loadSettings().lyricsModel)
  })
```

- [ ] **Step 4: Add the `lyrics:get` handler**

Add right after the `stem:export`/`stems:export-all` handlers (or any other ipcMain.handle block — placement doesn't matter, group with the other song-data handlers near `song:buffers`):

```ts
  ipcMain.handle('lyrics:get', (_e, videoId: string) => readLyrics(videoId))
```

- [ ] **Step 5: Extend `src/preload/index.ts`**

Add to the `api` object, e.g. right after `getThumb`/`onThumbCached`:

```ts
  getLyrics: (videoId) => ipcRenderer.invoke('lyrics:get', videoId),
```

(`fetchEngine` stays as-is — its type already widens automatically via `StemKitApi` from Task 1, no code change needed there.)

- [ ] **Step 6: Typecheck**

Run: `npm run typecheck`
Expected: exits 0 across both `tsconfig.node.json` and `tsconfig.web.json` (the renderer side still doesn't call `getLyrics` yet, but the type is now fully wired end to end).

- [ ] **Step 7: Manual verification**

```bash
npm run dev
```
Open devtools console in the renderer and run `await window.stemkit.getLyrics('<a videoId with lyrics.lrc from Task 4's test>')` — expect the raw LRC text string back. For a song without lyrics, expect `null`.

- [ ] **Step 8: Commit**

```bash
git add src/main/index.ts src/preload/index.ts
git commit -m "feat: expose lyrics over IPC"
```

---

### Task 6: Extend the smoke test

**Files:**
- Modify: `src/main/smoke.ts`

**Interfaces:**
- Consumes: `smokeTestTranscribe` from `./lyrics` (Task 3); the existing `roformerOut` vocals path already produced earlier in `runSmoke()`.

- [ ] **Step 1: Import the helper**

Add to the imports at the top of `src/main/smoke.ts`:

```ts
import { smokeTestTranscribe } from './lyrics'
```

- [ ] **Step 2: Add the lyrics smoke block**

Insert right after the existing roformer block (after the `log('roformer vocals ok')` line and before the `// GPU plumbing` comment):

```ts
    // lyrics extraction on a pure sine-wave mix: must not hallucinate any
    // lines (no vocals content at all) — proves the no_speech/avg_logprob
    // filter in transcribe.py actually works
    const lyricsOut = join(dir, 'stems-lyrics')
    const lyricsResult = await smokeTestTranscribe(join(roformerOut, 'vocals.wav'), lyricsOut)
    if (!lyricsResult.ok) {
      log(`FAIL: lyrics extraction: ${lyricsResult.error}`)
      return false
    }
    if (lyricsResult.lines > 0) {
      log(`FAIL: lyrics extraction produced ${lyricsResult.lines} hallucinated lines from a tone-only mix`)
      return false
    }
    log('lyrics extraction ok (no hallucinated lines on silence/tone)')
```

- [ ] **Step 3: Typecheck**

Run: `npm run typecheck:node`
Expected: exits 0.

- [ ] **Step 4: Run the smoke test**

Run: `STEMKIT_SMOKE=1 npm run dev`
Expected: process exits 0; `/tmp/stemkit-smoke.log` (or platform tmp dir equivalent) contains `lyrics extraction ok (no hallucinated lines on silence/tone)` and ends with `SMOKE RESULT: PASS`. This is slow (full bootstrap + both existing engines + now the lyrics engine download) — expect several minutes on a clean run, seconds on a machine that already has everything cached.

- [ ] **Step 5: Commit**

```bash
git add src/main/smoke.ts
git commit -m "test: extend smoke test with lyrics hallucination check"
```

---

### Task 7: `Lyrics.tsx` renderer component

**Files:**
- Create: `src/renderer/src/components/Lyrics.tsx`

**Interfaces:**
- Consumes: `window.stemkit.getLyrics(videoId)` (Task 5).
- Produces: `export function parseLrc(text: string): LyricLine[]`, `export function Lyrics(props: { videoId: string; getPosition: () => number; onSeek: (seconds: number) => void }): React.ReactElement | null` — both consumed by Task 8.

- [ ] **Step 1: Write the component**

```tsx
import { useEffect, useRef, useState } from 'react'

export interface LyricLine {
  time: number
  text: string
}

const LRC_LINE = /^\[(\d+):(\d+(?:\.\d+)?)\](.*)$/

export function parseLrc(text: string): LyricLine[] {
  const lines: LyricLine[] = []
  for (const raw of text.split('\n')) {
    const m = LRC_LINE.exec(raw.trim())
    if (!m) continue
    const minutes = parseInt(m[1], 10)
    const seconds = parseFloat(m[2])
    const content = m[3].trim()
    if (!content) continue
    lines.push({ time: minutes * 60 + seconds, text: content })
  }
  return lines
}

interface Props {
  videoId: string
  getPosition: () => number
  onSeek: (seconds: number) => void
}

export function Lyrics({ videoId, getPosition, onSeek }: Props): React.ReactElement | null {
  const [lines, setLines] = useState<LyricLine[]>([])
  const [activeIndex, setActiveIndex] = useState(-1)
  const lineRefs = useRef<(HTMLButtonElement | null)[]>([])

  useEffect(() => {
    let cancelled = false
    setLines([])
    setActiveIndex(-1)
    void window.stemkit.getLyrics(videoId).then((text) => {
      if (cancelled || !text) return
      setLines(parseLrc(text))
    })
    return () => {
      cancelled = true
    }
  }, [videoId])

  useEffect(() => {
    if (lines.length === 0) return
    const id = setInterval(() => {
      const pos = getPosition()
      let idx = -1
      for (let i = 0; i < lines.length; i++) {
        if (lines[i].time <= pos) idx = i
        else break
      }
      setActiveIndex((prev) => (prev === idx ? prev : idx))
    }, 150)
    return () => clearInterval(id)
  }, [lines, getPosition])

  useEffect(() => {
    lineRefs.current[activeIndex]?.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [activeIndex])

  if (lines.length === 0) return null

  return (
    <div className="glass rounded-2xl w-72 shrink-0 max-h-[420px] overflow-y-auto px-3 py-4 space-y-1">
      {lines.map((line, i) => (
        <button
          key={i}
          ref={(el) => {
            lineRefs.current[i] = el
          }}
          onClick={() => onSeek(line.time)}
          className={`no-drag block w-full text-left px-3 py-1.5 rounded-lg text-[13px] leading-relaxed transition-colors ${
            i === activeIndex
              ? 'bg-violet-500/20 text-white font-medium'
              : 'text-white/45 hover:text-white/70 hover:bg-white/5'
          }`}
        >
          {line.text}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck:web`
Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add src/renderer/src/components/Lyrics.tsx
git commit -m "feat: add synced lyrics panel component"
```

---

### Task 8: Wire the panel into `Player.tsx` and the job-stage label into `App.tsx`

**Files:**
- Modify: `src/renderer/src/components/Player.tsx:1-10` (import), `:403-430` (layout)
- Modify: `src/renderer/src/App.tsx:24-37` (`stageLabel`)

**Interfaces:**
- Consumes: `Lyrics` component (Task 7); `song.lyrics`, `getPosition`, `seekTo` (all already present in `Player.tsx`).

- [ ] **Step 1: Import `Lyrics` in `Player.tsx`**

Add near the other component imports:

```ts
import { Lyrics } from './Lyrics'
```

- [ ] **Step 2: Wrap the stem-lane list and add the panel**

Find:
```tsx
          <div className="mt-4 space-y-2">
            {decoding
              ? [...Array(4)].map((_, i) => (
                  <div
                    key={i}
                    className="glass rounded-xl h-16 animate-pulse"
                    style={{ animationDelay: `${i * 120}ms` }}
                  />
                ))
              : stemMeta.map((meta) => (
              <StemLane
                key={meta.id}
                meta={meta}
                buffer={buffers[meta.id] ?? null}
                duration={duration}
                getPosition={getPosition}
                audible={!mutes.has(meta.id) && (solos.size === 0 || solos.has(meta.id))}
                volume={vols[meta.id] ?? 1}
                muted={mutes.has(meta.id)}
                soloed={solos.has(meta.id)}
                onToggleMute={() => toggleMute(meta.id)}
                onToggleSolo={() => toggleSolo(meta.id)}
                onVolume={(v) => setVols((prev) => ({ ...prev, [meta.id]: v }))}
                onSeek={seekTo}
                onExport={() => exportStem(meta.id)}
              />
            ))}
          </div>
```
Replace with:
```tsx
          <div className="mt-4 flex items-start gap-4">
            <div className="flex-1 min-w-0 space-y-2">
              {decoding
                ? [...Array(4)].map((_, i) => (
                    <div
                      key={i}
                      className="glass rounded-xl h-16 animate-pulse"
                      style={{ animationDelay: `${i * 120}ms` }}
                    />
                  ))
                : stemMeta.map((meta) => (
                <StemLane
                  key={meta.id}
                  meta={meta}
                  buffer={buffers[meta.id] ?? null}
                  duration={duration}
                  getPosition={getPosition}
                  audible={!mutes.has(meta.id) && (solos.size === 0 || solos.has(meta.id))}
                  volume={vols[meta.id] ?? 1}
                  muted={mutes.has(meta.id)}
                  soloed={solos.has(meta.id)}
                  onToggleMute={() => toggleMute(meta.id)}
                  onToggleSolo={() => toggleSolo(meta.id)}
                  onVolume={(v) => setVols((prev) => ({ ...prev, [meta.id]: v }))}
                  onSeek={seekTo}
                  onExport={() => exportStem(meta.id)}
                />
              ))}
            </div>
            {song.lyrics && !decoding && (
              <Lyrics videoId={song.videoId} getPosition={getPosition} onSeek={seekTo} />
            )}
          </div>
```

- [ ] **Step 3: Add the `'lyrics'` case to `stageLabel` in `App.tsx`**

Find:
```ts
function stageLabel(stage: JobStage, pct: number): string {
  switch (stage) {
    case 'metadata':
      return 'reading info…'
    case 'download':
      return `downloading ${Math.round(pct)}%`
    case 'convert':
      return 'converting…'
    case 'separate':
      return `separating ${Math.round(pct)}%`
    case 'finalize':
      return 'finishing…'
  }
}
```
Replace with:
```ts
function stageLabel(stage: JobStage, pct: number): string {
  switch (stage) {
    case 'metadata':
      return 'reading info…'
    case 'download':
      return `downloading ${Math.round(pct)}%`
    case 'convert':
      return 'converting…'
    case 'separate':
      return `separating ${Math.round(pct)}%`
    case 'lyrics':
      return `extracting lyrics ${Math.round(pct)}%`
    case 'finalize':
      return 'finishing…'
  }
}
```

- [ ] **Step 4: Typecheck**

Run: `npm run typecheck:web`
Expected: exits 0.

- [ ] **Step 5: Manual verification**

```bash
npm run dev
```
Open the song from Task 4's test (the one with a real `lyrics.lrc`). Expect the lyrics panel to appear to the right of the stem lanes, the current line to highlight and auto-scroll during playback, and clicking a line to seek + keep audio/video in sync. Open a song without lyrics and confirm no panel/layout shift appears (single-column layout unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/renderer/src/components/Player.tsx src/renderer/src/App.tsx
git commit -m "feat: show synced lyrics panel in the player"
```

---

### Task 9: Settings UI — toggle and model picker

**Files:**
- Modify: `src/renderer/src/components/Settings.tsx`

**Interfaces:**
- Consumes: `settings.extractLyrics`, `settings.lyricsModel` (Task 1); `window.stemkit.enginesStatus()` returning `lyricsDownloading`/`lyricsReady` (Task 5); `window.stemkit.fetchEngine('lyrics')` (Task 5); `window.stemkit.onEnvEvent` (existing).

- [ ] **Step 1: Add lyrics-related state**

In the `Settings` component, add alongside the existing `vocalsPct`/`ftPct`/`gpuPct` state:

```ts
  const [lyricsPct, setLyricsPct] = useState<number | null>(null)
  const [lyricsError, setLyricsError] = useState<string | null>(null)
  const [lyricsStarting, setLyricsStarting] = useState(false)
```

- [ ] **Step 2: Extend the `onEnvEvent` listener**

Inside the existing `useEffect` that subscribes to `onEnvEvent`, add alongside the gpu-engine regex block:

```ts
      const lyricsEngine = e.message.match(/lyrics engine: (\d+)%/)
      if (lyricsEngine) {
        setLyricsPct(parseInt(lyricsEngine[1], 10))
        setLyricsError(null)
      }
      if (/Lyrics engine ready/.test(e.message)) setLyricsPct(100)
      if (/Lyrics engine download failed/.test(e.message)) setLyricsError(e.message)
```

- [ ] **Step 3: Extend the polling `useEffect`**

Inside the existing polling `tick()` function, add alongside the other `if (s...) set...Starting(false)` lines:

```ts
        if (s.lyricsDownloading || s.lyricsReady) setLyricsStarting(false)
```

- [ ] **Step 4: Add derived state and the start handler**

Alongside the existing `vocalsBusy`/`ftBusy`/`gpuBusy`/`showVocalsConfirm`/etc. block:

```ts
  const lyricsBusy = lyricsStarting || (engines?.lyricsDownloading ?? false)
  const showLyricsConfirm =
    !!engines && settings.extractLyrics && !engines.lyricsReady && !lyricsBusy
```

Alongside `startVocals`/`startFt`/`startGpu`:

```ts
  const startLyrics = (): void => {
    setLyricsStarting(true)
    setLyricsPct(null)
    setLyricsError(null)
    void window.stemkit.fetchEngine('lyrics')
  }
```

- [ ] **Step 5: Add the Lyrics section to the JSX**

Add a constant above the component (or inline where convenient) for the size labels:

```ts
const LYRICS_MODEL_SIZES: Record<AppSettings['lyricsModel'], string> = {
  small: '~500 MB',
  medium: '~1.5 GB',
  'large-v3': '~3 GB'
}
```

Insert a new `<section>` between the existing "Separation" section and the "Playback" section (i.e. right after the closing `</section>` of Separation, before `<section className="pt-5 border-t border-white/[0.06] space-y-5">` that starts Playback):

```tsx
          <section className="pt-5 border-t border-white/[0.06] space-y-5">
            <SectionHeader label="Lyrics" />
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-[13px] font-medium">Extract lyrics</p>
                <p className="text-[11.5px] text-white/40 leading-relaxed mt-0.5">
                  Transcribes the vocals stem locally into synced lyrics you can follow along
                  with while playing. Fully offline.
                </p>
                <DownloadBar
                  pct={lyricsPct}
                  starting={lyricsBusy && lyricsPct === null}
                  error={lyricsError}
                />
                {showLyricsConfirm && (
                  <ConfirmButton
                    label={`Download now · ${LYRICS_MODEL_SIZES[settings.lyricsModel]}`}
                    onClick={startLyrics}
                  />
                )}
              </div>
              <Toggle
                on={settings.extractLyrics}
                disabled={lyricsBusy}
                loading={lyricsBusy}
                onClick={() => onChange({ extractLyrics: !settings.extractLyrics })}
              />
            </div>

            {settings.extractLyrics && (
              <div className="flex items-center justify-between gap-4">
                <p className="text-[13px] font-medium">Model quality</p>
                <div className="flex shrink-0 rounded-lg bg-white/[0.06] p-0.5 border border-white/[0.08]">
                  {(
                    [
                      { v: 'small' as const, label: 'Fast' },
                      { v: 'medium' as const, label: 'Balanced' },
                      { v: 'large-v3' as const, label: 'Best' }
                    ]
                  ).map((opt) => (
                    <button
                      key={opt.v}
                      onClick={() => onChange({ lyricsModel: opt.v })}
                      className={`no-drag px-2.5 h-6 rounded-md text-[12px] font-semibold transition-colors ${
                        settings.lyricsModel === opt.v
                          ? 'bg-white text-black'
                          : 'text-white/45 hover:text-white/80'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </section>
```

- [ ] **Step 6: Typecheck**

Run: `npm run typecheck:web`
Expected: exits 0.

- [ ] **Step 7: Manual verification**

```bash
npm run dev
```
Open Settings, confirm a new "Lyrics" section appears between "Separation" and "Playback" with a toggle and, once enabled, a 3-way Fast/Balanced/Best picker. Toggle it on, confirm the download bar/confirm-button behavior matches the existing vocals/ft/gpu sections (spinner while downloading, error text on failure). Split a song afterwards and confirm it produces lyrics (cross-check with Task 4/8's manual runs).

- [ ] **Step 8: Commit**

```bash
git add src/renderer/src/components/Settings.tsx
git commit -m "feat: add lyrics settings section (toggle + model picker)"
```

---

## Self-Review Notes

- **Spec coverage:** Machbarkeit/Ansatz A → Tasks 2–3. Settings (opt-in + global model choice + GPU-toggle reuse) → Tasks 1, 4, 9. Own panel → Tasks 7–8. LRC format + caching marker → Task 3. Fehlerbehandlung (Halluzinations-Filter, nicht-fataler Fehler) → Tasks 2, 3. Testing (Smoke-Test-Erweiterung) → Task 6. File-separation goal (fachliche Logik in neuen Dateien) → reflected throughout the File Structure section and every task's touch list.
- **Placeholder scan:** no TBD/TODO markers; every step has full runnable code or an exact command with expected output.
- **Type consistency:** `LyricsModel` (Task 3) = `AppSettings['lyricsModel']` (Task 1) throughout; `maybeExtractLyrics(videoId, settings, device, onProgress)` signature matches its Task 4 call site exactly; `lyricsPresent`/`lyricsPath`/`readLyrics`/`ensureLyricsEngine`/`lyricsEngineStatus`/`smokeTestTranscribe` names and signatures match between their Task 3 definition and every consumer (Tasks 4, 5, 6); `parseLrc`/`Lyrics` props match between Task 7's definition and Task 8's usage; `JobStage`'s `'lyrics'` member matches between Task 1, Task 4's `progress(job, 'lyrics', ...)` calls, and Task 8's `stageLabel` case.
