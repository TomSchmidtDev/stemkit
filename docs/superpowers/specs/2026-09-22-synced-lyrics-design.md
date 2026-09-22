# Synchronisierte Lyrics-Extraktion (lokal, opt-in)

Status: Approved (Design), bereit für Implementation-Plan
Datum: 2026-09-22

## Ziel

Beim Separieren der Stems soll optional — komplett lokal, ohne externe Services
oder Bezahl-APIs — der Songtext aus dem isolierten `vocals.wav`-Stem extrahiert
und synchron zur Wiedergabe in einem eigenen Panel im Player angezeigt werden
können.

## Nicht-Ziele (MVP)

- Wortgenaues Karaoke-Highlighting (nur zeilengenaues Sync via LRC).
- Nachträgliche Lyrics-Generierung für bereits fertig gesplittete Songs, wenn
  das Feature erst danach aktiviert wird (Song muss erneut gesplittet werden).
- Export der `.lrc`-Datei (das Format erlaubt das später trivial, ist aber
  kein Teil dieses Specs).

## Machbarkeit (kurz)

Lokale Spracherkennung (ASR) über das bereits isolierte `vocals.wav` ist mit
OpenAI Whisper möglich und läuft komplett offline. Gesang ist für ASR
schwerer als Sprache (Melismen, Pitch, Reverb), die Transkriptionsqualität
ist entsprechend nicht perfekt, aber durch die vorgelagerte Vocals-Isolation
deutlich besser als direkt auf dem Mix. Whisper liefert satz-/zeilenweise
Segmente mit Start-/End-Zeitstempeln — genau die Granularität, die ein
zeilensynchrones LRC-Format braucht, ohne dass ein zusätzliches
Forced-Alignment-Modell nötig wäre.

## Technischer Ansatz

**`openai-whisper` (torch-basiert)**, nicht `faster-whisper` (CTranslate2)
oder `WhisperX` (Forced Alignment via wav2vec2):

- `openai-whisper` baut auf **torch** auf, das im venv bereits installiert
  ist. Die vorhandene GPU-Logik in `src/main/env.ts` (CUDA für NVIDIA, ROCm
  für AMD auf Linux, MPS auf macOS) lässt sich 1:1 mitnutzen — der
  bestehende `gpuSplit`-Toggle wirkt automatisch auch auf die
  Lyrics-Extraktion, ohne eine zweite GPU-Engine-Verwaltung zu bauen.
- `faster-whisper` hätte eine von torch unabhängige, eigene
  CUDA/CTranslate2-Anbindung gebraucht — hätte den GPU-Toggle nicht
  wiederverwendbar gemacht.
- `WhisperX` liefert wortgenaue Zeitstempel, zieht aber zusätzliche
  Alignment-Modelle pro Sprache nach sich — mehr Komplexität für ein
  Feature (Wort-Highlighting), das nicht angefragt ist (YAGNI).

## Architektur

Leitprinzip: Der Code für dieses Feature soll so weit wie möglich in eigenen,
neuen Dateien liegen, damit sich die Änderung später sauber als eigenständiger
Beitrag gegen das Original-Repo (nicht im Besitz des Autors dieses Specs)
einreichen lässt. Bestehende Dateien bekommen nur kleine, additive,
mechanische Anfasser.

### Neue Dateien (enthalten die gesamte fachliche Logik)

- **`python/transcribe.py`**
  Nimmt `--input vocals.wav --out <dir> --model <small|medium|large-v3>
  --device <device>`. Lädt Whisper, transkribiert mit Sprach-Autodetect,
  filtert Halluzinations-Segmente (siehe Fehlerbehandlung), schreibt
  `lyrics.lrc`. Gibt dasselbe NDJSON-Progress-Protokoll aus wie
  `separate.py`/`roformer.py` (`{"type":"progress",...}`,
  `{"type":"done",...}`, `{"type":"error",...}`), damit
  `src/main/pipeline.ts` es mit der bestehenden `runProcess`/`onLine`-Logik
  genauso parsen kann wie die vorhandenen Skripte.

- **`src/main/lyrics.ts`**
  Das Herzstück auf der Main-Process-Seite:
  - `ensureLyricsEngine(model, onProgress)` — lädt das gewählte
    Whisper-Modell einmalig herunter (analog zu `ensureVocalsEngine`/
    `ensureFtWeights` in `env.ts`, via `whisper.load_model(name,
    download_root=modelsDir())` als Python-Einzeiler, nach dem Muster von
    `torchHubFetch`).
  - `lyricsPath(videoId)`, `lyricsModelMarkerPath(videoId)`,
    `lyricsPresent(videoId)` — Pfad-Helfer, analog zu `stemsDir`/
    `stemsPresent` in `library.ts`, aber hier lokal, um `library.ts`
    unangetastet zu lassen.
  - `maybeExtractLyrics(videoId, settings, deviceArg, onProgress)` —
    Caching-Entscheidung (siehe unten) + Aufruf von `transcribe.py`.
  - `lyricsEngineStatus()` — für die „lädt gerade"-Anzeige, analog zu
    `engineStatus()` in `env.ts`, aber lokal getrackt.
  - Importiert nur bereits exportierte Low-Level-Helfer aus `env.ts`
    (`venvPython()`, `modelsDir()`, `ensureGpuEngine()`) —
    **`env.ts` selbst bleibt unverändert.**

- **`src/renderer/src/components/Lyrics.tsx`**
  Eigenes Panel. Lädt die `.lrc`-Datei beim Öffnen eines Songs via
  `window.api.getLyrics(videoId)`, parst Zeilen+Timestamps, hebt die
  aktuelle Zeile anhand von `engine.expected()` hervor (derselbe
  Audio-Takt, der auch das YouTube-Video synchronisiert), Klick auf eine
  Zeile seekt wie die Waveform-Lanes.

### Bestehende Dateien — nur kleine, additive Anfasser

- **`src/main/pipeline.ts`**: ein `import { maybeExtractLyrics } from
  './lyrics'` + ein Aufruf direkt nach `runSeparation(...)` in `startJob`
  und `startLocalJob` (je eine Zeile). Fortschritt wird über einen an
  `maybeExtractLyrics` übergebenen Callback gemeldet
  (`(pct, msg) => progress(job, 'lyrics', pct, msg)`), damit `lyrics.ts`
  keine pipeline-internen Funktionen kennen muss. Kein Eingriff in
  `resolveEngine`/`runSeparation`.

- **`src/shared/types.ts`**: rein additive Felder:
  - `AppSettings.extractLyrics: boolean`
  - `AppSettings.lyricsModel: 'small' | 'medium' | 'large-v3'`
  - `Song.lyrics?: boolean`
  - `JobStage` erweitert um `'lyrics'`
  - `StemKitApi.getLyrics(videoId): Promise<string | null>`
  - `EngineStatus` erweitert um `lyricsDownloading`, `lyricsReady`

- **`src/main/settings.ts`**: zwei neue Zeilen in der bestehenden
  Sanitize-Kette (gleiches Muster wie die anderen Bool-/Enum-Felder).

- **`src/main/index.ts`**: ein neuer `ipcMain.handle('lyrics:get', ...)`
  -Block, delegiert komplett an `lyrics.ts`. `engines:status` mergt
  zusätzlich `lyricsEngineStatus()` ein. `engines:fetch` bekommt einen
  Case `'lyrics'`, der `ensureLyricsEngine()` aufruft.

- **`src/preload/index.ts`**: eine neue Zeile für `getLyrics`.

- **`src/renderer/src/components/Settings.tsx`**: ein neuer,
  abgeschlossener Block (Toggle + Modell-Dropdown), kein Umbau
  bestehender Blöcke.

- **`src/renderer/src/components/Player.tsx`** (o. ä.): eine neue,
  bedingte Einbindung von `<Lyrics />`, wenn `song.lyrics` true ist.

## Settings

- **Toggle:** „Songtext extrahieren" (`extractLyrics`, default `false`) —
  analog zu „Studio-Vocals"/„Fine-tuned Demucs".
- **Modell-Dropdown** (nur sichtbar/relevant wenn Toggle an):
  - `small` — ~500MB, schnell
  - `medium` — ~1.5GB, empfohlen (Default bei Aktivierung)
  - `large-v3` — ~3GB, beste Qualität
- Beides sind **globale** Einstellungen (wie `shifts`, `htdemucsFt`,
  `gpuSplit`), nicht pro Song wählbar.
- Nutzt den bestehenden `gpuSplit`-Toggle mit — keine eigene
  GPU-Einstellung für Lyrics.

## Datenfluss & Speicherformat

- **Format:** Standard-LRC, zeilenweise Zeitstempel:
  ```
  [00:12.34]Zeile eins des Songtexts
  [00:15.80]Zeile zwei
  ```
- **Ablage:** `songDir(videoId)/lyrics.lrc`, gleiche Ebene wie `mix.wav`
  und `stems/`.
- **Caching-Marker:** `songDir(videoId)/lyrics.model` (reiner Text, z. B.
  `medium`) — hält fest, mit welchem Whisper-Modell die vorhandene
  `lyrics.lrc` erzeugt wurde.
- **Caching-Entscheidung** (in `maybeExtractLyrics`, unabhängig von der
  Stem-Cache-Logik in `reuseOrPrepare`): läuft nur, wenn `extractLyrics`
  an ist, `vocals.wav` vorhanden ist, und (`lyrics.lrc` fehlt ODER
  `lyrics.model` ≠ aktuell eingestelltes Modell). Ein Wechsel der
  Modellgröße triggert damit **nur** eine Neu-Transkription, nicht den
  kompletten Stem-Split neu.
- **Bekannte Einschränkung (MVP):** Wird `extractLyrics` erst nachträglich
  für einen bereits fertig gesplitteten Song aktiviert, greift die
  Lyrics-Erzeugung erst beim nächsten Split-Lauf dieses Songs.

## Fehlerbehandlung

- **Halluzinationen bei Stille/Instrumental-Passagen** (bekanntes
  Whisper-Problem): `transcribe.py` filtert Segmente mit hoher
  `no_speech_prob` bzw. sehr niedrigem `avg_logprob` (beides von Whisper
  pro Segment geliefert) heraus, bevor die `.lrc` geschrieben wird.
  Bleiben keine Segmente übrig (z. B. reine Instrumentals), wird keine
  `lyrics.lrc` geschrieben, `Song.lyrics` bleibt `false` — kein Fehler.
- **Sprache:** Whisper läuft im Auto-Detect-Modus (kein festes
  Sprach-Flag) — passend dazu, dass Songs über die YouTube-Suche in
  beliebiger Sprache reinkommen.
- **Lyrics-Extraktion schlägt fehl** (Modell-Download bricht ab,
  Transkription crasht): **nicht fatal für den Job.** Anders als die
  Vocals-Engine (deren Ausfall den Job abbricht, weil der Stem fehlen
  würde) ist die Lyrics-Erzeugung rein additiv — `finalizeJob` läuft
  normal durch, nur ohne `lyrics.lrc`. Der Fehler wird als Info-Event
  gemeldet (gleiches Muster wie `sendEnvEvent` in `env.ts`), erscheint
  aber nicht als „Split fehlgeschlagen".

## Testing

Das Repo hat keinen Unit-Test-Runner, nur `npm run typecheck` und den
End-to-End-Smoke-Test `src/main/smoke.ts` (`STEMKIT_SMOKE=1`), der vor
jedem Release in CI läuft (`.github/workflows/*-smoke.yml`).

- **`smoke.ts` wird um einen dritten Durchlauf erweitert:**
  `transcribe.py` auf dem bereits vorhandenen generierten Sinuston-Mix
  laufen lassen. Da ein Sinuston keine Sprache enthält, prüft dieser
  Durchlauf genau den Halluzinations-Filter: das Skript muss exit 0
  liefern **und** darf keine (oder nur eine leere) `lyrics.lrc` erzeugen.
- **Manuelle Verifikation:** einen bekannten Song mit klaren Lyrics
  splitten, `.lrc`-Timing stichprobenartig gegen den tatsächlichen Song
  prüfen (keine automatisierte Ground-Truth-Prüfung, da keine
  Referenz-Lyrics im Repo vorliegen).

## Offene Punkte für den Implementierungsplan

- Genaue Whisper-Modellbezeichner/Download-URLs (Whisper lädt über die
  eigene Library von OpenAIs CDN — kein Bezahl-API-Call, nur ein
  Gewichte-Download wie bei den anderen optionalen Engines).
- Exakte UI-Platzierung des `<Lyrics />`-Panels in `Player.tsx` (neben
  den Stem-Fadern, wie in der Architektur-Diskussion festgelegt).
