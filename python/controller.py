import json
import os
import random
import re
import sys
import time
import hashlib
import urllib.request
import urllib.parse
import json as _json
import requests

MAX_IMAGE_DURATION = 3.0
MIN_IMAGE_DURATION = 3.0


def load_assets(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fake_transcription(script_path):
    """Simple deterministic fallback transcription.
    Splits the provided script text into words and assigns fixed-length
    word segments (0.5s each). Returns a list of {start,end,text} dicts.
    """
    if not script_path or not os.path.exists(script_path):
        return []

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            text = f.read() or ""
    except Exception:
        text = ""

    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return []

    seg_len = 0.5
    segments = []
    t = 0.0
    for w in words:
        start = t
        end = round(t + seg_len, 3)
        segments.append({"start": round(start, 3), "end": end, "text": w})
        t += seg_len

    return segments


def transcribe_with_assemblyai(audio_path, api_key=None, timeout=600):
    """Upload audio to AssemblyAI, request transcription with word timestamps,
    poll until complete, and return a list of word-level segments.
    Returns None on failure.
    """
    # Prefer the official AssemblyAI Python SDK for robustness.
    key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
    if not key:
        return None

    try:
        import assemblyai as aai
    except Exception:
        sys.stderr.write("AssemblyAI SDK not installed. Install with: pip install assemblyai\n")
        return None

    try:
        # configure SDK
        try:
            aai.settings.api_key = key
        except Exception:
            try:
                aai.api_key = key
            except Exception:
                pass

        transcriber = aai.Transcriber()
        # build SDK config using provided types
        try:
            cfg = aai.types.TranscriptionConfig(punctuate=True, format_text=True, word_timestamps=True)
        except Exception:
            cfg = None

        if cfg is not None:
            transcript = transcriber.transcribe(audio_path, config=cfg)
        else:
            transcript = transcriber.transcribe(audio_path)

        try:
            status = transcript.status
        except Exception:
            status = None

        if status and str(status).lower() == "error":
            try:
                err = getattr(transcript, 'error', None)
                sys.stderr.write(f"AssemblyAI SDK transcription error: {err}\n")
            except Exception:
                pass
            return None

        words = []
        # SDK returns a Transcript object with a `words` attribute
        if hasattr(transcript, 'words') and transcript.words:
            words = transcript.words
        else:
            # try utterances
            if hasattr(transcript, 'utterances') and transcript.utterances:
                for u in transcript.utterances:
                    for w in getattr(u, 'words', []):
                        words.append(w)

        segments = []
        for w in words:
            try:
                s_ms = getattr(w, 'start', None) or (w.get('start') if isinstance(w, dict) else 0)
            except Exception:
                s_ms = 0
            try:
                e_ms = getattr(w, 'end', None) if not isinstance(w, dict) else w.get('end')
            except Exception:
                e_ms = None

            s = float(s_ms) / 1000.0 if s_ms is not None else 0.0
            if e_ms is None:
                e = s + 0.5
            else:
                e = float(e_ms) / 1000.0

            try:
                text = getattr(w, 'text', None) or (w.get('text') if isinstance(w, dict) else '')
            except Exception:
                text = ''

            segments.append({"start": s, "end": e, "text": (text or '').strip()})

        return segments
    except Exception as e:
        try:
            import traceback
            sys.stderr.write("AssemblyAI SDK error: " + str(e) + "\n")
            traceback.print_exc(file=sys.stderr)
        except Exception:
            pass
        return None


def call_openai_editor(transcript_text, api_key=None, model="gpt-3.5-turbo"):
    """Send the transcription to OpenAI with the editor prompt and return parsed JSON mapping.
    Returns list of dicts: [{"start":..,"end":..,"image":..,"split":bool}, ...] or None on error.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        sys.stderr.write("OpenAI API key not set (OPENAI_API_KEY).\n")
        return None

    system_prompt = (
        "You are a video editor. I will give you a transcription with timestamps.\n"
        "Your job is to decide which image appears at which time.\n\n"
        "Rules you must follow:\n"
        "Output only valid JSON. Nothing else.\n"
        "The video must always start at 00:00.\n"
        "The hook must always be split screen, starting at 00:00.\n"
        "Each image must stay on screen for maximum 3–3.5 seconds.\n"
        "Do not describe images in detail.\n"
        "Use only short entity names (e.g., \"andy reid\").\n"
        "If a single line talks about two different subjects, use split screen.\n"
        "Match visuals tightly with what’s being spoken.\n"
        "No filler, no creativity outside the context.\n"
        "If anything is out of context, use \"misc\" images.\n"
        "Keep transitions clean and logical.\n"
        "You will receive timestamps and dialogue.\n"
        "Return structured JSON showing: start time, end time, image subject(s), split screen (true/false).\n"
    )

    user_msg = "TRANSCRIPTION:\n" + transcript_text + "\n\nRespond with a JSON array like: [{\"start\":0.0,\"end\":2.5,\"image\":\"andy reid\",\"split\":false}, ...]"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 1500
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            },
            data=_json.dumps(payload),
            timeout=60
        )
    except Exception as e:
        sys.stderr.write("OpenAI request failed: " + str(e) + "\n")
        return None

    if resp.status_code != 200:
        sys.stderr.write(f"OpenAI API error {resp.status_code}: {resp.text}\n")
        return None

    try:
        j = resp.json()
        content = j["choices"][0]["message"]["content"]
    except Exception:
        sys.stderr.write("Failed to parse OpenAI response JSON.\n")
        return None

    # Try to extract JSON array from content
    try:
        # find first [ and last ]
        s = content
        si = s.find('[')
        ei = s.rfind(']')
        if si == -1 or ei == -1:
            # maybe the model returned raw JSON object
            parsed = _json.loads(s)
        else:
            parsed = _json.loads(s[si:ei+1])
        # normalize entries
        out = []
        for it in parsed:
            rec = {}
            rec['start'] = float(it.get('start'))
            rec['end'] = float(it.get('end'))
            rec['image'] = it.get('image')
            rec['split'] = bool(it.get('split'))
            out.append(rec)
        # return both parsed mapping and raw content
        return out, content
    except Exception as e:
        sys.stderr.write("Failed to parse model output as JSON: " + str(e) + "\n")
        return None


def detect_entity(text, assets, last_entity):
    """Detect which entity (key in assets) is referenced by `text`.
    Checks the entity name and any provided `aliases` in the assets entry.
    Returns the matched key, or `last_entity` or 'crowd' as fallback.
    """
    # Legacy function kept for compatibility: delegate to word-level matcher
    return detect_entity_for_word(text, assets) or last_entity or "crowd"


def normalize(s):
    if not s:
        return ""
    s = s.lower()
    # remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def detect_entity_for_word(text, assets):
    """Determine the best-matching entity for a single word/text.
    Priority:
      1. exact match to entity name or alias
      2. alias/name substring match (longest match wins)
      3. None if no match
    """
    needle = normalize(text)
    if not needle:
        return None

    # exact name or alias match
    for name, info in assets.items():
        try:
            if normalize(name) == needle:
                return name
            aliases = info.get("aliases", []) if isinstance(info, dict) else []
            for a in aliases:
                if a and normalize(a) == needle:
                    return name
        except Exception:
            continue

    # substring match: prefer longest alias/name match
    best = (None, 0)
    for name, info in assets.items():
        try:
            candidates = [name]
            if isinstance(info, dict):
                candidates += info.get("aliases", []) or []
            for c in candidates:
                if not c:
                    continue
                cn = normalize(c)
                if cn and cn in needle:
                    if len(cn) > best[1]:
                        best = (name, len(cn))
        except Exception:
            continue

    if best[0]:
        return best[0]

    return None


def download_asset(url, cache_dir):
    try:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or "asset"
        safe = hashlib.md5(url.encode("utf-8")).hexdigest() + "_" + name
        local_path = os.path.join(cache_dir, safe)

        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        # download
        urllib.request.urlretrieve(url, local_path)
        if os.path.exists(local_path):
            return local_path
    except Exception:
        return None
    return None


def build_timeline(transcript, assets, cache_dir, max_image_duration=3.5):
    """Build timeline by grouping consecutive words by detected entity/topic.
    Each group is split into chunks no longer than `max_image_duration` seconds.
    Images are downloaded into `cache_dir` and timeline `src` points to local files
    when possible so After Effects can import them reliably.
    """
    timeline = []
    if not transcript:
        return timeline


    spans = []

    # build entity spans from word-level transcript; keep word dicts
    current = None
    for w in transcript:
        text = (w.get("text") or "").strip()
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start + 0.0))
        entity = detect_entity_for_word(text, assets) or "crowd"

        if current is None:
            current = {"entity": entity, "start": start, "end": end, "words": [ {"text": text, "start": start, "end": end} ]}
            continue

        if entity == current["entity"]:
            current["end"] = end
            current["words"].append({"text": text, "start": start, "end": end})
        else:
            spans.append(current)
            current = {"entity": entity, "start": start, "end": end, "words": [{"text": text, "start": start, "end": end}]}

    if current is not None:
        spans.append(current)

    # deterministic asset selection per entity (cycle), no random picks
    timeline = []
    used_per_entity = {}
    animations = ["zoom_in", "zoom_out", "slide_left", "slide_right"]

    for si, span in enumerate(spans):
        ent = span.get("entity") or "crowd"
        pool = []
        if isinstance(assets.get(ent), dict):
            pool = assets.get(ent, {}).get("assets", []) or []
        if not pool:
            pool = assets.get("crowd", {}).get("assets", []) or []
        if not pool:
            # no assets at all; skip
            continue

        # allocate sequential chunks for this span
        span_start = float(span.get("start", 0.0))
        span_end = float(span.get("end", span_start + max_image_duration))
        t = span_start
        chunk_index = 0
        while t < span_end:
            chunk_end = min(span_end, t + max_image_duration)

            # deterministic round-robin per-entity
            idx = used_per_entity.get(ent, 0) % len(pool)
            candidate = pool[idx]
            used_per_entity[ent] = used_per_entity.get(ent, 0) + 1

            src = candidate
            if isinstance(candidate, str) and candidate.lower().startswith('http'):
                local = download_asset(candidate, cache_dir)
                if local:
                    src = local

            # collect words overlapping this chunk
            chunk_words = []
            for w in span.get("words", []):
                ws = float(w.get("start", 0.0))
                we = float(w.get("end", ws + 0.0))
                # include if any overlap
                if not (we <= t or ws >= chunk_end):
                    chunk_words.append({"text": w.get("text"), "start": round(ws, 3), "end": round(we, 3)})

            # deterministic animation choice
            anim = animations[(si + chunk_index) % len(animations)]

            timeline.append({
                "type": "image",
                "entity": ent,
                "src": src,
                "start": round(t, 3),
                "end": round(chunk_end, 3),
                "animation": anim,
                "words": chunk_words
            })

            t = chunk_end
            chunk_index += 1

    return timeline


def coalesce_segments(segments, min_duration=MIN_IMAGE_DURATION):
    """Merge consecutive short segments so each output segment lasts at least min_duration.
    Chooses the most frequent `src` in the merged group as the representative image.
    """
    if not segments:
        return []

    out = []
    i = 0
    n = len(segments)
    while i < n:
        start = segments[i]["start"]
        end = segments[i]["end"]
        srcs = [segments[i]["src"]]
        animations = [segments[i].get("animation")]
        j = i + 1
        # accumulate until duration >= min_duration or we run out
        while (end - start) < min_duration and j < n:
            end = segments[j]["end"]
            srcs.append(segments[j]["src"])
            animations.append(segments[j].get("animation"))
            j += 1

        # if still shorter than min_duration, extend end to start+min_duration (but don't exceed last segment end)
        if (end - start) < min_duration:
            end = start + min_duration

        # pick most common src
        rep_src = max(set(srcs), key=srcs.count)
        rep_animation = animations[0] if animations else "zoom_in"

        out.append({
            "type": "image",
            "src": rep_src,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "animation": rep_animation
        })

        i = j

    return out


def main():
    # determine project root (parent of this file's folder)
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, ".."))

    # If the AssemblyAI SDK isn't importable in this Python, try re-execing
    # the script using the project's virtualenv python so AE (which may use
    # a system Python without SDK) still works. Prevent infinite re-exec loops
    # by using an environment marker.
    if not os.getenv("PAYLOAD_REEXECED"):        
            venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
            if os.path.exists(venv_python) and os.path.abspath(venv_python) != os.path.abspath(sys.executable):
                # load common keys from .env if present so child inherits
                env_file = os.path.join(project_root, ".env")
                if os.path.exists(env_file):
                    try:
                        with open(env_file, "r", encoding="utf-8") as ef:
                            for ln in ef:
                                if not ln or '=' not in ln:
                                    continue
                                k, v = ln.split('=', 1)
                                k = k.strip()
                                v = v.strip()
                                if not k:
                                    continue
                                # only set keys that are useful here
                                if k in ("ASSEMBLYAI_API_KEY", "OPENAI_API_KEY") and not os.getenv(k):
                                    os.environ[k] = v
                    except Exception:
                        pass

                # set marker and re-exec
                os.environ["PAYLOAD_REEXECED"] = "1"
                try:
                    os.execv(venv_python, [venv_python] + sys.argv)
                except Exception:
                    # fallback: continue in current interpreter and let code handle missing SDK
                    pass

    # default config path
    default_cfg = os.path.join(project_root, "data", "job_config.json")

    cfg_path = default_cfg
    audio_arg = None
    if len(sys.argv) >= 2:
        audio_arg = sys.argv[1]
    if len(sys.argv) >= 3:
        cfg_path = sys.argv[2]

    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            try:
                cfg = json.load(f)
            except Exception:
                cfg = {}

    # resolve common paths (make relative cfg entries relative to project root)
    assets_path = cfg.get("asset_json", os.path.join(project_root, "data", "assets.json"))
    script_file = cfg.get("script_file", os.path.join(project_root, "data", "script.txt"))
    output_dir = cfg.get("output_dir", os.path.join(project_root, "jobs"))

    # if cfg provides relative paths (e.g., "data/assets.json"), make them absolute
    if not os.path.isabs(assets_path):
        assets_path = os.path.abspath(os.path.join(project_root, assets_path))
    else:
        assets_path = os.path.abspath(assets_path)

    if not os.path.isabs(script_file):
        script_file = os.path.abspath(os.path.join(project_root, script_file))
    else:
        script_file = os.path.abspath(script_file)

    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(os.path.join(project_root, output_dir))
    else:
        output_dir = os.path.abspath(output_dir)

    voiceover = audio_arg or cfg.get("voiceover")
    if voiceover and not os.path.isabs(voiceover):
        voiceover = os.path.abspath(os.path.join(project_root, voiceover))

    assets = load_assets(assets_path)

    # Ensure common API keys are available when After Effects calls this script
    # AE may not inherit shell environment variables, so allow config or .env fallback.
    for env_key, cfg_key in (("ASSEMBLYAI_API_KEY", "assemblyai_api_key"), ("OPENAI_API_KEY", "openai_api_key")):
        if not os.getenv(env_key):
            key = cfg.get(cfg_key) if isinstance(cfg, dict) else None
            if not key:
                env_file = os.path.join(project_root, ".env")
                try:
                    if os.path.exists(env_file):
                        with open(env_file, "r", encoding="utf-8") as ef:
                            for ln in ef:
                                if not ln or '=' not in ln:
                                    continue
                                k, v = ln.split('=', 1)
                                if k.strip() == env_key:
                                    key = v.strip()
                                    break
                except Exception:
                    key = None

            if key:
                os.environ[env_key] = key
                sys.stderr.write(f"Loaded {env_key} from config/.env for AE invocation.\n")

    # Transcribe: prefer AssemblyAI for word-level timestamps; otherwise fake
    transcript = None
    warnings = []
    if voiceover and os.path.exists(voiceover):
        transcript = transcribe_with_assemblyai(voiceover, api_key=os.getenv("ASSEMBLYAI_API_KEY"))
        if transcript is None:
            sys.stderr.write("ERROR: AssemblyAI transcription failed or ASSEMBLYAI_API_KEY not set.\n")
            sys.stderr.write("Set ASSEMBLYAI_API_KEY environment variable and retry.\n")
            sys.exit(2)
    else:
        transcript = fake_transcription(script_file)

    cache_dir = os.path.join(output_dir, "cache")
    max_img_dur = float(cfg.get("max_image_duration", 3.5))
    segments = build_timeline(transcript, assets, cache_dir, max_image_duration=max_img_dur)

    # Build a top-level words list
    words_list = []
    for w in transcript:
        words_list.append({
            "text": w.get("text", ""),
            "start": round(float(w.get("start", 0.0)), 3),
            "end": round(float(w.get("end", 0.0)), 3)
        })
    # Save the AssemblyAI transcript to jobs/transcript.json so OpenAI can read it
    transcript_path = os.path.join(output_dir, "transcript.json")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(transcript_path, "w", encoding="utf-8") as tf:
            json.dump(transcript, tf, indent=2)
    except Exception:
        pass

    # Prepare transcript text for the model: use JSON representation of the transcript file
    try:
        transcript_text = _json.dumps(transcript)
    except Exception:
        transcript_lines = []
        for w in words_list:
            transcript_lines.append(f"{w['start']:.3f}-{w['end']:.3f}: {w['text']}")
        transcript_text = "\n".join(transcript_lines)

    # Ensure OPENAI_API_KEY is loaded from .env as a last resort
    if not os.getenv("OPENAI_API_KEY"):
        env_file = os.path.join(project_root, ".env")
        try:
            if os.path.exists(env_file):
                with open(env_file, "r", encoding="utf-8") as ef:
                    for ln in ef:
                        if not ln or '=' not in ln:
                            continue
                        k, v = ln.split('=', 1)
                        if k.strip() == 'OPENAI_API_KEY':
                            os.environ['OPENAI_API_KEY'] = v.strip()
                            sys.stderr.write('Loaded OPENAI_API_KEY from .env for this invocation.\n')
                            break
        except Exception:
            pass

    mapping_result = call_openai_editor(transcript_text, api_key=os.getenv("OPENAI_API_KEY"))
    mapping = None
    mapping_raw = None
    if mapping_result:
        try:
            mapping, mapping_raw = mapping_result
        except Exception:
            mapping = mapping_result


    final_segments = []
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    def resolve_subject_to_src(subject):
        # find matching asset pool by name or alias (case-insensitive)
        if not subject:
            return None
        s = normalize(subject)
        for name, info in assets.items():
            if normalize(name) == s:
                pool = info.get('assets') if isinstance(info, dict) else None
                if pool:
                    candidate = pool[0]
                    if candidate.lower().startswith('http'):
                        local = download_asset(candidate, cache_dir)
                        return local or candidate
                    return candidate
            if isinstance(info, dict):
                for a in info.get('aliases', []) or []:
                    if normalize(a) == s:
                        pool = info.get('assets')
                        if pool:
                            candidate = pool[0]
                            if candidate.lower().startswith('http'):
                                local = download_asset(candidate, cache_dir)
                                return local or candidate
                            return candidate
        # fallback to crowd or misc
        pool = assets.get('misc', {}).get('assets') or assets.get('crowd', {}).get('assets') or []
        if pool:
            candidate = pool[0]
            if candidate.lower().startswith('http'):
                local = download_asset(candidate, cache_dir)
                return local or candidate
            return candidate
        return None

    # rotation state for deterministic selection
    used_per_entity = {}

    def select_asset_for_subject(subject, avoid=None):
        """Round-robin select an asset for the given subject (name or alias).
        Tries to avoid returning the `avoid` value when possible.
        Returns local path (downloaded) or URL/string, or None.
        """
        if not subject:
            return None
        s = normalize(subject)
        # find pool
        pool = None
        pool_name = None
        for name, info in assets.items():
            if normalize(name) == s:
                pool = info.get('assets') if isinstance(info, dict) else None
                pool_name = name
                break
            if isinstance(info, dict):
                for a in info.get('aliases', []) or []:
                    if normalize(a) == s:
                        pool = info.get('assets')
                        pool_name = name
                        break
                if pool:
                    break

        if not pool:
            pool = assets.get('misc', {}).get('assets') or assets.get('crowd', {}).get('assets') or []
            pool_name = 'misc' if pool else None

        if not pool:
            return None

        # try up to len(pool) entries to find one that isn't equal to avoid
        start_idx = used_per_entity.get(pool_name, 0) % len(pool)
        for attempt in range(len(pool)):
            idx = (start_idx + attempt) % len(pool)
            candidate = pool[idx]
            candidate_val = None
            if isinstance(candidate, str) and candidate.lower().startswith('http'):
                candidate_val = download_asset(candidate, cache_dir) or candidate
            else:
                candidate_val = candidate

            if avoid is None or not same_src(candidate_val, avoid):
                # commit rotation index
                used_per_entity[pool_name] = used_per_entity.get(pool_name, 0) + (attempt + 1)
                return candidate_val

        # fallback: return the start_idx candidate
        candidate = pool[start_idx]
        if isinstance(candidate, str) and candidate.lower().startswith('http'):
            return download_asset(candidate, cache_dir) or candidate
        return candidate

    def same_src(a, b):
        if a is None or b is None:
            return False
        if isinstance(a, list) and isinstance(b, list):
            return a == b
        return str(a) == str(b)

    # save mapping raw text and parsed mapping for inspection
    mapping_json_path = os.path.join(output_dir, "mapping.json")
    mapping_raw_path = os.path.join(output_dir, "mapping_raw.txt")
    if mapping_raw:
        try:
            with open(mapping_raw_path, "w", encoding="utf-8") as mf:
                mf.write(mapping_raw)
        except Exception:
            pass
    if mapping:
        try:
            with open(mapping_json_path, "w", encoding="utf-8") as mj:
                json.dump(mapping, mj, indent=2)
        except Exception:
            pass

    if mapping:
        for item in mapping:
            start = float(item.get('start', 0))
            end = float(item.get('end', start + 0.0))
            image_field = item.get('image')
            split = bool(item.get('split', False))

            # handle multi-subjects when split requested
            sources = []
            if split:
                # try to split by comma or & or ' and '
                parts = []
                if isinstance(image_field, list):
                    parts = image_field
                elif isinstance(image_field, str):
                    if ',' in image_field:
                        parts = [p.strip() for p in image_field.split(',') if p.strip()]
                    elif ' & ' in image_field:
                        parts = [p.strip() for p in image_field.split(' & ') if p.strip()]
                    elif ' and ' in image_field:
                        parts = [p.strip() for p in image_field.split(' and ') if p.strip()]
                    else:
                        parts = [image_field.strip()]
                for p in parts[:2]:
                    src = select_asset_for_subject(p)
                    if src:
                        sources.append(src)
                # ensure two slots
                while len(sources) < 2:
                    sources.append(select_asset_for_subject('misc'))
            else:
                if isinstance(image_field, list):
                    src = select_asset_for_subject(image_field[0])
                else:
                    src = select_asset_for_subject(image_field)
                sources = [src]

            # enforce max duration (split into consecutive chunks if needed)
            # strict max duration 3.0s (never exceed)
            max_cfg = float(cfg.get('max_image_duration', 3.0))
            max_d = min(3.0, max_cfg)
            t0 = start
            while t0 < end:
                t1 = min(end, t0 + max_d)
                seg = {
                    'type': 'image',
                    'entity': image_field if not split else ', '.join(parts[:2]),
                    'src': sources if split else (sources[0] if sources else None),
                    'start': round(t0, 3),
                    'end': round(t1, 3),
                    'animation': 'zoom_in',
                    'words': []
                }
                # avoid consecutive duplicates: if last has same src, pick next asset
                if final_segments:
                    last_src = final_segments[-1].get('src')
                    if not split and same_src(last_src, seg['src']):
                        # pick next asset for this subject
                        alt = select_asset_for_subject(image_field)
                        if alt and not same_src(alt, last_src):
                            seg['src'] = alt
                    if split and isinstance(seg['src'], list) and isinstance(last_src, list) and last_src == seg['src']:
                        # try rotate each side
                        segA = select_asset_for_subject(parts[0])
                        segB = select_asset_for_subject(parts[1] if len(parts) > 1 else 'misc')
                        seg['src'] = [segA or seg['src'][0], segB or (seg['src'][1] if len(seg['src'])>1 else seg['src'][0])]

                final_segments.append(seg)
                t0 = t1
    else:
        # fallback to original entity-based segments
        final_segments = segments

    # Enforce no gaps: sort and fill any gaps with misc images, and ensure continuity
    if final_segments:
        final_segments = sorted(final_segments, key=lambda s: float(s.get('start', 0)))
        filled = []
        prev_end = 0.0
        for seg in final_segments:
            s = float(seg.get('start', 0))
            e = float(seg.get('end', s))
            if s > prev_end + 1e-6:
                # fill gap [prev_end, s)
                t0 = prev_end
                while t0 < s - 1e-6:
                    t1 = min(s, t0 + max_d)
                    # avoid consecutive randoms and split randoms
                    avoid_src = filled[-1].get('src') if filled else None
                    filler_src = select_asset_for_subject('misc', avoid=avoid_src)
                    # if previous was also misc/crowd, try to alternate
                    if filled and filled[-1].get('entity') in ('misc','crowd'):
                        alt_src = select_asset_for_subject('crowd', avoid=filler_src)
                        if alt_src and not same_src(alt_src, filler_src):
                            filler_src = alt_src
                    filled.append({
                        'type': 'image', 'entity': 'misc', 'src': filler_src,
                        'start': round(t0,3), 'end': round(t1,3), 'animation':'zoom_in', 'words':[]
                    })
                    t0 = t1
            # adjust segment start to prev_end if overlapping
            if s < prev_end:
                seg['start'] = round(prev_end,3)
            filled.append(seg)
            prev_end = float(seg.get('end', seg.get('start')))

        # if timeline doesn't start at 0, ensure starts at 0 by prepending misc
        if filled and filled[0].get('start',0) > 0:
            # prepend filler from 0 to first.start
            first_start = filled[0]['start']
            t0 = 0.0
            new_pre = []
            while t0 < first_start:
                t1 = min(first_start, t0 + max_d)
                new_pre.append({'type':'image','entity':'misc','src':select_asset_for_subject('misc'),'start':round(t0,3),'end':round(t1,3),'animation':'zoom_in','words':[]})
                t0 = t1
            filled = new_pre + filled

        final_segments = filled

    # add word_indices to each segment referencing the global words_list
    for seg in final_segments:
        seg_indices = []
        for sw in seg.get('words', []):
            found = -1
            for idx, gw in enumerate(words_list):
                if abs(gw.get('start', 0) - float(sw.get('start', 0))) < 0.001 and abs(gw.get('end', 0) - float(sw.get('end', 0))) < 0.001 and (gw.get('text', '') == sw.get('text', '')):
                    found = idx
                    break
            if found == -1:
                # fallback nearest by start
                best = None
                for idx, gw in enumerate(words_list):
                    if abs(gw.get('start', 0) - float(sw.get('start', 0))) < 0.05:
                        best = idx
                        break
                if best is None:
                    continue
                found = best
            seg_indices.append(found)
        seg['word_indices'] = seg_indices

    output = {
        'meta': {'fps': 30, 'aspect': '9:16', 'warnings': warnings},
        'audio': {'src': voiceover or ''},
        'cache_dir': os.path.join(output_dir, 'cache') + os.sep,
        'words': words_list,
        'segments': final_segments,
        'transcript_json': transcript_path,
        'openai_mapping': mapping,
        'openai_mapping_raw': mapping_raw_path if mapping_raw else None
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "timeline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Print only the timeline path marker for AE to parse
    sys.stdout.write("TIMELINE_PATH=" + out_path + "\n")


if __name__ == "__main__":
    main()
