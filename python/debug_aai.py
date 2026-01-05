import os
import assemblyai as aai
key = os.getenv('ASSEMBLYAI_API_KEY') or None
print('KEY present:', bool(key))
aai.settings.api_key = key
print('settings.api_key set')
transcriber = aai.Transcriber()
print('Transcriber created')
try:
    cfg = aai.types.TranscriptionConfig(punctuate=True, format_text=True, word_timestamps=True)
    print('Config created')
except Exception as e:
    print('Config creation failed:', e)
    cfg = None

path = r'D:\VSCode Projects\Payload\test.wav'
print('Submitting:', path)
try:
    if cfg:
        transcript = transcriber.transcribe(path, config=cfg)
    else:
        transcript = transcriber.transcribe(path)
    print('Transcript object:', type(transcript))
    try:
        print('status:', transcript.status)
        print('error:', getattr(transcript,'error', None))
    except Exception as e:
        print('attr access error:', e)
    try:
        d = transcript.to_dict()
        print('to_dict keys:', list(d.keys()))
    except Exception as e:
        print('to_dict failed:', e)
except Exception as e:
    import traceback
    print('Transcribe call failed:', e)
    traceback.print_exc()
