import os
import assemblyai as aai

key = os.getenv('ASSEMBLYAI_API_KEY')
if not key:
    # try to read from .env in project root
    try:
        with open(os.path.join(os.path.dirname(__file__), '..', '.env'), 'r') as f:
            for l in f:
                if l.strip().startswith('ASSEMBLYAI_API_KEY'):
                    key = l.strip().split('=',1)[1].strip()
                    break
    except Exception:
        pass
print('KEY present:', bool(key))
aai.settings.api_key = key
transcriber = aai.Transcriber()
transcript = transcriber.transcribe(r'D:\VSCode Projects\Payload\test.wav')
print('Transcript type:', type(transcript))
attrs = [x for x in dir(transcript) if not x.startswith('_')]
print('attrs count:', len(attrs))
print('\nSome attrs:\n', attrs[:60])
print('\nHas words:', hasattr(transcript, 'words'))
if hasattr(transcript, 'words'):
    print('words sample count:', len(transcript.words))
    if transcript.words:
        w = transcript.words[0]
        print('sample word keys:', [k for k in dir(w) if not k.startswith('_')])
        try:
            print('sample word as dict via __dict__:', w.__dict__)
        except Exception as e:
            print('word __dict__ error', e)
print('status:', transcript.status)
print('text length:', len(getattr(transcript,'text','')))
