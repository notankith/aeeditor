import assemblyai as aai
print('module attrs:')
print([x for x in dir(aai) if not x.startswith('_')])
print('\nTranscriber attr:')
print(getattr(aai, 'Transcriber', None))
# try common config class names
for name in ('TranscriptionConfig','TranscriptConfig','TranscriptionConfigModel','Transcript','TranscriberConfig'):
    print(name, getattr(aai, name, None))
