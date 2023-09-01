import whisper

model = whisper.load_model("medium")
result = model.transcribe('/home/koros/PycharmProjects/whisper_lecture/My recording 2.m4a',
                          verbose=True,
                          initial_prompt="This is a lecture at a university Biology Department."
                          )

print(result["text"])