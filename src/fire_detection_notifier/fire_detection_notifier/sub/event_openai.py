from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "warning.wav"
response = client.audio.speech.create(
  model="tts-1",
  voice="onyx",
  input="화재가 감지되었습니다. 몸을 낮추고 신속하게 대피하십시오."
)


response.stream_to_file(speech_file_path)