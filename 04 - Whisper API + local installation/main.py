from pydub import AudioSegment
import whisper

# Load the audio file
#audio = AudioSegment.from_file("ITPU_MS_Degree_Session_5_-_Generative_AI-20241213_153714-Meeting_Recording.mp3")

# Extract the desired segment (e.g., 10 minutes to 15 minutes)
#start_time = 10 * 60 * 1000  # Convert minutes to milliseconds
#end_time = 15 * 60 * 1000
#segment = audio[start_time:end_time]

# Save the extracted segment
#segment.export("output_segment.mp3", format="mp3")

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the extracted segment
result = model.transcribe("output_segment.mp3")

# Print or save the transcription
print(result["text"])
with open("transcription.txt", "w") as f:
    f.write(result["text"])