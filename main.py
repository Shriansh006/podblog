
import os
import tempfile
from flask import Flask, request, render_template, send_file
import whisper
from transformers import pipeline
from pytube import YouTube
from pydub import AudioSegment


app = Flask(__name__)
model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        yt_link = request.form.get("yt_link")
        if not yt_link:
            return "No YouTube link provided.", 400

       
        try:
            yt = YouTube(yt_link)
            stream = yt.streams.filter(only_audio=True).first()
            out_file = stream.download(output_path=tempfile.gettempdir())
        except Exception as e:
            return f"Failed to download audio: {str(e)}", 500

        
        wav_path = out_file.replace(".mp4", ".wav").replace(".webm", ".wav")
        sound = AudioSegment.from_file(out_file)
        sound = sound.set_channels(1).set_frame_rate(16000)
        sound.export(wav_path, format="wav")

       
        result = model.transcribe(wav_path)
        transcript = result["text"]

        
        chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
        blog_parts = [summarizer(chunk)[0]['summary_text'] for chunk in chunks]
        full_blog = "\n\n".join(blog_parts)

       
        blog_path = os.path.join(tempfile.gettempdir(), "blog.txt")
        with open(blog_path, "w", encoding="utf-8") as f:
            f.write(full_blog)

        return send_file(blog_path, as_attachment=True, download_name="blog.txt")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
