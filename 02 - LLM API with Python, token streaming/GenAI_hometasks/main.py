
import openai
from flask import Flask, render_template_string

# Flask application
app = Flask(__name__)

# Set your API key
openai.api_key = "***"
# Prepare the file for upload
file_path = "lesson-1-transcript.txt"

# Read file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        transcript_content = file.read()
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    exit()

prompt = (
    "I am taking a Generative AI course in my Master's program. "
    "The tutor requested a blog post as a home task based on the lecture 1 transcripts. "
    "Here is the transcript content:\n\n"
    f"{transcript_content}\n\n"
    "Please create a nice and informative blog post focusing on the main points from the transcript. "
    "Take into consideration the following:\n"
    "1. The transcript content is in a conversational style where the tutor is talking with students.\n"
    "2. Speaker 8 is the tutor.\n"
    "The response should use html for formatting."
)

def chat_with_gpt(model="gpt-4o", max_tokens=2000, temperature=0.7):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        return f"An error occurred: {e}"

# Flask route to display the blog post
@app.route("/")
def home():
    # Get the GPT response
    blog_post_html = chat_with_gpt()
    print(blog_post_html)
    return render_template_string(blog_post_html)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)