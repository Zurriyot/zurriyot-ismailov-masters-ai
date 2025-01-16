
import openai
from flask import Flask, render_template_string
import os
from bs4 import BeautifulSoup

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

#Prompt for generating blogpost
generate_summary_prompt = (
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
#Generate blogpost
def generate_blogpost(model="gpt-4o", max_tokens=2000, temperature=0.7):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": generate_summary_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except openai.APIError as e:
        return f"An error occurred: {e}"

def read_html(file_path):
    """Reads an HTML file and returns its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def generate_prompts(blog_content, model="gpt-4o", max_tokens=2000, temperature=0.7):
    """Uses ChatGPT to generate 9 prompts based on the blog content."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates descriptive prompts for DALL-E based on blog content."},
            {"role": "user", "content": f"Generate 9 detailed and creative DALL-E prompts based on this blog content (each prompt should be one line):\n{blog_content}"}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    prompts = response.choices[0].message.content.strip().split('\n')
    return [prompt for prompt in prompts if prompt.strip()]  # Clean and return prompts

def generate_images(prompts):
    """Uses DALL-E to generate images based on the given prompts."""
    images = []
    for prompt in prompts:
        response = openai.images.generate(
            model = "dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        images.append((prompt, image_url))
        print(prompt)
        print(image_url)
    return images

def embed_images_in_html(blog_html, images):
    """Embeds images into the blog HTML at different parts."""
    soup = BeautifulSoup(blog_html, 'html.parser')

    # Find all paragraphs and evenly distribute images among them
    paragraphs = soup.find_all('p')
    num_paragraphs = len(paragraphs)
    step = max(1, num_paragraphs // len(images))

    for i, (prompt, image_url) in enumerate(images):
        if i * step < num_paragraphs:
            img_tag = soup.new_tag('img', src=image_url, alt=prompt, style="display:block; margin:20px auto;")
            paragraphs[i * step].insert_after(img_tag)

    return str(soup)

def save_html(file_path, content):
    """Saves modified HTML content to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# Flask route to display the blog post
@app.route("/")
def home():
    # Step 1: Read the blogpost HTML content
    blog_post_html = generate_blogpost()

    # Step 2: Generate prompts based on the blog content
    prompts = generate_prompts(blog_post_html)

    # Step 3: Generate images using DALL-E
    images = generate_images(prompts)

    # Step 4: Embed images into the HTML content
    updated_html = embed_images_in_html(blog_post_html, images)

    return render_template_string(updated_html)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)