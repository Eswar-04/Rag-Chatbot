import fitz  # PyMuPDF for handling PDF files
import os
from PIL import Image  # For image processing
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from app.qa import clean_text  # Custom function to clean text
from app.supabase_client import store_qa  # Function to store QA results in Supabase

# Load BLIP model (Large) for generating image captions
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load FLAN-T5 model for text-based question answering
t5 = pipeline("text2text-generation", model="google/flan-t5-base")

# Function to extract all images from a PDF
def extract_all_images(pdf_path, output_dir="images"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the PDF file
        doc = fitz.open(pdf_path)
        image_paths = []

        # Loop through each page in the PDF
        for p in range(len(doc)):
            page = doc[p]
            imgs = page.get_images(full=True)
            
            # Extract each image found on the page
            for idx, img in enumerate(imgs, start=1):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base["ext"]
                
                # Save the image to disk
                path = os.path.join(output_dir, f"page{p+1}_img{idx}.{ext}")
                with open(path, "wb") as f:
                    f.write(img_bytes)
                
                # Store the page number and image path
                image_paths.append((p + 1, path))

        # Return list of (page_number, image_path)
        return image_paths

    except Exception as e:
        print(" Error extracting images:", str(e))
        return []

# Function to describe an image using BLIP
def describe_image(path):
    try:
        # Open and convert image to RGB
        image = Image.open(path).convert("RGB")
        
        # Process image and generate caption
        inputs = blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        
        # Decode and clean the generated caption
        desc = blip_processor.decode(out[0], skip_special_tokens=True)
        return clean_text(desc)

    except Exception as e:
        print(f" Error describing image {path}:", str(e))
        return "No caption available"

# Main function to answer a user question based on images in the PDF
def image_question_answering(user_question, pdf_path):
    try:
        # Extract images from the given PDF
        images = extract_all_images(pdf_path)
        if not images:
            return "No images found in the PDF."

        # Use only top 3 images to reduce processing time
        top_images = images[:3]

        # Generate captions for each of the top images
        captions = [(p, describe_image(path)) for p, path in top_images]

        # Format all captions with page number for prompt construction
        joined = "\n".join([f"Page {p}: {c}" for p, c in captions])

        # Build a prompt for the QA model using the image descriptions and user question
        prompt = (
            f"You are an expert in analyzing visuals from AI textbooks. "
            f"Based on the following image descriptions, answer the user's question clearly and informatively.\n\n"
            f"{joined}\n\n"
            f"Question: {user_question}"
        )

        # Generate the final answer using FLAN-T5
        output = t5(prompt, max_new_tokens=250, do_sample=False)
        final_answer = clean_text(output[0]["generated_text"])

        # Save the question and answer to Supabase (for storage/logging)
        store_qa(user_question, "image", final_answer)

        # Return the generated answer
        return final_answer

    except Exception as e:
        print(" Error in image_question_answering:", str(e))
        return "Image-based QA failed."
