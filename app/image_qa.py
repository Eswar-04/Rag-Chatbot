import fitz
import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from app.qa import clean_text
from app.supabase_client import store_qa

# Load BLIP Large model (better captions)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load FLAN-T5 for answering
t5 = pipeline("text2text-generation", model="google/flan-t5-base")

def extract_all_images(pdf_path, output_dir="images"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_paths = []
        for p in range(len(doc)):
            page = doc[p]
            imgs = page.get_images(full=True)
            for idx, img in enumerate(imgs, start=1):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base["ext"]
                path = os.path.join(output_dir, f"page{p+1}_img{idx}.{ext}")
                with open(path, "wb") as f:
                    f.write(img_bytes)
                image_paths.append((p + 1, path))
        return image_paths
    except Exception as e:
        print(" Error extracting images:", str(e))
        return []

def describe_image(path):
    try:
        image = Image.open(path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        desc = blip_processor.decode(out[0], skip_special_tokens=True)
        return clean_text(desc)
    except Exception as e:
        print(f" Error describing image {path}:", str(e))
        return "No caption available"

def image_question_answering(user_question, pdf_path):
    try:
        images = extract_all_images(pdf_path)
        if not images:
            return "No images found in the PDF."

        # Use top 3 images to reduce load
        top_images = images[:3]
        captions = [(p, describe_image(path)) for p, path in top_images]

        # Build prompt
        joined = "\n".join([f"Page {p}: {c}" for p, c in captions])
        prompt = (
            f"You are an expert in analyzing visuals from AI textbooks. "
            f"Based on the following image descriptions, answer the user's question clearly and informatively.\n\n"
            f"{joined}\n\n"
            f"Question: {user_question}"
        )

        output = t5(prompt, max_new_tokens=250, do_sample=False)
        final_answer = clean_text(output[0]["generated_text"])

        # Save to Supabase
        store_qa(user_question, "image", final_answer)

        return final_answer
    except Exception as e:
        print(" Error in image_question_answering:", str(e))
        return "Image-based QA failed."
