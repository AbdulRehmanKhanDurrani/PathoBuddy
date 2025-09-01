import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Model & processor setup
MODEL_ID = "Salesforce/blip2-opt-2.7b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load processor and model
processor = Blip2Processor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)

def caption_image(image_path: str, max_new_tokens: int = 50) -> str:
    """
    Generate a caption/summary for the given image.

    Args:
        image_path (str): Path to the image file.
        max_new_tokens (int): Maximum length of generated caption.

    Returns:
        str: The model's caption of the image.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(generated_ids[0], skip_special_tokens=True)

def answer_question_on_image(image_path: str, question: str, max_new_tokens: int = 50) -> str:
    """
    Answer a natural language question about the image.

    Args:
        image_path (str): Path to the image file.
        question (str): The question to ask.
        max_new_tokens (int): Maximum length of generated answer.

    Returns:
        str: The model's answer.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return processor.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Test inputs
    sample_img = "/content/4.-TUBERCULOID-LEPROSY.jpg"
    question = (
    "Question: This is a histopathology image slide of a skin lesion. "
    "Please describe what you see in the image. "
    "Then tell me which stain is likely used in this slide (e.g., H&E, Ziehl-Neelsen, PAS, etc.). "
    "Be specific to pathology. Answer:"
)
    print("Caption:", caption_image(sample_img))
    print("Q:", question)
    print("A:", answer_question_on_image(sample_img, question))