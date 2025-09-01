# llm_module.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------
# Configs
# ---------------------------
BASE_MODEL = "EleutherAI/pythia-1b-deduped"   # must match training
LORA_WEIGHTS = "/content/drive/MyDrive/Colab Notebooks/PathoBuddy/lora_weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------
# Load Model + Tokenizer
# ---------------------------
def load_model():
    print("Loading tokenizer...")
    # CRITICAL: Load tokenizer from LoRA weights directory first
    # This ensures we get the same tokenizer config used during training
    tokenizer = AutoTokenizer.from_pretrained(LORA_WEIGHTS)
    
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    
    # CRITICAL: Resize embeddings BEFORE loading LoRA weights
    # This matches the embedding size used during training
    if len(tokenizer) != model.get_input_embeddings().weight.size(0):
        print(f"Resizing embeddings from {model.get_input_embeddings().weight.size(0)} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    
    return model, tokenizer

# ---------------------------
# Generate Answer Function
# ---------------------------
def generate_answer(prompt, mode="descriptive"):
    """
    mode = "descriptive" → longer, detailed answer
    mode = "short" → concise 1–2 sentence answer
    """
    
    if not hasattr(generate_answer, "model"):
        generate_answer.model, generate_answer.tokenizer = load_model()
    
    model, tokenizer = generate_answer.model, generate_answer.tokenizer
    
    # Format prompt to match training format - add context for longer responses in descriptive mode
    if mode == "descriptive":
        formatted_prompt = f"Question: {prompt}\nAnswer: Let me provide a detailed explanation."
    else:
        formatted_prompt = f"Question: {prompt}\nAnswer:"
    
    # Mode-based settings
    if mode == "descriptive":
        max_new_tokens = 512
        temperature = 0.8
        gen_kwargs = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 2,
        }
    else:  # short mode
        max_new_tokens = 60
        gen_kwargs = {
            "do_sample": True,
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
        }
    
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Clean up response
    eos_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    if eos_token in response:
        response = response.split(eos_token)[0]
    
    # Remove the original prompt from response
    if response.startswith(formatted_prompt):
        response = response[len(formatted_prompt):].strip()
    
    return response.strip()

# ---------------------------
# Example Test
# ---------------------------
if __name__ == "__main__":
    test_prompt = "What is liquefactive necrosis?"
    print("Descriptive mode:\n", generate_answer(test_prompt, mode="descriptive"))
    print("\nShort mode:\n", generate_answer(test_prompt, mode="short"))