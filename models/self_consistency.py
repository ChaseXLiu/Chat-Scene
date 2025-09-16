from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import Counter

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def self_consistency(prompt, num_samples=5, max_len=10):
    samples = []
    for _ in range(num_samples):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=max_len, do_sample=True)
        sample = tokenizer.decode(outputs[0], skip_special_tokens=True)
        samples.append(sample)
    
    vote_counts = Counter(samples)
    return vote_counts.most_common(1)[0][0]

prompt = "What is 2 + 3?"
final_answer = self_consistency(prompt)
print(f"Final answer: {final_answer}")
