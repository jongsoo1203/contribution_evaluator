from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

# Load tokenizer and GPT-2 model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2').to('cuda')  # Assuming CUDA is available

# Define the watermark processor with appropriate parameters
watermark_processor = WatermarkLogitsProcessor(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,
    delta=2.0,
    seeding_scheme="selfhash"  # Equivalent to `ff-anchored_minhash_prf-4-True-15485863`
)

# Input text for the model
input_text = 'tell me about world war 2'
tokenized_input = tokenizer(input_text, return_tensors='pt').to(model.device)  # Ensure tensors are on the correct device

# Generate output tokens with watermarking
output_tokens = model.generate(
    **tokenized_input,
    logits_processor=LogitsProcessorList([watermark_processor]),
    max_new_tokens=300  # Specify a limit to control the number of generated tokens
)

# Isolate the newly generated tokens (prompt tokens are not watermarked)
output_tokens = output_tokens[:, tokenized_input["input_ids"].shape[-1]:]

# Decode the generated tokens to text
output_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
print("Generated Text:")
print(output_text)

# Initialize the watermark detector
watermark_detector = WatermarkDetector(
    vocab=list(tokenizer.get_vocab().values()),
    gamma=0.25,  # Should match original setting
    seeding_scheme="selfhash",  # Should match original setting
    device=model.device,  # Must match the original RNG device type
    tokenizer=tokenizer,
    z_threshold=4.0,
    normalizers=[],  # Add normalizers if needed
    ignore_repeated_ngrams=True
)

# Detect the watermark in the generated text
score_dict = watermark_detector.detect(output_text)
print("Watermark Detection Score:")
print(score_dict)
