from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("tarudesu/ViHateT5-base-HSD")
model = AutoModelForSeq2SeqLM.from_pretrained("tarudesu/ViHateT5-base-HSD").to(device)
model.eval()

def generate_output(input_text, prefix):
    start = time.time()

    # Add prefix
    prefixed_input_text = prefix + ': ' + input_text

    # Tokenize input text
    inputs = tokenizer(prefixed_input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=256)

    # Decode the generated output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    end = time.time()
    print('Time taken: ', end - start)

    return output_text

sample = 'Đm nó, phim lởm quá trời, diễn viên diễn như cc, nội dung thì xàm, đạo diễn như ***, làm phim dở ẹc, tốn tiền vé vcl'
prefix = 'hate-speech-detection' # Choose 1 from 3 prefixes ['hate-speech-detection', 'toxic-speech-detection', 'hate-spans-detection']

result = generate_output(sample, prefix)
print(result)
