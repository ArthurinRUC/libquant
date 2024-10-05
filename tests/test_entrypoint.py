from transformers import AutoModelForCausalLM, AutoTokenizer

from libquant import QuantArgs, quant


def test_model_loading():
    model = AutoModelForCausalLM.from_pretrained(
        "/mnt/d/linuxdata/HuggingFace/Qwen2-0.5B-Instruct/", device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("/mnt/d/linuxdata/HuggingFace/Qwen2-0.5B-Instruct/")

    # Before quantization
    data = ["Hello, my dog is cute", "Hello, my cat is cute"]
    inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    output1 = model(**inputs)

    # After quantization
    quant_args = QuantArgs(nbits=8)
    quant(model, quant_args=quant_args)
    output2 = model(**inputs)


if __name__ == "__main__":
    test_model_loading()
