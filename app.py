import gradio as gr
from speechbrain.pretrained import EncoderDecoderASR
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


# Load ASR Model
asr_model = EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    savedir="pretrained_models/asr-transformer-transformerlm-librispeech",
    run_opts={"device":"cuda"}
)

# Download Code Generation Model Repo
repo_dir = snapshot_download("NovelAI/genji-python-6B-split")

# Load Code Generation Model + Tokenizer
model = AutoModelForCausalLM.from_pretrained(repo_dir + "/model").half().eval().cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")


def speech_to_code(file):
    text = asr_model.transcribe_file(file.name)
    text = f"""'''{text.lower()}'''\ndef ("""

    tokens = tokenizer(text, return_tensors="pt").input_ids
    generated_tokens = model.generate(
        tokens.long().cuda(),
        use_cache=True,
        do_sample=True,
        top_k=50,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.125,
        min_length=1,
        max_length=len(tokens[0]) + 400,
        pad_token_id=tokenizer.eos_token_id
    )
    last_tokens = generated_tokens[0][len(tokens[0]):]
    generated_text = tokenizer.decode(last_tokens)
    return text + generated_text

iface = gr.Interface(
    fn=speech_to_code,
    inputs=gr.inputs.Audio(source="microphone", type="file"),
    outputs="text"
)
iface.launch(share=True)
