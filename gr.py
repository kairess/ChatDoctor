import gradio as gr
import torch
import transformers
import torch

model_name = "/content/drive/MyDrive/ChatDoctor_weights"
print("Loading "+model_name+"...")

tokenizer = transformers.LLaMATokenizer.from_pretrained(model_name)
model = transformers.LLaMAForCausalLM.from_pretrained(
    model_name,
    #device_map=device_map,
    #device_map="auto",
    torch_dtype=torch.float16,
    #max_memory = {0: "14GB", 1: "14GB", 2: "14GB", 3: "14GB",4: "14GB",5: "14GB",6: "14GB",7: "14GB"},
    #load_in_8bit=eight_bit,
    #from_tf=True,
    low_cpu_mem_usage=True,
    load_in_8bit=False,
    cache_dir="cache"
).cuda()

generator = model.generate

def answer(state, state_chatbot, text):
    state = state + [f"Patient: {text}"] # TODO

    fulltext = "If you are a doctor, please answer the medical questions based on the patient's description. \n\n" + "\n\n".join(state) + "\n\n" + "ChatDoctor: "

    generated_text = ""
    gen_in = tokenizer(fulltext, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_new_tokens=200,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            repetition_penalty=1.1, # 1.0 means 'off'. unfortunately if we penalize it it will not output Sphynx:
            temperature=0.5, # default: 1.0
            top_k = 50, # default: 50
            top_p = 1.0, # default: 1.0
            early_stopping=True,
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] # for some reason, batch_decode returns an array of one element?

        text_without_prompt = generated_text[len(fulltext):]

    response = text_without_prompt
    response = response.split("Patient: ")[0].strip()

    state = state + [f"ChatDoctor: {response}"]
    state_chatbot = state_chatbot + [(text, response)]

    return state, state_chatbot, state_chatbot


with gr.Blocks(css='#chatbot .overflow-y-auto{height:500px}') as demo:
    state = gr.State(["ChatDoctor: I am ChatDoctor, what medical questions do you have?"])
    state_chatbot = gr.State([(None, "I am ChatDoctor, what medical questions do you have?")])

    with gr.Row():
        gr.HTML("""<div style="text-align: center; margin: 0 auto;">
            <div>
                <h1>ChatDoctor</h1>
            </div>
            <p style="margin-bottom: 10px; font-size: 94%">
                YouTube <a href="https://www.youtube.com/@bbanghyong">빵형의 개발도상국</a>
            </p>
        </div>""")

    with gr.Row():
        chatbot = gr.Chatbot(
            [(None, "I am ChatDoctor, what medical questions do you have?")],
            elem_id='chatbot')

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder='불편한 곳을 영어로 설명해주세요').style(container=False)

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: '' , None, txt)


demo.launch(debug=True, share=False)
