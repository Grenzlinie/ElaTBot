# load our finetuned model as ChatModel
import transformers
import torch
from langchain_huggingface import HuggingFacePipeline
from hf_chat import ChatHuggingFace
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_core.prompts import PromptTemplate
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
import re
from langchain_community.document_loaders.csv_loader import CSVLoader

model_path = "../Models/ElaTBoT"

device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# set quantization config
bng_config = transformers.BitsAndBytesConfig(
    load_in_8bit=False,
    load_in_4bit=False,
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
model_config = transformers.AutoConfig.from_pretrained(model_path)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True,
    config=model_config, 
    quantization_config=bng_config if bng_config.load_in_8bit or bng_config.load_in_4bit else None,
    device_map='auto',
    )

model.generation_config.pad_token_id = tokenizer.eos_token_id

with torch.inference_mode(mode=True):
    model.eval()
print(f"Model loaded on {device}")

stop_tokens = ['\nHuman']


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_token in stop_tokens:
            if stop_token in tokenizer.decode(input_ids[0][-5:], skip_special_tokens=True, clean_up_tokenization_spaces=True):
                return True
            return False
    
stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    task='text-generation',
    temperature= 0.01,
    max_new_tokens=512,
    repetition_penalty= 1.0,
    stopping_criteria=stopping_criteria,

)

llm = HuggingFacePipeline(pipeline=generate_text)
chat_llm = ChatHuggingFace(llm=llm, tokenizer=tokenizer, verbose=True)

loader = CSVLoader(file_path='./merged_data_for_test.csv')
data = loader.load()

# load local embeddings
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_name = "intfloat/multilingual-e5-large"
model_kwargs = {'device': 'cpu', 'trust_remote_code': True}
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)



prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know.

{context}

Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# OpenAI embeddings usually gives better results if your api-key is available.
# vectordb = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
vectordb = Chroma.from_documents(documents=data, embedding=embeddings)
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={'k': 5})
qa = RetrievalQA.from_chain_type(
    llm=chat_llm, 
    chain_type="stuff", 
    retriever=retriever, 
    chain_type_kwargs={"verbose":True, "prompt":PROMPT}
)

# def qa(query: str) -> str:
#     docs = vectordb.similarity_search(query, k=4)
#     context = ""
#     for doc in docs:
#         context += doc.page_content + "\n"
#     print(context)
#     prompt = PROMPT.format(context=context, question=query)
#     response = chat_llm.invoke(prompt).content.strip()
#     return response
    
# def query_with_plain_text(messages, history):
#     if len(history) >= 5:
#         history.pop(0)
#     inputs = ""
#     if history == []:
#         inputs = "Human:\n" + messages + "\n" + "Assistant:\n"
#     else:
#         for m in history:
#             inputs += "Human:\n" + m[0] + "\n" + "Assistant:\n" + m[1] + "\n"
#         inputs += "Human:\n" + messages + "\n" + "Assistant:\n"
#     response = llm.invoke(inputs).strip()
#     return response


def query_without_rag(messages, history):
    if len(history) >= 5:
        history.pop(0)
    if history == []:
        inputs = [HumanMessage(messages)]
    else:
        inputs = []
        for m in history:
            inputs.append(HumanMessage(m[0]))
            inputs.append(AIMessage(m[1]))
        inputs.append(HumanMessage(messages))
    response = chat_llm.invoke(inputs).content.strip()
    return response


def query(messages, history):
    if len(history) >= 5:
        history.pop(0)
    if re.search(r'bulk modulus', messages) and re.search(r'Ni3Al|gamma-PE16|gamma-TiAl', messages):
        print('using rag')
        qa_response = qa.run(messages)
        print(qa_response)
        if re.search(r"(don't know|do not know)", qa_response, re.IGNORECASE):
            print('fall back')
            qa_response = "please use our formatter function to get prompt!"
    else:
        qa_response = query_without_rag(messages, history)
    return qa_response

examples = [
            "Given a material description, predict its elastic tensor at 300K temperature accurately and directly using scientific logic. Provide the answer as a 6x6 Python matrix without additional comments, descriptions, or explanations. The material is Al with crystal system cubic and composition ratio {'Al': 100.0} (total is 100%). The information about the material is as follows. Al has an electronegativity of 1.61, an ionization energy of 5.986 eV, a bulk modulus of 76.0 GPa, a Young's modulus of 70.0 GPa, a Poisson's ratio of 0.35, an atomic radius of 1.18 Å.",
            "Generate a material chemical formula and its crystal system with a Voigt bulk modulus of 250 GPa at a temperature of 0K. Use scientific reasoning step-by-step and directly output the answer without additional comments, descriptions, or explanations.",
            "What‘s the Voigt bulk modulus of cubic Ni3Al at temperature 300 K?", 
            "What‘s the Voigt bulk modulus of cubic gamma-PE16 at temperature 113 K?",
            "What‘s the Voigt bulk modulus of tetragonal gamma-TiAl at temperature 170 K?",
            ]

demo = gr.ChatInterface(fn=query, examples=examples, title="ElaTBot")
demo.launch(share=True, inline=True, inbrowser=True)