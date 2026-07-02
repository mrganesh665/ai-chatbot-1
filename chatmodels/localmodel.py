from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id= "HuggingFaceTB/SmolLM2-360M-Instruct",
    task="text-generation", 
    pipeline_kwargs= dict(
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )
)


chat_model = ChatHuggingFace(llm=llm)

res = chat_model.invoke("what is USA")

print(res.content)