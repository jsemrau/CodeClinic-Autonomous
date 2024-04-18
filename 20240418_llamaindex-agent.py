from huggingface_hub import hf_hub_download

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import PromptTemplate
import time


react_system_header_str = """ 
                            You are a pirate. Reply to all user queries as if 
                            you were a swashbuckling, seafaring opportunitiests.
                            The heavy use of ARrrrrrrr is encouraged.
                             
                          """

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

reader=WikipediaReader()
wikireader = OnDemandLoaderTool.from_defaults(
    reader,
    name="Wikipedia Tool",
    description="A tool for loading data and querying articles from Wikipedia",
)

model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

llm = LlamaCPP(
    model_url=None,
    model_path = model_path,
    temperature=0.11,
    max_new_tokens=4096,
    context_window=4096,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 32},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

react_system_prompt = PromptTemplate(react_system_header_str)

agent = ReActAgent.from_tools([wikireader,multiply_tool, add_tool], llm=llm, verbose=True)
agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})


while True:
  print("\n=========Please type in your question=========================\n")
  user_content = input("\nQuestion: ") # User question

  t1= time.perf_counter()

  response = agent.chat(user_content)

  t2= time.perf_counter()
  print(f"Inferencing using the model: took {t2-t1} seconds to execute.")
  print(response)
