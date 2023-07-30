import pinecone
import os
import openai
import gradio as gr
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationBufferMemory, ConversationSummaryMemory
from langchain.agents import initialize_agent
from langchain.chains import ConversationChain
from langchain import PromptTemplate

pinecone.init(api_key="XXXX", environment="XXXX")

index = pinecone.Index("squareselect")

os.environ['OPENAI_API_KEY'] = 'XXXX'



  
desc = """

Use this tool when you need to tell some information regarding properties to the users. 

This tool only takes a single parameter called user query and returns five properties data at once.

"""

class PropertySearchTool(BaseTool):

    name = "Properties database"
    description = desc

    def _run(self, query: str) -> str:


        response = openai.Embedding.create(
          input=query,
          model="text-embedding-ada-002",
        )

        res = response["data"][0]["embedding"]

        result = index.query(
            
          vector=res,
          top_k=5,
          include_values = True,
          include_metadata = True,
          namespace = "squareselect",

        )

        data = ""

        data += result['matches'][0]['metadata']['text']
        data += result['matches'][1]['metadata']['text']
        data += result['matches'][2]['metadata']['text']
        data += result['matches'][3]['metadata']['text']
        data += result['matches'][4]['metadata']['text']

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")



conversational_memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        k=2,
        return_messages=True,
)


# initialize LLM (we use ChatOpenAI because we'll later define a `chat` agent)
llm = ChatOpenAI(
        openai_api_key="XXXX,
        temperature=0,
        model_name='gpt-3.5-turbo',
)



tools = [PropertySearchTool()]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True
    
)


def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = ' '.join(s)

    output = agent(input)['output']
    history.append((input, output))
    return history, history



css = """
    
    .chat {font-size: 60px !important;} 
    
    .msg {font-size: 60px !important;}
"""


block = gr.Blocks(css=css)


with block:

    gr.Markdown("""<h1><center>Square Select</center></h1>
    """)

    with gr.Row():
      text1 = gr.Textbox(label="History")

      with gr.Column(scale=1, min_width=600):

        chatbot = gr.Chatbot(elem_classes="chat")
        message = gr.Textbox(placeholder="Help me buy a property in...", elem_classes="msg")

        state = gr.State()

        submit = gr.Button("SEND")
        submit.click(chatgpt_clone, inputs=[message, state], outputs=[chatbot, state])



block.launch(debug = True, share=True, server_name="0.0.0.0")
