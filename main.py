from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine  # Updated import path   
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import canada_engine




# Retrieve and set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please check your .env file or environment settings.")

# Load population data
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# Initialize PandasQueryEngine with updated prompt
population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Define tools for the agent
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="Provides information on world population and demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=canada_engine,
        metadata=ToolMetadata(
            name="canada_data",
            description="Provides detailed information about Canada",
        ),
    ),
]

# Initialize OpenAI model
llm = OpenAI(model="gpt-3.5-turbo-0613")


llm.embed_model = 'local'  

# Create ReActAgent with tools and model
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# Start user input loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    try:
        result = agent.query(prompt)
        print(result)
    except Exception as e:
        print(f"An error occurred: {e}")

