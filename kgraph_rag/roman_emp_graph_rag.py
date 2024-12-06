from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

#from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_aws import ChatBedrock  # Changed from langchain_openai
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_aws import BedrockEmbeddings
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# AWS Bedrock configuration
AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize Bedrock chat and embedding models
chat = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # You can change this to another Bedrock model
    region_name=AWS_REGION_NAME,
    credentials_profile_name="rch"  # or use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name=AWS_REGION_NAME,
    credentials_profile_name="rch"
)

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

# # read the wikipedia page for the Roman Empire
# raw_documents = WikipediaLoader(query="The Roman empire").load()

# # Define chunking strategy
# text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
# documents = text_splitter.split_documents(raw_documents[:30])

# print(documents)
# print("=" * 50)


# llm_transformer = LLMGraphTransformer(llm=chat)
# graph_documents = llm_transformer.convert_to_graph_documents(documents)

# print(documents)
# print("=" * 50)


# # store to neo4j
# res = kg.add_graph_documents(
#     graph_documents,
#     include_source=True,
#     baseEntityLabel=True,
# )

# print("=" * 50)

# exit(0)

# Hybrid Retrieval for RAG
# create vector index

vector_index = Neo4jVector.from_existing_graph(
    embeddings,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        default_factory=list,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract specific named entities (people, places, organizations) from the text. "
            "Always return a list of strings, even if there's only one entity."
        ),
        (
            "human",
            "Extract named entities from this text: {question}\n"
            "Return ONLY a list of entity names. If no clear entities, return an empty list. "
            "Format: ['Entity1', 'Entity2']"
        ),
    ]
)

def parse_entities(model_output):
    # Extract the content from the AIMessage
    text = model_output.content
    # Split the output by comma and strip whitespace
    entity_names = [name.strip() for name in text.split(',') if name.strip()]
    return Entities(names=entity_names)



#entity_chain = prompt | chat.with_structured_output(Entities, include_raw=False)

entity_chain = (
    prompt 
    | chat
    | RunnableLambda(parse_entities)
)

# Test it out:
res = entity_chain.invoke(
    {"question": "In the year of 123 there was an emperor who did not like to rule."}
)
print(res)


# Retriever
kg.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.
    """
    words = [el for el in remove_lucene_chars(input).split() if el]
    
    # Return empty string if no words found
    if not words:
        return ""
    
    # If only one word, return that word with fuzzy match
    if len(words) == 1:
        return f"{words[0]}~2"
    
    # Multiple words: combine with AND and fuzzy match
    full_text_query = ""
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


# Fulltext index query
def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    
    print(f"Entities: {entities.names}")

    # Early return if no entities found
    if not entities.names:
        return result

    for entity in entities.names:
        print(f" Getting Entity: {entity}")
        query = generate_full_text_query(entity)
        
        # Skip if query is empty
        if not query:
            continue

        response = kg.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": query},
        )
        result += "\n".join([el["output"] for el in response])
    return result


#print(structured_retriever("Who is Octavian?"))
#q = "In the year of 123 there was an emperor who did not like to rule."
q = "Who is Octavian?"
print(f"====> Search query: {q}")
print(structured_retriever(q))



print("=" * 150)


# Final retrieval step
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    print(f"\nFinal Data::: ==>{final_data}")
    return final_data


# Define the RAG chain
# Condense a chat history and follow-up question into a standalone question
_template = """Convert the follow-up question into a standalone question that captures the full context from the previous conversation.

Chat History: {chat_history}
Follow Up Input: {question}
Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer


# _search_query = RunnableBranch(
#     # If input includes chat_history, we condense it with the follow-up question
#     (
#         RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
#             run_name="HasChatHistoryCheck"
#         ),  # Condense follow-up question and chat into a standalone_question
#         RunnablePassthrough.assign(
#             chat_history=lambda x: _format_chat_history(x["chat_history"])
#         )
#         | CONDENSE_QUESTION_PROMPT
#         | chat
#         | StrOutputParser(),
#     ),
#     # Else, we have no chat history, so just pass through the question
#     RunnableLambda(lambda x: x["question"]),
# )


# Use a dedicated model or adjust the existing chat model for condensing
_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        # Improved condensing step
        RunnablePassthrough.assign(
            standalone_question=lambda x: (
                CONDENSE_QUESTION_PROMPT | chat | StrOutputParser()
            ).invoke({
                "chat_history": _format_chat_history(x["chat_history"]),
                "question": x["question"]
            })
        )
        | RunnableLambda(lambda x: x["standalone_question"])
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x: x["question"]),
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

def log_prompt(input_dict):
    print("\n=== Generated Prompt ===")
    print(prompt.format(**input_dict))
    print("=====================\n")
    return input_dict


chain1 = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | RunnableLambda(log_prompt)  # Add logging here
    | prompt
    | chat
    | StrOutputParser()
)


chain = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": _search_query,  # Use the standalone question here
        }
    )
    | RunnableLambda(log_prompt)  # Add logging here
    | prompt 
    | chat 
    | StrOutputParser()
)

print("====== chat without history ======")
# TEST it all out!
# res_simple = chain.invoke(
#     {
#         "question": "How did the Roman empire fall?",
#     }
# )

# print("=" * 150)

# print(f"\n Results === {res_simple}\n\n")

print("=" * 150)
print("=" * 150)


print("====== chat with history ======")

q = "When did he become the first emperor?"
hist =  [
            ("Who was the first emperor?", "Augustus was the first emperor.")
        ]

res_hist = chain.invoke(
    {
        "question": q,
        "chat_history": hist,
    }
)

print(f"\n === {res_hist}\n\n")