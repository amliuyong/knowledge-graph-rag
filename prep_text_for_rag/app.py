from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph


load_dotenv()

AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
NEO4J_DATABASE = os.environ["NEO4J_DATABASE"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

accessKeyId = os.getenv("accessKeyId")
secretAccessKey = os.getenv("secretAccessKey")


kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)

kg.query(
    """
    CREATE VECTOR INDEX health_providers_embeddings IF NOT EXISTS
    FOR (hp:HealthcareProvider) ON (hp.comprehensiveEmbedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
      }
    }
    """
)

# # test to see if the index was created
res = kg.query(
    """
  SHOW VECTOR INDEXES
  """
)
print(res)

print("=" * 50)

# kg.query(
#     """
#     MATCH (hp:HealthcareProvider)-[:TREATS]->(p:Patient)
#     WHERE hp.bio IS NOT NULL
#     WITH hp, genai.vector.encode(
#         hp.bio,
#         "Bedrock",
#         {
#           accessKeyId: $accessKeyId,
#           secretAccessKey: $secretAccessKey,
#           model: 'amazon.titan-embed-text-v1',
#           region: 'us-west-2'
#         }) AS vector
#     WITH hp, vector
#     WHERE vector IS NOT NULL
#     CALL db.create.setNodeVectorProperty(hp, "comprehensiveEmbedding", vector)
#     """,
#     params={
#         "accessKeyId": accessKeyId,
#         "secretAccessKey": secretAccessKey,
#     },
# )

result = kg.query(
    """
    MATCH (hp:HealthcareProvider)
    WHERE hp.bio IS NOT NULL
    RETURN hp.bio, hp.name, hp.comprehensiveEmbedding
    LIMIT 5
    """
)
# loop through the results
for record in result:
    print(f" bio: {record["hp.bio"]}, name: {record["hp.name"]}")

print("=" * 50)


# == Queerying the graph for a healthcare provider
question = "give me a list of healthcare providers in the area of dermatology"

print(f"Question: {question}")
print("=" * 25)
# # Execute the query
result = kg.query(
    """
    WITH genai.vector.encode(
        $question,
        "Bedrock",
        {
          accessKeyId: $accessKeyId,
          secretAccessKey: $secretAccessKey,
          model: 'amazon.titan-embed-text-v1',
          region: 'us-west-2'
        }) AS question_embedding
    CALL db.index.vector.queryNodes(
        'health_providers_embeddings',
        $top_k,
        question_embedding
        ) YIELD node AS healthcare_provider, score
    RETURN healthcare_provider.name, healthcare_provider.bio, score
    """,
    params={
        "accessKeyId": accessKeyId,
        "secretAccessKey": secretAccessKey,
        "question": question,
        "top_k": 3,
    },
)

# # Print the encoded question vector for debugging
# print("Encoded question vector:", result)

# # Print the result
for record in result:
    print(f"Name: {record['healthcare_provider.name']}")
    print(f"Bio: {record['healthcare_provider.bio']}")
    # print(f"Specialization: {record['healthcare_provider.specialization']}")
    # print(f"Location: {record['healthcare_provider.location']}")
    print(f"Score: {record['score']}")
    print("---")
