

# docker run -d \
#     --restart=always \
#     --name neo4j-database \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/Industry123! \
#     --volume=./data:/data \
#     neo4j:5.25.1



# docker run -d \
#     --restart=always \
#     --name neo4j-database \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/Industry123! \
#     --volume=$(pwd)/data:/data \
#     --volume=$(pwd)/plugins:/plugins \
#     --env NEO4JLABS_PLUGINS='["apoc"]' \
#     --env NEO4J_dbms_security_procedures_unrestricted="apoc.*" \
#     --env NEO4J_apoc_export_file_enabled=true \
#     --env NEO4J_apoc_import_file_enabled=true \
#     neo4j:5.25.1




# docker run -d \
#     --restart=always \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/Industry123! \
#     --volume=$(pwd)/data:/data \
#     --volume=$(pwd)/plugins:/plugins \
#     --env NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
#     --env NEO4J_dbms_security_procedures_unrestricted="apoc.*,gds.*" \
#     --env NEO4J_apoc_export_file_enabled=true \
#     --env NEO4J_apoc_import_file_enabled=true \
#     neo4j:5.25.1


cd /home/ec2-user/work/llm/knowledge-graph-rag/neo4j

docker run -d \
    --restart=always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/Industry123! \
    --volume=$(pwd)/data:/data \
    --volume=$(pwd)/plugins:/plugins \
    --env NEO4J_PLUGINS='["apoc", "graph-data-science", "genai"]' \
    --env NEO4J_dbms_security_procedures_unrestricted="apoc.*,gds.*,genai.*" \
    --env NEO4J_apoc_export_file_enabled=true \
    --env NEO4J_apoc_import_file_enabled=true \
    neo4j:5.25.1




MATCH (n) DETACH DELETE n;
