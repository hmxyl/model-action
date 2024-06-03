from pymilvus import FieldSchema, DataType, CollectionSchema
from pymilvus import connections, Collection
from pymilvus.orm import utility


def create_aivideo_collection(collection_name):
    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='vector_id', dtype=DataType.VARCHAR, description='向量id', max_length=200),
            FieldSchema(name='file_md5', dtype=DataType.VARCHAR, description='文件md5', max_length=200),
            FieldSchema(name='embedding_name', dtype=DataType.VARCHAR, description='嵌入算法名称', max_length=200),
            FieldSchema(name='page_num', dtype=DataType.INT64, description='文档页码', max_length=11),
            # FieldSchema(name='chunk', dtype=DataType.VARCHAR, description='原始块', max_length=2000),
            FieldSchema(name='create_time', dtype=DataType.VARCHAR, description='入库时间', max_length=200),
            FieldSchema(name='chunk_embedding', dtype=DataType.FLOAT_VECTOR, description='向量', dim=1536),

        ]
        schema = CollectionSchema(fields=fields)
        collection = Collection(name=collection_name, schema=schema)

        index_params = {
            'index_type': 'IVF_FLAT',
            'metric_type': 'L2',
            'params': {'nlist': 1024}
        }
        collection.create_index(field_name="chunk_embedding", index_params=index_params)
        collection.create_index(field_name="file_md5", index_name="file_md5_index")
        collection.create_index(field_name="embedding_name", index_name="embedding_name_index")
        collection.create_index(field_name="page_num", index_name="page_num_index")


connections.connect(
    user="root",
    password="JlJ1cfcpLPHbllmO",
    host="yunzhou.com",
    port="19530"
)

create_aivideo_collection('aivideo_dev')