from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    """
    拆分段落: https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/summary
    """
    """
    问题记录：cannot import name '_datasets_server' from 'datasets.utils' (D:\Developer\anaconda3\envs\model-action\Lib\site-packages\datasets\\utils\\__init__.py)
    处理：pip install datasets==2.18.0
    """
    file_path = "D:\\Workspace\\test_model\\paragraph\\auto_paragraph_origin.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph_words = file.readline()
    print(paragraph_words)
    output = pipeline(
        task=Tasks.document_segmentation,
        model='damo/nlp_bert_document-segmentation_chinese-base')
    result = output(paragraph_words)
    with open("D:\\Workspace\\test_model\\paragraph\\auto_paragraph_result.txt", 'w') as file_result:
        line = result[OutputKeys.TEXT]
        file_result.write(line)
        print(line)
