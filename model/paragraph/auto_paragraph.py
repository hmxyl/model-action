from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def test_auto_paragraph():
    """
    拆分段落
    """
    file_path = "D:\\Workspace\\test_model\\paragraph\\auto_paragraph_origin.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        paragraph_words = file.readline()
    p = pipeline(
        task=Tasks.document_segmentation,
        model='damo/nlp_bert_document-segmentation_chinese-base')
    result = p(documents=paragraph_words)
    with open("D:\\Workspace\\test_model\\paragraph\\auto_paragraph_result.txt", 'w') as file_result:
        file_result.write(result[OutputKeys.TEXT])


if __name__ == '__main__':
    test_auto_paragraph()
