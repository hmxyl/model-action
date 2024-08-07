import re

from funasr import AutoModel


def auto_punctuation():
    """
    标点回填
    """
    with open("D:\\Workspace\\test_model\\punctuation\\auto_punctuation_clean.txt", 'r', encoding='utf-8') as file:
        test_text = file.readline()
    model = AutoModel(model="ct-punc", model_revision="v2.0.4")
    res = model.generate(input=test_text)
    punctuation_result = res[0]['text']
    print("【回填后】")
    print(punctuation_result)
    with open("D:\\Workspace\\test_model\\punctuation\\auto_punctuation_result.txt", 'w') as result:
        result.write(punctuation_result)


def clean_punctuation():
    """
    清理标点
    """
    with open("D:\\Workspace\\test_model\\punctuation\\auto_punctuation.txt", 'r', encoding='utf-8') as file:
        test_text = file.readline()
    # 删除中文标点符号
    punctuation = r"[，。？！：；‘’“”（）【】《》、!#$%&'()*+,-./:;<=>?@[\]^_`{|}~\"]"
    test_text_clean = re.sub(punctuation, "", test_text)
    print("【清洗前】")
    print(test_text)
    print("【清洗后】")
    print(test_text_clean)
    with open("D:\\Workspace\\test_model\\punctuation\\auto_punctuation_clean.txt", 'w', encoding='utf-8') as file_clean:
        file_clean.write(test_text_clean)


if __name__ == '__main__':
    clean_punctuation()
    auto_punctuation()
