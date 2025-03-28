import re
import numpy as np
from jieba import cut
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def get_words(filename):
    """优化后的预处理函数"""
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.read().strip()
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
        return ' '.join([word for word in cut(text) if len(word) > 1])

def build_feature_extractor(feature_type, X_train, top_num):
    """构建特征转换器（仅在训练集上拟合）"""
    if feature_type == 'frequency':
        all_terms = ' '.join(X_train).split()
        top_words = [w for w, _ in Counter(all_terms).most_common(top_num)]
        return CountVectorizer(vocabulary=top_words)
    elif feature_type == 'tfidf':
        return TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            max_features=top_num,
            token_pattern=None
        )
    raise ValueError("非法特征类型")

def main(feature_type='frequency', top_num=100, test_size=0.2):
    # 加载数据并划分训练测试集
    corpus = [get_words(f'邮件_files/{i}.txt') for i in range(151)]
    labels = np.array([1]*127 + [0]*24)
    X_train, X_test, y_train, y_test = train_test_split(
        corpus, labels,
        test_size=test_size,
        stratify=labels,
        random_state=42
    )

    # 特征工程
    vectorizer = build_feature_extractor(feature_type, X_train, top_num)
    X_train = vectorizer.fit_transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    # 处理样本不平衡
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    # 训练模型
    model = MultinomialNB().fit(X_res, y_res)

    # 模型评估
    print("\n" + "="*50)
    print(f"{'TF-IDF' if feature_type=='tfidf' else '高频词'}特征评估报告")
    print("="*50)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred,
                               target_names=['普通邮件', '垃圾邮件'],
                               digits=4))

    # 新邮件预测
    print("\n预测未知邮件：")
    for fid in range(151, 156):
        text = get_words(f'邮件_files/{fid}.txt')
        vec = vectorizer.transform([text]).toarray()
        res = '垃圾邮件' if model.predict(vec)[0] == 1 else '普通邮件'
        print(f"{fid}.txt -> {res}")

if __name__ == '__main__':
    main(feature_type='tfidf',   # 切换特征类型
         top_num=150,            # 特征维度
         test_size=0.3)          # 测试集比例