# Recommendation-Algorithm-KG
基于知识图谱的推荐算法实现

data是数据集；entity-list.txt的第一列是数据预处理后的id，第二列是数据预处理之前的id；
kg.txt是知识图谱文件，第一列是头实体，第二列是尾实体，第三列是关系；
ratings.txt是按照1:1采集负样本之后的数据集；
relation-list.txt的第一列是关系id，第二列是关系；
其余文件可以忽视。

对于每一个算法来说，如果存在preprocess.py，则先要运行preprocess.py再运行main.py。否则，直接运行main.py。

在以下环境代码可正常运行。
python==3.7.11
torch==1.10.1
tqdm==4.62.3
sklearn==0.0
pandas==1.3.5
numpy==1.12.5
networkx==2.6.3
