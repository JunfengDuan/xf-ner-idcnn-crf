
# Chinese Named Entity Recognition using IDCNN + CRF

## Requirements

* Python (>=3.5)

* TensorFlow (>=r1.0)

* jieba (>=0.37)


## Usage


### * Training:

1. Prepare data at data/, including training data (example.train), validation data (example.dev), testing data (example.test), and Chinese character embedding data (vec.txt).


Current sample data includes 3 types of Named Entities, including ORG(单位), PER(人) and LOC(地址)
对于问题属地，可通过识别出来的简写地址去数据库匹配全名

2. Training, models with best F1 score on validation data will be saved at ckpt/


To train with IDCNN+CRF (default), run:


python3 main.py --train=True --clean=True

用时 3.5 h

We have provided with pre-trained NER models. 

To test the pre-trained IDCNN+CRF model, run

python3 main.py


## TensorBoard
TensorBoard生成图形的流程框架，简单概括起来就两点：

TensorFlow运行并将log信息记录到文件；
TensorBoard读取文件并绘制图形。

### 启动TensorBoard Server

启动TensorBoard Server可以与前面的记录写入并行，TensorBoard会自动的扫描日志文件的更新。

重新生成并绘制，只需手工删除现有数据或者目录即可。

新启动一个命令行窗口，键入命令tensorboard，其参数logdir指出log文件的存放目录，
可以只给出其上级目录，TensorBoard会自动递归扫描目录：
tensorboard --logdir=summary --port=6006

### TensorBoard Server

当TensorBoard服务器顺利启动后，即可打开浏览器输入地址：http://127.0.0.1:6006/查看。
注意在Windows环境下输入http://DESKTOP-S2Q1MOS:6006



## Sample Results


INFO:tensorflow:Restoring parameters from ckpt_IDCNN/ner.ckpt

{'string': '香港的房价已经到达历史巅峰,乌溪沙地铁站上盖由新鸿基地产公司开发的银湖天峰,现在的尺价已经超过一万五千港币。'，
'entities': [{'word': '香港', 'end': 2, 'start': 0, 'type': 'LOC'}, {'word': '乌溪沙地铁站', 'end': 20, 'start': 14, 'type': 'LOC'}, {'word': '新鸿基地产公司', 'end': 30, 'start': 23, 'type': 'ORG'}, {'word': '银湖天峰', 'end': 37, 'start': 33, 'type': 'LOC'}]}

## 参考致谢
https://github.com/crownpku/Information-Extraction-Chinese/tree/master/NER_IDCNN_CRF


## ModuleNotFoundError: No module named '_sqlite3'

ubuntu: sudo apt-get install libsqlite3-dev
重新编译python