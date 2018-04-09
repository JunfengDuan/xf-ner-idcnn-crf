
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


#python3 main.py --train=True --clean=True

用时 3.5 h

We have provided with pre-trained NER models. 

To test the pre-trained IDCNN+CRF model, run

#python3 main.py


# TensorBoard
TensorBoard生成图形的流程框架，简单概括起来就两点：

TensorFlow运行并将log信息记录到文件；
TensorBoard读取文件并绘制图形。

## 启动TensorBoard Server

启动TensorBoard Server可以与前面的记录写入并行，TensorBoard会自动的扫描日志文件的更新。

重新生成并绘制，只需手工删除现有数据或者目录即可。

新启动一个命令行窗口，键入命令tensorboard，其参数logdir指出log文件的存放目录，
可以只给出其上级目录，TensorBoard会自动递归扫描目录：
tensorboard --logdir=summary --port=6006

## TensorBoard Server

当TensorBoard服务器顺利启动后，即可打开浏览器输入地址：http://127.0.0.1:6006/查看。
注意在Windows环境下输入http://DESKTOP-S2Q1MOS:6006



## Sample Results

```
INFO:tensorflow:Restoring parameters from ckpt_IDCNN/ner.ckpt

{'string': '香港的房价已经到达历史巅峰,乌溪沙地铁站上盖由新鸿基地产公司开发的银湖天峰,现在的尺价已经超过一万五千港币。'，
'entities': [{'word': '香港', 'end': 2, 'start': 0, 'type': 'LOC'}, {'word': '乌溪沙地铁站', 'end': 20, 'start': 14, 'type': 'LOC'}, {'word': '新鸿基地产公司', 'end': 30, 'start': 23, 'type': 'ORG'}, {'word': '银湖天峰', 'end': 37, 'start': 33, 'type': 'LOC'}]}

{'string': '联想集团的总部位于北京,首席执行官是杨元庆先生', 
'entities': [{'end': 4, 'start': 0, 'word': '联想集团', 'type': 'ORG'}, {'end': 11, 'start': 9, 'word': '北京', 'type': 'LOC'}, {'end': 21, 'start': 18, 'word': '杨元庆', 'type': 'PER'}]}

{'string': '在万达集团的老总王健林的著名采访之后,深圳出现了一家公司叫做赚它一个亿网络科技有限公司', 
'entities': [{'end': 5, 'start': 1, 'word': '万达集团', 'type': 'ORG'}, {'end': 11, 'start': 8, 'word': '王健林', 'type': 'PER'}, {'end': 21, 'start': 19, 'word': '深圳', 'type': 'LOC'}]}

{'string': '律师解读郭敬明性骚扰事件:若无证据 对李枫不利', 
'entities': [{'end': 7, 'start': 4, 'word': '郭敬明', 'type': 'PER'}, {'end': 21, 'start': 19, 'word': '李枫', 'type': 'PER'}]}

{'string': '南开大学党委书记魏大鹏、校长龚克,中科院院士白以龙、陈和生、陈十一、陈永川、邓小刚、杜江峰、方守贤、葛墨林、贺贤土、洪家兴、江松、李家明、李树深、罗俊、罗民兴、莫毅明、欧阳钟灿、潘建伟、孙昌璞、向涛、谢心澄、邢定钰、杨国桢、张维岩、张伟平、张肇西、赵政国、赵忠贤、周向宇、朱邦芬、邹广田,著名书画家、南开大学终身教授范曾,南开大学副校长严纯华出席。', 
'entities': [{'end': 6, 'start': 0, 'word': '南开大学党委', 'type': 'ORG'}, {'end': 11, 'start': 8, 'word': '魏大鹏', 'type': 'PER'}, {'end': 17, 'start': 14, 'word': '龚克,', 'type': 'PER'}, {'end': 20, 'start': 17, 'word': '中科院', 'type': 'ORG'}, {'end': 25, 'start': 22, 'word': '白以龙', 'type': 'PER'}, {'end': 29, 'start': 26, 'word': '陈和生', 'type': 'PER'}, {'end': 33, 'start': 30, 'word': '陈十一', 'type': 'PER'}, {'end': 37, 'start': 34, 'word': '陈永川', 'type': 'PER'}, {'end': 41, 'start': 38, 'word': '邓小刚', 'type': 'PER'}, {'end': 45, 'start': 42, 'word': '杜江峰', 'type': 'PER'}, {'end': 49, 'start': 46, 'word': '方守贤', 'type': 'PER'}, {'end': 53, 'start': 50, 'word': '葛墨林', 'type': 'PER'}, {'end': 57, 'start': 54, 'word': '贺贤土', 'type': 'PER'}, {'end': 61, 'start': 58, 'word': '洪家兴', 'type': 'PER'}, {'end': 64, 'start': 62, 'word': '江松', 'type': 'PER'}, {'end': 68, 'start': 65, 'word': '李家明', 'type': 'PER'}, {'end': 72, 'start': 69, 'word': '李树深', 'type': 'PER'}, {'end': 75, 'start': 73, 'word': '罗俊', 'type': 'PER'}, {'end': 79, 'start': 76, 'word': '罗民兴', 'type': 'PER'}, {'end': 83, 'start': 80, 'word': '莫毅明', 'type': 'PER'}, {'end': 86, 'start': 84, 'word': '欧阳', 'type': 'LOC'}, {'end': 88, 'start': 86, 'word': '钟灿', 'type': 'PER'}, {'end': 92, 'start': 89, 'word': '潘建伟', 'type': 'PER'}, {'end': 96, 'start': 93, 'word': '孙昌璞', 'type': 'PER'}, {'end': 99, 'start': 97, 'word': '向涛', 'type': 'PER'}, {'end': 103, 'start': 100, 'word': '谢心澄', 'type': 'PER'}, {'end': 107, 'start': 104, 'word': '邢定钰', 'type': 'PER'}, {'end': 111, 'start': 108, 'word': '杨国桢', 'type': 'PER'}, {'end': 115, 'start': 112, 'word': '张维岩', 'type': 'PER'}, {'end': 119, 'start': 116, 'word': '张伟平', 'type': 'PER'}, {'end': 123, 'start': 120, 'word': '张肇西', 'type': 'PER'}, {'end': 127, 'start': 124, 'word': '赵政国', 'type': 'PER'}, {'end': 131, 'start': 128, 'word': '赵忠贤', 'type': 'PER'}, {'end': 135, 'start': 132, 'word': '周向宇', 'type': 'PER'}, {'end': 139, 'start': 136, 'word': '朱邦芬', 'type': 'PER'}, {'end': 143, 'start': 140, 'word': '邹广田', 'type': 'PER'}, {'end': 154, 'start': 150, 'word': '南开大学', 'type': 'ORG'}, {'end': 165, 'start': 158, 'word': '范曾,南开大学', 'type': 'ORG'}, {'end': 171, 'start': 168, 'word': '严纯华', 'type': 'PER'}]}

{'string': '陈省身先生的好朋友、原英国皇家学会会长迈克尔•阿蒂亚曾为在爱丁堡广场捐建价值约200万英镑麦克斯韦铜像,花费了很大力气。', 
'entities': [{'end': 3, 'start': 0, 'word': '陈省身', 'type': 'PER'}, {'end': 17, 'start': 11, 'word': '英国皇家学会', 'type': 'ORG'}, {'end': 22, 'start': 19, 'word': '迈克尔', 'type': 'PER'}, {'end': 26, 'start': 23, 'word': '阿蒂亚', 'type': 'PER'}, {'end': 34, 'start': 29, 'word': '爱丁堡广场', 'type': 'LOC'}, {'end': 49, 'start': 45, 'word': '麦克斯韦', 'type': 'LOC'}]}

{'string': '当地时间25日(周五)下午2点30分,韩国法院将对三星电子副会长李在镕行贿案作出一审判决。今年49岁、三星集团的实际领导人李在镕,即将迎来他的“命运星期五”。', 
'entities': [{'end': 21, 'start': 19, 'word': '韩国', 'type': 'LOC'}, {'end': 29, 'start': 25, 'word': '三星电子', 'type': 'ORG'}, {'end': 35, 'start': 32, 'word': '李在镕', 'type': 'PER'}, {'end': 55, 'start': 51, 'word': '三星集团', 'type': 'ORG'}, {'end': 64, 'start': 61, 'word': '李在镕', 'type': 'PER'}]}
```

# ModuleNotFoundError: No module named '_sqlite3'

ubuntu: sudo apt-get install libsqlite3-dev
重新编译python