# face_rec
基于OpenCV+TensorFlow+LBP的人脸识别系统

## core.py
获取摄像头权限，进行人脸识别，并从数据库中获取识别人脸的身份信息，返回姓名/Unknown。

## dataRecord.py
获取摄像头权限，点击“开始采集人脸数据”后，会自动采集300张人脸图像。采集完成后输入身份信息，将人脸与身份匹配输入数据库。

## dataManage.py
手动管理数据库，对身份信息进行删改查。

## index.py
初始化。
