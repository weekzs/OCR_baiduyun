# OCR学习笔记

# 1、进入百度云，然后登陆，点击控制台，然后搜文字识别，新建应用

[百度智能云-云智一体深入产业](https://cloud.baidu.com/?track=f62df187950751b38cff2b6f5d3ab241aaa899d0f8955d32)

![image-20250415110657628](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415110657628.png)

> 搜OCR文字识别

![image-20250415110729865](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415110729865.png)

> 新建应用

![image-20250415110806589](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415110806589.png)

![image-20250415110847190](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415110847190.png)

# 2、复制密钥，为了生成之后的taken

![image-20250415110959441](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415110959441.png)

![image-20250414223413162](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250414223413162.png)

```吃
LUxKJ2sTQu4ltj2eOmt5bGIK

2w8pUU3Zt7tkQTH8GVKGQLSOEUH93t3p

6715044
```



# 3、查看API接口文档

![image-20250415111106940](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415111106940.png)



# 4、我选的是OCR返回坐标的post请求

![image-20250415111152458](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415111152458.png)



# 5、获取taken



## 5、1 第一种方法，按照操作就行，但是要下一个==Postman==

![image-20250415111236999](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415111236999.png)

## 5、2 直接在线调试就行

![image-20250415111645167](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415111645167.png)



# 6、可以进行在线的代码调试



![image-20250415111755398](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415111755398.png)

# 7、代码如下，这里用了模糊匹配和文字帧率的显示，传坐标值

```python
import requests
import base64
import cv2
import time
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 通用文字识别
request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general"
access_token = '24.973bae9d3a347091d6d7eaa2c8e23643.2592000.1747274046.282335-118494985'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}

cap = cv2.VideoCapture(1)
# 这个交互的话，就是视觉里面的onn，这里用的模糊匹配
myobject = ['微积分']
while True:
    success, img0 = cap.read()
    if success:
        t1 = time.time()
        # 将图像转换为 Base64 编码
        _, buffer = cv2.imencode('.png', img0)
        img_base64 = base64.b64encode(buffer)

        # 添加额外的请求参数
        params = {
            "image": img_base64,
            "language_type": "CHN_ENG",
            "detect_direction": "false",
            "detect_language": "false",
            "vertexes_location": "true",
            "paragraph": "true",
            "probability": "true"
        }

        # 调用百度 OCR API
        response = requests.post(request_url, data=params, headers=headers)
        if response:
            result = response.json()
            if 'words_result' in result:
                for item in result['words_result']:
                    text = item['words']
                    for obj in myobject:
                        if re.search(obj, text):
                            vertexes = item['vertexes_location']
                            x1 = vertexes[0]['x']
                            y1 = vertexes[0]['y']
                            x2 = vertexes[2]['x']
                            y2 = vertexes[2]['y']
                            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            print(f"匹配文字: {text}, 坐标: 左上角({x1}, {y1}), 右下角({x2}, {y2})")
                            # 在窗口左上角显示匹配文字和坐标
                            # 强制类型转换
                            info_text = f"匹配文字: {str(text)}, 坐标: 左上角({str(x1)}, {str(y1)}), 右下角({str(x2)}, {str(y2)})"
                            # 将 OpenCV 图像转换为 PIL 图像
                            pil_image = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            # 加载中文字体文件，这里使用的是黑体，你可以根据需要更换
                            font = ImageFont.truetype("simhei.ttf", 18)
                            draw.text((10, 60), info_text, font=font, fill=(0, 255, 0))
                            # 将 PIL 图像转换回 OpenCV 图像
                            img0 = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        fps = 1 / (time.time() - t1)
        cv2.putText(img0, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Detection", img0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

# 8、结果如下

![5ece5c331a757a16658f3007f9148d8](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/5ece5c331a757a16658f3007f9148d8.png)

![image-20250415112139661](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415112139661.png)

# 9、注意事项

> 这里调用之后记得关掉，免费的有额度，这里每个月有免费的额度，付费的话，50块1万次吧

![image-20250415112339953](./OCR%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0.assets/image-20250415112339953.png)