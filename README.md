# 使用方法

[TOC]

## 服务端

开启服务：

```bash
cd /home/jz/PortraitMatting
gunicorn -b 0.0.0.0:9111 server:app
```



## 客户端

### 一、base64编码图像请求方式

#### 请求格式

URL：

```
http://59.41.16.115:9111/api/matting_base64
```

请求头:

```
"Content-Type": "application/json;charset=utf8"
```

请求体（json格式）：

```json
{
	"image":"YpFU9rZtVTiJiKoCQLAAAAABJRU5ErkJggg==",  // 图像的base64编码
	"bg":[33, 150, 242]                               // 背景颜色，顺序RGB，取值为0-255整数
}
```

返回值（json格式）：

```json
{
	"result":"YpFU9rZtVTiJiKoCQLAAAAABJRU5ErkJggg=="  // 图像的base64编码
}
```

#### Python样例

```python
import base64
import json
import cv2
import requests
import numpy as np

if __name__ == '__main__':
    url = 'http://59.41.16.115:9111/api/matting_base64'
    img_path = 'D:/Mission/photos_png/1.png'
    bg_color = [33, 150, 242]

    with open(img_path, 'rb') as f:
        img_base64 = base64.b64encode(f.read())

    request_data = json.dumps(
        {
            'image': img_base64.decode('utf-8'), 
            'bg': json.dumps(bg_color)
        }
    )

    headers = {
        "Content-Type": "application/json;charset=utf8"
    }

    r = requests.post(url, data=request_data, headers=headers)
    response_data = r.json()

    result_base64 = base64.b64decode(response_data['result'])
    result_numpy = cv2.imdecode(np.fromstring(result_base64, np.uint8), cv2.IMREAD_COLOR)

    print(response_data)
    cv2.imwrite('./test.png', result_numpy)

```





### 二、矩阵编码图像请求方式

#### 请求格式

URL：

```
http://59.41.16.115:9111/api/matting
```

请求头:

```
"Content-Type": "application/json;charset=utf8"
```

请求体（json格式）：

```json
{
	"image":[[[1, 1, 1], [2, 2, 2], [3, 3, 3], ...]],  // 图像矩阵（按opencv的格式，shape=[高,宽,颜色（顺序为BGR）]，取值为0-255整数）
	"bg":[33, 150, 242]                                // 背景颜色，顺序RGB，取值为0-255整数
}
```

返回值（json格式）：

```json
{
	"result":[[[1, 1, 1], [2, 2, 2], [3, 3, 3], ...]]  // 图像矩阵（按opencv的格式，shape=[高,宽,颜色（顺序为BGR）]，取值为0-255整数）
}
```

#### Python样例

```python
import json
import cv2
import requests
import numpy as np

if __name__ == '__main__':
    url = 'http://59.41.16.115:9111/api/matting'
    img_path = 'D:/Mission/photos_png/1.png'
    bg_color = [33, 150, 242]

    img = cv2.imread(img_path)
    request_data = json.dumps(
        {
            'image': json.dumps(img.tolist()), 
            'bg': json.dumps(bg_color)
        }
    )

    headers = {
        "Content-Type": "application/json;charset=utf8"
    }

    r = requests.post(url, data=request_data, headers=headers)
    response_data = r.json()

    result_numpy = np.array(json.loads(response_data['result']), dtype=np.uint8)

    cv2.imwrite('./test.png', result_numpy)


```





