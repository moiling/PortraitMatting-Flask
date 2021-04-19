import base64
import json
import time

import cv2
import requests
import numpy as np

if __name__ == '__main__':
    url = 'http://59.41.16.115:9111/api/matting_base64'
    img_path = 'D:/Mission/all_photos/200501705736451697934336.png'
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

    start_time = time.time()
    r = requests.post(url, data=request_data, headers=headers)
    response_data = r.json()
    end_time = time.time()
    print(f'time:{end_time - start_time:.2f}s')

    result_base64 = base64.b64decode(response_data['result'])
    result_numpy = cv2.imdecode(np.fromstring(result_base64, np.uint8), cv2.IMREAD_COLOR)

    cv2.imwrite('./test.png', result_numpy)

