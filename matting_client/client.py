import json
import time

import cv2
import requests
import numpy as np

if __name__ == '__main__':
    url = 'http://59.41.16.115:9111/api/matting'
    img_path = 'D:/Mission/all_photos/200501705736451697934336.png'
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

    start_time = time.time()
    r = requests.post(url, data=request_data, headers=headers)
    response_data = r.json()
    end_time = time.time()
    print(f'time:{end_time - start_time:.2f}s')

    result_numpy = np.array(json.loads(response_data['result']), dtype=np.uint8)

    cv2.imwrite('./test.png', result_numpy)

