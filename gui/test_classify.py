import urllib.request
import io
import json
from PIL import Image
import numpy as np

img = Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8), 'L')
buf = io.BytesIO()
img.save(buf, 'PNG')
img_bytes = buf.getvalue()

boundary = 'boundary123'
header1 = b'--boundary123\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n'
header2 = b'\r\n--boundary123\r\nContent-Disposition: form-data; name="model_name"\r\n\r\nBaseline\r\n--boundary123--'
body = header1 + img_bytes + header2

req = urllib.request.Request(
    'http://localhost:8000/classify',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=boundary123'},
    method='POST'
)
r = urllib.request.urlopen(req)
print(json.loads(r.read()))
