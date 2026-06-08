import base64
from projects.common import Project
import requests
from io import BytesIO
from PIL import Image


class ReverseImageSearch(Project):
    def __init__(self, id, name, description, port=5050, **kwargs):
        kwargs.setdefault('type', 'ml')
        super().__init__(id, name, description, **kwargs)
        self.port = port

    # /projects/reverse_image_search?image_url=https://cdn.pixabay.com/photo/2015/06/03/13/13/cats-796437_1280.jpg
    # /projects/reverse_image_search?image_url=https://www.tensorflow.org/images/cropped_panda.jpg
    def fill_dict(self, request, d):
        if request.FILES.get('painting'):
            data = request.FILES['painting'].read()
        elif request.GET.get('image_url'):
            data = requests.get(request.GET['image_url']).content
        else:
            return
        url = 'http://douwe.com:%s/' % self.port
        files = {'file': data}
        reply = requests.post(url, files=files).json()
        res = reply['results']
        for image, _ in res:
            if 'thumbnail_width' in image:
                image['thumbnail_width'] /= 2
                image['thumbnail_height'] /= 2
        guesses = reply.get('guesses', {})
        if reply.get('year'):
            guesses['year'] = reply['year']
        d['results'] = res[:3]
        d['guesses'] = list(guesses.items())
        stream = BytesIO(data)
        img = Image.open(stream)
        img.thumbnail((240, 240))
        stream_out = BytesIO()
        img.save(stream_out, 'jpeg')
        d['image'] = base64.encodebytes(stream_out.getvalue())
        d['image_width'], d['image_height'] = img.size
