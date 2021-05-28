from flask import Flask
from flask_restful import Api,Resource
from keras.preprocessing.image import img_to_array
from keras.models import  load_model
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import json

#Server side
app = Flask(__name__)
api = Api(app)

#design class
class roast(Resource):
    def get(self, url):
        out = predicturl(url)
        return out

#call api
api.add_resource(roast,"/roast/<path:url>")

#function
def predicturl(url):
    my_url = url
    response = requests.get(my_url)
    img = Image.open(BytesIO(response.content))
    new_image = img.resize((224, 224))

    img = img_to_array(new_image)/255.0
    img = np.expand_dims(img, axis=0)
    saved_model = load_model("models/model_vgg16_coffee_beans.h5")
    probs = saved_model.predict(img)[0]
    probs_r = np.round(probs*100,decimals=2)
            
    output = {'Light:': probs_r[0], 'Dark': probs_r[1], 'Medium': probs_r[2]}
    out = json.dumps(str(output))
    return out

#route
@app.route('/')
def index():
    return "Hello, World!"

if __name__ == "__main__":
    app.run(debug=True, threaded=False)