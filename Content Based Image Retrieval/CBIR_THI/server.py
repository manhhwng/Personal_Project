import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from FeatureExtractorVGG19 import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path
from gevent.pywsgi import WSGIServer

app = Flask(__name__, template_folder='template')

# Read image features
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./static/FeaturesVGG19").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./static/database") / (feature_path.stem + ".jpg"))
features = np.array(features)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['query_img']

        # Save query image
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "./static/Query/" + file.filename
        img.save(uploaded_img_path)

        # Run search
        query = fe.extract(img)
        print(query.shape)
        dists = np.linalg.norm(features - query, axis=1)  # L2 distances to features
        ids = np.argsort(dists)[:12]  # Top 12 results
        scores = [(dists[id], img_paths[id]) for id in ids]
        scores.sort(reverse=True)

        return render_template('index2.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index2.html')


if __name__=="__main__":
    app.run(debug=True)
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()