from flask import Flask, request, jsonify
from flask_cors import *
import argparse
import io

app = Flask(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8800, type=int)
    args = parser.parse_args()

    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', args.port)
    

if __name__ == '__main__':
    main()

@app.route('/api/upload', methods=['POST'])
def upload_img():
    global predictor
    bio = io.BytesIO()
    data = request.form.to_dict()
    file = request.files['file']
    file.save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    code,rst = get_predictor(east_path,crnn_path)(img)
    if code == 0:
        result = save_result(img, rst)
        res = jsonify({
            'code': 0,
            'msg': 'ok',
            'data': {
                'imginput': result['img_input'],
                'imgoutput': result['img_output'],
                'text': result
            }
        })
    else:
        res = jsonify({
            'code': code,
            'msg': 'error',
        })

    res.headers['Access-Control-Allow-Origin'] = '*'
    res.headers['Access-Control-Allow-Methods'] = 'POST,GET,OPTIONS'
    res.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return res