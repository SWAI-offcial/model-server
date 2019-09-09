#!/usr/bin/env python
# encoding: utf-8
'''
@author: tianxiaomo
@license: (C) Apache.
@contact: huguanghao520@gmail.com
@software: 
@file: run_demo_server.py.py
@time: 2019/9/9 14:11
@desc:
'''
#!/usr/bin/env python3
import argparse,requests
from keras.preprocessing import image
import os,datetime,cv2,uuid,json,functools,logging,collections,math,re,time
from sklearn.cluster import KMeans
from sklearn import metrics
from PIL import Image
import numpy as np

### the webserver
from flask import Flask, request, render_template, jsonify
from flask_cors import *
import argparse

from .eval import resize_image, sort_poly, detect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#define every error
OK = 0
DetectionError = 5001
RecognitionError = 5002
PostProcessorError = 5003

def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu

def save_result(img, rst):
    session_id = str(uuid.uuid1())
    dirpath = os.path.join(config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    rst['img_input'] = os.path.join(dirpath,'input.png')
    rst['img_output'] = os.path.join(dirpath,'output.png')
    return rst

# 放射变换
def adjust(img,boxes,gray=False):
    #TODO https://www.jianshu.com/p/05374b86e85b
    if gray:
        images = np.zeros([boxes,shape[0],32,320,1])
    else:
        images = np.zeros([boxes.shape[0],32,320,3])
    t_boxes = boxes.copy()
    for i,box in enumerate(t_boxes):
        # 原图的四个角点    661,449,731,467,725,488,656,469
        pts1 = box.astype('float32')
        # 变换后分别在左上、右上、左下、右下四个点
        w = box[:,0].max() - box[:,0].min()
        h = box[:, 1].max() - box[:, 1].min()

        pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        # 生成透视变换矩阵
        M = cv2.getPerspectiveTransform(pts1, pts2)
        # 进行透视变换
        dst = cv2.warpPerspective(img, M, (w, h))
        h, w,_ = dst.shape

        n_w = int(w * (32 / h) + 0.5)
        if n_w > 320:n_w = 320
        dst = Image.fromarray(dst)
        image = dst.resize([n_w, 32])
        # print(image.size)
        if gray:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = image.convert('L')
        image = np.asarray(image)
        images[i,:,:n_w,:] = image
    return images

# 检测
def detector(img):
    im_resized, (ratio_h, ratio_w) = resize_image(img)
    im_resized = np.asarray(im_resized)
    rtparams = collections.OrderedDict()
    rtparams['start_time'] = datetime.datetime.now().isoformat()
    rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])
    
    payload = {"instances": [{'input_image': im_resized.tolist()}]}

    try:
        start = time.time()
        # sending post request to TensorFlow Serving server
        r = requests.post('http://localhost:8501/v1/models/east:predict', json=payload)
        logger.debug(r)
        pred = json.loads(r.content.decode('utf-8'))
        score = np.asarray(pred.get('predictions')[0].get('feature_fusion/Conv_7/Sigmoid'))
        geometry = np.asarray(pred.get('predictions')[0].get('feature_fusion/concat_3'))
        geometry = np.expand_dims(geometry,axis=0)
        score = np.expand_dims(score,axis=0)
        logger.debug(str(score.shape)+str(geometry.shape))

        timer['net'] = time.time() - start
        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))
        logger.debug(boxes)
        if boxes is not None:
            scores = boxes[:,8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        b = boxes[:,0,1]
        boxes = boxes[b.argsort()]
    except BaseException as e:
        print(e)
        return DetectionError,None
    return boxes,scores


def postprocessor(boxes,scores,text):
    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score, t in zip(boxes, scores,text):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            tl['text'] = t
            text_lines.append(tl)

    #后处理
    try:
        result = []
        Euclidean_distance = 10e10
        num1,num2,date,time,dis,fare = 0,0,0,0,0,0

        # 使用江苏的发票出错,因为江苏的发票代码长度为12位导致匹配错误.(可以把不符合规则的发票返回一个值表示发票不支持)
        point = [[t['x0'],t['y0'],t['x1'],t['y1'],t['x2'],t['y2'],t['x3'],t['y3']] for t in text_lines if re.search(r"\d{11}", t['text'])][0]
        x = np.mean(point[::2])
        y = np.mean(point[1::2])

        for t in text_lines:
            i = t['text']
            point = [t['x0'],t['y0'],t['x1'],t['y1'],t['x2'],t['y2'],t['x3'],t['y3']]

            if '¥' in i or '元' in i:
                temp = float(i.strip('¥').strip('元').strip('k').strip('m').replace(':', '.').replace('..', '.'))
                if temp > fare:
                    fare = float(math.ceil(temp))
            if re.search(r"(\d{4}-\d{2}-\d{2})", i) or re.search(r"(\d{2}-\d{2}-\d{2})", i):
                date = i
            if re.search(r"(\d{2}:\d{2}-\d{2}:\d{2})", i):
                time = re.search(r"(\d{2}:\d{2}-\d{2}:\d{2})", i).group()
            if re.search(r"^\d+\.\d{1}$", i) or re.search(r"^\d+\.\d{1}km$", i) or re.search(r"^\d+\.\d{1}m元$", i):
                dis = float(i.strip('元').strip('m').strip('k'))
            if re.search(r"\d{11}", i):
                try:
                    num1 = re.search(r"11100+\d{7}", i).group()
                except BaseException:
                    num1 = i
            if re.search(r"^\d{7}$", i) or re.search(r"^\d{8}$", i):
                if (np.mean(point[::2]) - x) ** 2 + (np.mean(point[1::2]) - y) ** 2 < Euclidean_distance:
                    Euclidean_distance = (np.mean(point[::2]) - x) ** 2 + (np.mean(point[1::2]) - y) ** 2
                    num2 = i

        result.append({'number':num1+num2})
        result.append({'number1': num1})
        result.append({'number2': num2})
        result.append({'date':date})
        result.append({'time':time})
        result.append({'distance':dis})
        result.append({'fare':fare})
    except BaseException as e:
        print(e)
        return PostProcessorError,None
    return result,text_lines


@functools.lru_cache(maxsize=100)
def get_predictor(crnn_path):
    logger.debug('loading model')
    from .eval import sort_poly
    from .CRNN.crnn_predict import GluonNet_v1

    crnn = GluonNet_v1(crnn_path,gpu_id=0)

    logger.debug('model loaded')
    def predictor(img):
        import time
        # Argument parser for giving input image_path from command line
        #image_path = img 
        # Preprocessing our input image
        #img = cv2.imread(image_path)
        timer = {}

        # 检测
        start = time.time()
        boxes,scores = detector(img)
        if str(boxes) == str(DetectionError):
            return DetectionError,None
        timer['detect'] = time.time() - start
        logger.debug('检测完成')

        # 矫正
        start = time.time()
        images = adjust(img,boxes)
        timer['adjust'] = time.time() - start

        # 识别
        start = time.time()
        try:
            images = images[:,:,:,::-1]
            images = np.transpose(images, (0, 3, 1, 2)) / 255
            text,_ = crnn.predict(images)
        except BaseException as e:
            print(e)
            return RecognitionError,None
        timer['recognition'] = time.time() - start

        start = time.time()
        result,text_lines = postprocessor(boxes,scores,text)
        if str(result) == str(PostProcessorError):
            return PostProcessorError,None
        timer['postprocess'] = time.time() - start
        
        ret = {
            'result':result,
            'text_lines': text_lines,
           # 'rtparams': rtparams,
            'timing': timer,
        }
        print(timer)
        logger.debug(ret)
        #ret.update(get_host_info())
        return OK,ret

    return predictor


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret

class Config:
    SAVE_DIR = 'static/results'

config = Config()
app = Flask(__name__)

CORS(app, resources=r'/*')
@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')

crnn_path = '/home/guoningyan/data/CRNN/crnn'

@app.route('/api/upload', methods=['POST'])
def upload_img():
    print ('get img!!!')
    global predictor
    import io
    bio = io.BytesIO()
    data = request.form.to_dict()
    username = data.get('username')
    file = request.files['file']
    file.save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    code,rst = predictor(img)
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

def main():
    global crnn_path
    global predictor
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8801, type=int)
    parser.add_argument('--crnn_path', default=crnn_path)
    args = parser.parse_args()
    crnn_path = args.crnn_path
    
    predictor = get_predictor(crnn_path)

    app.debug = False  # change this to True if you want to debug
    app.run('0.0.0.0', args.port)

if __name__ == '__main__':
    main()
