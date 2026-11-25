# DrawNet drawing demo. Run after training, or use the packaged trained_results folder.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

OUT = Path(__file__).parent / 'trained_results'
if not OUT.exists():
    OUT = Path(__file__).parent / 'DrawNet_results'
if not (OUT / 'classes.json').exists():
    raise FileNotFoundError('Run the notebook first, or extract the full package with trained_results.')
class_names = json.loads((OUT / 'classes.json').read_text())
paths = {'M0': OUT / 'm0_float.tflite', 'M1': OUT / 'm1_float.tflite', 'M2': OUT / 'm2_int8.tflite'}
results = pd.read_csv(OUT / 'results.csv', index_col='model')
RUN_DEMO = True
SHARE_DEMO = False

def get_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter

def prepare_input(x, detail):
    if np.issubdtype(detail['dtype'], np.integer):
        scale, zero = detail['quantization']
        if scale <= 0:
            raise ValueError('Invalid input quantization scale')
        limits = np.iinfo(detail['dtype'])
        return np.clip(np.round(x / scale + zero), limits.min, limits.max).astype(detail['dtype'])
    return x.astype(detail['dtype'])

def read_output(values, detail):
    if np.issubdtype(detail['dtype'], np.integer):
        scale, zero = detail['quantization']
        return (values.astype(np.float32) - zero) * scale
    return values

def predict_lite(interpreter, data):
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    predictions = []
    for x in data:
        interpreter.set_tensor(inp['index'], prepare_input(x[np.newaxis], inp))
        interpreter.invoke()
        predictions.append(read_output(interpreter.get_tensor(out['index'])[0], out))
    return np.array(predictions)

def preprocess_drawing(editor_value):
    if editor_value is None:
        raise ValueError('Draw an object first.')
    picture = editor_value.get('composite') if isinstance(editor_value, dict) else editor_value
    if picture is None:
        raise ValueError('Draw an object first.')
    image = Image.fromarray(np.asarray(picture).astype('uint8')) if not isinstance(picture, Image.Image) else picture
    rgba = image.convert('RGBA')
    white = Image.new('RGBA', rgba.size, 'white')
    white.alpha_composite(rgba)
    # Quick Draw bitmaps have light ink on a dark background.
    ink = ImageOps.invert(white.convert('RGB').convert('L'))
    array = np.asarray(ink)
    points = np.argwhere(array > 30)
    if len(points) < 5:
        raise ValueError('The canvas is empty.')
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    ink = ink.crop((int(left), int(top), int(right), int(bottom)))
    scale = 24 / max(ink.size)
    size = (max(1, round(ink.width*scale)), max(1, round(ink.height*scale)))
    ink = ink.resize(size, Image.Resampling.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(ink, ((28-size[0])//2, (28-size[1])//2))
    return np.asarray(canvas, dtype=np.float32)[..., None] / 255.0

demo_interpreters = {name: get_interpreter(path) for name, path in paths.items()}
def recognize_drawing(value):
    import gradio as gr
    try:
        image = preprocess_drawing(value)
    except ValueError as error:
        raise gr.Error(str(error))
    predictions = []
    for name in ['M0', 'M1', 'M2']:
        scores = predict_lite(demo_interpreters[name], image[np.newaxis])[0]
        predictions.append({label: float(score) for label, score in zip(class_names, scores)})
    return *predictions, image[:, :, 0]

if RUN_DEMO:
    import gradio as gr
    with gr.Blocks(title='DrawNet') as demo:
        gr.Markdown('# DrawNet\nDraw an object and compare three small neural network models.')
        gr.Markdown('Classes: ' + ', '.join(class_names))
        with gr.Row():
            sketch = gr.Sketchpad(type='numpy', image_mode='RGBA', canvas_size=(360, 360),
                                 brush=gr.Brush(colors=['#000000'], color_mode='fixed', default_size=8),
                                 label='Draw here')
            with gr.Column():
                outputs = [gr.Label(num_top_classes=3, label=name) for name in ['M0 Baseline', 'M1 Augmentation', 'M2 INT8']]
                processed = gr.Image(label='28 by 28 model input', image_mode='L')
        button = gr.Button('Recognize', variant='primary')
        button.click(recognize_drawing, inputs=sketch, outputs=outputs+[processed], concurrency_limit=1)
        gr.Dataframe(value=results.reset_index(), interactive=False, label='Measured results')
    demo.launch(share=SHARE_DEMO)
