# DrawNet drawing demo. Run after training, or use the packaged trained_results folder.
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps

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
    return (*predictions, (image[:, :, 0] * 255).round().astype(np.uint8))

def build_demo():
    import secrets
    import gradio as gr

    def blank_canvas():
        return Image.new('RGB', (360, 360), 'white')

    def clear_drawing():
        return blank_canvas(), None, None, None, None, ''

    def new_challenge(previous):
        target = secrets.choice([name for name in class_names if name != previous])
        return target, f'### Draw **{target}**', *clear_drawing()

    def compare_drawing(value, target):
        outputs = recognize_drawing(value)
        if not target:
            return (*outputs, '')
        matches = [name for name, scores in zip(('M0', 'M1', 'M2'), outputs[:3])
                   if max(scores, key=scores.get) == target]
        picked = ', '.join(matches)
        feedback = (f'**{len(matches)} of 3 models guessed {target}.** {picked}'
                    if matches else f'**0 of 3 models guessed {target}.** Try another sketch.')
        return (*outputs, feedback)

    css = """
    #drawnet-studio { max-width: 1080px; margin: 0 auto; }
    #drawnet-heading { padding: 1.05rem 1.25rem; border-radius: 16px;
        background: linear-gradient(120deg, #251c57, #15485a);
        color: #fff; }
    #drawnet-heading h1 { color: #fff; margin: 0 0 .3rem; }
    #drawnet-heading p { color: #eef1ff; margin: 0; }
    #drawnet-workspace { display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
        align-items: start; }
    #drawnet-predictions { gap: .8rem; }
    #drawnet-actions { flex-wrap: wrap; }
    #drawnet-actions button { min-width: 105px; }
    @media (max-width: 740px) {
        #drawnet-workspace { grid-template-columns: minmax(0, 1fr); }
    }
    """
    theme = gr.themes.Soft(primary_hue='indigo', neutral_hue='slate')
    with gr.Blocks(title='DrawNet', theme=theme, css=css) as demo:
        with gr.Column(elem_id='drawnet-studio'):
            gr.Markdown('# DrawNet\nSketch a class and compare three model guesses.',
                        elem_id='drawnet-heading')
            with gr.Row(elem_id='drawnet-workspace'):
                with gr.Column():
                    target = gr.State('')
                    prompt = gr.Markdown('### Free draw')
                    with gr.Row():
                        challenge = gr.Button('New challenge', variant='secondary')
                    sketch = gr.Sketchpad(
                        value=blank_canvas(), type='numpy', image_mode='RGBA',
                        canvas_size=(360, 360), height=375, layers=False,
                        show_download_button=False, show_fullscreen_button=False,
                        brush=gr.Brush(colors=['#000000'], color_mode='fixed', default_size=8),
                        label='Drawing canvas',
                    )
                    with gr.Row(elem_id='drawnet-actions'):
                        recognize = gr.Button('Compare guesses', variant='primary')
                        clear = gr.Button('Clear drawing', variant='secondary')
                    gr.Markdown('Classes · ' + ', '.join(class_names))
                with gr.Column(elem_id='drawnet-predictions'):
                    verdict = gr.Markdown()
                    predictions = [
                        gr.Label(num_top_classes=3, label=label)
                        for label in ('M0 · Baseline CNN', 'M1 · Augmented CNN', 'M2 · Full INT8')
                    ]
                    processed = gr.Image(label='28 × 28 model input', image_mode='L',
                                         height=150, show_download_button=False)
            with gr.Accordion('Measured test results', open=False):
                gr.Dataframe(value=results.reset_index(), interactive=False, label='Metrics')
        drawing_outputs = [sketch, *predictions, processed, verdict]
        recognize.click(compare_drawing, inputs=[sketch, target],
                        outputs=[*predictions, processed, verdict],
                        concurrency_limit=1, api_name='recognize')
        clear.click(clear_drawing, outputs=drawing_outputs, api_name='clear')
        challenge.click(new_challenge, inputs=target, outputs=[target, prompt, *drawing_outputs],
                        api_name='challenge')
    return demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the DrawNet demo with trained models.')
    parser.add_argument('--results-dir', type=Path,
                        help='Folder containing the three TFLite models, classes.json and results.csv.')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    packaged = project / 'trained_results'
    folder = args.results_dir
    if folder is None:
        folder = packaged if packaged.is_dir() else project / 'DrawNet_results'
    paths = {'M0': folder / 'm0_float.tflite', 'M1': folder / 'm1_float.tflite',
             'M2': folder / 'm2_int8.tflite'}
    required = [*paths.values(), folder / 'classes.json', folder / 'results.csv']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Missing saved files: ' + ', '.join(missing))
    class_names = json.loads((folder / 'classes.json').read_text(encoding='utf-8'))
    results = pd.read_csv(folder / 'results.csv', index_col='model')
    demo_interpreters = {name: get_interpreter(path) for name, path in paths.items()}
    demo = build_demo()
    demo.launch()
