import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json, time, random, platform, hashlib, shutil, gc
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import psutil
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from IPython.display import display

SEED = 42
QUICK_RUN = False
USE_DRIVE = False
RUN_DEMO = True
SHARE_DEMO = False
IN_COLAB = 'COLAB_RELEASE_TAG' in os.environ
class_names = ['tree', 'cup', 'bicycle', 'apple', 'cat', 'car', 'house', 'fish', 'flower', 'umbrella']
SAMPLES_PER_CLASS = 150 if QUICK_RUN else 3000
EPOCHS = 1 if QUICK_RUN else 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
ROTATION = 0.05  # fraction of one full turn, so at most 18 degrees
TRANSLATION = 0.08

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = Path('/content/drive/MyDrive/DrawNet_results')
else:
    OUT = Path('DrawNet_results')
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'
DATA.mkdir(exist_ok=True)

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
set_seed()
tf.config.experimental.enable_op_determinism()
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except RuntimeError:
    pass
config = dict(seed=SEED, quick_run=QUICK_RUN, classes=class_names,
              samples_per_class=SAMPLES_PER_CLASS, epochs=EPOCHS,
              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE,
              rotation=ROTATION, translation=TRANSLATION)
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
environment = dict(python=platform.python_version(), tensorflow=tf.__version__,
                   platform=platform.platform(), cpu=platform.processor(),
                   logical_cpus=os.cpu_count(), gpu=[d.name for d in tf.config.list_physical_devices('GPU')])
(OUT / 'environment.json').write_text(json.dumps(environment, indent=2))
print(environment)
print('Mode:', 'PIPELINE CHECK' if QUICK_RUN else 'FULL EXPERIMENT')

def read_bitmaps(path):
    drawings = np.load(path, mmap_mode='r', allow_pickle=False)
    if drawings.ndim != 2 or drawings.shape[1] != 784 or drawings.dtype != np.uint8:
        raise ValueError(f'Unexpected Quick Draw bitmap format: {path.name}')
    return drawings


def load_bitmaps(name):
    path = DATA / f'{name}.npy'
    if path.exists():
        try:
            return read_bitmaps(path)
        except (OSError, ValueError):
            print(f'Replacing invalid cached data for {name}')
    url = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/' + quote(name) + '.npy'
    temporary = path.with_suffix('.part')
    for attempt in range(3):
        try:
            print(f'Downloading {name} ({attempt + 1}/3)', flush=True)
            with urlopen(url, timeout=60) as response, temporary.open('wb') as output:
                shutil.copyfileobj(response, output)
                length = response.headers.get('Content-Length')
                if length is not None and output.tell() != int(length):
                    raise OSError('Incomplete download')
            drawings = read_bitmaps(temporary)
            del drawings  # Release the memory map before renaming on Windows.
            temporary.replace(path)
            return read_bitmaps(path)
        except (OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise RuntimeError(f'Could not download valid Quick Draw data for {name}: {url}') from error
            time.sleep(2)


images, labels = [], []
seen = set()
audit = []
for label, name in enumerate(class_names):
    drawings = load_bitmaps(name)
    rng = np.random.default_rng(SEED + label)
    order = rng.permutation(len(drawings))
    selected = []
    rejected = 0
    for index in order:
        drawing = np.asarray(drawings[index], dtype=np.uint8).reshape(28, 28)
        key = hashlib.sha256(drawing.tobytes()).digest()
        if drawing.max() == 0 or key in seen:
            rejected += 1
            continue
        seen.add(key)
        selected.append(drawing.copy())
        if len(selected) == SAMPLES_PER_CLASS:
            break
    if len(selected) != SAMPLES_PER_CLASS:
        raise ValueError('Not enough unique nonblank drawings for ' + name)
    images.extend(selected)
    labels.extend([label] * len(selected))
    audit.append(dict(category=name, selected=len(selected), rejected=rejected))
X_uint8 = np.stack(images)
y = np.array(labels, dtype=np.int64)
del images, labels, seen
pd.DataFrame(audit).to_csv(OUT / 'data_audit.csv', index=False)
display(pd.DataFrame(audit))
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_uint8[np.flatnonzero(y == i)[0]], cmap='gray', vmin=0, vmax=255)
    ax.set_title(class_names[i])
    ax.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'sample_drawings.png', dpi=180)
plt.show()


all_indices = np.arange(len(y))
train_ids, remaining = train_test_split(all_indices, test_size=0.30, stratify=y, random_state=SEED)
val_ids, test_ids = train_test_split(remaining, test_size=0.50, stratify=y[remaining], random_state=SEED)
X = X_uint8.astype('float32')[..., np.newaxis] / 255.0
X_train, y_train = X[train_ids], y[train_ids]
X_val, y_val = X[val_ids], y[val_ids]
X_test, y_test = X[test_ids], y[test_ids]
assert not set(train_ids) & set(val_ids)
assert not set(train_ids) & set(test_ids)
assert not set(val_ids) & set(test_ids)
np.savez_compressed(OUT / 'splits.npz', X_train=X_train, y_train=y_train,
                    X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)
(OUT / 'classes.json').write_text(json.dumps(class_names))
print('Train:', X_train.shape, 'Validation:', X_val.shape, 'Test:', X_test.shape)


def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

set_seed()
initial_model = build_cnn()
initial_weights = initial_model.get_weights()
model_m0 = build_cnn()
model_m0.set_weights(initial_weights)
model_m0.summary()

def make_train_data(augmentation=None):
    ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    ds = ds.shuffle(len(X_train), seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    if augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=1)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    return ds.with_options(options).prefetch(1)

training_info = {}
def train_model(model, name, augmentation=None):
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoint = OUT / (name + '.keras')
    callback = tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor='val_loss', save_best_only=True)
    process = psutil.Process()
    rss_start = process.memory_info().rss / 1024**2
    start = time.perf_counter()
    history = model.fit(make_train_data(augmentation), epochs=EPOCHS,
                        validation_data=(X_val, y_val), callbacks=[callback], verbose=2)
    training_info[name] = dict(training_seconds=time.perf_counter() - start,
                               process_rss_start_MiB=rss_start,
                               process_rss_end_MiB=process.memory_info().rss / 1024**2,
                               best_validation_epoch=int(np.argmin(history.history['val_loss'])) + 1)
    (OUT / (name + '_history.json')).write_text(json.dumps(history.history, indent=2))
    return tf.keras.models.load_model(checkpoint), history.history

model_m0, history_m0 = train_model(model_m0, 'm0')


augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(ROTATION, fill_mode='constant', fill_value=0, seed=SEED),
    tf.keras.layers.RandomTranslation(TRANSLATION, TRANSLATION, fill_mode='constant', fill_value=0, seed=SEED + 1)
])
preview = augmentation(X_train[:8], training=True).numpy()
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for i in range(8):
    axes[0, i].imshow(X_train[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].imshow(preview[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    axes[1, i].axis('off')
axes[0, 0].set_title('Original')
axes[1, 0].set_title('Augmented')
plt.tight_layout()
plt.savefig(OUT / 'augmentation.png', dpi=180)
plt.show()
# Recreate after preview so preview does not change the training random state.
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(ROTATION, fill_mode='constant', fill_value=0, seed=SEED),
    tf.keras.layers.RandomTranslation(TRANSLATION, TRANSLATION, fill_mode='constant', fill_value=0, seed=SEED + 1)
])
set_seed()
model_m1 = build_cnn()
model_m1.set_weights(initial_weights)
assert model_m0.count_params() == model_m1.count_params()
model_m1, history_m1 = train_model(model_m1, 'm1', augmentation)
(OUT / 'training_info.json').write_text(json.dumps(training_info, indent=2))


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, history in [('M0', history_m0), ('M1', history_m1)]:
    epochs = range(1, len(history['loss']) + 1)
    axes[0].plot(epochs, history['accuracy'], label=name + ' train')
    axes[0].plot(epochs, history['val_accuracy'], '--', label=name + ' validation')
    axes[1].plot(epochs, history['loss'], label=name + ' train')
    axes[1].plot(epochs, history['val_loss'], '--', label=name + ' validation')
for ax, title in zip(axes, ['Accuracy', 'Loss']):
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / 'learning_curves.png', dpi=180)
plt.show()


def export_tflite(model, name, integer=False):
    # SavedModel route avoids Keras conversion differences.
    folder = OUT / (name + '_savedmodel')
    model.export(str(folder))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(folder))
    if integer:
        rng = np.random.default_rng(SEED)
        ids = rng.choice(len(X_train), min(500, len(X_train)), replace=False)
        def representative_data():
            for i in ids:
                yield [X_train[i:i+1].astype(np.float32)]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    result = converter.convert()
    path = OUT / (name + '.tflite')
    path.write_bytes(result)
    return path

paths = {'M0': export_tflite(model_m0, 'm0_float'),
         'M1': export_tflite(model_m1, 'm1_float'),
         'M2': export_tflite(model_m1, 'm2_int8', integer=True)}

def get_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    return interpreter

check = get_interpreter(paths['M2'])
assert check.get_input_details()[0]['dtype'] == np.int8
assert check.get_output_details()[0]['dtype'] == np.int8
assert all(t['dtype'] not in (np.float32, np.float16, np.float64) for t in check.get_tensor_details())
print('M2 has integer tensors and INT8 input/output.')


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

def benchmark(path):
    # Same runtime, one thread and batch size 1. Timing excludes Python preprocessing.
    process = psutil.Process()
    gc.collect()
    rss_before = process.memory_info().rss
    interpreter = get_interpreter(path)
    inp = interpreter.get_input_details()[0]
    samples = [prepare_input(x[np.newaxis], inp) for x in X_val[:min(100, len(X_val))]]
    for i in range(30):
        interpreter.set_tensor(inp['index'], samples[i % len(samples)])
        interpreter.invoke()
    times = []
    for i in range(300):
        interpreter.set_tensor(inp['index'], samples[i % len(samples)])
        start = time.perf_counter_ns()
        interpreter.invoke()
        times.append((time.perf_counter_ns() - start) / 1e6)
    rss_after = process.memory_info().rss
    return dict(latency_median_ms=float(np.median(times)), latency_p90_ms=float(np.percentile(times, 90)),
                process_rss_MiB=rss_after / 1024**2,
                observed_rss_change_MiB=(rss_after - rss_before) / 1024**2)

# Analytical MAC count for this exact architecture, batch size 1.
# Conv1: 28*28*16*(3*3*1), Conv2: 14*14*32*(3*3*16), Dense: 32*classes.
MACS = 28*28*16*9 + 14*14*32*3*3*16 + 32*len(class_names)
# Convention: FLOPs = 2 * MACs, excludes pooling, bias, activations and GAP.
rows, probabilities = [], {}
for name, path in paths.items():
    interpreter = get_interpreter(path)
    probabilities[name] = predict_lite(interpreter, X_test)
    predictions = probabilities[name].argmax(axis=1)
    np.save(OUT / (name + '_test_predictions.npy'), predictions)
    row = dict(model=name, accuracy=accuracy_score(y_test, predictions),
               macro_f1=f1_score(y_test, predictions, average='macro'),
               parameters=model_m0.count_params(), size_KiB=path.stat().st_size/1024,
               MACs=MACS, FLOPs_2xMACs=2*MACS)
    row.update(benchmark(path))
    rows.append(row)
results = pd.DataFrame(rows).set_index('model')
results.to_csv(OUT / 'results.csv')
display(results)
print('Memory is whole-process RSS in this notebook, NOT an isolated model peak or tensor arena measurement.')
print('RSS changes can be affected by allocator reuse. MAC/FLOP counts are analytical, not measured CPU instructions.')
print('INT8 changes representation, not the architecture parameter count or theoretical MAC count.')
# Confirm float export predictions are close to Keras on validation data.
for name, model in [('M0', model_m0), ('M1', model_m1)]:
    lite = predict_lite(get_interpreter(paths[name]), X_val[:32])
    keras = model(X_val[:32], training=False).numpy()
    np.testing.assert_allclose(lite, keras, atol=1e-4, rtol=1e-3)
print('Float export parity checks passed.')


fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ['#7b71c7', '#38a6a0', '#d38c4c']
for ax, metric, title in zip(axes, ['accuracy', 'size_KiB', 'latency_median_ms'],
                            ['Test accuracy', 'File size (KiB)', 'CPU invoke latency (ms)']):
    ax.bar(results.index, results[metric], color=colors)
    ax.set_title(title)
plt.tight_layout()
plt.savefig(OUT / 'model_comparison.png', dpi=180)
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(19, 5))
for ax, name in zip(axes, paths):
    prediction = probabilities[name].argmax(axis=1)
    matrix = confusion_matrix(y_test, prediction, labels=range(len(class_names)))
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(ax=ax, colorbar=False, xticks_rotation=90)
    ax.set_title(name)
    report = classification_report(y_test, prediction, labels=range(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    (OUT / (name + '_classification_report.json')).write_text(json.dumps(report, indent=2))
plt.tight_layout()
plt.savefig(OUT / 'confusion_matrices.png', dpi=180)
plt.show()
wrong = np.flatnonzero(probabilities['M1'].argmax(axis=1) != y_test)[:12]
if len(wrong):
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(X_test[i, :, :, 0], cmap='gray')
        ax.set_title('True: ' + class_names[y_test[i]] + '\nPred: ' + class_names[probabilities['M1'][i].argmax()])
    plt.tight_layout()
    plt.savefig(OUT / 'm1_errors.png', dpi=180)
    plt.show()


a0, a1, a2 = results['accuracy']
gain = a1 - a0
size_reduction = 100 * (1 - results.loc['M2', 'size_KiB'] / results.loc['M1', 'size_KiB'])
# Compare integer counts and file bytes so exact threshold cases do not depend on rounding.
correct = {name: int(np.count_nonzero(scores.argmax(axis=1) == y_test))
           for name, scores in probabilities.items()}
gain_count = correct['M1'] - correct['M0']
h1 = 100 * gain_count >= len(y_test)
h2 = (5 * paths['M2'].stat().st_size <= 2 * paths['M1'].stat().st_size
      and gain_count > 0 and 2 * (correct['M2'] - correct['M0']) >= gain_count)
summary = dict(H1_supported_this_run=h1, H2_supported_this_run=h2,
               M1_gain_percentage_points=float(100*gain),
               M2_accuracy_change_vs_M1_percentage_points=float(100*(a2-a1)),
               size_reduction_percent=float(size_reduction), quick_run=QUICK_RUN)
(OUT / 'hypotheses.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
print('Exploratory targets: H1 +1 percentage point; H2 >=60% smaller and retains >=half of positive M1 gain.')
print('One training seed is not proof of statistical significance. Repeat with predeclared seeds for stronger evidence.')


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


if RUN_DEMO:
    if IN_COLAB and not SHARE_DEMO:
        print('To launch in hosted Colab, set SHARE_DEMO=True and rerun this cell.')
    else:
        demo = build_demo()
        demo.launch(share=SHARE_DEMO, prevent_thread_lock=True)


# The archive contains models, figures, configuration, splits and real results.
# Raw downloaded class files are excluded to keep it manageable.
archive_dir = OUT / 'export'
archive_dir.mkdir(exist_ok=True)
for path in OUT.iterdir():
    if path.is_file() and path.suffix in ['.keras', '.tflite', '.json', '.csv', '.png', '.npz', '.npy']:
        shutil.copy2(path, archive_dir / path.name)
archive = shutil.make_archive(str(OUT / 'DrawNet_results'), 'zip', archive_dir)
print('Saved:', archive)
if IN_COLAB:
    from google.colab import files
    files.download(archive)

if RUN_DEMO and not IN_COLAB:
    demo.block_thread()
