# DrawNet

Team: Yara Mohammad, Lama Alturki, Amal Shuaileh.

Complete student-style implementation for CS4085.

[Open in Colab](https://colab.research.google.com/github/iiYARA/CS4085-Deep-Learning/blob/main/Project/DrawNet/DrawNet.ipynb)

## Run the full experiment in Google Colab

1. Open `DrawNet.ipynb` with the link above. Start a fresh Python runtime if the old one has already imported a different TensorFlow version. A GPU is optional.
2. Choose **Runtime → Run all**. The setup cell reuses compatible Colab packages and installs Gradio if needed.
3. Leave `QUICK_RUN = False` for the experiment. `QUICK_RUN = True` uses 150 drawings per class and one epoch for a pipeline check. Keep separate result folders when running both modes.
4. Enable `USE_DRIVE` before running to keep files in your own Google Drive, or download the ZIP offered by the last cell before disconnecting.

The first run downloads ten official Quick Draw bitmap files (about 1 GB) into `DrawNet_results/data`. A valid cache is reused on subsequent runs. The final ZIP contains trained checkpoints, TFLite models, plots, splits, and measured results. It excludes the raw bitmap downloads.

## Models
- M0: two Conv2D blocks (16 and 32 filters), global average pooling and softmax.
- M1: exactly the same network and starting weights, with training-only rotation and translation.
- M2: trained M1 converted to full INT8 TFLite, using training-only calibration.

Classes are tree, cup, bicycle, apple, cat, car, house, fish, flower and umbrella.

## Fair experiment
The notebook samples 3,000 unique nonblank bitmaps per class, then performs a stratified 70/15/15 split. Exact duplicates are removed globally before splitting. Near-duplicates and artist overlap are not ruled out. M0/M1 share training budget, data order and initial weights. Best checkpoints use validation loss. Test data is evaluated after both models are fixed.

Augmentation settings are fixed defaults (rotation factor 0.05 and translation 0.08). Review or tune them using validation only before final testing. The notebook does not automatically search settings or promise improved accuracy. Running the default configuration once does not establish statistical significance. Repeat a predeclared set of seeds for a stronger research claim.

## Measurements
- Test accuracy, macro F1 and classification reports.
- Float TFLite M0/M1 and INT8 M2 sizes (KiB).
- Same CPU TFLite runtime, one thread, batch one, 30 warm-up invokes, 300 timed invokes. Median and p90 exclude preprocessing, file loading and network time.
- Analytical convolution/dense MACs; FLOPs = twice MACs, excluding pooling, bias, activation and global-average operations. This is not an instruction-level measurement.
- Process RSS and observed RSS change: whole-process measurements, not isolated model peak RAM. Allocator reuse affects them; do not present them as a precise model memory footprint.
- One architecture means identical parameter counts and theoretical MACs after quantization.

Exploratory H1: M1 gains >=1 percentage point. H2: M2 is >=60% smaller than M1 float and retains at least half of M1's positive gain over M0. These are targets, not guaranteed outcomes.

## Run locally or open the demo without training

Use CPython 3.12 or 3.13. From `Project/DrawNet`, create a virtual environment and use its Python to install and run the app.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

**macOS or Linux**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

On Windows, use a short path for the virtual environment if pip reports a long-path error while unpacking TensorFlow.

Open the local URL printed by Gradio. Draw one of the ten classes or click **New challenge** for a random prompt. **Compare guesses** shows each model’s top three predictions and a 28×28 preview of its input. Challenge mode counts how many models guessed the prompt. Use **Clear drawing** to try again and expand **Measured test results** to see the saved metrics.

`python app.py` uses the repository's packaged models. To use a new experiment, extract the notebook's results ZIP and pass its directory to the app. Use the same virtual environment Python as above.

```sh
python app.py --results-dir DrawNet_results
```

The directory must contain `m0_float.tflite`, `m1_float.tflite`, `m2_int8.tflite`, `classes.json`, and `results.csv` from the same run. The app loads those artifacts directly, with no dataset download or training.

To run the notebook locally, install JupyterLab with `python -m pip install jupyterlab`, start it with `python -m jupyterlab`, then open the notebook and run all cells. Alternatively use `python drawnet.py`. For a script run that exits after exporting, set `RUN_DEMO = False` in the script. In hosted Colab, set `SHARE_DEMO = True` in the setup cell for a temporary public Gradio URL. By default, hosted Colab skips launch and the exported models can be used in the local demo.

## Attribution and AI use
Dataset: Google Creative Lab Quick Draw, https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0 drawings).
Implementation references are listed at the end of the notebook. The code was drafted with ChatGPT assistance. Students should understand, run and review it and disclose this assistance according to the course instructions.
