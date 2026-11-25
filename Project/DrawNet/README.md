# DrawNet

Team: Yara Mohammad, Lama Alturki, Amal Shuaileh.

Complete student-style implementation for CS4085.

[Open in Colab](https://colab.research.google.com/github/iiYARA/CS4085-Deep-Learning/blob/main/Project/DrawNet/DrawNet_Complete.ipynb)

## Easiest way: Google Colab
1. Open https://colab.research.google.com/ and choose Upload notebook.
2. Upload DrawNet_Complete.ipynb.
3. Optional: Runtime > Change runtime type > GPU.
4. Run the installation cell first, then run the remaining cells in order.
5. Keep QUICK_RUN=False for the full experiment. QUICK_RUN=True uses 150 images per class and one epoch to check the pipeline only.
6. Set USE_DRIVE=True to persist results in your own Google Drive.
7. The last cell downloads a ZIP of models, measured results and charts.

The first run downloads the full bitmap file for each selected category (roughly 1 GB total). Downloads are cached in DrawNet_results/data. CPU training also works, but can take longer. Pinning the dependencies helps reproducibility; a runtime restart can be needed after the installation cell.

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

## Demo
The Gradio interface accepts black drawings on a white canvas and compares the three model predictions. It shows the processed 28x28 input and the measured results table. Predictions are limited to the ten classes; it does not detect unknown objects.
RUN_DEMO=True starts the interface. In Colab, set SHARE_DEMO=True if you need a temporary external URL. This publishes a temporary demo only when the user runs the cell with that option. This package is not a permanently hosted website.

## Run the packaged trained demo
From this folder, install requirements.txt and run `python app.py`. The app reads trained_results and does not retrain the models. It starts a local URL. For a temporary external link change SHARE_DEMO to True in app.py.

## Local Python
Use Python 3.10-3.12. Install requirements.txt, then run python drawnet.py. Adjust the settings at the top. With RUN_DEMO=False the script runs all experiments and exports the results without serving the interface.

## Attribution and AI use
Dataset: Google Creative Lab Quick Draw, https://github.com/googlecreativelab/quickdraw-dataset (CC BY 4.0 drawings).
Implementation references are listed at the end of the notebook. The code was drafted with ChatGPT assistance. Students should understand, run and review it and disclose this assistance according to the course instructions.
