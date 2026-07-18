# WORKFLOW — saving Colab work so outputs + code persist

**Core fact:** a notebook's output cells are stored *inside* the `.ipynb`. Saving/committing the notebook
after a run is what preserves the outputs. Nothing extra needed.

## Every time you run a notebook in Colab
Pick one:
- **Colab → File → Save a copy in GitHub** — commits the `.ipynb` *with outputs* to your repo. Use path
  `colab/<name>.ipynb`. (Best: versioned + shareable.)
- **File → Download → Download .ipynb** — local copy with outputs; commit it from your Mac.

## One-time repo setup (do once)
```bash
cd ~/dev
cp -R "~/Documents/Obsidian Vault/Career/Projects/triton-serving-lab" .
cd triton-serving-lab
git init && git add . && git commit -m "Triton serving lab"
# make an empty repo 'triton-serving-lab' on github.com, then:
git remote add origin https://github.com/ViSaReVe/triton-serving-lab.git
git push -u origin main
```

## What goes in git vs not
- **In git:** code, configs (`config.pbtxt`), notebooks (with outputs), README, results tables/CSVs, small `.onnx` (~2 MB).
  - keep the onnx: `git add -f model_repository/audio_cnn/1/model.onnx`
- **NOT in git** (already in `.gitignore`): `*.pt` checkpoints, datasets, `__pycache__`. Large files → Google Drive.

## Save big artifacts from Colab to Drive
```python
from google.colab import drive
drive.mount('/content/drive')
!cp audio_cnn.onnx "/content/drive/MyDrive/triton-serving-lab/"
```

## Suggested repo layout
```
triton-serving-lab/
  colab/            # notebooks WITH outputs (Step1 export, Step3 benchmark, ...)
  model.py  export_onnx.py  client.py  benchmark.py
  model_repository/ # config.pbtxt (+ optional committed model.onnx)
  README.md  LEARN.md  WORKFLOW.md
  results/          # benchmark tables/CSVs you generate
```
