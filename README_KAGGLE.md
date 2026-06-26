Kaggle Run Instructions for BEVPlace++

1) Prepare datasets on Kaggle:
   - Upload the repository files (or a zip) as a dataset (name it e.g. `project-repo`).
   - Upload the teacher model `model_best.pth.tar` as a dataset (name it e.g. `teacher-model`).
   - Optionally upload a small subset of BEV images as `bev-dataset` for fast smoke-testing.

2) In Kaggle Kernel, add the uploaded datasets via Add Data.

3) Use the provided `kaggle_run.ipynb` notebook to install deps and run a 1-epoch smoke-test.

Notes:
- The notebook expects the teacher model at `/kaggle/input/<teacher-dataset>/model_best.pth.tar`.
- `train_distill.py` now auto-searches `/kaggle/input/` for `model_best.pth.tar` if the specified `--teacher_path` does not exist.
- For CPU runs, ensure `faiss-cpu` and CPU PyTorch are installed (not GPU builds). See the notebook for commands.
