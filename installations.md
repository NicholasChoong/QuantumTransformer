# Installations

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --user

pip install pennylane --upgrade

pip install -U scikit-learn tqdm torchtext torchdata seaborn ipykernel 'portalocker>=2.0.0' nbconvert papermill

pip install tensorcircuit tensorflow==2.14.0

pip install datasets transformers accelerate

pip install -U qiskit cirq pylatexenc

conda activate F:\web-dev\qt



python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print([torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())])"


pip install tensorflow[and-cuda]

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -U pennylane scikit-learn tqdm seaborn ipykernel nbconvert papermill

pip install tensorcircuit

pip install datasets transformers accelerate

pip install -U qiskit cirq pylatexenc














```
