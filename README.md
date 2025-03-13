# Chinese Cultural Bias In LLMs

To install the necessary packages, run the following:

    pip install -r requirements.txt

Our best models after hyperparameter optimization are in the 'models' folder. To run the WVS score SFT, from within `WVS Score/finetune` run

    python SFT.py

To evaluate the WVS score of a model, from within `WVS Score/evaluate` run 

    python evaluate_WVS_score.py --qa_file [path to JSONL data file] --model_type ['huggingface' or 'local'] --model_path [model path]
