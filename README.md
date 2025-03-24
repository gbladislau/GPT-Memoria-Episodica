# 🧠 Episodic Memory for Generative Pre-Trained Transformers

## 👨‍💻 Developers
- Gabriel Braga Ladislau
- Guilherme Silveira Gomes Brotto
- Marlon Moratti de Amaral  

## 🎯 Objective
The goal of this project is to enhance Generative Pre-Trained Transformers (GPTs) with **episodic memory**, allowing them to recall specific facts from past interactions. We implement a memory system that stores and retrieves information, enabling the GPT to provide **more contextually relevant responses** based on previous exchanges.

## 🛠️ Methodology
We use **two pre-trained LLMs**: one with an episodic memory system and another without. Both models are fed with factual data, and later, we prompt them with questions about these facts. The answers are then **compared against expected results** using **Sentence-BERT (SBERT) and cosine similarity** to measure accuracy and relevance.

The episodic memory system is built using **ChromaDB** as a database for storing and retrieving facts efficiently.

## 💻 Technologies
- **ChromaDB** – for storing episodic memory
- **PyTorch** – for working with LLMs
- **Sentence-BERT (SBERT)** – for evaluating generated answers
- **NarrativeQA** – as an optional dataset for testing

## 🚀 How to Use

### 🧑‍💻 Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
### 🔧 Install dependencies in a Virtual Enviroment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
### 🗂️ Prepare your dataset (for inference mode only):
   - You can use your own dataset or **NarrativeQA**.
   - NarrativeQA is fully available on HuggingFace at [NarrativeQA](https://huggingface.co/datasets/deepmind/narrativeqa).

### 🤖 Choose your model
   - To use different models from HuggingFace you need to have all their dependencies previously installed.
   - By default all of `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` dependencies are in our [requirements](./requirements.txt).
   - For the SBERT models all of them under [Sentence Transformers](https://huggingface.co/sentence-transformers).

### 🦙 Running the LLM

The `app.py` script allows you to launch the LLM and interact with it, with or without the episodic memory module.

In chat mode, you can exit by typing `exit`. If you want to exit without saving any data to the memory module, simply type `exit_quiet`.

```text
usage: app.py [-h] [-m MODEL] [-r REFLECTION_PROMPT] [--results RESULTS] [-e] [-v] [-s SBERT] [-i]

Run the LLM and begin your conversation

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model's Name
  -r REFLECTION_PROMPT, --reflection_prompt REFLECTION_PROMPT
                        Reflection Prompt Template
  --results RESULTS     Results output path
  -e, --episodic        Run the LLM with the episodic memory module
  -v, --verbose         Verbose
  -s SBERT, --sbert SBERT
                        SBERT Model to evaluate the similarity scores
  -i, --inference_mode  Use inference mode instead of chat mode
```

#### Arguments Explained:
- `-m, --model`: Specifies the name of the model to be used.
- `-r, --reflection_prompt`: Defines the reflection prompt template to be used.
- `--results`: Specifies the file path where results should be saved.
- `-e, --episodic`: Runs the LLM with the episodic memory module enabled.
- `-v, --verbose`: Enables verbose mode for additional logging and debugging information.
- `-s, --sbert`: Specifies the SBERT model to be used for evaluating similarity scores.
- `-i, --inference_mode`: Runs the model in inference mode instead of interactive chat mode.

#### Example Usage:
To run the model in standard chat mode:
```bash
python app.py -m hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

To run the model with episodic memory enabled:
```bash
python app.py -e -m hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
```

To specify a reflection prompt template:
```bash
python app.py -r prompts/reflection_prompt_template.txt
```

To use a specific SBERT model for evaluation:
```bash
python app.py -s all-mpnet-base-v2
```

To save results to a specific file:
```bash
python app.py --results results/results.json
```


### 📊 Evaluation Script
The evaluation script (`evaluate.py`) includes several command-line options:

```text
usage: evaluate.py [-h] [--plot] [--dont_rerun] [--gen_prompt] [--result RESULT] [--prompt PROMPT]

Run the evaluation module to calculate the LLMs results

options:
  -h, --help       show this help message and exit
  --plot, -p       Generate plot for quantitativeanalysis
  --dont_rerun     Use the mem.npy and no_mem.npyscores previously calculated
  --gen_prompt     Generate qualitative prompt
  --result RESULT  LLM anwsers results input file path (json)
  --prompt PROMPT  Prompt file output path
```

#### Arguments Explained:
- `--plot, -p`: Generates a boxplot comparing the similarity scores of the memory-enabled and memoryless models.
- `--dont_rerun`: Uses precomputed scores stored in `mem.npy` and `no_mem.npy` instead of recalculating them.
- `--gen_prompt`: Generates a qualitative prompt for evaluation.
- `--result`: Specifies the path to the JSON file containing both LLM generated anwsers (default: `results/results.json`).
- `--prompt`: Specifies the output path for the qualitative prompt file (default: `prompts/evaluation_prompt.txt`).

To run the evaluation with a plot:
```bash
python evaluate.py --plot
```

To use previously calculated scores:
```bash
python evaluate.py --dont_rerun
```

To generate a qualitative prompt:
```bash
python evaluate.py --gen_prompt
```

## 📄 You can check our paper:
The paper is available in this repository: [ Episodic Memory in Large Language Models](./Episodic_Memory_in_Large_Language_Models.pdf)

