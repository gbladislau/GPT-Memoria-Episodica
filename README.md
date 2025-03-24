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
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install dependencies in a Virtual Enviroment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Prepare your dataset:**
   - You can use your own dataset or **NarrativeQA**.
   - NarrativeQA is fully available on HuggingFaces at:
     https://huggingface.co/datasets/deepmind/narrativeqa
4. **Run the LLM with episodic memory:**
   ```bash
   python app.py
   ```

### App Script


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
The paper is available in this repository: [Paper Title](./Episodic_Memory_in_Large_Language_Models.pdf)

