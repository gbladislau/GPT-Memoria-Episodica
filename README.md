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
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare your dataset:**
   - You can use your own dataset or **NarrativeQA**.
   - NarrativeQA is available on HuggingFaces at: 
     https://huggingface.co/datasets/deepmind/narrativeqa
4. **Run the LLM with episodic memory:**
   ```bash
   python app.py
   ```
5. **Evaluate the results:**
   ```bash
   python evaluate.py
   ```

### You can check our paper: 
