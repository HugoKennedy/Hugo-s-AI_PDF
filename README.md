# RAG GUI Generative AI Application
Full Stack RAG Application Designed and Built with Warp Agentic Development Environment

## 🎥 Video Tutorial
This repository is based on a video tutorial, where I share how this app was built step by step
<a href="https://youtu.be/bPNmmDPyGzk"><img width="600" alt="Agentic RAG App Template" src="https://github.com/user-attachments/assets/ce7c141b-7428-4f85-930c-4de9998451f0"/></a>
<br>
Watch it here: https://youtu.be/bPNmmDPyGzk

## 💻 Run Application

1. Clone Application:
```
git clone https://github.com/MariyaSha/RAG_GUI_GenAIApp.git
cd RAG_GUI_GenAIApp
```

2. Setup Environment:
```
conda create -n rag_env python=3.12
conda activate rag_env
```

3. Install Dependancies:
```
pip install -U "langchain>=0.2" "langchain-community>=0.2" langchain-core langchain-huggingface transformers torch sentence-transformers faiss-cpu pypdf flask
```

4. Run app:
```
python vectorize.py
python app.py
```

5. Enjoy! 🙂
