QA_PAIRS = [
    {
        "question": "What are the three main types of machine learning?",
        "reference": "The three main types of machine learning are Supervised Learning, Unsupervised Learning, and Reinforcement Learning."
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": "Overfitting occurs when a model learns the training data too well, including its noise and outliers, making it perform poorly on new, unseen data."
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": "This is the balance between the error introduced by approximating a real-world problem (bias) and the error introduced by the model's sensitivity to small fluctuations in the training set (variance)."
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": "Techniques like L1 and L2 regularization prevent overfitting by adding a penalty term to the loss function based on the magnitude of the model's weights."
    },
    {
        "question": "What is cross-validation?",
        "reference": "A technique to evaluate the performance of a model by partitioning the data into multiple subsets and training/testing the model several times."
    },
    {
        "question": "What is backpropagation?",
        "reference": "The primary algorithm for training neural networks, calculating the gradient of the loss function with respect to each weight."
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": "Convolutional Neural Networks are primarily used for image recognition and computer vision tasks."
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": "LSTM networks use a gating mechanism to control the flow of information, addressing the vanishing gradient problem in RNNs."
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": "Common activation functions include ReLU, Sigmoid, and Tanh."
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": "Pooling layers reduce the spatial dimensions of the input, decreasing the number of parameters and computation."
    },
    {
        "question": "What is the transformer architecture?",
        "reference": "The transformer architecture relies entirely on self-attention mechanisms without using recurrence or convolutions."
    },
    {
        "question": "What are word embeddings?",
        "reference": "Numerical representations of words in a continuous vector space where semantically similar words are nearby."
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": "A technique where a model developed for one task is reused as the starting point for a model on a related task."
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": "BERT is trained on both left and right context simultaneously to handle language understanding."
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": "A mechanism that allows each position in a sequence to attend to all other positions."
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": "Generative Pre-trained Transformer is trained using a causal language modeling objective to predict the next token."
    },
    {
        "question": "What is instruction tuning?",
        "reference": "Fine-tuning an LLM on a dataset of (instruction, output) pairs to improve instruction following."
    },
    {
        "question": "What is RLHF?",
        "reference": "Reinforcement Learning from Human Feedback is used to align LLMs with human preferences."
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": "A technique that encourages the LLM to generate intermediate reasoning steps."
    },
    {
        "question": "What is the context length of GPT-4?",
        "reference": "GPT-4 supports context lengths from 8k up to 128k tokens in the Turbo versions."
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": "RAG combines LLMs with external knowledge retrieval for more accurate answers."
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": "A typical RAG pipeline includes a retriever, a prompt, and an LLM."
    },
    {
        "question": "What is dense retrieval?",
        "reference": "A retrieval method that uses dense vector embeddings and similarity search."
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": "Chunking is important to fit the LLM's context window and improve retrieval relevance."
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": "Techniques like query expansion, reranking, and hybrid search."
    },
    {
        "question": "What are vector databases used for?",
        "reference": "They are designed to store and search through high-dimensional vector embeddings."
    },
    {
        "question": "What is FAISS?",
        "reference": "Facebook AI Similarity Search is a library for efficient similarity search of dense vectors."
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": "By mapping text to a vector space where distance represents semantic similarity."
    },
    {
        "question": "What is HNSW?",
        "reference": "Hierarchical Navigable Small World is an algorithm for efficient approximate nearest neighbor search."
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": "Combining vector search with traditional keyword search."
    },
    {
        "question": "What is LangChain?",
        "reference": "A framework for building applications powered by large language models."
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": "A declarative way to compose chains and runnables."
    },
    {
        "question": "What is LangGraph?",
        "reference": "A library for building stateful, multi-agent applications with LLMs."
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": "Supports various types like ConversationBufferMemory and ConversationSummaryMemory."
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": "Interfaces that return documents given an unstructured query."
    },
    {
        "question": "What is LangSmith?",
        "reference": "A platform for debugging, testing, evaluating, and monitoring LLM applications."
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": "Inputs, outputs, latency, token usage, and intermediate steps."
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": "A central repository for sharing and versioning prompts."
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": "Helps monitor performance, detect regressions, and debug production issues."
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": "Used for systematic testing and evaluation of LLM pipelines."
    },
    {
        "question": "What is RAGAS?",
        "reference": "A framework for automated evaluation of RAG pipelines."
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": "Measures how much the answer is derived from the retrieved context."
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": "Measures how relevant the generated answer is to the original question."
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": "Measures if necessary information is present in the retrieved context."
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": "Typically requires question, answer, contexts, and reference."
    },
    {
        "question": "What is Guardrails AI?",
        "reference": "A framework for adding structure and quality guarantees to LLM outputs."
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": "Personally Identifiable Information; detecting it is crucial for privacy."
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": "Ensures LLM responses follow a specific schema like JSON."
    },
    {
        "question": "What is Constitutional AI?",
        "reference": "An approach to training AI to be helpful, honest, and harmless using principles."
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": "Concerns include bias, toxicity, hallucinations, and alignment."
    }
]
