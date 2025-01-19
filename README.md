# FastReader

FastReader is an application designed to process and summarize documents. Text compression is happening in an iterative manner, enabling fastReader to compress even huge documents or books into a few sentences.
![alt text](assets/graph.png)

## Installation

To set up the environment:

```sh
make env-create
```

## Ollama Installation

Ollama is required for running local language models. To install Ollama, follow the instructions on the [Ollama GitHub page](https://github.com/ollama/ollama).

## Usage

To run the application, use the following command:

```sh
python src/run.py --URL https://www.zenml.io/blog/production-llm-security-real-world-strategies-from-industry-leaders --document_type text
```

out:
```sh
python src/run.py --URL https://www.zenml.io/blog/production-llm-security-real-world-strategies-from-industry-leaders --document_type text
INFO     | 2025-01-19 07:09:08,157 | app.main | 10 | Running summarization workflow with state: {'URL': 'https://www.zenml.io/blog/production-llm-security-real-world-strategies-from-industry-leaders', 'document_type': 'text', 'model_class': 'ollama', 'model_name': 'phi4', 'chunk_size': 2000, 'chunk_size_overlap': 50, 'chunk_size_decay': 0.8, 'target_pre_summary_text_length': 15000}
INFO     | 2025-01-19 07:09:08,406 | app.text_extraction | 44 | Text extraction took 0 seconds. Text length: 19992
INFO     | 2025-01-19 07:09:08,408 | app.text_splitter | 17 | Text splitting took 0 seconds. Number of chunks: 11
INFO     | 2025-01-19 07:09:08,408 | app.compress_text | 17 | Text compression: Iteration: 0, Chunk size: 2000
INFO     | 2025-01-19 07:09:11,680 | app.compress_text | 35 | Compression took 3 seconds. Compressed text length: 2867
INFO     | 2025-01-19 07:09:14,569 | app.short_summary | 29 | Short summary took 2 seconds. Short summary length: 1056
INFO     | 2025-01-19 07:09:19,329 | app.extensive_summary | 29 | Extensive summary took 7 seconds. Extensive summary length: 3023
INFO     | 2025-01-19 07:09:19,330 | app.main | 16 | The blog post highlights the necessity of securing large language models (LLMs) in production environments through multi-layered defense strategies due to emerging security challenges. It emphasizes enhancing data privacy and security throughout the LLMOps lifecycle by implementing robust controls over data access, integration practices for plugins, fine-tuning, adversarial training, input validation, output filtering with anomaly detection, and effective prompt engineering. Moreover, it stresses the importance of using traditional security principles alongside LLM-specific measures such as real-time threat monitoring, human oversight, and explainability to ensure safe AI applications.

The post also notes the risks associated with fine-tuning models on sensitive data and suggests mitigating these through careful data handling, deployment strategies, and monitoring. Additionally, it underscores the role of platforms like ZenML in simplifying MLOps by providing various integrations and resources for effective large language model management.
INFO     | 2025-01-19 07:09:19,330 | app.main | 17 | ### Summary of the Blog Post on Securing Large Language Models (LLMs)

The blog post provides an in-depth discussion on strategies for securing large language models (LLMs) within production environments. As LLMs become more prevalent across industries, their adoption brings forth new security challenges that must be addressed to prevent vulnerabilities and protect sensitive information.

#### Key Security Challenges
1. **Prompt Injection Attacks**: One of the primary risks highlighted is prompt injection attacks, where malicious inputs can manipulate LLM outputs.
2. **Data Leakage in RAG Systems**: Retrieval-Augmented Generation (RAG) systems face threats from indirect prompt injections and data leakage, emphasizing the need for thorough vetting of external sources.

#### Strategies to Enhance Security
- **Multi-Layered Defense Approach**:
  - Implementing input sanitization and validation.
  - Output filtering with anomaly detection and content classification.
  - Using retry mechanisms for handling safety triggers.

- **Data Privacy and Secure Integration**: 
  - Controls around data access, secure integration practices, especially for plugins.
  - Techniques like fine-tuning and adversarial training to fortify models against malicious inputs.

#### Enhancing Data Privacy
- **Privacy-Preserving Methods**:
  - Secure storage of data with robust access controls.
  - Anonymization techniques to protect sensitive information.
  - Exploring new privacy-preserving training methods to mitigate risks associated with fine-tuning on sensitive data.

#### Monitoring and Threat Detection
- Employing anomaly detection systems and real-time threat monitoring.
- Adversarial training and human oversight as additional layers of security.
- Explainability to ensure the safe deployment of AI applications by making model decisions transparent.

#### Effective Prompt Engineering
- Clear instructions in prompts improve user experience while also enhancing security against vulnerabilities, thereby reducing risks from improper LLM interactions.

#### Organizational Security Principles
- The blog emphasizes that securing LLMs involves traditional security principles such as transparency and privacy.
- Defense-in-depth strategies are crucial, which include comprehensive monitoring tailored to specific organizational needs.

#### MLOps Platform: ZenML
- The post briefly mentions ZenML, a platform designed to simplify machine learning operations (MLOps) by providing various integrations and resources for users.
- It compares ZenML with other tools and highlights the availability of documentation and community engagement opportunities as valuable resources for practitioners in the field.

Overall, the blog underscores that securing LLMs is an ongoing process requiring a combination of technical strategies, robust data handling practices, and continuous monitoring to adapt to evolving threats. These measures are essential to ensure safe and trustworthy AI applications in production environments.
```

There are three types of documents that can be processed:
- youtube videos: `youtube` - https://www.youtube.com/watch?v=gF341XMN8cY
- pdfs: `pdf` - https://arxiv.org/pdf/2501.04227
- text: `text` - https://www.zenml.io/blog/production-llm-security-real-world-strategies-from-industry-leaders