from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Artificial intelligence is transforming the way humans interact with technology. From recommendation systems to autonomous vehicles, AI-powered tools are increasingly embedded in everyday life. These systems rely on large amounts of data, sophisticated algorithms, and significant computational resources.

The concept of artificial intelligence dates back to the mid-twentieth century. Early researchers believed that human-level intelligence could be achieved by encoding logical rules into machines, but progress was slow due to limited computing power and insufficient data. Over time, this led to frustration and reduced funding during several so-called AI winters.

In the nineteen nineties and early two thousands, advances in hardware and the rapid growth of the internet changed the field dramatically. Machine learning techniques based on statistical methods began to outperform rule-based systems in tasks such as speech recognition and image classification. As data became more abundant, models grew larger and more accurate.

Today, artificial intelligence is used across many industries. In healthcare, it supports medical imaging, diagnostics, and drug discovery. In finance, it is applied to fraud detection and algorithmic trading. In education, it enables personalized learning experiences, while in entertainment it drives content recommendations and game design.

Despite these successes, AI systems still face significant challenges. They often struggle to generalize beyond their training data, misunderstand context or nuance, and behave unpredictably in real-world environments. Large language models, in particular, may generate fluent responses that are misleading or incorrect.

Looking ahead, researchers are working to improve the reliability, interpretability, and ethical alignment of AI systems. Progress in areas such as reinforcement learning, multimodal modeling, and human oversight will shape how these technologies are developed and deployed. The long-term impact of artificial intelligence will depend not only on technical advances, but also on responsible and thoughtful use.

"""
# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

# Perform the split
chunks = splitter.split_text(text)

print(len(chunks))
print(chunks)