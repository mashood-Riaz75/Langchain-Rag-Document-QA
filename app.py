
import logging
import os

import gradio as gr
from dotenv import load_dotenv

from rag_pipeline import (
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_RETRIEVAL_MODE,
    DEFAULT_TOP_K,
    RAGConfigurationError,
    answer_question,
    validate_config,
)

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def respond(message, history, top_k, threshold, retrieval_mode):
    message = (message or "").strip()
    history = list(history or [])

    if not message:
        return (
            history,
            "",
            "Enter a question to retrieve document sources.",
            "No question submitted.",
        )

    try:
        result = answer_question(
            question=message,
            top_k=int(top_k),
            min_relevance=float(threshold),
            mode=retrieval_mode,
        )

        answer = result["answer"]
        sources = result["sources"]

        if sources:
            citations = "\n".join(
                f"- [{item['source']} | {item['chunk_id']}]"
                for item in sources
            )
            answer += "\n\n**Retrieved sources:**\n" + citations

            previews = ["## Retrieved source previews"]

            for number, item in enumerate(sources, start=1):
                source = item["source"]
                chunk_id = item["chunk_id"]
                score = item["score"]
                content = item["text"].replace("```", "'''")

                previews.append(
                    f"### Source {number}: `{source}`\n\n"
                    f"- **Chunk ID:** `{chunk_id}`\n"
                    f"- **Relevance score:** `{score:.3f}`\n\n"
                    f"```text\n{content}\n```"
                )

            preview_text = "\n\n---\n\n".join(previews)
        else:
            preview_text = (
                "No source chunks passed the relevance threshold."
            )

        status = (
            f"Status: {result['status']}  \n"
            f"Retrieved chunks: {len(sources)}  \n"
            f"Retrieval mode: {retrieval_mode}  \n"
            f"Elapsed time: "
            f"{result.get('timings', {}).get('total_seconds', 'N/A')} seconds"
        )

    except FileNotFoundError:
        logger.exception("Vector index is unavailable.")
        answer = (
            "The document index is missing. Run `python ingestion.py` "
            "and then restart the app."
        )
        preview_text = "Source previews are unavailable."
        status = "Index unavailable."

    except (ValueError, RAGConfigurationError) as error:
        logger.warning("Invalid request or configuration: %s", error)
        answer = "Configuration or input error: " + str(error)
        preview_text = "Fix the configuration before retrieving sources."
        status = "Configuration/input error."

    except Exception:
        logger.exception("Unexpected RAG application error.")
        answer = (
            "Something went wrong while processing your question. "
            "Check the terminal logs for details."
        )
        preview_text = "Source previews are unavailable."
        status = "Request failed."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return history, "", preview_text, status


def clear_all():
    return (
        [],
        "",
        "Ask a question to inspect the retrieved document chunks.",
        "Chat cleared.",
    )


with gr.Blocks(title="Smart Document Q&A") as demo:
    gr.Markdown(
        """
        # Smart Document Q&A

        Ask questions about your attendance, grading, and course schedule.

        Answers are generated from retrieved document context.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=440,
    )

    with gr.Row():
        message = gr.Textbox(
            label="Your question",
            placeholder="Ask a question about your documents...",
            scale=5,
        )

        top_k = gr.Slider(
            minimum=1,
            maximum=8,
            value=DEFAULT_TOP_K,
            step=1,
            label="Maximum retrieved chunks",
            scale=2,
        )

    with gr.Row():
        threshold = gr.Slider(
            minimum=0,
            maximum=1,
            value=DEFAULT_MIN_RELEVANCE,
            step=0.05,
            label="Minimum relevance score",
        )

        retrieval_mode = gr.Radio(
            choices=["similarity", "mmr"],
            value=DEFAULT_RETRIEVAL_MODE,
            label="Retrieval strategy",
        )

    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        clear_button = gr.Button("Clear chat")

    status = gr.Markdown("Ready.")

    with gr.Accordion("Retrieved source previews", open=False):
        source_preview = gr.Markdown(
            "Ask a question to inspect the retrieved document chunks."
        )

    inputs = [message, chatbot, top_k, threshold, retrieval_mode]
    outputs = [chatbot, message, source_preview, status]

    ask_button.click(
        fn=respond,
        inputs=inputs,
        outputs=outputs,
    )

    message.submit(
        fn=respond,
        inputs=inputs,
        outputs=outputs,
    )

    clear_button.click(
        fn=clear_all,
        inputs=[],
        outputs=outputs,
    )


if __name__ == "__main__":
    try:
        validate_config()
        demo.launch()
    except RAGConfigurationError as error:
        logger.error("%s", error)
        raise SystemExit(str(error))
