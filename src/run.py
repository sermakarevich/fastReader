import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Process document for summarization.")

    parser.add_argument("--URL", type=str, required=True, help="The URL of the document")
    parser.add_argument(
        "--document_type", type=str, required=True, choices=["pdf", "text", "youtube"], help="The type of the document"
    )
    parser.add_argument("--model_class", type=str, default="ollama", help="The class of the model used")
    parser.add_argument("--model_name", type=str, default="phi4", help="The name of the model used")
    parser.add_argument("--chunk_size", type=int, default=2000, help="The size of each chunk")
    parser.add_argument("--chunk_size_overlap", type=int, default=50, help="The overlap of each chunk")
    parser.add_argument("--chunk_size_decay", type=float, default=0.8, help="The decay rate of the chunk size")
    parser.add_argument(
        "--target_pre_summary_text_length", type=int, default=15000, help="The target length of the pre-summary text"
    )

    args = parser.parse_args()
    return vars(args)


def main():
    from app.logger import initialize_logging

    initialize_logging()

    from app.main import run

    state = parse_args()

    run(state=state)


if __name__ == "__main__":
    main()
