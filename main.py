#!/usr/bin/env python3
"""CLI for the RAG pipeline."""
import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    sub = parser.add_subparsers(dest="command")

    index_p = sub.add_parser("index", help="Index documents")
    index_p.add_argument("sources", nargs="+", help="File paths or URLs")
    index_p.add_argument("--strategy", default="recursive", choices=["recursive", "token"])

    query_p = sub.add_parser("query", help="Query indexed documents")
    query_p.add_argument("question", help="Natural language question")

    sub.add_parser("serve", help="Start FastAPI server")

    args = parser.parse_args()

    if args.command == "index":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline(chunk_strategy=args.strategy)
        count = pipeline.index(args.sources)
        print(f"Indexed {count} chunks")

    elif args.command == "query":
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        response = pipeline.query(args.question)
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources: {response.sources}")

    elif args.command == "serve":
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
