import argparse
import os

import uvicorn
from dotenv import load_dotenv


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Start the Triage Dataset Manager web server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind the server to (default: 8000 or PORT env var)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    uvicorn.run(
        "triage_webserver.app:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
