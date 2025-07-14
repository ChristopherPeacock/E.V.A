import argparse
import os
import json
from pathlib import Path
from utils.scrape import scrape_url
from utils.embed import get_vectorstore, query_jarvis
from dotenv import load_dotenv
from visuals import eyes

load_dotenv()

def load_config(path='config.json'):
    if Path(path).exists():
        print("found file")
        with open(path) as f:
            return json.load(f)


def interactive_loop():
    
    while True:
        try:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit']:
                print("[🔌] Shutting down. See you soon, Commander.")
                break
            
            print("\nE.V.A: ", end="", flush=True)
            answer = query_jarvis(query, stream=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n[✋] Interrupted. Exiting.")
            break

def main():
    config = load_config()
    eyes.animate_eyes()
    print(f"\n[E.V.A] Hello {os.getenv('NAME')}. What are we doing today?")


    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', help='URL to scrape and vectorise')
    parser.add_argument('--ask', help='Ask a question to your assistant')
    parser.add_argument('--debug', help='Do you want to put the debug mode on' )
    args = parser.parse_args()
    
    if args.ingest:
        print(f"[+] Scraping and indexing: {args.ingest}")
        text = scrape_url(args.ingest)
        get_vectorstore(text, args.ingest)
        print("[✓] Knowledge ingested and vectorised.")
    
    if args.ask:
        print(f"[?] Asking: {args.ask}")
        print("\n[LLM] ", end="", flush=True)
        answer = query_jarvis(args.ask, stream=True)
        print("\n")
    
    if args.debug:
        debug = args.debug if args.debug else config.get("debug", False)
        print(f"debug mode is on:")
        
    interactive_loop()

if __name__ == '__main__':
    main()