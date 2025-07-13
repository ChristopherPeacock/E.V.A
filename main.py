import argparse
import os
from utils.scrape import scrape_url
from utils.embed import get_vectorstore, query_jarvis
from dotenv import load_dotenv
from visuals import eyes

load_dotenv()

def interactive_loop():
    
    while True:
        try:
            query = input("\n> ")
            if query.lower() in ['exit', 'quit']:
                print("[🔌] Shutting down. See you soon, Commander.")
                break
            
            print("\nJARVIS: ", end="", flush=True)
            answer = query_jarvis(query, stream=True)
            print("\n")
            
        except KeyboardInterrupt:
            print("\n[✋] Interrupted. Exiting.")
            break

def main():
    eyes.animate_eyes()
    print(f"\n[JARVIS] Hello {os.getenv('NAME')}. What are we doing today?")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ingest', help='URL to scrape and vectorise')
    parser.add_argument('--ask', help='Ask a question to your assistant')
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
    
    interactive_loop()

if __name__ == '__main__':
    main()