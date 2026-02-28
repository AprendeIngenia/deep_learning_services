from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os

HOST = "0.0.0.0"
PORT = 5500

def main():
    web_dir = Path(__file__).resolve().parent
    os.chdir(web_dir)

    print(f"Serving {web_dir} on http://localhost:{PORT}")
    httpd = ThreadingHTTPServer((HOST, PORT), SimpleHTTPRequestHandler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()
