import http.server
import ssl


handler = http.server.SimpleHTTPRequestHandler


httpd = http.server.HTTPServer(('0.0.0.0', 8443), handler)

httpd.socket = ssl.wrap_socket(httpd.socket,
                               certfile='cert.pem',
                               keyfile='key.pem')

print("Serving on https://0.0.0.0:8443")
httpd.serve_forever()

# import http.server
# import ssl
# import socketserver

# # Redirect handler
# class RedirectHandler(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(301)
#         new_url = f"https://{self.headers['Host'].split(':')[0]}:8443{self.path}"
#         self.send_header('Location', new_url)
#         self.end_headers()

# # Start HTTP redirect server on 9090
# def start_http_redirect():
#     with socketserver.TCPServer(("", 9090), RedirectHandler) as httpd:
#         print("Redirecting all HTTP traffic from port 9090 → 8443")
#         httpd.serve_forever()

# # Start HTTPS server on 8443
# def start_https_server():
#     handler = http.server.SimpleHTTPRequestHandler
#     httpd = http.server.HTTPServer(('0.0.0.0', 8443), handler)
#     httpd.socket = ssl.wrap_socket(httpd.socket, certfile='cert.pem', keyfile='key.pem', server_side=True)
#     print("Serving securely on https://0.0.0.0:8443")
#     httpd.serve_forever()

# # Run both servers
# import threading
# threading.Thread(target=start_http_redirect, daemon=True).start()
# start_https_server()