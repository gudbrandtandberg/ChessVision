#!/usr/bin/env python3
''' Copyright (C) 2016  Povilas Kanapickas <povilas@radix.lt>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

Taken from https://github.com/p12tic/simple-http-file-server
'''

from http.server import SimpleHTTPRequestHandler, HTTPServer
import argparse
import base64
import json
import os
import queue
import socket
import sys
import time
import threading
import cgi

class SimpleUploadHandler(SimpleHTTPRequestHandler):
    """Overriding GET"""
    def do_GET(self):
        """Serve a GET request."""
        f = self.send_head()
        print("Custom get")
        if f:
            try:
                self.copyfile(f, self.wfile)
            finally:
                f.close()

    def do_POST(self):
        self.log_headers_if_needed()
        print("Custom POST")
        
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'}
        )
        
        #imagedata = base64.b64decode(form["image"])
        imagedata = base64.b64decode(form["image"].value)
        #print(imagedata)
        
        filename = 'uploaded_image.jpg'
        with open(filename, 'wb') as f:
            f.write(imagedata)
        
        self.send_response(200)
        self.end_headers()

    def log_write(self, msg):
        if hasattr(self.server, 'log_file') and self.server.log_file is not None:
            self.server.log_file.write(msg)
        else:
            sys.stderr.write(msg)

    def log_headers_if_needed(self):
        if hasattr(self.server, 'log_headers') and self.server.log_headers == True:
            self.log_write(str(self.headers))

    def log_message(self, format, *args):
        msg = "%s - - [%s] %s\n" % (self.address_string(),
                                    self.log_date_time_string(),
                                    format % args)
        self.log_write(msg)

def encode_http_auth_password(user, psw):
    txt = user + ':' + psw
    txt = base64.b64encode(txt.encode('UTF-8')).decode('UTF-8')
    return txt

def decode_http_auth_password(txt):
    txt = base64.b64decode(txt.encode('UTF-8')).decode('UTF-8')
    items = txt.split(':')
    if len(items) != 2:
        return None
    return (items[0], items[1])

class PathConfig:
    def __init__(self, filename):
        if '/' in filename:
            raise Exception()
        self.filename = filename
        self.perms = {}
        self.children = {}

class AuthConfig:

    def __init__(self, log_file=sys.stdout):
        self.root = PathConfig('')
        self.users = {}
        self.log_file = log_file

    def add_path_config(self, path, user, perms):
        path_items = [ p for p in path.split('/') if p not in [ '', '.', '..' ] ]

        p = self.root
        for i in path_items:
            if i not in p.children:
                p.children[i] = PathConfig(i)
            p = p.children[i]

        p.perms[user] = perms

    def load_config(self, config_file_path):
        try:
            config = json.load(open(config_file_path, 'r'))
            config_paths = config['paths']
            for config_path in config_paths:
                path = config_path['path']
                user = config_path['user']
                perms = config_path['perms']
                self.add_path_config(path, user, perms)

            config_users = config['users']
            for config_user in config_users:
                user = config_user['user']
                psw = config_user['psw']
                self.users[user] = psw

        except Exception as e:
            self.log_write("Error reading config file " + config_file_path)
            self.log_write(str(e))

    def check_perm(self, perms, user, perm):
        if user in perms:
            if perm in perms[user]:
                return True
            return False

        if '*' in perms:
            if perm in perms['*']:
                return True
            return False
        return None

    def combine_perm(self, prev, next):
        if next == None:
            return prev
        return next

    def check_path_for_perm(self, path, perm, user, psw):
        if user not in self.users:
            user = '*'
        elif self.users[user] != psw:
            return False

        p = self.root
        items = path.split('/')

        result = self.combine_perm(True, self.check_perm(p.perms, user, perm))

        for i in items:
            if i not in p.children:
                return result
            p = p.children[i]

            result = self.combine_perm(result, self.check_perm(p.perms, user, perm))

        return result

class AuthUploadHandler(SimpleUploadHandler):

    def do_AUTHHEAD(self):
        self.log_headers_if_needed()

        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm=\"Test\"')
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Not authenticated\n')

    def check_auth_impl(self, perm):
        try:
            path = self.translate_path(self.path)
            path = os.path.relpath(path)
            if path.startswith('..'):
                return False

            auth_header = self.headers.get('Authorization')
            if auth_header == None:
                (user, psw) = ('*', None)
            else:
                if not auth_header.startswith('Basic '):
                    return False
                decode_result = decode_http_auth_password(auth_header[6:].strip())
                if decode_result == None:
                    return False
                (user, psw) = decode_result
            return self.server.auth_config.check_path_for_perm(path, perm, user, psw)

        except Exception as e:
            self.log_message("%s", str(e))
            self.wfile.write(str(e))
            return False

    def check_auth(self, perm):
        if not self.check_auth_impl(perm):
            self.do_AUTHHEAD()
            return False
        return True

    def do_HEAD(self):
        if self.check_auth('r'):
            self.log_headers_if_needed()
            SimpleHTTPRequestHandler.do_HEAD(self)

    def do_GET(self):
        if self.check_auth('r'):
            self.log_headers_if_needed()
            SimpleHTTPRequestHandler.do_GET(self)

    def do_PUT(self):
        if self.check_auth('w'):
            SimpleUploadHandler.do_PUT(self)

class PrintThread(threading.Thread):
    def __init__(self, log_file, should_flush=False):
        super().__init__()
        self.log_file = log_file
        self.should_flush = should_flush
        self.queue = queue.Queue()

    def run(self):
        while True:
            self.log_file.write(self.queue.get())
            if self.should_flush:
                self.log_file.flush()
            self.queue.task_done()

class FileQueueWrapper:
    def __init__(self, queue):
        self.queue = queue

    def write(self, data):
        self.queue.put(data)

def setup_log(log_path, should_flush_log):
    if log_path is not None:
        log_file = open(log_path, 'w')
    else:
        log_file = sys.stdout

    log_thread = PrintThread(log_file, should_flush=should_flush_log)
    log_thread.setDaemon(True)
    log_thread.start()
    return FileQueueWrapper(log_thread.queue)

def create_socket(host, port):
    addr = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    sock.listen(5)
    return sock

class ExternalSocketHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, socket):
        super().__init__(server_address, RequestHandlerClass)
        self.socket = socket

    def server_bind(self):
        pass

    def server_close(self):
        pass

class ListenerThread(threading.Thread):
    def __init__(self, host, port, socket, log_file, log_headers, auth_config):
        threading.Thread.__init__(self)
        self.host = host
        self.port = port
        self.socket = socket
        self.log_file = log_file
        self.log_headers = log_headers
        self.auth_config = auth_config

    def run(self):
        if self.auth_config is None:
            server = ExternalSocketHTTPServer((self.host, self.port),
                                              SimpleUploadHandler, self.socket)
        else:
            server = ExternalSocketHTTPServer((self.host, self.port),
                                              AuthUploadHandler, self.socket)
            server.auth_config = self.auth_config

        server.log_file = self.log_file
        server.log_headers = self.log_headers
        server.serve_forever()

def setup_and_start_http_server(host, port, access_config_path,
                                should_log_headers, log_path, should_flush_log,
                                num_threads):
    log_file = setup_log(log_path, should_flush_log)

    socket = create_socket(host, port)

    auth_config = None
    if access_config_path is not None:
        if not os.path.exists(access_config_path):
            log_file.write('No such file: {0}\n'.format(access_config_path))
            sys.exit(1)
        log_file.write('Setting up access restrictions\n')
        auth_config = AuthConfig()
        auth_config.load_config(access_config_path)

    log_file.write('listening on {0}:{1} using {2} threads\n'.format(
        host, port, num_threads))

    for i in range(num_threads):
        listener = ListenerThread(host, port, socket, log_file,
                                  should_log_headers, auth_config)
        listener.setDaemon(True)
        listener.start()
    time.sleep(90000000)

def main():
    parser = argparse.ArgumentParser(prog='server.py')
    parser.add_argument('port', type=int, help="The port to listen on")
    parser.add_argument('--access_config', type=str, default=None,
                        help="Path to access config")
    parser.add_argument('--log_headers', action='store_true', default=False,
                        help="If set logs headers of all requests")
    parser.add_argument('--log', type=str, default=None,
                        help="Path to log file")
    parser.add_argument('--should_flush_log', action='store_true', default=False,
                        help="If set, flushes log to disk after each entry")
    parser.add_argument('--threads', type=int, default=2,
                        help="The number of threads to launch")
    args = parser.parse_args()

    setup_and_start_http_server('localhost', args.port, args.access_config,
                                args.log_headers, args.log,
                                args.should_flush_log, args.threads)

if __name__ == '__main__':
    main()