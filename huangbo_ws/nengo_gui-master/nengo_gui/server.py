"""HTTP and WebSocket server implementation."""

import base64
import errno
import hashlib
import logging
import socket
import struct
import ssl
import sys
import threading
import traceback

try:
    from http import server
    from http.cookies import SimpleCookie
    import socketserver
    from urllib.parse import parse_qs, urlparse
except ImportError:  # Python 2.7
    import BaseHTTPServer as server
    from Cookie import SimpleCookie
    import SocketServer as socketserver
    from urlparse import parse_qs, urlparse


logger = logging.getLogger(__name__)


WS_MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'


class SocketClosedError(IOError):
    pass


class HttpError(Exception):
    def __init__(self, code, msg, headers=(), data=None):
        super(HttpError, self).__init__(msg)
        self.code = code
        self.msg = msg
        self.headers = headers
        if data is None:
            data = b'<h1>' + bytes(self.code) + b'</h1><p>' + msg.encode(
                'utf-8') + b'</p>'
        self.data = data

    def to_response(self):
        return HtmlResponse(self.data, code=self.code, headers=self.headers)


class BadRequest(HttpError):
    def __init__(self):
        super(BadRequest, self).__init__(400, 'Bad request')


class Forbidden(HttpError):
    def __init__(self):
        super(Forbidden, self).__init__(403, 'Forbidden')


class InvalidResource(HttpError):
    def __init__(self, path):
        super(InvalidResource, self).__init__(404, 'Invalid resource: ' + path)


class UpgradeRequired(HttpError):
    def __init__(self, headers):
        super(UpgradeRequired, self).__init__(426, 'Upgrade required', headers)


class InternalServerError(HttpError):
    def __init__(self, msg):
        super(InternalServerError, self).__init__(
            500, 'Internal server error', data=msg.encode('utf-8'))


class HttpResponse(object):
    def __init__(self, data, mimetype='text/html', code=200, headers=()):
        self.data = data
        self.mimetype = mimetype
        self.code = code
        self.headers = headers

    def send(self, request):
        request.send_response(self.code)
        request.send_header('Content-type', self.mimetype)
        if hasattr(request, 'flush_headers'):
            request.flush_headers()
        request.wfile.write(request.cookie.output().encode('utf-8'))
        request.wfile.write(b'\r\n')
        for header in self.headers:
            request.send_header(*header)
        request.end_headers()
        request.wfile.write(self.data)


class HttpRedirect(HttpResponse):
    def __init__(
            self, location, data=b'', mimetype='text/html', code=303,
            headers=()):
        super(HttpRedirect, self).__init__(
            data=data, mimetype=mimetype, code=code,
            headers=headers + (('Location', location),))
        self.location = location


class HtmlResponse(HttpResponse):
    def __init__(self, body, code=200, headers=()):
        data = b'<html><body>' + body + b'</body></html>'
        super(HtmlResponse, self).__init__(data, code=code, headers=headers)


class ManagedThreadHttpServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Threaded HTTP and WebSocket server that keeps track of its connections
    to allow a proper shutdown."""

    daemon_threads = True  # this ensures all spawned threads exit

    def __init__(self, *args, **kwargs):
        server.HTTPServer.__init__(self, *args, **kwargs)

        # keep track of open threads, so we can close them when we exit
        self._requests = []
        self._websockets = []

        self._shutting_down = False

    @property
    def requests(self):
        return self._requests[:]

    @property
    def websockets(self):
        return self._websockets[:]

    def create_websocket(self, socket):
        ws = WebSocket(socket)
        self._websockets.append(ws)
        return ws

    def process_request_thread(self, request, client_address):
        thread = threading.current_thread()
        self._requests.append((thread, request))
        socketserver.ThreadingMixIn.process_request_thread(
            self, request, client_address)
        self._requests.remove((thread, request))

    def handle_error(self, request, client_address):
        exc_type, exc_value, _ = sys.exc_info()
        if (exc_type is socket.error and
                exc_value.args[0] in
                [errno.EPIPE, errno.EBADF, errno.ECONNRESET]):
            return  # Probably caused by a server shutdown
        else:
            logger.exception("Server error.")
            server.HTTPServer.handle_error(self, request, client_address)

    def shutdown(self):
        if self._shutting_down:
            return
        self._shutting_down = True

        for ws in self.websockets:
            ws.close()
        for _, request in self.requests:
            self.shutdown_request(request)

        server.HTTPServer.shutdown(self)

    def wait_for_shutdown(self, timeout=None):
        """Wait for all request threads to finish.

        Parameters
        ----------
        timeout : float, optional
            Maximum time in seconds to wait for each thread to finish.
        """
        for thread, _ in self.requests:
            if thread.is_alive():
                thread.join(timeout)


class HttpWsRequestHandler(server.BaseHTTPRequestHandler):
    """Base class for request handler that can handle normal and websocket
    requests.

    `http_commands` and `ws_commands` are dictionaries mapping resource names
    (with leading '/') to function names (as string) in this class. These
    functions do not take any arguments except for `self`. All required data
    is defined as attributes on the instance. In addition to the attributes
    provided by the `BaseHTTPRequestHandler` the resource name is provided as
    `resource`, the parsed query string (as dictionary) as `query` and combined
    query string and post fields as `db`. In a websocket command handler
    function `ws` provides access to the websocket.

    If no handler function for a resource was defined, the `http_default` and
    `ws_default` functions will be used.
    """
    http_commands = {}
    ws_commands = {}

    def __init__(self, *args, **kwargs):
        self.resource = None
        self.query = None
        self.db = {}
        self.cookie = SimpleCookie()
        self.ws = None
        server.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def do_POST(self):
        data = self.rfile.read(
            int(self.headers['Content-Length'])).decode('ascii')

        if 'multipart/form-data' in self.headers['Content-Type']:
            raise NotImplementedError()  # TODO
        else:
            self.db = {k: v[0] for k, v in parse_qs(data).items()}

        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        self.resource = parsed.path
        self.query = parse_qs(parsed.query)
        self.db.update(
            {k: v[0] for k, v in self.query.items() if k not in self.db})
        if 'Cookie' in self.headers:
            self.cookie.load(self.headers['Cookie'])

        try:
            connection = self.headers.get('Connection', 'close').lower()
            if 'upgrade' in connection:
                self.handle_upgrade()
            else:
                self.http_GET()
        except HttpError as err:
            logger.warning(
                'Error response (%i): %s', err.code, err.msg, exc_info=True)
            err.to_response().send(self)
        except Exception as err:
            logger.exception('Error response')
            err = InternalServerError(
                '<pre>' + traceback.format_exc() + '</pre>')
            err.to_response().send(self)

    def http_GET(self):
        command = self._get_command(self.http_commands, self.resource)
        if command is None:
            response = self.http_default()
        else:
            response = getattr(self, command)()
        response.send(self)

    def http_default(self):
        raise InvalidResource(self.path)

    def handle_upgrade(self):
        upgrade = self.headers.get('Upgrade').lower()
        if upgrade == 'websocket':
            self.upgrade_to_ws()
        else:
            raise BadRequest()

    def get_expected_origins(self):
        raise NotImplementedError()

    def upgrade_to_ws(self):
        response = '''HTTP/1.1 101 Switching Protocols\r\n\
Upgrade: websocket\r\n\
Connection: Upgrade\r\n\
Sec-WebSocket-Accept: {sec}\r\n\
\r\n\
'''
        valid_srv_addrs = self.get_expected_origins()

        try:
            origin = urlparse(self.headers['Origin'])
            assert origin.netloc.lower() in valid_srv_addrs
        except KeyError:
            raise Forbidden()
        except AssertionError:
            raise Forbidden()

        try:
            host = self.headers['Host'].lower()
            assert host in valid_srv_addrs
            key = self.headers['Sec-WebSocket-Key']
            assert len(base64.b64decode(key)) == 16
        except KeyError:
            raise BadRequest()
        except AssertionError:
            raise BadRequest()

        if self.headers['Sec-WebSocket-Version'] != '13':
            raise UpgradeRequired(['Sec-WebSocket-Version: 13'])

        sec = base64.b64encode(hashlib.sha1(
            (key + WS_MAGIC).encode('ascii')).digest()).decode('ascii')
        _sendall(self.request, response.format(sec=sec).encode('utf-8'))

        self.ws = self.server.create_websocket(self.request)
        self.ws.set_blocking(False)

        command = self._get_command(self.ws_commands, self.resource)
        if command is None:
            self.ws_default()
        else:
            getattr(self, command)()
        self.ws.close()

    def ws_default(self):
        raise InvalidResource(self.path)

    @classmethod
    def _get_command(cls, commands, path):
        if not path.startswith('/'):
            path = '/' + path

        while len(path) > 0:
            if path in commands:
                return commands[path]
            path = path.rsplit('/', 1)[0]
        if '/' in commands:
            return commands['/']
        return None


class WebSocket(object):
    ST_OPEN, ST_CLOSING, ST_CLOSED = range(3)

    def __init__(self, socket):
        self.socket = socket
        self._buf = bytearray([])
        self.state = self.ST_OPEN

    def set_timeout(self, timeout):
        self.socket.settimeout(timeout)

    def set_blocking(self, flag):
        self.socket.setblocking(flag)

    def _read(self):
        try:
            self._buf = self._buf + bytearray(self.socket.recv(512))
        except ssl.SSLError as e:
            if e.errno == 2:
                # Corresponds to SSLWantReadError which only exists in Python
                # 2.7.9+ and 3.3+.
                pass
            else:
                raise
        except socket.error as e:
            if e.errno in [errno.EDEADLK, errno.EAGAIN, 10035]:
                # no data available
                pass
            elif e.errno == errno.EBADF:
                raise SocketClosedError("Cannot read from closed socket.")
            else:
                raise

    def read_frame(self):
        try:
            self._read()
            frame, size = WebSocketFrame.parse(self._buf)
            self._buf = self._buf[size:]
            if not self._handle_frame(frame):
                return frame
        except ValueError:
            return None
        except socket.timeout:
            return None


    def _handle_frame(self, frame):
        if frame.opcode == WebSocketFrame.OP_CLOSE:
            if self.state not in [self.ST_CLOSING, self.ST_CLOSED]:
                self.close()
            raise SocketClosedError("Websocket has been closed")
        elif frame.opcode == WebSocketFrame.OP_PING:
            if self.state == self.ST_OPEN:
                pong = WebSocketFrame(
                    fin=1, rsv=0, opcode=WebSocketFrame.OP_PONG, mask=0,
                    data=frame.data)
                _sendall(self.socket, pong.pack())
            return True
        elif frame.opcode == WebSocketFrame.OP_PONG:
            return True
        else:
            return False

    def close(self):
        if self.state not in [self.ST_CLOSING, self.ST_CLOSED]:
            self.state = self.ST_CLOSING
            close_frame = WebSocketFrame(
                fin=1, rsv=0, opcode=WebSocketFrame.OP_CLOSE, mask=0, data=b'')
            try:
                _sendall(self.socket, close_frame.pack())
            except socket.error as err:
                if err.errno in [errno.EPIPE, errno.EBADF]:
                    pass

    def write_frame(self, frame):
        if self.state != self.ST_OPEN:
            raise SocketClosedError("Connection not open.")

        try:
            _sendall(self.socket, frame.pack())
        except socket.error as e:
            if e.errno == errno.EPIPE:  # Broken pipe
                raise SocketClosedError("Cannot write to socket.")
            else:
                raise

    def write_text(self, text):
        self.write_frame(WebSocketFrame.create_text_frame(text))

    def write_binary(self, data):
        self.write_frame(WebSocketFrame.create_binary_frame(data))


def _sendall(socket, data):
    bytes_sent = 0
    while bytes_sent < len(data):
        bytes_sent += socket.send(data[bytes_sent:])



class WebSocketFrame(object):
    __slots__ = ['fin', 'rsv', 'opcode', 'mask', 'data']

    OP_CONT = 0x0
    OP_TEXT = 0x1
    OP_BIN = 0x2
    OP_CLOSE = 0x8
    OP_PING = 0x9
    OP_PONG = 0xA

    def __init__(self, fin, rsv, opcode, mask, data):
        self.fin = fin
        self.rsv = rsv
        self.opcode = opcode
        self.mask = mask
        self.data = data

    @classmethod
    def parse(cls, data):
        try:
            offset = 0

            fin = (data[0] >> 7) & 0x1
            rsv = (data[0] >> 4) & 0x07
            opcode = data[0] & 0x0F
            masked = (data[1] >> 7) & 0x01
            datalen = data[1] & 0x7F
            mask = b'\x00\x00\x00\x00'

            offset += 2

            if datalen == 126:
                datalen = cls._to_int(data[offset:offset+2])
                offset += 2
            elif datalen == 127:
                datalen = cls._to_int(data[offset:offset+8])
                offset += 8

            if masked:
                mask = data[offset:offset+4]
                offset += 4

            size = offset + datalen
            masked_data = data[offset:size]
            if len(masked_data) < datalen:
                raise IndexError()
            unmasked_data = [masked_data[i] ^ mask[i % 4]
                             for i in range(len(masked_data))]
            data = bytearray(unmasked_data)
            if opcode == cls.OP_TEXT:
                data = data.decode('ascii')


            return cls(fin, rsv, opcode, mask, data), size
        except IndexError:
            raise ValueError('Frame incomplete.')

    @classmethod
    def _to_int(cls, data):
        value = 0
        for b in data:
            value = (value << 8) + b
        return value

    def pack(self):
        code = (self.fin & 0x01) << 7
        code |= (self.rsv & 0x07) << 4
        code |= self.opcode & 0x0F

        datalen = len(self.data)
        mask_bit = ((self.mask != 0) & 0x01) << 7
        if datalen < 126:
            header = struct.pack('!BB', code, datalen | mask_bit)
        elif datalen <= 0xFFFF:
            header = struct.pack('!BBH', code, 126 | mask_bit, datalen)
        else:
            header = struct.pack('!BBQ', code, 127 | mask_bit, datalen)

        data = self.data

        return header + data

    @classmethod
    def create_text_frame(cls, text, mask=0):
        return cls(1, 0, cls.OP_TEXT, mask, text.encode('utf-8'))

    @classmethod
    def create_binary_frame(cls, data, mask=0):
        return cls(1, 0, cls.OP_BIN, mask, data)
