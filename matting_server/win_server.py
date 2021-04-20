from tornado.httpserver import HTTPServer
from tornado.wsgi import WSGIContainer
from server import app
from tornado.ioloop import IOLoop

s = HTTPServer(WSGIContainer(app))
s.listen(9111)
IOLoop.current().start()
