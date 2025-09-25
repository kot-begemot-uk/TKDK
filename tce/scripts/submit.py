#!/usr/bin/python3

'''CGI python script to accept cache entry archive submissions'''

import os
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from multipart import parse_form_data

FILENAME_REGEXP = re.compile(r"\w{52}\.tar\.gz")
SPOOL = "/var/www/cache-test"
OK_RESULT = '''Content-type:text/html\r\n\r\n
<html>
<head>
<title>Succesful Submission</title>
</head>
<body>
<h2>Submitted OK</h2>
</body>
</html>
'''

NOK_RESULT = '''Content-type:text/html\r\n\r\n
<html>
<head>
<title>Submission Failed</title>
</head>
<body>
<h2>Fail</h2>
</body>
</html>
'''


class TarEntry():
    '''Tar entry submitted via a CGI upload'''
    def __init__(self, filename=None, fsname=None):
        self.filename = filename
        self.fsname = fsname

    def receive(self):
        '''Receive tar data and store it as a temporary file'''
        environ = dict(os.environ.items())
        environ['wsgi.input'] = sys.stdin.buffer
        _, files = parse_form_data(environ)
        for file_part in files.getall("tarfile"):
            with NamedTemporaryFile(dir=SPOOL, delete=False, delete_on_close=False) as tar_file:
                tar_file.write(file_part.file.read())
                self.fsname = tar_file.name
                self.filename = file_part.filename

    def verify(self):
        '''Verify that the incoming file is cache entry'''
        if self.filename is None:
            return False
        if FILENAME_REGEXP.match(self.filename) is None:
            return False
        return True

    def move_to_spool(self, spool_dir):
        '''Move file to spool directory so it can be served'''
        if self.verify():
            os.rename(self.fsname, Path(spool_dir, self.filename).as_posix())
            return True
        return False


def main():
    '''When invoked as a CGI script'''
    entry = TarEntry()
    entry.receive()
    if entry.move_to_spool(SPOOL):
        print(OK_RESULT)
    else:
        print (NOK_RESULT)

if __name__ == "__main__":
    main()
