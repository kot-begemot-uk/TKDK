#!/usr/bin/python3
'''
Open API (sorta) compliant cache web service
Bottle openapi plugin is suffering from severe bitrot, so we cannot
use it at present. Thus, this codes the OpenAPI schema manually.
Strict openAPI validation will be added at a later date if and when
needed.
'''

import re
import json
from pathlib import Path
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory
import gzip
import tarfile
import shutil
import bottle


def transform_index(cache_index):
    '''Open API cannot return a typed dict of items. The semantics of the
       API types mandate that it is returned as a list with the key being added as
       a property value.
    '''
    temp_result = []
    for (key, value) in cache_index.items():
        temp_result.append({"key":key} | value)
    return {"items":temp_result}


class CacheServer():
    '''Cache Server instance'''

    prefix = ""
    METADATA_EXT = ".meta"

    def __init__(self, cache_path, cache_index=None):
        self.cache_path = cache_path
        if cache_index is None:
            self.build_index()
        else:
            self.cache_index = cache_index
        self.schema = [
            ("{}/api/version", ["GET"], self.get_version),
            ("{}/api/v1/cache", ["GET"], self.get_cache),
            ("{}/api/v1/cache/<cache_key>", ["GET"], self.get_cache_item),
            ("{}/api/v1/cache/<cache_key>", ["PATCH"], self.set_cache_item),
            ("{}/api/v1/cache/<cache_key>", ["DELETE"], self.delete_cache_item),
            ("{}/api/v1/cache/<cache_key>/binary", ["GET"], self.get_cache_archive),
            ("{}/api/v1/cache/<cache_key>/binary", ["PUT"], self.set_cache_archive),
        ]

    def build_index(self):
        '''Build index from stored metadata files'''
        self.cache_index = {}
        print(f"building index for {self.cache_path}")
        for meta in Path(self.cache_path).glob("*" + self.METADATA_EXT):
            print(f"parsing {meta}")
            try:
                with open(meta, encoding="ascii") as metafile:
                    self.cache_index[meta.stem] = json.load(metafile)
            except FileNotFoundError:
                pass # will log later
            except json.JSONDecodeError:
                pass # will log later

    def cache_tmp(self):
        '''Cache temporary directory'''
        return self.cache_path + "/tmp"

    def create_tar(self, temp_dir):
        '''Tar Sanitized artefact'''
        with NamedTemporaryFile(dir=self.cache_tmp(),
                                delete=False,
                                delete_on_close=False) as tar_fd:
            with tarfile.TarFile(tar_fd.name, mode="w") as tarf:
                for artefact in Path(temp_dir).glob("*"):
                    tarf.add(artefact.as_posix(),
                             arcname=artefact.stem + "".join(artefact.suffixes))
        target = Path(tar_fd.name + ".tar.gz")
        gzip_file = gzip.GzipFile(target.as_posix(), mode="wb")
        with open(tar_fd.name, mode="rb") as to_compress:
            gzip_file.write(to_compress.read())
        gzip_file.close()
        os.unlink(tar_fd.name)

        return target

    def sanitize_artefact(self, temp_dir):
        '''Do nothing - implement in descendants'''

    # pylint: disable=unused-argument
    def validate_key(self, key):
        '''Do nothing - implement in descendants'''
        return True

    def status_code(self, code):
        '''Return expanded operation status'''
        return bottle.HTTPResponse(
            body=json.dumps({"status":code}),
            status=code
        )

    def setup_routing(self, application, config):
        '''Set up www request routing'''
        for (path, methods, handler) in config:
            application.route(path.format(self.prefix), methods, handler)

    def get_version(self):
        '''API Version'''
        return {"version":"v1"}

    def get_cache(self):
        '''Get all of cache'''
        return transform_index(self.cache_index)

    def get_cache_item(self, cache_key=None):
        '''Get/Set for a cache item data'''
        if not self.validate_key(cache_key):
            return self.status_code(400)
        try:
            return self.cache_index[cache_key]
        except KeyError:
            return self.status_code(404)
        return self.status_code(400)

    def validate_merge(self, data):
        '''Validate a merge operation'''
        return True

    def set_cache_item(self, cache_key=None):
        '''Get/Set for a cache item data'''
        if not self.validate_key(cache_key):
            return self.status_code(400)
        try:
            data_to_merge = json.loads(bottle.request.body)
        except json.JSONDecodeError:
            return self.status_code(400)
        if not self.validate_merge(data_to_merge):
            return self.status_code(400)
        self.cache_index[cache_key] = self.cache_index[cache_key] | data_to_merge
        self.store_metadata(cache_key)
        return self.cache_index[cache_key]

    def delete_cache_item(self, cache_key=None):
        '''Get/Set for a cache item data'''
        if not self.validate_key(cache_key):
            return self.status_code(400)
        try:
            Path(self.cache_path, cache_key + ".meta").unlink()
            Path(self.cache_path, cache_key + ".tar.gz").unlink()
            del self.cache_index[cache_key]
        except FileNotFoundError:
            return self.status_code(404)
        except PermissionError:
            return self.status_code(403)
        return self.status_code(200)

    def get_cache_archive(self, cache_key=None):
        '''Get/Set for a cache item binary archive
           The binary archive must not contain the group file - it is handled separately.
        '''
        if not self.validate_key(cache_key):
            return self.status_code(400)
        try:
            with open(Path(self.cache_path, cache_key + ".tar.gz").as_posix(),
                      mode="rb") as data_file:
                return bottle.HTTPResponse(
                    body=data_file.read(),
                    status=200,
                    headers={'Content-Type': 'application/x-gzip-compressed-tar'}
                )
        except FileNotFoundError:
            return self.status_code(404)
        except PermissionError:
            return self.status_code(403)
        return self.status_code(400)

    def fetch_metadata(self, cache_key, tmp_dir):
        '''Fetch Metadata for an item - descendants override'''

    def store_metadata(self, cache_key):
        '''Store Metadata for an item - descendants override'''
        with Path(self.cache_path, cache_key + ".meta").open(encoding="ascii", mode="w") as meta:
            json.dump(self.cache_index[cache_key], meta)

    def set_cache_archive(self, cache_key=None):
        '''Get/Set for a cache item binary archive
           The binary archive must not contain the group file - it is handled separately.
        '''
        if not self.validate_key(cache_key):
            return self.status_code(400)
        if bottle.request.files is None or len(bottle.request.files) == 0:
            print(f"no files to put {len(bottle.request.files)}")
            return self.status_code(400)

        with NamedTemporaryFile(dir=self.cache_tmp(),
                                delete=False,
                                delete_on_close=False) as tar_file:
            tar_file.write(bottle.request.files["file"].file.read())
        tar_file.close()
        print(f"Tarfile written as {tar_file.name}")

        with TemporaryDirectory(dir=self.cache_tmp(), delete=False) as temp_dir:
            with tarfile.open(name=tar_file.name, mode="r:gz") as tar:
                tar.extractall(path=temp_dir, filter=tarfile.data_filter)
            os.unlink(tar_file.name)
            self.sanitize_artefact(temp_dir)
            self.fetch_metadata(cache_key, temp_dir)

            # leave any errors to throw a 503 from bottle itself
            self.create_tar(temp_dir).rename(Path(self.cache_path, cache_key + ".tar.gz"))
            self.store_metadata(cache_key)

            shutil.rmtree(temp_dir)

        return self.status_code(200)

class TritonCacheServer(CacheServer):
    '''Triton Cache version'''

    METADATA_NAME = "__grp__add_kernel.json"
    VALID_KEY = re.compile(r"\w{52}")
    CACHE_PATH = "/var/cache/triton"
    key_length = 52
    prefix="/triton"

    def __init__(self, cache_path=None, cache_index=None):
        if cache_path is None:
            cache_path = self.CACHE_PATH
        super().__init__(cache_path, cache_index)

    def sanitize_artefact(self, temp_dir):
        '''Sanitize and verify artefact'''
        # add checks here, for now just rewrite the __grp__add_kernel file
        with Path(temp_dir, self.METADATA_NAME).open(mode="r", encoding="ascii") as meta:
            data = json.load(meta)
        new_paths = {}
        for key, value in data["child_paths"].items():
            new_paths[key] = Path(value).stem
        data["child_paths"] = new_paths
        with Path(temp_dir, self.METADATA_NAME).open(encoding="ascii", mode="w") as new_meta:
            json.dump(data, new_meta)

    def validate_key(self, key):
        '''Check if the key matches the allowed pattern(s)'''
        if len(key) > self.key_length:
            return False
        if self.VALID_KEY.match(key) is None:
            return False
        return True

    def fetch_metadata(self, cache_key, tmp_dir):
        '''Fetch Triton version of metadata'''
        with Path(tmp_dir, self.METADATA_NAME).open(encoding="ascii") as meta:
            self.cache_index[cache_key] = json.load(meta)


    def validate_merge(self, data):
        '''Check if the key matches the allowed pattern(s)'''
        return data.get("child_paths", None) is None


app = bottle.Bottle()
cache_server = TritonCacheServer()
cache_server.setup_routing(app, cache_server.schema)

app.run(host="192.168.2.198", port=7080)
