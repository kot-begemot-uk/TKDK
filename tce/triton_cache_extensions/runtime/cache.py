'''Hierarchical cache support for Triton'''
import json
import os
import random
import tarfile
from typing import Dict, Optional
from tempfile import TemporaryDirectory, NamedTemporaryFile
import requests_openapi

from enhanced_pathlib import EPath
from triton.runtime.cache import CacheManager, FileCacheManager
try:
    from Crypto.PublicKey import RSA
except ModuleNotFoundError:
    from Cryptodome.PublicKey import RSA

class HierarchicalCacheManager(CacheManager):
    '''Hierarchical Path Manager class path element'''
    config = None
    managers = []
    needs_init = True
    rsa_keys = []

    def __init__(self, key,  override=False, dump=False):
        # dump and override are ignored - they are here for compatibility
        self.key = key
        if self.needs_init:
            try:
                with open(os.environ["TCE_CONFIG"],"r", encoding="ascii") as tce_config:
                    HierarchicalCacheManager.config = json.load(tce_config)
            except (
                    json.JSONDecodeError,OSError, IOError,
                    PermissionError, FileNotFoundError) as exc:
                raise RuntimeError("Could not open and parse Cache Manager configuration") from exc

            for conf in HierarchicalCacheManager.config["cache_managers"]:
                HierarchicalCacheManager.managers.append(CACHE_TYPES[conf["type"]](key, conf))
                try:
                    with open(conf["rsa_key"], mode="rb") as keyfile:
                        rsa_key = RSA.import_key(keyfile.read())
                        HierarchicalCacheManager.rsa_keys.append(rsa_key)
                        HierarchicalCacheManager.managers[-1].set_key(rsa_key)
                except KeyError:
                    HierarchicalCacheManager.rsa_keys.append(None)
            if HierarchicalCacheManager.config["fallback"]:
                HierarchicalCacheManager.managers.append(
                    FileCacheManager(key, override=override, dump=dump))
            HierarchicalCacheManager.needs_init = False

    def has_file(self, filename) -> bool:
        '''Has file'''
        index = 0
        for manager in HierarchicalCacheManager.managers:
            result = manager.has_file(filename)
            if HierarchicalCacheManager.config["debug"]:
                print(f"has file result from manager {index} {result}")
                index = index + 1
            if result:
                return result
        return False

    def get_file(self, filename) -> Optional[str]:
        '''Return filename from first manager in the list which agrees to handle it'''
        index = 0
        for manager in HierarchicalCacheManager.managers:
            result = manager.get_file(filename)
            if HierarchicalCacheManager.config["debug"]:
                print(f"get file result from manager {index} {result}")
                index = index + 1
            if result is not None:
                return result
        return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        '''Return group from first manager in the list which agrees to handle it'''
        index = 0
        for manager in HierarchicalCacheManager.managers:
            result = manager.get_group(filename)
            if HierarchicalCacheManager.config["debug"]:
                print(f"get group result from manager {index} {result}")
                index = index + 1
            if result is not None:
                return result
        return None

    # Note a group of pushed files as being part of a group
    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        '''Put group'''
        if HierarchicalCacheManager.config["fallback"]:
            # Only the fallback manager of last resort is expected to be able to do write ops
            # all others are read-only
            return HierarchicalCacheManager.managers[-1].put_group(filename, group)
        raise RuntimeError("No fallback manager, cache put is not supported")


    def put(self, data, filename, binary=True) -> str:
        '''Put file'''
        if HierarchicalCacheManager.config["fallback"]:
            # Ditto - only the fallback manager is expected to be able to do put
            return HierarchicalCacheManager.managers[-1].put(data, filename, binary)
        raise RuntimeError("No fallback manager, cache put is not supported")



class EPathCacheManager(CacheManager):
    '''Cache manager using EPath, base class for compressed/signed cache''' 

    def __init__(self, key, config):
        self.key = key
        self.config = config

    def filepath(self, filename, **kargs):
        '''Return an EPath object for the filename'''
        if filename is not None:
            return EPath(self.config["cache_dir"], self.key, filename, **kargs)
        return EPath(self.config["cache_dir"], self.key, **kargs)

    def has_file(self, filename) -> bool:
        '''EPath based has_file - checks if file exists'''
        return self.filepath(filename).exists()

    def get_file(self, filename) -> Optional[str]:
        '''EPath based get_file - returns full path for a file'''
        path = self.filepath(filename)
        if path.exists():
            return path.as_posix()
        return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        '''EPath based get_group'''

        grp = self.filepath(f"__grp__{filename}")
        if not grp.exists():
            return None
        grp_data = json.loads(grp.read_text())
        try:
            result = {}
            for code, path in grp_data["child_paths"].items():
                if os.path.exists(path):
                    result[code] = path
            return result
        except KeyError:
            return None

    # Note a group of pushed files as being part of a group
    def put_group(self, filename: str, group: Dict[str, str]) -> str:
        '''EPath based put_group'''
        return self.put(json.dumps({"child_paths": group}), f"__grp__{filename}", binary=False)

    def put(self, data, filename, binary=True) -> str:
        '''Atomic put file. Create all dirs if needed'''
        os.makedirs(str(self.filepath(None)), exist_ok=True)
        binary = binary or isinstance(data, bytes)
        path = self.filepath(f"{filename}.temp-{os.getpid()}-{int(random.uniform(0, 32768))}")
        if binary:
            path.write_bytes(data)
        else:
            path.write_text(str(data))
        filepath = self.filepath(filename)
        os.rename(str(path), str(filepath))
        return str(filepath)

TARNAME = "{}/{}.tar.gz"

class WebClientCacheManager(EPathCacheManager):
    '''Cache which fetches cache entries from a configured web server.
       Note - this class does not perform web server submission.
    '''

    IDL = None
    METADATA_NAME = "__grp__add_kernel.json"

    def __init__(self, key, config):
        super().__init__(key, config)
        if WebClientCacheManager.IDL is None:
            WebClientCacheManager.IDL = requests_openapi.Client().load_spec_from_file(config["openapi"])
            WebClientCacheManager.IDL.set_server(requests_openapi.Server(url=config["url"]))
        self.fetch_if_needed()

    def fetch_if_needed(self):
        '''Fetch a cache entry from a web server if it is not present locally'''
        path = EPath(self.config["cache_dir"], self.key)
        if path.is_dir():
            return
        # we do not want to use with on temp_dir as it is intended to stay after we are done
        # and we want to handle the -EEXISTS
        # pylint: disable=consider-using-with

        temp_dir = TemporaryDirectory(dir=self.config["cache_dir"], delete=False)
        with NamedTemporaryFile(dir="/tmp", delete=False, delete_on_close=False) as temp_tar:
            resp = self.IDL.get_binary_artefacts(name=self.key)
            if resp.status_code == 200:
                for data in resp.iter_content(chunk_size=2048):
                    temp_tar.write(data)
                temp_tar.close()
                tar = tarfile.open(name=temp_tar.name, mode="r:gz")
                tar.extractall(path=temp_dir.name, filter=tarfile.data_filter)
                self.update_group_file(temp_dir.name)
                try:
                    os.rename(temp_dir.name, path.as_posix())
                except FileExistsError:
                    pass
            os.unlink(temp_tar.name)

    def update_group_file(self, temp_dir):
        '''Update group file'''
        with EPath(temp_dir, self.METADATA_NAME).open(mode="r", encoding="ascii") as meta:
            data = json.load(meta)
        new_paths = {}
        for key, value in data["child_paths"].items():
            new_paths[key] = EPath(self.config["cache_dir"], self.key, value).name
        data["child_paths"] = new_paths
        with EPath(temp_dir, self.METADATA_NAME).open(encoding="ascii", mode="w") as new_meta:
            json.dump(data, new_meta)

class SignedCacheManager(EPathCacheManager):
    '''Signed artifact cache manager using EPath'''

    def __init__(self, key, config, rsa_key=None):
        super().__init__(key, config)
        self.rsa_key = rsa_key

    def set_key(self, rsa_key):
        '''Set RSA Key'''
        self.rsa_key = rsa_key

    def _verify_group(self, group):
        '''Check if all files in the group have valid signatures'''
        if group is None:
            return
        for filename in group.values():
            EPath(
                filename,
                signed='sha384',
                key = self.rsa_key,
                signature=EPath(filename + ".sig").read_bytes()
            ).read_bytes()

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        '''EPath based get_group with integrated verify'''
        result = super().get_group(filename)
        self._verify_group(result)
        return result

CACHE_TYPES = {
    "file":EPathCacheManager,
    "sig":SignedCacheManager,
    "web":WebClientCacheManager
}
