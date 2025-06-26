'''Hierarchical cache support for Triton'''
import json
import os
import random
from typing import Dict, Optional

from enhanced_pathlib import EPath
from triton.runtime.cache import CacheManager, FileCacheManager

class HierarchicalCacheManager(CacheManager):
    '''Hierarchical Path Manager class path element'''
    config = None
    managers = []
    needs_init = True

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
                HierarchicalCacheManager.managers.append(EPathCacheManager(key, conf))

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

        print(f"asking exist for {self.filepath(filename).as_posix()}")

        return self.filepath(filename).exists()

    def get_file(self, filename) -> Optional[str]:
        '''EPath based get_file - returns full path for a file'''
        path = self.filepath(filename)
        print(f"asking get for {path.as_posix()}")
        if path.exists():
            return path.as_posix()
        return None

    def get_group(self, filename: str) -> Optional[Dict[str, str]]:
        '''EPath based get_group'''

        grp = self.filepath(f"__grp__{filename}")

        print(f"asking group for {grp.as_posix()}")

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

class SignedCacheManager(EPathCacheManager):
    '''Signed artifact cache manager using EPath'''
    rsa_key = None

    def filepath(self, filename, **kargs):
        '''Return an EPath object for the filename'''
        epath = super().filepath(filename, **kargs)
        if filename is not None and self.rsa_key is not None:
            sigpath = epath.with_suffix(".sig")
            epath.signature = sigpath.read_bytes()
            epath.key = self.rsa_key
        return epath
