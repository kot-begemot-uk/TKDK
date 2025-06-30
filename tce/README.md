# Triton Cache Extensions (TCE)

Hierarchical Cache Manager for Triton Lang

The Hierarchical Cache manager allows Triton to use multiple cache locations with different properties. All cache is by default READ ONLY. This is by design - it is intended for deployment/rollout of a tested Triton application. It is possible to configure a fallthrough to the default Triton jit compile/cache as a last resourt caching mechanism.

### Example Deployment

To have a verified read-only cache in /var/cache/triton/

1. Put directories with the verified cache items under /var/cache/triton 
1. Rewrite the file names in the group metadata file \_\_grp\_\_add\_kernel.json to point to the new location
1. Write a config file for the hierarchical cache manager
1. Enable the new cache manager in Triton. Up to 3.4 - by setting the TRITON\_CACHE\_MANAGER environment variable. Its expected value is "module:class". The TCE module must be in the module load path. 3.4 and later - by setting the cache manager knob in the configuration.
1. Point the cache manager to the config file by setting the TCE\_CONFIG environment variable.
1. Run Triton with the added new functionality

### Configuration File Format

```json
{
    "cache_managers": [
        {
            "id":"test_manager",
            "cache_dir":"//home/fedora/anivanov/src/test-cache"
        }
    ],
    "fallback":true,
    "debug":true
}
```
1. fallback - enable fallback to default jit compile/per-user cache
1. debug - enable debug output
1. cache\_managers - list of RO cache managers and their cache dirs (keying and compression options will be added in future versions).

---

## Installation

```bash
pip install -e .
```

---

## Quick start

---

## TODO

1. Cryptographic signatures on cache items
1. Network cache/true database backend
1. Cache item compression
1. Packaging scripts to rewrite metadata to new locations


---

## Requirements

- Python ≥ 3.9
- Triton > 3.3.0
