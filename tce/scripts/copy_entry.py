'''
    Script to move an entry from user cache to a dedicated cache directory
    used by the hierarchical cache manager.
'''
import json
from argparse import ArgumentParser
from pathlib import Path

try:
    from Crypto.PublicKey import RSA
    from Crypto.Signature import pkcs1_15
    from Crypto.Hash import SHA384
except ModuleNotFoundError:
    from Cryptodome.PublicKey import RSA
    from Cryptodome.Signature import pkcs1_15
    from Cryptodome.Hash import SHA384

def load_rsa_key(keyfile_name):
    '''Load signature key'''
    return RSA.import_key(Path(keyfile_name).read_bytes())

def sign_data(data, key):
    '''Sign data with key, return signature using SHA384'''
    sig = pkcs1_15.new(key)
    return sig.sign(SHA384.new(data))

def copy_entry(source, dest, rsa_key):
    '''Copy files from source to dest and adjust the contensts of the
       group file
    '''
    source_path = Path(source).resolve()
    dest_path = Path(dest, source_path.name).resolve()
    dest_path.mkdir(parents=True)
    if not (source_path.is_dir() and dest_path.is_dir()):
        raise RuntimeError("Source and Destination must be directories")

    group_file = None
    for source_file in source_path.iterdir():
        if source_file.name.startswith("__grp__") and \
           source_file.name.endswith(".json"):
            group_file = source_file
        else:
            data = source_file.read_bytes()
            if rsa_key is not None:
                Path(dest_path, source_file.name + ".sig").write_bytes(
                   sign_data(data, rsa_key)
                )
            Path(dest_path, source_file.name).write_bytes(data)

    grp = json.loads(group_file.read_text())

    new_grp = {}
    for key, value in grp["child_paths"].items():
        new_grp[key] = Path(dest_path, Path(value).name).as_posix()

    Path(dest, *group_file.parts[-2:]).write_text(
            json.dumps({"child_paths":new_grp}), encoding="ascii")

def main():
    '''Copy one or more directory to a destination cache adjusting
       group files and/or compressing/signing them in the process
    '''

    aparser = ArgumentParser(description=main.__doc__)
    aparser.add_argument(
        'src',
        help='cache entries',
        nargs="+")

    aparser.add_argument(
        'dst',
        help='destination')

    aparser.add_argument(
        '--sign-with',
        help='sign with key',
        type=str)

    args = vars(aparser.parse_args())

    if len(args["src"]) < 1 or len(args["dst"]) < 1:
        raise RuntimeError("Ivalid arguments need source(s) and destination")

    rsa_key = None
    try:
        rsa_key = load_rsa_key(args["sign_with"])
    except KeyError:
        pass

    for source in args["src"]:
        copy_entry(source, args["dst"], rsa_key)
        # sign

if __name__ == "__main__":
    main()
