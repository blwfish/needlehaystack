from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("needlestack")
except PackageNotFoundError:
    __version__ = "unknown"
