import argparse
import os
import re
import time
from pathlib import Path
from typing import Callable, Literal, Sequence, Type, TypeVar, Union, overload

from loguru import logger

T = TypeVar("T")
N1 = TypeVar("N1", bound="Namespace")
N2 = TypeVar("N2", bound="Namespace")
N3 = TypeVar("N3", bound="Namespace")

ENV = os.environ
HTTP_PROXY = ENV.get("HTTP_PROXY") or ENV.get("http_proxy", "")
HTTPS_PROXY = ENV.get("HTTPS_PROXY") or ENV.get("https_proxy", "")
NO_PROXY = "127.0.0.1"  # ENV.get("NO_PROXY") or ENV.get("no_proxy", "")
PORT = 8081

ROOT_DIR = Path(__file__).parents[2]


class Namespace(argparse.Namespace):
    """
    Custom namespace class that offers useful utilities.

    `self.validate()` can be implemented to validate the inputs.
    Inherited class must call `super().validate()` to pass validation downstream.

    `self.separate_into(...)` splits all annotated variables into separate namespace classes.
    """

    def validate(self):
        if hasattr(super(), "validate"):
            super().validate()  # type: ignore

    def as_json_dict(self):
        d = self.__dict__.copy()
        remove_keys = []
        for k, v in d.items():
            if k.startswith("_"):
                remove_keys.append(k)
                continue
            if isinstance(v, Path):
                d[k] = str(v.resolve())
            elif isinstance(v, set) or isinstance(v, list):
                d[k] = sorted(v)

        for k in remove_keys:
            del d[k]
        return d

    @overload
    def separate_into(
        self, namespaces: tuple[Type[N1]], *, warn_unused: bool = True
    ) -> N1: ...
    @overload
    def separate_into(
        self, namespaces: tuple[Type[N1], Type[N2]], *, warn_unused: bool = True
    ) -> tuple[N1, N2]: ...
    @overload
    def separate_into(
        self,
        namespaces: tuple[Type[N1], Type[N2], Type[N3]],
        *,
        warn_unused: bool = True,
    ) -> tuple[N1, N2, N3]: ...
    def separate_into(
        self,
        namespaces: Union[
            tuple[Type[N1]],
            tuple[Type[N1], Type[N2]],
            tuple[Type[N1], Type[N2], Type[N3]],
            tuple[Type["Namespace"], ...],
        ],
        *,
        warn_unused: bool = True,
    ) -> Union["Namespace", tuple["Namespace", ...]]:
        """
        Splits all the variables placed by `parse_args()` in `args` into an instance of each class in `namespaces`.

        @param namespaces: `tuple[Type[Namespace]]` Tuple of namespace types to separate into (Not instances).
        @param warn_unused: `bool` Warn of unused items from the original namespace. Default: True
        @returns: `tuple[Namespace]` Returns a tuple of namespaces with fields populated from the original namespace.

        Example:
        ```python
        class A(Namespace):...
        class B(Namespace):...
        class C(A, B):...
        parser = argparse.ArgumentParser()
        ...
        args = parser.parse_args(namespace=C())
        args_a, args_b = args.separate_into(A, B)
        ```

        NOTE:
            Only the variables explicitly listed
        """
        assert len(namespaces) > 0

        kwarg_namespaces = [{} for _ in range(len(namespaces))]
        for k, v in vars(self).items():
            used = False
            for i, ns in enumerate(namespaces):
                if k in ns._get_annotated_vars():
                    used = True
                    kwarg_namespaces[i][k] = v
            if not used and warn_unused:
                logger.warning(f"Field {k}={v} was parsed but not used")
        if len(namespaces) == 1:
            return namespaces[0](**kwarg_namespaces[0])
        return tuple(ns(**kwargs) for ns, kwargs in zip(namespaces, kwarg_namespaces))

    @classmethod
    def _get_annotated_vars(cls):
        out = set(cls.__annotations__.keys())
        for c in cls.__bases__:
            if hasattr(c, "_annotated_vars"):
                out.update(c._annotated_vars)
        return out


def timed(
    fn=None,
    /,
    *,
    enabled=True,
    units: Literal["s", "ms", "us", "ns"] = "s",
) -> Callable:
    divisors = {"s": 10e9, "ms": 10e6, "us": 10e3, "ns": 1}
    print_name = {
        "s": "seconds",
        "ms": "milli-seconds",
        "us": "micro-seconds",
        "ns": "nano-seconds",
    }
    if units not in divisors.keys():
        raise ValueError(
            f"Units provided {units} is not recognized. Valid units: {list(divisors.keys())}"
        )
    use_ns = units in ["us", "ns"]

    def decorator(fn):
        def time_wrap(*args, **kwargs):
            if not enabled:
                return fn(*args, **kwargs)
            name = f"{fn.__module__}:{fn.__qualname__}"
            logger.info(f"Starting {name}")
            # Save time in nanoseconds
            start_time = (
                time.perf_counter_ns() if use_ns else time.perf_counter() * 10e9
            )
            out = fn(*args, **kwargs)
            end_time = time.perf_counter_ns() if use_ns else time.perf_counter() * 10e9

            logger.info(
                f"Completed {name} in {(end_time - start_time) / divisors[units]:.3f} {print_name[units]}"
            )
            return out

        return time_wrap

    if fn is None:
        return decorator
    return decorator(fn)


def parse_time(time_str: str) -> float:
    """Parses time with format `######[s|ms|µs]` into a floating point time in milliseconds."""
    match = re.match(r"^([0-9]*.?[0-9]*)(.*)", time_str)
    if match is None:
        raise ValueError(f"Could not match time {time_str}")
    value = float(match.group(1))
    units = match.group(2)

    if units == "s":
        return value * 1e3
    if units == "ms":
        return value
    if units == "µs":
        return value / 1e3
    raise ValueError(f"Invalid units {units} for time string {time_str}")


def dict_has_diff(dict1, dict2) -> bool:
    """Determines recursively if two dictionaries have any differences."""
    if len(dict1) != len(dict2):
        logger.error(f"Different dictionary sizes: {len(dict1)} != {len(dict2)}")
        return True
    for k, v in dict1.items():
        if isinstance(v, dict) and isinstance(dict2[k], dict):
            inner_dict_result = dict_has_diff(v, dict2[k])
            if inner_dict_result:
                return True
            continue
        else:
            if v != dict2.get(k):
                logger.warning(f"Different values for {v} and {dict2.get(k)}")
                return True
    return False


def batchify(items: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    num_batches, extra = divmod(len(items), batch_size)
    if extra:
        num_batches += 1
    return [
        items[i * batch_size : min((i + 1) * batch_size, len(items))]
        for i in range(num_batches)
    ]
