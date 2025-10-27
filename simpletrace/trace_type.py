from collections import UserList

from torch import Tensor


# Define a custom data structure for trace data
# Used in collate function to handle trace data properly
class Trace_Container(UserList):
    pass


class Trace_Data:
    start: Trace_Container
    end: Trace_Container
    pid: Trace_Container
    child: Trace_Container
    args: dict[str, Trace_Container]

    def __init__(self):
        self.start: Trace_Container[float] = Trace_Container()
        self.end: Trace_Container[float] = Trace_Container()
        self.pid: Trace_Container[int] = Trace_Container()
        self.child: Trace_Container[dict[str, "Trace_Data"]] = Trace_Container()
        self.args: dict[str, Trace_Container] = dict()

    def append(
        self,
        start: float,
        end: float,
        pid: int,
        childs: dict,
        args: dict[str, any] = {},
    ):
        self.start.append(start)
        self.end.append(end)
        self.pid.append(pid)
        self.child.append(childs)
        for key, value in args.items():
            if key not in self.args:
                self.args[key] = Trace_Container()
            self.args[key].append(value)


class Collated_Trace_Data:
    def __init__(self, start, end, pid, child, args):
        self.start: Tensor = start
        self.end: Tensor = end
        self.pid: Tensor = pid
        self.child: dict[str, "Collated_Trace_Data"] = child
        self.args: dict[str, Tensor] = args


class Batch_Trace_Data:
    def __init__(self, start, end, pid, tid=-1):
        self.start: float = start
        self.end: float = end
        self.pid: int = pid
        self.tid: int = tid


def check_data(data):
    if (
        isinstance(data, dict)
        and "raw_data" in data
        and ("trace_data" in data or "train_profile_enable" in data)
    ):
        return True
    return False
