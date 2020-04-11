from _ctypes import PyObj_FromPtr
import json
import re
from collections import OrderedDict


class NoIndent(object):
    """ Value wrapper. """

    def __init__(self, value):
        self.value = value


class NoIndentEncoder(json.JSONEncoder):
    """ Encoder for json that allows for a NoIndent wrapper on lists
    
    Based upon the StackOverflow answer: https://stackoverflow.com/a/13252112/200663
    
    Extends:
        json.JSONEncoder
    
    Variables:
        regex {[type]} -- [description]
    """

    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get("sort_keys", None)
        super(NoIndentEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(NoIndentEncoder, self).default(obj)
        )

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(NoIndentEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr
            )

        return json_repr


def read_json(json_loc):
    return json.load(open(json_loc), object_pairs_hook=OrderedDict)
