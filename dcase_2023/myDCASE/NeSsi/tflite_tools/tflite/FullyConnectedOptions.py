# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class FullyConnectedOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFullyConnectedOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FullyConnectedOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def FullyConnectedOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # FullyConnectedOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # FullyConnectedOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # FullyConnectedOptions
    def WeightsFormat(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # FullyConnectedOptions
    def KeepNumDims(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def FullyConnectedOptionsStart(builder): builder.StartObject(3)
def FullyConnectedOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def FullyConnectedOptionsAddWeightsFormat(builder, weightsFormat): builder.PrependInt8Slot(1, weightsFormat, 0)
def FullyConnectedOptionsAddKeepNumDims(builder, keepNumDims): builder.PrependBoolSlot(2, keepNumDims, 0)
def FullyConnectedOptionsEnd(builder): return builder.EndObject()