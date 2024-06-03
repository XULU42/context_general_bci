import numpy as np
from context_general_bci.subjects import SubjectName, SortedArrayInfo, SubjectInfo, SubjectArrayRegistry, GeometricArrayInfo

@SubjectArrayRegistry.register
class Bohr(SubjectInfo):
    name = SubjectName.bohr
    _arrays = {
        'main': SortedArrayInfo(_max_channels=192),  # TODO: get exact geometry info of array
    }
