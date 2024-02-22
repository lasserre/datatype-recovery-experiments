from typing import List, Dict

class TypeSequenceProjection:
    '''
    Encapsulates a projection of our very granular type sequences down
    to a more coarse-grained type system
    '''
    def __init__(self, name:str, projection:Dict[int,Dict[str,str]]) -> None:
        self.name = name
        self._projection = projection

    @property
    def projection(self) -> Dict[int, Dict[str, str]]:
        '''
        Returns the projection as a mapping from typeseqlen: {typename: newname}

        Maps seqlen -> type name remapping; for each seqlen, its remap is
        used to convert any keys appearing in the sequence to the values they
        are being remapped to.

        Keys not appearing are left alone.
        Sequence lengths not appearing in projection are left alone.
        '''
        return self._projection

    def project_typesequence(self, tseq:List[str], drop_after_len:int=None) -> List[str]:
        '''
        tseq: Sequence to be converted
        drop_after_len: If specified, sequences longer than this will be truncated to this length before projection

        CLS: gotta be a better way of doing this, but it's getting late and I
        just need it done, so python-smashing my way to victory :)
        '''
        seqlen = len(tseq)

        trunc_seq = tseq[:drop_after_len] if drop_after_len is not None else tseq

        if seqlen in self.projection:
            return [self.projection[seqlen][x] if x in self.projection[seqlen] else x for x in trunc_seq]
        return trunc_seq     # no mapping for this sequence length

class DatasetBalanceProjection(TypeSequenceProjection):
    def __init__(self) -> None:

        projection_1stlevel = {
            'int128': 'int64',      # don't worry about ensuring int128 is in our train set
            'uint128': 'uint64',
            'uint256': 'uint64',
            'uint512': 'uint64',
            # these don't occur in isolation
            'PTR': 'signed_int',
            'ARR': 'signed_int',
            'void': 'signed_int',
        }

        projection_2ndlevel = {
            'long double': 'double',
            'char': 'signed_int',
            'short': 'signed_int',
            'int32': 'signed_int',
            'int64': 'signed_int',
            'int128': 'signed_int',
            'uchar': 'unsigned_int',
            'ushort': 'unsigned_int',
            'uint32': 'unsigned_int',
            'uint64': 'unsigned_int',
            'uint128': 'unsigned_int',
            'uint256': 'unsigned_int',
            'uint512': 'unsigned_int',
            'UNION': 'STRUCT',  # group in with struct
            'ENUM': 'unsigned_int',
        }

        projection = {
            1: projection_1stlevel,
            2: projection_2ndlevel,
        }

        super().__init__('DatasetBalance', projection)

