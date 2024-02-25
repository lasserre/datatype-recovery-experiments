from typing import List, Dict

class TypeSequenceProjection:
    '''
    Encapsulates a projection of our very granular type sequences down
    to a more coarse-grained type system
    '''
    def __init__(self, name:str, hier_projection:Dict[int,Dict[str,str]], final_remaps:Dict[str,str]) -> None:
        self.name = name
        self._hier_projection = hier_projection
        self._final_remaps = final_remaps

    @property
    def hier_projection(self) -> Dict[int, Dict[str, str]]:
        '''
        Returns the projection as a mapping from typeseqlen: {typename: newname}

        Maps seqlen -> type name remapping; for each seqlen, its remap is
        used to convert any keys appearing at that level in the sequence to the values they
        are being remapped to.

        Keys not appearing are left alone.
        Sequence lengths not appearing in projection are left alone.
        '''
        return self._hier_projection

    @property
    def final_remaps(self) -> Dict[str,str]:
        '''
        Maps comma-joined type sequence strings to their projected versions, but
        is applied after the hierarchical projection
        '''
        return self._final_remaps

    def project_typesequence(self, tseq:List[str], drop_after_len:int=None) -> List[str]:
        '''
        tseq: Sequence to be converted
        drop_after_len: If specified, sequences longer than this will be truncated to this length before projection

        CLS: gotta be a better way of doing this, but it's getting late and I
        just need it done, so python-smashing my way to victory :)
        '''
        trunc_seq = tseq[:drop_after_len] if drop_after_len is not None else tseq

        seqlen = len(trunc_seq)
        out_seq = []
        for i in range(seqlen):
            if i+1 in self.hier_projection:
                level_proj = self.hier_projection[i+1]
                if trunc_seq[i] in level_proj:
                    out_seq.append(level_proj[trunc_seq[i]])
                    continue
            out_seq.append(trunc_seq[i])

        if self.final_remaps:
            out_csv = ','.join(out_seq)
            if out_csv in self.final_remaps:
                return self.final_remaps[out_csv].split(',')

        return out_seq

        # return [self.hier_projection[i+1][trunc_seq[i]] if i+1 in self.hier_projection and trunc_seq[i] in self.hier_projection[i+1] else trunc_seq[i] for i in range(seqlen)]

        # if seqlen in self.projection:
        #     return [self.projection[seqlen][x] if x in self.projection[seqlen] else x for x in trunc_seq]
        # return trunc_seq     # no mapping for this sequence length

class DatasetBalanceProjection(TypeSequenceProjection):
    def __init__(self) -> None:

        projection_1stlevel = {
            'int128': 'int64',      # don't worry about ensuring int128 is in our train set
            'uint128': 'uint64',
            'uint256': 'uint64',
            'uint512': 'uint64',
        }

        projection_2ndlevel = {
            'float': 'floating',
            'double': 'floating',
            'long double': 'floating',
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
            # 'STRUCT': 'Other',
            'UNION': 'Other',
            'ENUM': 'Other',
            # 'PTR': 'Other',
            'ARR': 'Other',
        }

        hier_projection = {
            1: projection_1stlevel,
            2: projection_2ndlevel,
        }

        final_remaps = {
            # 'ARR,floating': 'ARR,Other',
            # 'ARR,PTR': 'ARR,Other',
            # 'ARR,STRUCT': 'ARR,Other'
        }

        super().__init__('DatasetBalance', hier_projection, final_remaps)

