import numpy as np


def pad_sequence(seq_diagnosis_codes, maxlen, maxcode):
    diagnosis_codes = np.zeros((maxlen, maxcode), dtype=np.int64)
    seq_mask_code = np.zeros((maxlen, maxcode), dtype=np.float32)
    for pid, subseq in enumerate(seq_diagnosis_codes):
        for tid, code in enumerate(subseq):
            diagnosis_codes[pid, tid] = code
            seq_mask_code[pid, tid] = 1
    return diagnosis_codes, seq_mask_code

def keep_last_one_in_columns(a):
    # 결과 배열 초기화
    result = np.zeros_like(a)
    # 각 열에 대해 반복
    for col_index in range(a.shape[1]):
        # 현재 열 추출
        column = a[:, col_index]
        # 이 열에서 마지막 '1' 찾기
        last_one_idx = np.max(np.where(column == 1)[0]) if 1 in column else None
        if last_one_idx is not None:
            result[last_one_idx, col_index] = 1
    return result