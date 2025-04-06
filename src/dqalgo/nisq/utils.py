
def get_register_counts(counts: dict[str, int], creg_sizes: list[int], target_reg_name: str,
                        reg_names: list[str]) -> dict[str, int]:
    """If the circuit has multiple target registers, this function can be used to get the counts of a specific target register.

    Args:
        counts (dict[str, int]): the counts of the circuit, e.g. {'0000000': 100, '0000001': 100, '0000010': 100, '0000011': 100, '0000100': 100, '0000101': 100, '0000110': 100, '0000111': 100}
        reg_names (list[str]): the names of the classical registers, e.g. ['a', 't']
        creg_sizes (list[int]): the sizes of the corresponding classical registers of `reg_names`, e.g. [3, 4]
        target_reg_name (str): the name of the register that we want to get the counts of, e.g. 't'

    Returns:
        dict[str, int]: the counts of the target register; the rightmost bit is the least significant bit (0-th bit)
    """
    idx = reg_names.index(target_reg_name)
    start = sum(creg_sizes[:idx])
    end = start + creg_sizes[idx]

    reg_counts = {}
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")  # remove spaces
        bitstring = bitstring[::-1]  # little endian
        reg_bits = bitstring[start:end][::-1]
        reg_counts[reg_bits] = reg_counts.get(reg_bits, 0) + count
    return reg_counts
