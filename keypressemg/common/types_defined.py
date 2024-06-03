from enum import Enum


class Participant(Enum):
    P1 = 'P1'
    P2 = 'P2'
    P3 = 'P3'
    P4 = 'P4'
    P5 = 'P5'
    P6 = 'P6'
    P7 = 'P7'
    P8 = 'P8'
    P9 = 'P9'
    P10 = 'P10'
    P11 = 'P11'
    P12 = 'P12'
    P13 = 'P13'
    P14 = 'P14'
    P15 = 'P15'
    P16 = 'P16'
    P17 = 'P17'
    P18 = 'P18'
    P19 = 'P19'

    def to_num(self) -> str:
        return self.value.replace('P', '')


class DayT1T2(Enum):
    T1 = 'T1'
    T2 = 'T2'

    def to_num(self) -> str:
        return self.value.replace('T', '')

    def other_day(self):
        return DayT1T2.T1 if self == DayT1T2.T2 else DayT1T2.T2


class KeyPress(Enum):
    A = 'a'
    B = 'b'
    C = 'c'
    D = 'd'
    E = 'e'
    F = 'f'
    G = 'g'
    H = 'h'
    I = 'i'
    J = 'j'
    K = 'k'
    L = 'l'
    M = 'm'
    N = 'n'
    O = 'o'
    P = 'p'
    Q = 'q'
    R = 'r'
    S = 's'
    T = 't'
    U = 'u'
    V = 'v'
    W = 'w'
    X = 'x'
    Y = 'y'
    Z = 'z'
    SPACE = 'Key.space'

    def to_num(self) -> int:
        _TO_NUM_DICT = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10,
                        'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20,
                        'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, 'Key.space': 26}
        return _TO_NUM_DICT[self.value]

    @staticmethod
    def from_str(s: str):
        key_press = next(filter(lambda k: k.value == s, [k for k in KeyPress]))
        return key_press


if __name__ == '__main__':
    k = KeyPress.from_str('o').to_num()
    print(k)
    k = KeyPress.O.to_num()
    print(k)
