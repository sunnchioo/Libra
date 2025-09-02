#pragma once

namespace isecfhe {
    enum PolyFormat { EVALUATION = 0, COEFFICIENT = 1 };

    inline std::ostream& operator<<(std::ostream& s, PolyFormat f) {
        switch (f) {
            case EVALUATION:
                s << "EVALUATION";
                break;
            case COEFFICIENT:
                s << "COEFFICIENT";
                break;
            default:
                s << "UNKNOWN";
                break;
        }
        return s;
    }
}
