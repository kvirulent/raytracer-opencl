#include "fetch.h"

std::string fetch_src(std::string kernel_name) {
    std::fstream f;
    f.open(std::string(KERNELS_DIR) + "/" + kernel_name + ".cl");
    if(!f.is_open()) {
        throw std::runtime_error("Failed to open kernel file.");
    }

    std::string res;
    char c;
    while (f.get(c)) {
        res += c;
    }

    f.close();
    return res;
}